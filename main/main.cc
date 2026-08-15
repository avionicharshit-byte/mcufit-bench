// Times TFLite Micro inference on real silicon, one JSON record per model.

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "esp_chip_info.h"
#include "esp_heap_caps.h"
#include "esp_idf_version.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"

namespace {

// Generous: an allocation failure should mean something real, not a small arena.
constexpr size_t kArenaBytes = 160 * 1024;
constexpr int kWarmupRuns = 3;
constexpr int kTimedRuns = 30;

alignas(16) uint8_t g_arena[kArenaBytes];

struct Model {
  const char* name;
  const unsigned char* data;
  unsigned int len;
  const char* sha256;
};

const Model kModels[] = {
    {"person_detect.tflite", g_person_detect, g_person_detect_len, g_person_detect_sha256},
    {"kws_ref_model.tflite", g_kws, g_kws_len, g_kws_sha256},
    {"pretrainedResnet_quant.tflite", g_ic_resnet, g_ic_resnet_len, g_ic_resnet_sha256},
    {"ad01_int8.tflite", g_ad, g_ad_len, g_ad_sha256},
};

#if defined(CONFIG_NN_OPTIMIZED)
constexpr const char* kKernels = "esp-nn-optimized";
#elif defined(CONFIG_NN_ANSI_C)
constexpr const char* kKernels = "ansi-c-reference";
#else
constexpr const char* kKernels = "unknown";
#endif

#if defined(CONFIG_COMPILER_OPTIMIZATION_PERF)
constexpr const char* kOptLevel = "-O2";
#elif defined(CONFIG_COMPILER_OPTIMIZATION_SIZE)
constexpr const char* kOptLevel = "-Os";
#else
constexpr const char* kOptLevel = "other";
#endif

const char* chip_name(const esp_chip_info_t& info) {
  switch (info.model) {
    case CHIP_ESP32:    return "ESP32";
    case CHIP_ESP32S2:  return "ESP32-S2";
    case CHIP_ESP32S3:  return "ESP32-S3";
    case CHIP_ESP32C3:  return "ESP32-C3";
    case CHIP_ESP32C6:  return "ESP32-C6";
    default:            return "unknown";
  }
}

void run_one(const Model& m, const esp_chip_info_t& chip) {
  const tflite::Model* model = tflite::GetModel(m.data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("SKIP %s: schema mismatch", m.name);
    return;
  }

  // Union of the operators across all four models.
  tflite::MicroMutableOpResolver<7> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddFullyConnected();
  resolver.AddAveragePool2D();
  resolver.AddAdd();
  resolver.AddReshape();
  resolver.AddSoftmax();

  tflite::MicroInterpreter interpreter(model, resolver, g_arena, kArenaBytes);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    MicroPrintf("SKIP %s: AllocateTensors failed", m.name);
    return;
  }

  const size_t arena_used = interpreter.arena_used_bytes();
  TfLiteTensor* input = interpreter.input(0);
  // Content does not change the work done, only the answer.
  std::memset(input->data.data, 0, input->bytes);

  for (int i = 0; i < kWarmupRuns; ++i) {
    interpreter.Invoke();
    vTaskDelay(pdMS_TO_TICKS(10));
  }

  std::vector<int64_t> samples;
  samples.reserve(kTimedRuns);
  int64_t total = 0;
  for (int i = 0; i < kTimedRuns; ++i) {
    const int64_t start = esp_timer_get_time();
    const TfLiteStatus status = interpreter.Invoke();
    const int64_t elapsed = esp_timer_get_time() - start;
    if (status != kTfLiteOk) {
      MicroPrintf("SKIP %s: Invoke failed on run %d", m.name, i);
      return;
    }
    samples.push_back(elapsed);
    total += elapsed;
    // Yield or the task watchdog fires on long inferences.
    vTaskDelay(pdMS_TO_TICKS(10));
  }
  std::sort(samples.begin(), samples.end());

  printf(
      "\nMCUFIT_RESULT "
      "{\"schema\":1,\"model\":\"%s\",\"model_sha256\":\"%s\",\"model_bytes\":%u,"
      "\"target\":\"%s\",\"chip\":\"%s\",\"chip_revision\":%d,\"cores\":%d,"
      "\"cpu_mhz\":%d,\"kernels\":\"%s\",\"opt_level\":\"%s\",\"idf_version\":\"%s\","
      "\"arena_used_bytes\":%u,\"runs\":%d,\"min_us\":%" PRId64 ",\"p50_us\":%" PRId64
      ",\"mean_us\":%" PRId64 ",\"max_us\":%" PRId64 "}\n",
      m.name, m.sha256, m.len, CONFIG_IDF_TARGET, chip_name(chip), chip.revision,
      chip.cores, CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ, kKernels, kOptLevel,
      esp_get_idf_version(), static_cast<unsigned>(arena_used), kTimedRuns,
      samples.front(), samples[kTimedRuns / 2], total / kTimedRuns, samples.back());

  MicroPrintf("%s: median %lld ms, arena %u B", m.name,
              static_cast<long long>(samples[kTimedRuns / 2] / 1000),
              static_cast<unsigned>(arena_used));
}

}  // namespace

extern "C" void app_main(void) {
  esp_chip_info_t chip{};
  esp_chip_info(&chip);

  MicroPrintf("");
  MicroPrintf("mcufit-bench: %s, %d core(s), rev %d, %d MHz, kernels=%s, %s",
              chip_name(chip), chip.cores, chip.revision,
              CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ, kKernels, kOptLevel);

  for (const Model& m : kModels) run_one(m, chip);

  MicroPrintf("");
  MicroPrintf("done. %d models.", (int)(sizeof(kModels) / sizeof(kModels[0])));
}
