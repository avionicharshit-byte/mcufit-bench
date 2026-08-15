// Times a TFLite Micro inference on real silicon and prints one JSON record.

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
constexpr size_t kArenaBytes = 136 * 1024;
constexpr int kWarmupRuns = 3;
constexpr int kTimedRuns = 30;

alignas(16) uint8_t g_arena[kArenaBytes];

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
#elif defined(CONFIG_COMPILER_OPTIMIZATION_DEBUG)
constexpr const char* kOptLevel = "-Og";
#elif defined(CONFIG_COMPILER_OPTIMIZATION_NONE)
constexpr const char* kOptLevel = "-O0";
#else
constexpr const char* kOptLevel = "unknown";
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

int64_t percentile(std::vector<int64_t> sorted, double p) {
  const size_t index = static_cast<size_t>(p * (sorted.size() - 1));
  return sorted[index];
}

}  // namespace

extern "C" void app_main(void) {
  esp_chip_info_t chip{};
  esp_chip_info(&chip);

  MicroPrintf("");
  MicroPrintf("mcufit-bench: %s, %d core(s), rev %d, %d MHz, kernels=%s, %s",
              chip_name(chip), chip.cores, chip.revision,
              CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ, kKernels, kOptLevel);

  const tflite::Model* model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("FATAL: schema %lu, expected %d",
                static_cast<unsigned long>(model->version()), TFLITE_SCHEMA_VERSION);
    return;
  }

  // The five operators person_detect uses.
  tflite::MicroMutableOpResolver<5> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddAveragePool2D();
  resolver.AddReshape();
  resolver.AddSoftmax();

  tflite::MicroInterpreter interpreter(model, resolver, g_arena, kArenaBytes);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    MicroPrintf("FATAL: AllocateTensors failed, arena of %u B is too small",
                static_cast<unsigned>(kArenaBytes));
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
  for (int i = 0; i < kTimedRuns; ++i) {
    const int64_t start = esp_timer_get_time();
    const TfLiteStatus status = interpreter.Invoke();
    const int64_t elapsed = esp_timer_get_time() - start;
    if (status != kTfLiteOk) {
      MicroPrintf("FATAL: Invoke failed on run %d", i);
      return;
    }
    samples.push_back(elapsed);
    // Yield or the task watchdog fires on long inferences.
    vTaskDelay(pdMS_TO_TICKS(10));
  }

  std::sort(samples.begin(), samples.end());
  const int64_t total = [&] {
    int64_t sum = 0;
    for (int64_t s : samples) sum += s;
    return sum;
  }();

  // One JSON line, ready to append to results/results.jsonl.
  printf(
      "\nMCUFIT_RESULT "
      "{\"schema\":1,"
      "\"model\":\"person_detect.tflite\","
      "\"model_sha256\":\"%s\","
      "\"model_bytes\":%u,"
      "\"target\":\"%s\","
      "\"chip\":\"%s\","
      "\"chip_revision\":%d,"
      "\"cores\":%d,"
      "\"cpu_mhz\":%d,"
      "\"kernels\":\"%s\","
      "\"opt_level\":\"%s\","
      "\"idf_version\":\"%s\","
      "\"arena_used_bytes\":%u,"
      "\"free_heap_bytes\":%u,"
      "\"runs\":%d,"
      "\"min_us\":%" PRId64 ","
      "\"p50_us\":%" PRId64 ","
      "\"mean_us\":%" PRId64 ","
      "\"max_us\":%" PRId64 "}\n",
      g_model_sha256, g_model_len, CONFIG_IDF_TARGET, chip_name(chip),
      chip.revision, chip.cores, CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ, kKernels,
      kOptLevel, esp_get_idf_version(), static_cast<unsigned>(arena_used),
      static_cast<unsigned>(esp_get_free_heap_size()), kTimedRuns,
      samples.front(), percentile(samples, 0.5), total / kTimedRuns,
      samples.back());

  MicroPrintf("");
  MicroPrintf("median %lld ms per inference (%s)",
              static_cast<long long>(percentile(samples, 0.5) / 1000), kKernels);
  MicroPrintf("arena actually used: %u B", static_cast<unsigned>(arena_used));
  MicroPrintf("done. copy the MCUFIT_RESULT line above.");
}
