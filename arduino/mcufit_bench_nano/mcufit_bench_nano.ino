// Times TFLite Micro inference on a Nano 33 BLE, one JSON record per model.
// Same measurement as main/main.cc, different chip.

#include <Chirale_TensorFlowLite.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"

namespace {

constexpr size_t kArenaBytes = 100 * 1024;
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

void sortAscending(uint32_t* v, int n) {
  for (int i = 1; i < n; ++i) {
    uint32_t key = v[i];
    int j = i - 1;
    while (j >= 0 && v[j] > key) { v[j + 1] = v[j]; --j; }
    v[j + 1] = key;
  }
}

void runOne(const Model& m) {
  const tflite::Model* model = tflite::GetModel(m.data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("SKIP schema "); Serial.println(m.name);
    return;
  }

  // Union of the operators across all four models.
  static tflite::MicroMutableOpResolver<7> resolver;
  static bool built = false;
  if (!built) {
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddFullyConnected();
    resolver.AddAveragePool2D();
    resolver.AddAdd();
    resolver.AddReshape();
    resolver.AddSoftmax();
    built = true;
  }

  tflite::MicroInterpreter interpreter(model, resolver, g_arena, kArenaBytes);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    Serial.print("SKIP alloc "); Serial.println(m.name);
    return;
  }

  const size_t arenaUsed = interpreter.arena_used_bytes();
  TfLiteTensor* input = interpreter.input(0);
  // Content does not change the work done, only the answer.
  memset(input->data.data, 0, input->bytes);

  for (int i = 0; i < kWarmupRuns; ++i) interpreter.Invoke();

  static uint32_t samples[kTimedRuns];
  uint64_t total = 0;
  for (int i = 0; i < kTimedRuns; ++i) {
    const uint32_t start = micros();
    const TfLiteStatus status = interpreter.Invoke();
    const uint32_t elapsed = micros() - start;
    if (status != kTfLiteOk) {
      Serial.print("SKIP invoke "); Serial.println(m.name);
      return;
    }
    samples[i] = elapsed;
    total += elapsed;
  }
  sortAscending(samples, kTimedRuns);

  Serial.print("\nMCUFIT_RESULT {\"schema\":1,\"model\":\""); Serial.print(m.name);
  Serial.print("\",\"model_sha256\":\""); Serial.print(m.sha256);
  Serial.print("\",\"model_bytes\":"); Serial.print(m.len);
  Serial.print(",\"target\":\"nano33ble\",\"chip\":\"nRF52840\",\"cores\":1");
  Serial.print(",\"cpu_mhz\":64,\"kernels\":\"cmsis-nn-or-reference\"");
  Serial.print(",\"opt_level\":\"arduino-default\"");
  Serial.print(",\"arena_used_bytes\":"); Serial.print((unsigned)arenaUsed);
  Serial.print(",\"runs\":"); Serial.print(kTimedRuns);
  Serial.print(",\"min_us\":"); Serial.print(samples[0]);
  Serial.print(",\"p50_us\":"); Serial.print(samples[kTimedRuns / 2]);
  Serial.print(",\"mean_us\":"); Serial.print((uint32_t)(total / kTimedRuns));
  Serial.print(",\"max_us\":"); Serial.print(samples[kTimedRuns - 1]);
  Serial.println("}");

  Serial.print(m.name); Serial.print(": median ");
  Serial.print(samples[kTimedRuns / 2] / 1000);
  Serial.print(" ms, arena "); Serial.println((unsigned)arenaUsed);
}

}  // namespace

void setup() {
  Serial.begin(115200);
  // Native USB: block until the host opens the port, or the run is printed
  // to nobody and lost.
  while (!Serial) {}
  delay(200);

  Serial.println();
  Serial.println("mcufit-bench: nRF52840 (Nano 33 BLE), 64 MHz");

  for (const Model& m : kModels) runOne(m);

  Serial.println();
  Serial.println("done. 4 models.");
}

void loop() {}
