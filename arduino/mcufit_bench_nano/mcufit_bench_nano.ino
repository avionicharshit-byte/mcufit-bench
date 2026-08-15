// Times a TFLite Micro inference on a Nano 33 BLE and prints one JSON record.
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

tflite::MicroInterpreter* g_interpreter = nullptr;

void sortAscending(uint32_t* v, int n) {
  for (int i = 1; i < n; ++i) {
    uint32_t key = v[i];
    int j = i - 1;
    while (j >= 0 && v[j] > key) { v[j + 1] = v[j]; --j; }
    v[j + 1] = key;
  }
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

  const tflite::Model* model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("FATAL: schema mismatch");
    return;
  }

  // The five operators person_detect uses.
  static tflite::MicroMutableOpResolver<5> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddAveragePool2D();
  resolver.AddReshape();
  resolver.AddSoftmax();

  static tflite::MicroInterpreter interpreter(model, resolver, g_arena, kArenaBytes);
  g_interpreter = &interpreter;

  if (interpreter.AllocateTensors() != kTfLiteOk) {
    Serial.print("FATAL: AllocateTensors failed with arena ");
    Serial.println((unsigned)kArenaBytes);
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
      Serial.print("FATAL: Invoke failed on run ");
      Serial.println(i);
      return;
    }
    samples[i] = elapsed;
    total += elapsed;
  }
  sortAscending(samples, kTimedRuns);

  // One JSON line, ready to append to results/results.jsonl.
  Serial.print("\nMCUFIT_RESULT {\"schema\":1,\"model\":\"person_detect.tflite\"");
  Serial.print(",\"model_sha256\":\""); Serial.print(g_model_sha256); Serial.print("\"");
  Serial.print(",\"model_bytes\":"); Serial.print(g_model_len);
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

  Serial.print("\nmedian "); Serial.print(samples[kTimedRuns / 2] / 1000);
  Serial.println(" ms per inference");
  Serial.print("arena actually used: "); Serial.println((unsigned)arenaUsed);
  Serial.println("done. copy the MCUFIT_RESULT line above.");
}

void loop() {}
