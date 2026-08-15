<div align="center">

# ⏱️ mcufit-bench

**What TensorFlow Lite Micro actually costs on real microcontrollers, per model and per layer.**

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![build](https://github.com/avionicharshit-byte/mcufit-bench/actions/workflows/build.yml/badge.svg)](https://github.com/avionicharshit-byte/mcufit-bench/actions/workflows/build.yml)
![boards](https://img.shields.io/badge/boards-3-blue)
![models](https://img.shields.io/badge/models-4-blue)
![measurements](https://img.shields.io/badge/measurements-54-blue)

Ground truth for [mcufit](https://github.com/avionicharshit-byte/mcufit).
Every number here came off a physical chip.

</div>

---

## the finding

**Each chip is slow at a different operator, and nothing tells you which.**

Throughput in MACs per clock cycle, measured layer by layer:

| operator | ESP32 (esp-nn) | Nano 33 BLE (CMSIS-NN) |
| --- | --- | --- |
| `CONV_2D` | 0.073 | 0.190 |
| `DEPTHWISE_CONV_2D` | 0.044 | 0.063 |
| `FULLY_CONNECTED` | **0.022** | 0.186 |

Relative to its own convolution, the ESP32 is **3.2x worse at fully-connected**
and 1.6x worse at depthwise. The nRF52840 is the other way round: 1.14x at
fully-connected and **3.3x worse at depthwise**.

esp-nn ships optimised convolution and depthwise for the Xtensa LX6 but no
fully-connected kernel, so those layers fall back to reference C. CMSIS-NN
covers fully-connected and is comparatively weak on depthwise. Neither
toolchain warns you, and neither publishes this.

So a chip cannot be ranked without knowing the model, and a model cannot be
costed without measuring the chip. The Nano beats the ESP32 by 2.7x on
depthwise-heavy person detection and by 8.4x on the all-fully-connected
anomaly detector.

**Small layers also cost a flat fee.** On the ESP32, `SOFTMAX` over 2 values
takes 385 µs and `AVERAGE_POOL_2D` 415 µs per call, independent of size.
Invisible to any estimate based on operation counts.

## measurements

Median of 30 timed inferences after 3 warmups, `Invoke()` only.

| model | operators | MACs | ESP32 @ 240 MHz | Nano 33 BLE @ 64 MHz |
| --- | --- | --- | --- | --- |
| person_detect | depthwise-heavy | 7.16 M | 474 ms | 657 ms |
| kws_ref_model | conv + depthwise | 2.66 M | 162 ms | 225 ms |
| pretrainedResnet_quant | pure conv + add | 12.5 M | 724 ms | 1242 ms |
| ad01_int8 | 10x fully-connected | 0.26 M | 49 ms | 22 ms |

Raw records in [`results/results.jsonl`](results/results.jsonl) (whole model)
and [`results/layers.jsonl`](results/layers.jsonl) (per operator). Layer times
sum to the whole-model median within 0.2%.

### it is deterministic

Run-to-run spread is 41 µs out of 474 ms. Two ESP32 boards of **different
silicon revisions** (D0WDQ6 rev v1.0 and D0WD-V3 rev v3.1) returned medians
**0-1 µs apart** on all four models.

One reading per configuration is enough. Repeats and per-device records buy
nothing.

The limit: that holds *within one binary*. Adding profiler code elsewhere in
the firmware moved the ResNet result by 2.3% while the other three moved under
0.2%, so store the build alongside the measurement. Every CI artifact ships a
`provenance.txt` for that reason.

## what this disproved

**mcufit's own latency estimate was 3.2x optimistic** and ranked the two chips
backwards. It has since been rewritten to use these measurements, and to
return nothing at all for boards that have none.

**Espressif's published figures did not reproduce.** Their README quotes
person_detect at 380 ms with esp-nn and 4084 ms without. We measured 474 ms and
600 ms, a 1.27x gap rather than 11x. `CONFIG_NN_ANSI_C` still uses esp-nn's own
C kernels rather than stock TFLM reference kernels, so their 4084 ms is likely
a third configuration.

**Measuring the arena on a laptop over-reports it.** A 64-bit host build needs
more interpreter bookkeeping than a 32-bit chip:

| arena section | host, 64-bit | wasm32 | real device |
| --- | --- | --- | --- |
| activations | 55,296 | 55,296 | 55,296 |
| interpreter overhead | 33,952 | 29,132 | 27,004 |
| **total** | **89,248** (+8.4%) | **84,428** (+2.6%) | **82,300** |

Activations come from the model and are identical everywhere. Everything else
follows pointer width. mcufit now uses the wasm32 build because of this.

## running it

Nothing to install. GitHub builds the firmware, esptool flashes it.

**ESP32**

1. Push, or hit **Run workflow** on the `build` action.
2. Download the `mcufit-bench-fast` artifact. It holds one merged `.bin` plus
   the `provenance.txt` recording exactly how it was built.
3. Flash to offset `0x0` and read the output at 115200 after a reset.

```bash
pip install esptool
esptool --port /dev/cu.usbserial-0001 --chip esp32 --baud 460800 \
  write-flash 0x0 mcufit-bench-fast.bin
```

Or flash from Chrome at <https://espressif.github.io/esptool-js/>, same offset,
no Python needed.

**Nano 33 BLE**

```bash
arduino-cli core install arduino:mbed_nano
arduino-cli lib install "Chirale_TensorFLowLite"
arduino-cli compile --fqbn arduino:mbed_nano:nano33ble arduino/mcufit_bench_nano
arduino-cli upload -p /dev/cu.usbmodem114101 \
  --fqbn arduino:mbed_nano:nano33ble arduino/mcufit_bench_nano
```

Append the `MCUFIT_RESULT` and `MCUFIT_LAYER` lines it prints to the files in
`results/`.

## how it measures

Each model runs twice. Once plain, timing only `Invoke()` with
`esp_timer_get_time()` (or `micros()` on Arduino), and once with a
`TagProfiler` attached.

The profiler implements `tflite::MicroProfilerInterface`. TFLM wraps every
operator's invoke in a `ScopedMicroProfiler` tagged with the operator name, so
accumulating per tag gives the cost of each layer type in about 40 lines. The
hook is compiled out by `-DTF_LITE_STRIP_ERROR_STRINGS`, which esp-tflite-micro
never defines.

`scripts/macs_by_op.py` counts MACs per operator on the host using mcufit's own
parser, and `scripts/layer_throughput.py` joins the two into the MACs/cycle
table above.

Two ESP32 build variants differ **only** in the kernel library, with 240 MHz and
`-O2` pinned in `sdkconfig.defaults` so nothing else can vary:

- `sdkconfig.fast` sets `CONFIG_NN_OPTIMIZED=y`
- `sdkconfig.slow` sets `CONFIG_NN_ANSI_C=y`

## adding a board

Roughly 15 minutes. Build the firmware for it, flash, capture the
`MCUFIT_LAYER` lines, run them through `scripts/layer_throughput.py`, and open a
PR with the raw records. Numbers that land in `results/` feed mcufit's
`measured.yaml`.

The most interesting board still unmeasured is the **ESP32-S3**, where esp-nn
ships hand-written assembly rather than generic C.

<details>
<summary><b>Notes that cost time</b></summary>
<br>

- **`esp-idf-ci-action` builds inside Docker as root**, so `build/` is not
  writable by the runner afterwards. Write artifacts to the repo root, and read
  `sdkconfig` from the project root rather than `build/sdkconfig`.
- **Native-USB boards print into the void.** The Nano's first run was lost
  because `setup()` waited only 5 s for Serial. Use `while (!Serial) {}` and read
  back without toggling DTR/RTS.
- **A run ID fetched immediately after `git push` is the previous run.** Match
  `gh run view --json headSha` against the commit before downloading, or you
  will flash a stale binary.
- **Models are committed as 16-byte-aligned C arrays.** TFLM reads the model in
  place from flash and CMake's `EMBED_FILES` does not guarantee alignment.
- The ANSI-C build takes seconds per inference, so the task watchdog needs
  raising and the loop needs a `vTaskDelay` yield.
- **Wokwi cannot be used for this.** It caps the simulated CPU frequency, which
  corrupts the exact quantity being measured.

</details>

<details>
<summary><b>Regenerating the model arrays</b></summary>
<br>

```bash
for m in person_detect kws ic_resnet ad; do
  python3 scripts/gen_model_array.py models/$m.tflite main/model_$m.cc $m
  cp main/model_$m.cc arduino/mcufit_bench_nano/model_$m.cpp
done
```

</details>

## models

Redistributed so the benchmark is reproducible without fetching anything. All
Apache 2.0.

| file | source |
| --- | --- |
| `models/person_detect.tflite` | [tflite-micro](https://github.com/tensorflow/tflite-micro), visual wake words reference model |
| `models/kws.tflite` | [MLPerf Tiny](https://github.com/mlcommons/tiny), keyword spotting |
| `models/ic_resnet.tflite` | MLPerf Tiny, image classification, ResNet-8 on CIFAR-10 |
| `models/ad.tflite` | MLPerf Tiny, anomaly detection |

## licence

MIT, see [LICENSE](LICENSE). The models keep their own Apache 2.0 terms.
