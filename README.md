# mcufit-bench

Measuring what mcufit guesses, on real silicon.

## results

person_detect.tflite on an ESP32-D0WDQ6 (DevKit v1), 240 MHz, `-O2`, IDF v5.5,
30 timed inferences after 3 warmups. 2026-08-16.

| | latency | arena |
| --- | --- | --- |
| mcufit prediction | 149 ms | 76,147 B static / 89,248 B `--exact` |
| device, ESP-NN kernels | **474 ms** | **82,300 B** |
| device, ANSI-C kernels | **600 ms** | **82,300 B** |

Three findings.

**mcufit's latency estimate is 3.2x optimistic.** `macs_per_cycle: 0.2` for this
board was invented, and the real figure is closer to 0.06.

**The kernel library matters far less than Espressif's README implies.** They
publish 380 ms with ESP-NN and 4084 ms without. Neither reproduced: we measured
474 ms and 600 ms, a 1.27x gap rather than 11x. Note `CONFIG_NN_ANSI_C` still
uses esp-nn's own C kernels, which are not stock TFLM reference kernels, so
their 4084 ms may be measuring a third thing.

**mcufit's `--exact` arena was 8.4% high, and that is now fixed.** The cause was
pointer width: the host build is 64-bit, so its interpreter bookkeeping is
larger than the chip's. The wasm32 build lands at 84,428 B, within 2.6%, and
mcufit now uses it by default.

| arena section | host, 64-bit | wasm32 | device |
| --- | --- | --- | --- |
| activations | 55,296 | 55,296 | 55,296 |
| interpreter overhead | 33,952 | 29,132 | 27,004 |
| total | 89,248 | 84,428 | 82,300 |

**It is deterministic.** Run-to-run spread is 41 µs out of 474 ms. Two boards
of different silicon revisions, D0WDQ6 rev v1.0 and D0WD-V3 rev v3.1, returned
byte-identical min, median, mean and max on both builds. So one measurement
per chip, model, kernel library, clock and optimisation level is enough, and
collecting repeats or per-board data would buy nothing.

## what it does

Loads the model from flash and times only `Invoke()` with
`esp_timer_get_time()`, reporting min, median, mean, max plus the arena the
interpreter actually allocated, as one JSON line tagged `MCUFIT_RESULT`.

Two builds, identical but for the kernel library:

- `sdkconfig.fast` sets `CONFIG_NN_OPTIMIZED=y`
- `sdkconfig.slow` sets `CONFIG_NN_ANSI_C=y`

240 MHz and `-O2` are pinned in `sdkconfig.defaults` so nothing else varies.

## running it

Nothing to install locally. GitHub builds it, esptool flashes it.

1. Push, or hit **Run workflow** on the `build` action.
2. Download the `mcufit-bench-fast` and `mcufit-bench-slow` artifacts. Each has
   one `.bin` and a `provenance.txt` recording the settings it was built with.
3. Flash to offset `0x0`. It is a merged image, so that one file is everything.

```bash
pip install esptool
esptool --port /dev/cu.usbserial-0001 --chip esp32 --baud 460800 \
  write-flash 0x0 mcufit-bench-fast.bin
```

4. Read the output at 115200 after a reset and append the `MCUFIT_RESULT` line
   to `results/results.jsonl`.

Or flash from Chrome at <https://espressif.github.io/esptool-js/>, same offset,
which needs no Python at all.

If no serial port appears, check the cable carries data and go straight into
the machine rather than through a hub.

## models

Redistributed here so the benchmark is reproducible without fetching anything.
All Apache 2.0.

| file | source |
| --- | --- |
| `person_detect.tflite` | [tflite-micro](https://github.com/tensorflow/tflite-micro) visual wake words reference model |
| `kws_ref_model.tflite` | [MLPerf Tiny](https://github.com/mlcommons/tiny) keyword spotting |
| `pretrainedResnet_quant.tflite` | MLPerf Tiny image classification, ResNet-8 on CIFAR-10 |
| `ad01_int8.tflite` | MLPerf Tiny anomaly detection |

## licence

MIT, see [LICENSE](LICENSE). The models above keep their own Apache 2.0 terms.

## regenerating the model array

Committed as a 16-byte-aligned C array because TFLM reads it in place from
flash and CMake's `EMBED_FILES` does not guarantee alignment.

```bash
python3 scripts/gen_model_array.py models/person_detect.tflite main/model_data.cc
```
