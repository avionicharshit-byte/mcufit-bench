# mcufit-bench

Measuring how wrong mcufit's latency estimate is, on real silicon.

## the question

mcufit predicts inference speed from one hand-picked number per board,
`macs_per_cycle` in `boards.yaml`. For the classic ESP32 that number is 0.2,
and it was invented, not measured.

For `person_detect.tflite` on an ESP32 at 240 MHz:

| source | result |
| --- | --- |
| mcufit prediction | 149 ms |
| Espressif, ESP-NN kernels on | 380 ms |
| Espressif, ESP-NN kernels off | 4084 ms |

Two things follow if those numbers hold. mcufit is 2.5x optimistic at best and
27x at worst. And more importantly the same model on the same chip at the same
clock differs by 11x depending on one compile-time switch, which means latency
is not a property of the board and no single per-board number can ever be
right.

This repo builds both binaries and measures both, so the claim rests on our own
hardware rather than someone's README.

## what it does

Loads `person_detect.tflite` from flash, runs 3 warmup inferences, then times 30
with `esp_timer_get_time()` around `Invoke()` and nothing else. Prints min,
median, mean and max, plus the arena the interpreter actually allocated, as one
JSON line tagged `MCUFIT_RESULT`.

Two builds, identical except for the kernel library:

- `sdkconfig.fast` sets `CONFIG_NN_OPTIMIZED=y`, Espressif's optimised kernels
- `sdkconfig.slow` sets `CONFIG_NN_ANSI_C=y`, plain portable C

Everything else, 240 MHz and `-O2`, is pinned in `sdkconfig.defaults` so the
kernel library is the only difference between the two numbers.

## running it

Nothing to install. GitHub builds it, Chrome flashes it.

1. Push, or hit **Run workflow** on the `build` action.
2. Download both artifacts, `mcufit-bench-fast` and `mcufit-bench-slow`. Each
   holds a single `.bin` and a `provenance.txt` recording the settings it was
   built with.
3. Open <https://espressif.github.io/esptool-js/> in Chrome or Edge.
4. Connect, pick the board's serial port, add the `.bin` at offset **0x0**,
   and flash. It is a merged image, so that one file is the whole thing.
5. Switch to the console at 115200 baud and press the board's reset button.
6. Copy the `MCUFIT_RESULT` line into `results/results.jsonl`.

Repeat for the second binary, and for each of the three boards.

If no serial port appears, the board never enumerated. Check the cable is a
data cable and not a charge-only one before suspecting drivers.

## what we expect

Roughly 380 ms from the fast build and 4084 ms from the slow one, and about
89 KB of arena in both. The arena figure should match mcufit's exact mode
almost exactly, since arena allocation is deterministic. If it does not, that
is a separate and more alarming finding.

Three identical boards are deliberate. If they disagree by more than a few
percent, a single measurement is not trustworthy and the future database needs
repeats.

## regenerating the model array

The model is committed as a 16-byte-aligned C array because TFLM reads it in
place from flash and CMake's `EMBED_FILES` does not guarantee alignment.

```
python3 scripts/gen_model_array.py models/person_detect.tflite main/model_data.cc
```
