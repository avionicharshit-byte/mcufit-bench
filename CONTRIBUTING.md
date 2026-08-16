# adding your board

Every number in this repo came off a physical chip, and that is the only kind
we take. If you own a microcontroller that is not in `results/`, you can add it
in about fifteen minutes, and the numbers go straight into
[mcufit](https://github.com/avionicharshit-byte/mcufit)'s board database.

You need the board, a USB cable, and `arduino-cli`. Nothing else.

## which boards are worth the most

Inference speed is a property of **(chip, operator, kernel library)**, not of
the chip on its own, which is [the finding this repo
exists for](README.md#the-finding). So one board per family answers for the
whole family, and these families currently have nothing at all:

| family | boards this would answer for | example board |
| --- | --- | --- |
| Cortex-M7, CMSIS-NN | STM32F746, STM32H743, Teensy 4.0/4.1, Portenta H7 | Teensy 4.0 |
| Xtensa LX7, esp-nn | ESP32-S3, ESP32-S2, XIAO ESP32S3 | ESP32-S3 DevKit |
| RISC-V, esp-nn | ESP32-C3, ESP32-C6, ESP32-P4 | ESP32-C3 DevKit |
| Cortex-M0+, no DSP | RP2040, Pico W, Nano 33 IoT | Raspberry Pi Pico |
| Cortex-M33, CMSIS-NN | RP2350 | Raspberry Pi Pico 2 |
| Cortex-M3 | STM32F103 | Blue Pill |

**The ESP32-S3 is the one we want most.** esp-nn ships hand-written assembly
there rather than generic C, so it is the board most likely to break the
current model rather than confirm it, which makes it the most informative.

Already covered: Xtensa LX6 with esp-nn (ESP32) and Cortex-M4F with CMSIS-NN
(nRF52840). A second reading on either is still welcome as a cross-check.

## the run

**1. Install your board's core and a TFLM library.**

```bash
arduino-cli core install arduino:mbed_nano        # or rp2040, esp32, teensy, ...
arduino-cli lib install "Chirale_TensorFlowLite"  # generic ARM
# ESP32 family instead:
arduino-cli lib install "tflm_esp32"
```

**2. Build a sketch for your board.**

```bash
python3 scripts/prepare_arduino.py --target rp2040 --out /tmp/mcufit_bench
```

`--target` is a short id of your choosing, and it becomes the `target` field in
every record. The script picks the right library for you and tells you which
one to install.

Four models is about 4.5 MB of C arrays and will not fit a small board. Cut
down with `--models`, and lower the arena if the sketch still will not link:

```bash
python3 scripts/prepare_arduino.py --target stm32f103 --models ad --out /tmp/mcufit_bench
arduino-cli compile --fqbn <FQBN> \
  --build-property compiler.cpp.extra_flags=-DMCUFIT_ARENA_KB=32 /tmp/mcufit_bench
```

Compiling takes over ten minutes on some cores. That is the C arrays, not you.

**3. Flash and capture.** Save the whole serial output from reset, not just the
summary lines.

```bash
arduino-cli compile --fqbn <FQBN> /tmp/mcufit_bench
arduino-cli upload -p <PORT> --fqbn <FQBN> /tmp/mcufit_bench
arduino-cli monitor -p <PORT> --config baudrate=115200 | tee /tmp/capture.txt
```

On a native-USB board the sketch blocks until you open the port, so nothing is
lost if you are slow attaching. Do not toggle DTR or RTS while reading, it
resets the chip.

**4. Check it, then file it.**

```bash
python3 scripts/ingest.py /tmp/capture.txt          # checks only
python3 scripts/ingest.py /tmp/capture.txt --write  # appends to results/
```

`ingest.py` rejects anything that would poison the database: an unrecognised
chip, a clock of zero, a model whose bytes do not match `models/`. If it
complains about the chip, add the macro for your board to
`arduino/mcufit_bench_portable/mcufit_bench_portable.ino` and include that in
the PR. It is three lines and it helps the next person.

**5. Open a pull request** with the changed files in `results/` and one line
saying which board, at what clock, with which core and library versions.

No board and still want to help? Open an issue with your serial capture pasted
in and someone will file it.

## the rules, and why

- **Never guess a chip name or a clock.** Every throughput figure is
  `MACs / (seconds x hertz)`, so a wrong clock silently produces a wrong number
  that looks perfectly reasonable. If the sketch says `unknown` or `0`, find
  the real value or leave the record out.
- **One reading per configuration is enough.** Two ESP32s of different silicon
  revisions came back 0 to 1 microseconds apart across four models. Repeats and
  per-device records buy nothing. A configuration is (chip, model, kernel
  library, clock, optimisation level).
- **Say which build produced it.** Timings are deterministic within one binary
  and not across binaries: adding profiler code elsewhere in the firmware moved
  one model by 2.3%. Note the core and library versions in the PR.
- **Arduino and ESP-IDF numbers are kept apart, never averaged.** The two
  toolchains pick different optimisation levels and different kernel
  configurations, so they are two configurations of the same chip and not two
  readings of one. Both are welcome and both are recorded, each with its own
  `opt_level` field. The ESP32 rows in `results/` come from ESP-IDF at `-O2`
  and 240 MHz pinned; an Arduino run on the same chip belongs beside them as a
  separate row, not merged into them. If you find the two disagree by a lot on
  the same silicon, that is a finding, so please say so in the PR.
- **Simulator numbers are not accepted.** Renode and QEMU cost a fixed number
  of ticks per instruction, which is a guess with a config file. Wokwi caps the
  simulated clock, which corrupts precisely what is being measured. A host
  build is also not a device: it over-reports the arena by 8.4% because it is
  64-bit.
- **A negative result is a result.** If the model will not fit, or an operator
  falls back to reference C and runs terribly, that is the interesting half.
  Say so in the PR rather than dropping the board.

## the ESP32 path

The ESP32 family also has an ESP-IDF build, which is what produced the numbers
in the README. GitHub Actions compiles it, so you need no local toolchain, only
`esptool` to flash. See [running it](README.md#running-it).

Use it rather than the Arduino path when you want the two kernel variants
(`CONFIG_NN_OPTIMIZED` against `CONFIG_NN_ANSI_C`) pinned at the same clock and
optimisation level.
