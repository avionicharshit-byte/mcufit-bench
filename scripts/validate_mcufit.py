#!/usr/bin/env python3
"""Check mcufit's estimates against every device measurement in results/.

  validate_mcufit.py            # all estimators, all measurements
  validate_mcufit.py --exact    # only the wasm measurement path
  validate_mcufit.py --strict   # exit 1 if any bound is breached

Both bugs found in mcufit so far were the same mistake: an estimator validated
against one model, on one board. person_detect is large enough that an additive
error looks like a small percentage, so the error hid there twice. This runs
every (board, model) pair we have silicon numbers for and prints the worst case.

Needs mcufit importable, and node on PATH for the --exact path.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results" / "results.jsonl"
MODELS = REPO / "models"

# The firmware reports upstream model names; models/ uses short filenames.
FILENAME = {
    "person_detect.tflite": "person_detect.tflite",
    "kws_ref_model.tflite": "kws.tflite",
    "pretrainedResnet_quant.tflite": "ic_resnet.tflite",
    "ad01_int8.tflite": "ad.tflite",
}

# An estimate below the device figure makes mcufit say "fits" when it does not,
# so undershoot is the one failure that is never acceptable, from either path.
MAX_UNDERSHOOT_BYTES = 0

# Overshoot is the safe direction, so it is only bounded for the measurement
# path, which claims accuracy. The static estimator does not: it cannot see
# TFLM's interpreter bookkeeping, which was measured at between 52 and 303 bytes
# per tensor across the four models, a 6x spread that no single constant covers.
# That is why `check` measures with wasm whenever node is present.
MAX_OVERSHOOT_BYTES = {"exact": 4096, "static": None}

MAX_LATENCY_ERROR_PCT = 25.0


def load_devices() -> list[dict]:
    """One record per (target, model, kernels), deduped: repeats buy nothing."""
    seen: dict[tuple, dict] = {}
    for line in RESULTS.read_text().splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        r = json.loads(line)
        if "arena_used_bytes" not in r or r["model"] not in FILENAME:
            continue
        seen.setdefault((r["target"], r["model"], r.get("kernels", "")), r)
    return list(seen.values())


def static_arena(model: Path) -> int:
    """The static estimator, called directly.

    Not reachable through the CLI any more: `check` measures with wasm whenever
    node is present, because the static path reads low. It still runs for users
    without node, so it still has to be validated.
    """
    from mcufit.estimation.greedy import GreedyLifetimeEstimator
    from mcufit.parsing.tflite_parser import TFLiteModelParser

    return GreedyLifetimeEstimator().estimate(TFLiteModelParser().parse(model)).total_arena_bytes


def run_mcufit(model: Path, board: str, exact: bool) -> dict | None:
    cmd = ["mcufit", "check", str(model), "-b", board, "--json"]
    if exact:
        cmd.append("--exact")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode not in (0, 1):  # 1 just means "does not fit"
        print(f"  mcufit failed on {model.name}/{board}: {proc.stderr[-300:]}", file=sys.stderr)
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        print(f"  mcufit gave no JSON for {model.name}/{board}", file=sys.stderr)
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exact", action="store_true", help="only the wasm measurement path")
    ap.add_argument("--static", action="store_true", help="only the static estimator")
    ap.add_argument("--strict", action="store_true", help="exit 1 on any breach")
    args = ap.parse_args()

    if shutil.which("mcufit") is None:
        print("mcufit not on PATH. Install it, or activate its venv.", file=sys.stderr)
        return 2

    modes = []
    if not args.static:
        modes.append(("exact", True))
    if not args.exact:
        modes.append(("static", False))

    devices = load_devices()
    print(f"{len(devices)} device measurements, "
          f"{len({d['target'] for d in devices})} boards, "
          f"{len({d['model'] for d in devices})} models\n")

    breaches: list[str] = []
    deltas: dict[tuple[str, str], list[int]] = defaultdict(list)

    for mode, exact in modes:
        print(f"--- arena, {mode} estimator "
              f"{'(wasm32 under node)' if exact else '(static lifetime analysis)'}")
        print(f"{'board':11}{'model':24}{'estimate':>10}{'device':>9}{'delta':>9}{'':>4}")
        for d in sorted(devices, key=lambda r: (r["target"], r["model"])):
            model_file = MODELS / FILENAME[d["model"]]
            if exact:
                out = run_mcufit(model_file, d["target"], True)
                if out is None:
                    continue
                est = out["estimate"]["total_arena_bytes"]
            else:
                est = static_arena(model_file)
            dev = d["arena_used_bytes"]
            delta = est - dev
            deltas[(mode, d["target"])].append(delta)

            over_bound = MAX_OVERSHOOT_BYTES[mode]
            flag = ""
            if delta < -MAX_UNDERSHOOT_BYTES:
                flag = "UNDER"
                breaches.append(
                    f"{mode}/{d['target']}/{d['model']}: estimate {est:,} is {-delta:,} B "
                    f"BELOW the device's {dev:,}. mcufit would say this fits when it does not."
                )
            elif over_bound is not None and delta > over_bound:
                flag = "over"
                breaches.append(
                    f"{mode}/{d['target']}/{d['model']}: estimate {est:,} is {delta:,} B "
                    f"above the device's {dev:,}, past the {over_bound:,} B bound."
                )
            elif delta > 4096:
                flag = "(over)"  # safe direction, recorded but not a breach
            print(f"{d['target']:11}{d['model']:24}{est:>10,}{dev:>9,}{delta:>+9,}{flag:>6}")

        print()
        for (m, board), ds in sorted(deltas.items()):
            if m != mode:
                continue
            const = "constant" if len(set(ds)) == 1 else f"{min(ds):+,} to {max(ds):+,}"
            print(f"  {board:11} delta {const}")
        print()

    print("--- latency, against the measured device p50")
    print(f"{'board':11}{'model':24}{'estimate':>10}{'device':>9}{'err':>8}")
    for d in sorted(devices, key=lambda r: (r["target"], r["model"])):
        out = run_mcufit(MODELS / FILENAME[d["model"]], d["target"], False)
        if out is None:
            continue
        lat = out.get("latency") or {}
        if not lat.get("measured"):
            print(f"{d['target']:11}{d['model']:24}{'no data':>10}{d['p50_us']/1000:>8.1f}ms")
            continue
        # measured.yaml holds one kernel library per board. Comparing an esp-nn
        # estimate against an ANSI-C run is a 26% "error" that is really just two
        # different configurations, so skip the pairs that do not match.
        if lat.get("kernels") and d.get("kernels") and lat["kernels"] != d["kernels"]:
            print(f"{d['target']:11}{d['model']:24}{'skipped':>10}"
                  f"  built with {d['kernels']}, measured.yaml has {lat['kernels']}")
            continue
        est_ms, dev_ms = lat["milliseconds"], d["p50_us"] / 1000
        err = 100 * (est_ms - dev_ms) / dev_ms
        if abs(err) > MAX_LATENCY_ERROR_PCT:
            breaches.append(
                f"latency/{d['target']}/{d['model']}: {est_ms:.1f} ms vs the device's "
                f"{dev_ms:.1f} ms, {err:+.1f}%, past the {MAX_LATENCY_ERROR_PCT}% bound."
            )
        print(f"{d['target']:11}{d['model']:24}{est_ms:>9.1f}ms{dev_ms:>8.1f}ms{err:>+7.1f}%")

    if breaches:
        print(f"\n{len(breaches)} bound(s) breached:")
        for b in breaches:
            print(f"  {b}")
        return 1 if args.strict else 0

    print("\nevery estimate inside its bound.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
