#!/usr/bin/env python3
"""Check a pasted serial capture and file it into results/.

  ingest.py capture.txt          # check only, print what it would add
  ingest.py capture.txt --write  # append the new records to results/

Reads whatever the board printed, keeps the MCUFIT_RESULT and MCUFIT_LAYER
lines, and refuses anything that would poison the database: an unrecognised
chip, a missing clock, a model whose bytes do not match the one in models/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results" / "results.jsonl"
LAYERS = REPO / "results" / "layers.jsonl"
MODELS_DIR = REPO / "models"

RESULT_REQUIRED = ["model", "target", "chip", "cpu_mhz", "kernels",
                   "arena_used_bytes", "runs", "p50_us"]
LAYER_REQUIRED = ["model", "target", "chip", "cpu_mhz", "kernels", "op",
                  "calls_per_inference", "us_per_inference"]

# The firmware reports upstream model names; models/ uses short filenames.
FILENAME = {
    "person_detect.tflite": "person_detect.tflite",
    "kws_ref_model.tflite": "kws.tflite",
    "pretrainedResnet_quant.tflite": "ic_resnet.tflite",
    "ad01_int8.tflite": "ad.tflite",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse(text: str) -> tuple[list[dict], list[dict], list[str]]:
    results, layers, problems = [], [], []
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        for prefix, bucket in (("MCUFIT_RESULT ", results), ("MCUFIT_LAYER ", layers)):
            if not line.startswith(prefix):
                continue
            body = line[len(prefix):]
            try:
                bucket.append(json.loads(body))
            except json.JSONDecodeError as e:
                # Usually a dropped byte on the serial line. Re-run rather than
                # hand-repair; a truncated number looks like a valid one.
                problems.append(f"line {n}: not valid JSON ({e.msg}). Capture the run again.")
    return results, layers, problems


def check(rows: list[dict], required: list[str], kind: str) -> list[str]:
    problems = []
    for r in rows:
        where = f"{kind} {r.get('target', '?')}/{r.get('model', '?')}"
        for field in required:
            if r.get(field) in (None, ""):
                problems.append(f"{where}: missing {field}")
        if r.get("chip") == "unknown":
            problems.append(f"{where}: chip is 'unknown'. Add the macro to the sketch "
                            f"or set chip by hand, but do not guess.")
        if r.get("cpu_mhz") in (0, None):
            problems.append(f"{where}: cpu_mhz is 0. Every throughput figure divides "
                            f"by it, so a wrong clock is worse than no record.")
        if r.get("kernels_source") == "compile-time-guess" and r.get("kernels", "").startswith("cmsis"):
            # Not fatal. CMSIS-NN presence is inferred from the DSP extension,
            # which some cores have without linking the optimised kernels.
            print(f"  note  {where}: kernels '{r['kernels']}' is inferred from the "
                  f"target's features, not confirmed by the runtime.", file=sys.stderr)
    return problems


def check_models(results: list[dict]) -> list[str]:
    problems = []
    known = {}
    for reported, filename in FILENAME.items():
        path = MODELS_DIR / filename
        if path.exists():
            known[reported] = (sha256(path), path.stat().st_size)
    for r in results:
        name = r.get("model")
        if name not in known:
            problems.append(f"result {name}: not one of the bundled models. "
                            f"Custom models are welcome but belong in their own file.")
            continue
        want_sha, want_bytes = known[name]
        if r.get("model_sha256") and r["model_sha256"] != want_sha:
            problems.append(f"result {name}: model_sha256 does not match models/. "
                            f"The board ran a different file, so the timing is not comparable.")
        if r.get("model_bytes") and r["model_bytes"] != want_bytes:
            problems.append(f"result {name}: model_bytes {r['model_bytes']} != {want_bytes}.")
    return problems


def existing_keys(path: Path, fields: tuple[str, ...]) -> set[tuple]:
    keys = set()
    if not path.exists():
        return keys
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if all(f in r for f in fields):
            keys.add(tuple(r[f] for f in fields))
    return keys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture", nargs="?", type=Path,
                    help="file holding the serial output. Reads stdin if omitted.")
    ap.add_argument("--write", action="store_true", help="append the new records to results/")
    args = ap.parse_args()

    text = args.capture.read_text() if args.capture else sys.stdin.read()
    results, layers, problems = parse(text)

    if not results and not layers:
        print("no MCUFIT_RESULT or MCUFIT_LAYER lines found.", file=sys.stderr)
        print("Capture the whole serial output from reset, not just the summary.", file=sys.stderr)
        return 1

    problems += check(results, RESULT_REQUIRED, "result")
    problems += check(layers, LAYER_REQUIRED, "layer")
    problems += check_models(results)

    if problems:
        print(f"{len(problems)} problem(s):", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1

    seen_r = existing_keys(RESULTS, ("target", "model", "kernels", "cpu_mhz"))
    seen_l = existing_keys(LAYERS, ("target", "model", "kernels", "cpu_mhz", "op"))
    new_r = [r for r in results
             if (r["target"], r["model"], r["kernels"], r["cpu_mhz"]) not in seen_r]
    new_l = [r for r in layers
             if (r["target"], r["model"], r["kernels"], r["cpu_mhz"], r["op"]) not in seen_l]

    targets = sorted({r["target"] for r in results})
    print(f"target(s): {', '.join(targets)}")
    for r in sorted(results, key=lambda r: r["model"]):
        dup = "" if r in new_r else "   (already recorded)"
        print(f"  {r['model']:32} {r['p50_us'] / 1000:8.1f} ms   "
              f"arena {r['arena_used_bytes']:>7,}{dup}")
    print(f"\n{len(new_r)} new whole-model, {len(new_l)} new per-layer "
          f"({len(results) - len(new_r)} and {len(layers) - len(new_l)} already present)")

    if not args.write:
        print("\nlooks good. Re-run with --write to file it.")
        return 0

    if new_r:
        with RESULTS.open("a") as f:
            for r in new_r:
                f.write(json.dumps(r) + "\n")
    if new_l:
        with LAYERS.open("a") as f:
            for r in new_l:
                f.write(json.dumps(r) + "\n")
    print(f"\nwritten. Next: python3 scripts/macs_by_op.py models/*.tflite > /tmp/macs.jsonl "
          f"&& python3 scripts/layer_throughput.py /tmp/macs.jsonl results/layers.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
