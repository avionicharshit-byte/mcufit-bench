#!/usr/bin/env python3
"""Join device per-layer timings with host MAC counts to get throughput per operator.

  layer_throughput.py <macs_by_op.jsonl> <layers.jsonl>

macs_by_op.jsonl comes from macs_by_op.py, layers.jsonl from the MCUFIT_LAYER
lines the firmware prints.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

# The firmware reports the models under their upstream names.
ALIAS = {
    "kws_ref_model.tflite": "kws.tflite",
    "pretrainedResnet_quant.tflite": "ic_resnet.tflite",
    "ad01_int8.tflite": "ad.tflite",
}


def load(path: str) -> list[dict]:
    rows = []
    for line in open(path):
        line = line.strip()
        if line.startswith("MCUFIT_LAYER "):
            line = line[len("MCUFIT_LAYER ") :]
        if line.startswith("{"):
            rows.append(json.loads(line))
    return rows


def main() -> int:
    macs = {(ALIAS.get(r["model"], r["model"]), r["op"]): r["macs"] for r in load(sys.argv[1])}
    layers = load(sys.argv[2])

    by_target: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    print(f"{'target':10} {'model':30} {'op':20} {'us':>9} {'MACs':>11} {'MACs/cyc':>9}")
    for r in sorted(layers, key=lambda r: (r["target"], r["model"], -r["us_per_inference"])):
        key = (ALIAS.get(r["model"], r["model"]), r["op"])
        m = macs.get(key)
        us, mhz = r["us_per_inference"], r["cpu_mhz"]
        if not m or us <= 0:
            continue
        mc = m / (us / 1e6 * mhz * 1e6)
        by_target[r["target"]][r["op"]].append(mc)
        print(f"{r['target']:10} {r['model']:30} {r['op']:20} {us:>9,} {m:>11,} {mc:>9.3f}")

    print("\nthroughput per operator, averaged over the models that use it:")
    for target, ops in by_target.items():
        print(f"\n  {target}")
        for op, vals in sorted(ops.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
            avg = sum(vals) / len(vals)
            spread = f"  ({min(vals):.3f}-{max(vals):.3f} over {len(vals)})" if len(vals) > 1 else ""
            print(f"    {op:20} {avg:>7.3f} MACs/cycle{spread}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
