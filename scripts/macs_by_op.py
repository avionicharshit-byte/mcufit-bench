#!/usr/bin/env python3
"""Per-operator MAC counts for a model, so device timings become throughput.

Reuses mcufit's parser and MAC counter rather than reimplementing them.
Run inside mcufit's venv, or with mcufit installed.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from mcufit.analysis.latency import _layer_macs
from mcufit.parsing.tflite_parser import TFLiteModelParser


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: macs_by_op.py <model.tflite> [...]", file=sys.stderr)
        return 2

    for path in sys.argv[1:]:
        model = TFLiteModelParser().parse(Path(path))
        tensors = {t.index: t for t in model.tensors}
        macs: dict[str, int] = defaultdict(int)
        calls: dict[str, int] = defaultdict(int)
        for layer in model.layers:
            macs[layer.op_name] += _layer_macs(layer, tensors)
            calls[layer.op_name] += 1
        for op in sorted(macs, key=lambda o: -macs[o]):
            print(json.dumps({"model": Path(path).name, "op": op,
                              "calls": calls[op], "macs": macs[op]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
