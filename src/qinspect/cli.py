"""Command line entry point: `qinspect generate | inspect | demo`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .report import run_batch
from .spec import PartSpec
from .synth import generate_lot


def _load_spec(path: str | None) -> PartSpec:
    return PartSpec.from_json(path) if path else PartSpec()


def _print_summary(summary: dict) -> None:
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="qinspect", description="Defense-part quality inspection MVP")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="render a synthetic lot with ground truth")
    g.add_argument("--out", default="data/synth")
    g.add_argument("--n", type=int, default=200)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--lot", default="LOT-0001")
    g.add_argument("--no-drift", action="store_true", help="disable simulated tool wear")
    g.add_argument("--spec")

    i = sub.add_parser("inspect", help="inspect a folder of images")
    i.add_argument("--images", required=True)
    i.add_argument("--out", default="runs/latest")
    i.add_argument("--truth", help="ground_truth.csv for accuracy metrics")
    i.add_argument("--subgroup", type=int, default=5)
    i.add_argument("--baseline", type=int, default=10, help="subgroups used to set control limits (0 = all)")
    i.add_argument("--spec")

    d = sub.add_parser("demo", help="generate + inspect in one go")
    d.add_argument("--out", default="runs/demo")
    d.add_argument("--n", type=int, default=200)
    d.add_argument("--seed", type=int, default=0)

    args = ap.parse_args(argv)

    if args.cmd == "generate":
        spec = _load_spec(args.spec)
        df = generate_lot(spec, args.n, args.out, seed=args.seed, lot_id=args.lot,
                          drift_start=None if args.no_drift else 0.6)
        print(f"wrote {len(df)} images to {Path(args.out) / 'images'}")
        print(df.expected_verdict.value_counts().to_string())
    elif args.cmd == "inspect":
        spec = _load_spec(args.spec)
        summary = run_batch(args.images, args.out, spec, subgroup=args.subgroup,
                            baseline=args.baseline or None, truth_csv=args.truth)
        _print_summary(summary)
    elif args.cmd == "demo":
        spec = PartSpec()
        data = Path(args.out) / "data"
        generate_lot(spec, args.n, data, seed=args.seed)
        summary = run_batch(data / "images", args.out, spec, truth_csv=data / "ground_truth.csv")
        _print_summary(summary)
        print(f"\nresults: {args.out}/results.csv, chart: {args.out}/spc_outer_diameter.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
