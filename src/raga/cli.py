"""Command line entry point: ``raga manifest | train | predict``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from raga.config import Config, data_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="raga")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("manifest", help="Parse every filename and write a portable manifest")
    p.add_argument("--data-dir")
    p.add_argument("--out", default="artifacts/manifest.csv")

    p = sub.add_parser("train", help="Cross-validate, fit, and save a model")
    p.add_argument("--config", default="configs/baseline.yaml")
    p.add_argument("--data-dir")
    p.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Score the sealed test set. Every use is logged to artifacts/test_evaluations.jsonl",
    )

    p = sub.add_parser("predict", help="Predict the raga of one audio file")
    p.add_argument("audio")
    p.add_argument("--model", default="artifacts/runs/baseline/model.joblib")
    p.add_argument("--top-k", type=int, default=3)

    args = parser.parse_args(argv)

    if args.cmd == "manifest":
        from raga.data import build_manifest

        df = build_manifest(data_dir(args.data_dir))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"{len(df)} files -> {args.out}")
        print(df.groupby(["raga", "aug_type"]).size().unstack(fill_value=0))
    elif args.cmd == "train":
        from raga.train import run

        run(Config.from_yaml(args.config), data_dir(args.data_dir), args.evaluate_test)
    elif args.cmd == "predict":
        from raga.predict import RagaPredictor

        result = RagaPredictor(args.model).predict_file(args.audio, args.top_k)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
