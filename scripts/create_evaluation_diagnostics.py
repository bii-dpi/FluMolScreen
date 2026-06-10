"""Create evaluation model-comparison diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from flumolscreen.visualization import evaluation as evaluation_viz


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create evaluation model-comparison diagnostics.",
    )
    parser.add_argument("--results-dir", default="results", help="Output results directory.")
    parser.add_argument(
        "--round-id",
        default="round_synthetic",
        help="Result round id to process.",
    )
    parser.add_argument(
        "--dataset-labels",
        nargs="+",
        default=None,
        help="Dataset labels to process. Defaults to all complete evaluation outputs.",
    )
    parser.add_argument(
        "--metric",
        default=evaluation_viz.DEFAULT_METRIC,
        help="Metric used for feature-comparison plots and fold-winner selection.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Write CSV outputs only. By default, static PNG plots are written.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    outputs = evaluation_viz.create_evaluation_diagnostics(
        results_dir=args.results_dir,
        round_id=args.round_id,
        dataset_labels=args.dataset_labels,
        metric=args.metric,
        write_plots=not args.skip_plots,
    )

    print(f"Wrote evaluation diagnostics to: {outputs['output_dir']}")
    print("Data outputs:")
    for label, path in outputs["data_paths"].items():
        print(f"  {label}: {path}")
    if not args.skip_plots:
        print("Plot outputs:")
        for label, path in outputs["plot_paths"].items():
            print(f"  {label}: {path}")
    return outputs


if __name__ == "__main__":
    main()
