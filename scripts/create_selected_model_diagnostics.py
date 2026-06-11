"""Create selected-model OOF behavior diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from flumolscreen.visualization import selected_model as selected_model_viz


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create selected-model OOF behavior diagnostics.",
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
        help="Dataset labels to process. Defaults to all selected-model OOF outputs.",
    )
    parser.add_argument(
        "--disagreement-column",
        default=selected_model_viz.DEFAULT_DISAGREEMENT_COLUMN,
        choices=selected_model_viz.DISAGREEMENT_COLUMNS,
        help="Selected OOF disagreement column to use on the residual diagnostic x-axis.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Write CSV outputs only. By default, Plotly HTML plots are written.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    outputs = selected_model_viz.create_selected_model_diagnostics(
        results_dir=args.results_dir,
        round_id=args.round_id,
        dataset_labels=args.dataset_labels,
        disagreement_column=args.disagreement_column,
        write_plots=not args.skip_plots,
    )

    print(f"Wrote selected-model OOF diagnostics to: {outputs['output_dir']}")
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
