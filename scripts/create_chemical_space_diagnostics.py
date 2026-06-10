"""Create chemical-space and scaffold coverage diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from flumolscreen.target_registry import TARGET_CLASS_REGISTRY
from flumolscreen.visualization import chemical_space as chemical_space_viz


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create chemical-space and scaffold assay-coverage diagnostics.",
    )
    parser.add_argument("--data-dir", default="data", help="Input data directory.")
    parser.add_argument("--results-dir", default="results", help="Output results directory.")
    parser.add_argument(
        "--round-ids",
        nargs="+",
        default=None,
        help="Assay round ids to include. Defaults to all data/round_*/assay_data folders.",
    )
    parser.add_argument(
        "--target-classes",
        nargs="+",
        default=None,
        choices=sorted(TARGET_CLASS_REGISTRY),
        help="Target classes to process. Defaults to all registered target classes.",
    )
    parser.add_argument(
        "--top-n-scaffolds",
        type=int,
        default=chemical_space_viz.DEFAULT_TOP_N_SCAFFOLDS,
        help="Number of scaffolds to show in scaffold coverage plots.",
    )
    parser.add_argument(
        "--fingerprint-radius",
        type=int,
        default=chemical_space_viz.DEFAULT_FINGERPRINT_RADIUS,
        help="Morgan fingerprint radius.",
    )
    parser.add_argument(
        "--fingerprint-bits",
        type=int,
        default=chemical_space_viz.DEFAULT_FINGERPRINT_BITS,
        help="Morgan fingerprint bit length.",
    )
    parser.add_argument(
        "--umap-n-neighbors",
        type=int,
        default=chemical_space_viz.DEFAULT_UMAP_N_NEIGHBORS,
        help="UMAP n_neighbors value, adjusted downward for small libraries.",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=chemical_space_viz.DEFAULT_UMAP_MIN_DIST,
        help="UMAP min_dist value.",
    )
    parser.add_argument(
        "--umap-random-state",
        type=int,
        default=chemical_space_viz.DEFAULT_UMAP_RANDOM_STATE,
        help="UMAP random seed.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Write CSV outputs only. By default, static PNG and interactive HTML are written.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    outputs = chemical_space_viz.create_chemical_space_diagnostics(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        round_ids=args.round_ids,
        target_classes=args.target_classes,
        top_n_scaffolds=args.top_n_scaffolds,
        fingerprint_radius=args.fingerprint_radius,
        fingerprint_bits=args.fingerprint_bits,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
        umap_random_state=args.umap_random_state,
        write_plots=not args.skip_plots,
    )

    print(f"Wrote chemical-space diagnostics to: {outputs['output_dir']}")
    print(f"Wrote summary table to: {outputs['summary_path']}")
    for target_class, target_paths in outputs["target_outputs"].items():
        print(f"{target_class}:")
        for label, path in target_paths.items():
            if path.exists():
                print(f"  {label}: {path}")
    return outputs


if __name__ == "__main__":
    main()

