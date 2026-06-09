"""Canonical target and target-class metadata for FluMolScreen."""

from __future__ import annotations

TARGET_CLASS_REGISTRY = {
    "furin": {
        "display_name": "Furin",
        "targets": ["furin"],
        "reference_target": "furin",
        "compound_file": "compounds_furin_standardized.csv",
    },
    "fasn": {
        "display_name": "FASN",
        "targets": ["fasn"],
        "reference_target": "fasn",
        "compound_file": "compounds_fasn_standardized.csv",
    },
    "pa": {
        "display_name": "PA",
        "targets": ["pa_ph1n1", "pa_h3n2", "pa_h5n1"],
        "reference_target": "pa_ph1n1",
        "compound_file": "compounds_pa_standardized.csv",
    },
    "na": {
        "display_name": "NA",
        "targets": ["na_ph1n1", "na_h3n2", "na_h5n1"],
        "reference_target": "na_ph1n1",
        "compound_file": "compounds_na_standardized.csv",
    },
}

TARGET_REGISTRY = {
    "furin": {
        "target_class": "furin",
        "strain": None,
    },
    "fasn": {
        "target_class": "fasn",
        "strain": None,
    },
    "pa_ph1n1": {
        "target_class": "pa",
        "strain": "ph1n1",
    },
    "pa_h3n2": {
        "target_class": "pa",
        "strain": "h3n2",
    },
    "pa_h5n1": {
        "target_class": "pa",
        "strain": "h5n1",
    },
    "na_ph1n1": {
        "target_class": "na",
        "strain": "ph1n1",
    },
    "na_h3n2": {
        "target_class": "na",
        "strain": "h3n2",
    },
    "na_h5n1": {
        "target_class": "na",
        "strain": "h5n1",
    },
}


def resolve_target_class_targets(target_class: str) -> list[str]:
    """Return canonical targets for a target class."""
    if target_class not in TARGET_CLASS_REGISTRY:
        raise ValueError(f"Unknown target_class: {target_class}")
    return list(TARGET_CLASS_REGISTRY[target_class]["targets"])


def resolve_target_class_reference_target(target_class: str) -> str:
    """Return the reference target for target-class pooled models."""
    if target_class not in TARGET_CLASS_REGISTRY:
        raise ValueError(f"Unknown target_class: {target_class}")
    return TARGET_CLASS_REGISTRY[target_class]["reference_target"]


def resolve_target_to_label(
    target_class: str,
    targets: list[str] | None = None,
) -> dict[str, str]:
    """Return target -> short-label mapping for target-context feature names."""
    if target_class not in TARGET_CLASS_REGISTRY:
        raise ValueError(f"Unknown target_class: {target_class}")

    resolved_targets = (
        resolve_target_class_targets(target_class) if targets is None else targets
    )
    prefix = f"{target_class}_"
    return {
        target: target.removeprefix(prefix)
        for target in resolved_targets
    }


def resolve_target_class_for_target(target: str) -> str:
    """Return the target-class slug for one concrete target."""
    if target not in TARGET_REGISTRY:
        raise ValueError(f"Unknown target: {target}")
    return TARGET_REGISTRY[target]["target_class"]


def resolve_compound_file_for_target_class(target_class: str) -> str:
    """Return the shared compound-library filename for a target class."""
    if target_class not in TARGET_CLASS_REGISTRY:
        raise ValueError(f"Unknown target_class: {target_class}")
    return TARGET_CLASS_REGISTRY[target_class]["compound_file"]
