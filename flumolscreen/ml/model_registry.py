"""Model-specific hyperparameter search spaces for FluMolScreen."""

from __future__ import annotations

from typing import Any

__all__ = ["get_tuning_space", "suggest_model_params"]

SEARCH_SPACES: dict[str, dict[str, dict[str, Any]]] = {
    "ridge": {
        "alpha": {
            "type": "float",
            "low": 1e-4,
            "high": 1e3,
            "log": True,
        }
    },
    "xgboost": {
        "n_estimators": {"type": "int", "low": 100, "high": 600, "step": 50},
        "max_depth": {"type": "int", "low": 2, "high": 8, "step": 1},
        "learning_rate": {
            "type": "float",
            "low": 1e-3,
            "high": 3e-1,
            "log": True,
        },
        "subsample": {"type": "float", "low": 0.5, "high": 1.0},
    },
}


def get_tuning_space(model_type: str) -> dict[str, dict[str, Any]]:
    """Return the registered tuning space for a supported model."""
    model_key = "xgboost" if model_type == "xgb" else model_type
    return SEARCH_SPACES.get(model_key, {})


def suggest_model_params(trial: Any, model_type: str) -> dict[str, Any]:
    """Suggest one hyperparameter set for the requested model."""
    params = {}
    for name, spec in get_tuning_space(model_type).items():
        kind = spec["type"]
        if kind == "float":
            params[name] = trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                log=bool(spec.get("log", False)),
            )
        elif kind == "int":
            params[name] = trial.suggest_int(
                name,
                int(spec["low"]),
                int(spec["high"]),
                step=int(spec.get("step", 1)),
            )
        else:
            raise ValueError(f"Unsupported search-space type: {kind}")
    return params
