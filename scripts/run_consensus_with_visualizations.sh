#!/usr/bin/env bash
set -euo pipefail

DEFAULT_CONFIG="configs/runs/round_synthetic.yml"
PYTHON_BIN="${PYTHON:-python}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
invocation_dir="$(pwd)"

usage() {
  cat <<EOF
Usage: $(basename "$0") [CONFIG_PATH]

Run the consensus learner and then generate all visualization outputs.

Arguments:
  CONFIG_PATH  Run YAML config path. Defaults to ${DEFAULT_CONFIG}.

Environment:
  PYTHON       Python executable to use. Defaults to python.
EOF
}

format_duration() {
  local total_seconds="$1"
  printf "%02d:%02d:%02d" \
    "$((total_seconds / 3600))" \
    "$(((total_seconds % 3600) / 60))" \
    "$((total_seconds % 60))"
}

resolve_config_path() {
  local config_arg="$1"
  if [[ "${config_arg}" = /* ]]; then
    printf "%s\n" "${config_arg}"
  elif [[ -f "${invocation_dir}/${config_arg}" ]]; then
    (
      cd -- "$(dirname -- "${invocation_dir}/${config_arg}")"
      printf "%s/%s\n" "$(pwd)" "$(basename -- "${config_arg}")"
    )
  else
    printf "%s/%s\n" "${repo_root}" "${config_arg}"
  fi
}

run_step() {
  local label="$1"
  shift

  local step_start="${SECONDS}"
  printf "\n[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "${label}"
  "$@"
  local step_elapsed="$((SECONDS - step_start))"
  printf "[%s] Finished %s in %s\n" \
    "$(date '+%Y-%m-%d %H:%M:%S')" \
    "${label}" \
    "$(format_duration "${step_elapsed}")"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$#" -gt 1 ]]; then
  usage >&2
  exit 2
fi

config_path="$(resolve_config_path "${1:-${DEFAULT_CONFIG}}")"

if [[ ! -f "${config_path}" ]]; then
  printf "Config file not found: %s\n" "${config_path}" >&2
  exit 1
fi

cd -- "${repo_root}"

total_start="${SECONDS}"
printf "Using config: %s\n" "${config_path}"

mapfile -t visualization_contexts < <(
  "${PYTHON_BIN}" - "${config_path}" <<'PY'
from pathlib import Path
import sys

from flumolscreen.run_config import load_run_config

resolved_config = load_run_config(sys.argv[1])
seen = set()

for job in resolved_config["jobs"]:
    context = (
        str(Path(job["data_dir"])),
        str(Path(job["results_dir"])),
        str(job["train_round_id"]),
    )
    if context in seen:
        continue
    seen.add(context)
    print("\t".join(context))
PY
)

#run_step "consensus learner" \
#  "${PYTHON_BIN}" run_consensus_learner.py --config "${config_path}"

for context in "${visualization_contexts[@]}"; do
  IFS=$'\t' read -r data_dir results_dir round_id <<<"${context}"

  run_step "evaluation visualizations (${round_id})" \
    "${PYTHON_BIN}" scripts/create_evaluation_diagnostics.py \
      --results-dir "${results_dir}" \
      --round-id "${round_id}"

  run_step "selected-model visualizations (${round_id})" \
    "${PYTHON_BIN}" scripts/create_selected_model_diagnostics.py \
      --results-dir "${results_dir}" \
      --round-id "${round_id}"

  run_step "chemical-space visualizations (${round_id})" \
    "${PYTHON_BIN}" scripts/create_chemical_space_diagnostics.py \
      --data-dir "${data_dir}" \
      --results-dir "${results_dir}" \
      --round-ids "${round_id}"
done

total_elapsed="$((SECONDS - total_start))"
printf "\nCompleted learner and visualizations in %s\n" \
  "$(format_duration "${total_elapsed}")"
