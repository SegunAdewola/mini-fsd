#!/usr/bin/env bash
#
# Times every report candidate in one cool session (T4, E1, E3, E4).
#
# The fanless M1 slows down as it heats, and background work skews p50, so
# the report takes latency only from runs/harness/latency.jsonl (see
# minifsd.eval.report.with_latency). This script fills that log: before
# each run it waits for the machine to go quiet, then lets it cool for a
# fixed pause. Accuracy is logged too but the report ignores it here.
#
# Usage, from the repo root, plugged in, with other apps closed:
#   bash scripts/latency.sh
# Takes about 30 minutes. COOL_SECONDS and MAX_LOAD override the defaults.

set -euo pipefail

readonly OUT="runs/harness/latency.jsonl"
readonly COOL_SECONDS="${COOL_SECONDS:-60}"
readonly MAX_LOAD="${MAX_LOAD:-1.5}"
readonly NYU="nyuv2-val"
readonly SIM="coco:data/sim/v10/val"

# candidate and dataset, one pair per line. Each table reads latency from
# one dataset: floor and depth from nyuv2-val, detection from v10 val.
readonly RUNS="
floor-classical ${NYU}
floor-segformer-b0 ${NYU}
floor-segformer-b0-v10 ${NYU}
floor-dinov2-s ${NYU}
floor-dinov2-s-v10 ${NYU}
detect-yolo26n ${SIM}
detect-rfdetr-nano ${SIM}
depth-anything-v2-small ${NYU}
depth-student ${NYU}
hydra-floor ${NYU}
hydra-depth ${NYU}
hydra-detect ${SIM}
hydra-all ${NYU}
"

# Prints the 1-minute load average.
load_1m() {
  sysctl -n vm.loadavg | awk '{print $2}'
}

# Succeeds while the 1-minute load is at or over MAX_LOAD (awk does the
# float comparison, which bash can't).
is_busy() {
  awk -v load="$(load_1m)" -v max="${MAX_LOAD}" 'BEGIN { exit (load < max) }'
}

# Waits until the machine is quiet, then COOL_SECONDS more.
wait_until_quiet() {
  while is_busy; do
    echo "  load is $(load_1m), over ${MAX_LOAD}; waiting"
    sleep 15
  done
  sleep "${COOL_SECONDS}"
}

main() {
  if ! pmset -g batt | grep -q "AC Power"; then
    echo "On battery: plug in first (macOS throttles on battery)." >&2
    exit 1
  fi
  mkdir -p "$(dirname "${OUT}")"
  local candidate dataset
  while read -r candidate dataset; do
    [[ -z "${candidate}" ]] && continue
    echo "== ${candidate} on ${dataset}"
    wait_until_quiet
    pixi run -e cv python -m minifsd.eval.harness \
      --candidate "${candidate}" --dataset "${dataset}" --out "${OUT}"
  done <<<"${RUNS}"
  echo "done; lines in the log now:"
  wc -l "${OUT}"
}

main "$@"
