#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/train_chpm_384_sp2.yaml}"
RUN_NAME="${RUN_NAME:-chpm_$(date +%Y%m%d_%H%M%S)}"
LOGDIR="${LOGDIR:-${REPO_ROOT}/outputs/${RUN_NAME}}"

NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
RDZV_ID="${RDZV_ID:-${RUN_NAME}}"

ENABLE_WANDB="${ENABLE_WANDB:-0}"
RESUME_MODE="${RESUME_MODE:-auto}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
DRY_RUN="${DRY_RUN:-0}"

: "${WAN_MODEL_ROOT:?Set WAN_MODEL_ROOT to the Wan2.2-TI2V-5B directory}"
: "${LONGLIVE2_CHECKPOINT:?Set LONGLIVE2_CHECKPOINT to the LongLive2 generator checkpoint}"
: "${DATA_ROOT:?Set DATA_ROOT to the prompt dataset}"

case "${RESUME_MODE}" in
  none|auto) ;;
  explicit)
    : "${RESUME_CHECKPOINT:?RESUME_MODE=explicit requires RESUME_CHECKPOINT}"
    ;;
  *)
    echo "RESUME_MODE must be none, auto, or explicit" >&2
    exit 2
    ;;
esac

if [[ "${NODE_RANK}" == "0" ]]; then
  mkdir -p "${LOGDIR}"
  printf '%s\n' \
    "run_name=${RUN_NAME}" \
    "config=${CONFIG}" \
    "nnodes=${NNODES}" \
    "nproc_per_node=${NPROC_PER_NODE}" \
    "resume_mode=${RESUME_MODE}" \
    > "${LOGDIR}/run_metadata.txt"
  touch "${LOGDIR}/.ready"
else
  until [[ -f "${LOGDIR}/.ready" ]]; do sleep 1; done
fi

TRAIN_ARGS=(
  --config_path "${CONFIG}"
  --logdir "${LOGDIR}"
  --resume-mode "${RESUME_MODE}"
)
if [[ "${RESUME_MODE}" == "explicit" ]]; then
  TRAIN_ARGS+=(--resume-checkpoint "${RESUME_CHECKPOINT}")
fi
if [[ "${ENABLE_WANDB}" == "1" ]]; then
  TRAIN_ARGS+=(--enable-wandb)
else
  export WANDB_MODE=disabled
fi
TRAIN_ARGS+=("$@")

if [[ "${NNODES}" == "1" ]]; then
  TORCHRUN_ARGS=(--standalone --nnodes=1 --nproc-per-node="${NPROC_PER_NODE}")
else
  TORCHRUN_ARGS=(
    --nnodes="${NNODES}"
    --nproc-per-node="${NPROC_PER_NODE}"
    --node-rank="${NODE_RANK}"
    --master-addr="${MASTER_ADDR}"
    --master-port="${MASTER_PORT}"
    --rdzv-id="${RDZV_ID}"
  )
fi

COMMAND=(torchrun "${TORCHRUN_ARGS[@]}" train.py "${TRAIN_ARGS[@]}")
printf '[CHPM launcher]'
printf ' %q' "${COMMAND[@]}"
printf '\n'
if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

cd "${REPO_ROOT}"
"${COMMAND[@]}" 2>&1 | tee -a "${LOGDIR}/console_node${NODE_RANK}.log"
