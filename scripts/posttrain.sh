#!/usr/bin/env bash
# Copyright 2026 Chubin Zhang. All rights reserved.
# SPDX-License-Identifier: MIT
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RECIPE="${1:?Usage: scripts/posttrain.sh configs/posttrain/robocasa.yaml [--key value ...]}"
shift
PYTHON="${PYTHON:-python}"
NUM_GPUS="${NUM_GPUS:-$($PYTHON -c 'import torch; print(torch.cuda.device_count())')}"
NUM_NODES="${NUM_NODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
for key in NUM_GPUS NUM_NODES BATCH_SIZE GLOBAL_BATCH_SIZE; do
  [[ "${!key}" =~ ^[1-9][0-9]*$ ]] || { echo "$key must be a positive integer" >&2; exit 2; }
done
[[ "$NODE_RANK" =~ ^[0-9]+$ ]] && ((NODE_RANK < NUM_NODES)) || { echo 'Invalid NODE_RANK' >&2; exit 2; }
WORLD_SIZE=$((NUM_GPUS * NUM_NODES))
MICRO_BATCH=$((WORLD_SIZE * BATCH_SIZE))
((GLOBAL_BATCH_SIZE % MICRO_BATCH == 0)) || { echo 'GLOBAL_BATCH_SIZE must divide by GPUs * BATCH_SIZE' >&2; exit 2; }
[[ -f "$RECIPE" ]] || { echo "Missing recipe: $RECIPE" >&2; exit 2; }
export PYTHONPATH="$ROOT"
export TOKENIZERS_PARALLELISM=false NO_ALBUMENTATIONS_UPDATE=1 MPLBACKEND=Agg
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
# Do not impose a cluster-specific RDMA/HCA interface. Multi-node users configure NCCL themselves.
if ((NUM_NODES == 1)); then export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"; fi
CMD=("$PYTHON" -m accelerate.commands.launch --config_file "${ACCELERATE_CONFIG:-configs/distributed/zero2.yaml}"
  --num_processes "$WORLD_SIZE" --num_machines "$NUM_NODES" --machine_rank "$NODE_RANK"
  --main_process_ip "${MASTER_ADDR:-127.0.0.1}" --main_process_port "${MASTER_PORT:-29500}"
  starVLA/training/train_starvla.py --config_yaml "$RECIPE"
  --datasets.vla_data.per_device_batch_size "$BATCH_SIZE"
  --trainer.gradient_accumulation_steps "$((GLOBAL_BATCH_SIZE / MICRO_BATCH))" "$@")
printf 'Launching:'; printf ' %q' "${CMD[@]}"; printf '\n'
[[ "${DRY_RUN:-0}" == 1 ]] || exec "${CMD[@]}"
