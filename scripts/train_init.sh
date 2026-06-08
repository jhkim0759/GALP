#!/bin/bash
# Training launcher (3D-FUTURE + ScanNet)
# - Floor rotation prediction (xz2f_rot)
# - EMA model enabled
# - Point-map surface loss
# - Select GPUs via CUDA_VISIBLE_DEVICES (default: 6,7)

cd "$(dirname "$0")/.."

NUM_MACHINES=1
MACHINE_RANK=0
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=, read -r -a GPU_IDS <<< "$GPU_LIST"
NUM_LOCAL_GPUS="${#GPU_IDS[@]}"
LAUNCHER_PYTHON="${PYTHON:-python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib}"
mkdir -p "$MPLCONFIGDIR"

# NCCL debugging + longer timeout to survive slow rank-0 ops (NFS writes, wandb upload)
export TORCH_NCCL_TRACE_BUFFER_SIZE=8192
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
OUTPUT_DIR="${OUTPUT_DIR:-./output}"

# Bypass accelerate's CLI entrypoint because it imports a broken torch._dynamo path in trellis.
CUDA_VISIBLE_DEVICES="$GPU_LIST" "$LAUNCHER_PYTHON" -m accelerate.commands.launch \
  --num_machines $NUM_MACHINES \
  --num_processes $(( $NUM_MACHINES * $NUM_LOCAL_GPUS )) \
  --machine_rank $MACHINE_RANK \
  --main_process_port 29504 \
  -m src.train \
  --config configs/init_train.yaml \
  --pin_memory \
  --allow_tf32 \
  --gradient_accumulation_steps 4 \
  --output_dir "$OUTPUT_DIR" \
  --num_workers 12 \
  --dataset_mix future+scannet \
  --tag init_train \
  --mesh_aug_prob 1.0 \
  --use_high_pointmap \
  --use_pm_surface_loss \
  --pm_surface_loss_weight 0.1 \
  --max_val_steps 0 \
  "$@"
