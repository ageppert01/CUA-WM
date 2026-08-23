#!/bin/bash
# run_train.sh — install training deps, run LoRA fine-tuning of OpenCUA-7B.
#
# Set SMOKE_TEST=1 for a 10-step validation run; 0 for full 3-epoch training.
#
SMOKE_TEST=0
HF_TOKEN="${HF_TOKEN:?set HF_TOKEN in your environment; see .env.example}"
HUB_MODEL_ID="ageppert/world-model-7b-lora"

set -e

# shellcheck source=common.sh
. ./common.sh

section "Job started"
echo "Hostname: $(hostname)"
echo "Working directory: $(pwd)"

# Status file for HTCondor transfer — write early so the transfer always succeeds.
echo "status=started" > training_status.txt
echo "started=$(date)" >> training_status.txt

print_gpu_info

echo "Container transformers:"
python3 -c "import transformers; print(f'  transformers: {transformers.__version__}')" 2>&1

install_packages_with_strip \
    "trl>=0.14,<0.16" \
    "peft>=0.12,<0.14" \
    "accelerate>=0.34,<1.0" \
    bitsandbytes \
    datasets \
    "huggingface_hub>=0.25,<1.0"

[ -n "${HF_TOKEN}" ] && export HF_TOKEN

section "Package versions (torch/transformers/accelerate come from container)"
python3 -c "
import torch; print(f'torch: {torch.__version__} from {torch.__file__}')
import transformers; print(f'transformers: {transformers.__version__} from {transformers.__file__}')
import trl; print(f'trl: {trl.__version__}')
import peft; print(f'peft: {peft.__version__}')
import accelerate; print(f'accelerate: {accelerate.__version__}')
import datasets; print(f'datasets: {datasets.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
"

# Build train command
TRAIN_CMD="python3 train_world_model.py"
[ "${SMOKE_TEST}" = "1" ] && TRAIN_CMD="${TRAIN_CMD} --smoke-test"
[ -n "${HUB_MODEL_ID}" ] && [ -n "${HF_TOKEN}" ] && \
    TRAIN_CMD="${TRAIN_CMD} --push-to-hub --hub-model-id ${HUB_MODEL_ID}"

section "Starting training"
echo "Command: ${TRAIN_CMD}"
set +e  # Don't exit on training failure — still package outputs below
${TRAIN_CMD}
TRAIN_EXIT=$?
set -e
[ ${TRAIN_EXIT} -ne 0 ] && echo "WARNING: Training exited with code ${TRAIN_EXIT}"

# Package outputs for HTCondor transfer (touch empty tars as fallback).
section "Packaging outputs"
touch world_model_final.tar.gz training_logs.tar.gz

if [ -d "world_model_final" ]; then
    tar czf world_model_final.tar.gz world_model_final/
    echo "Packaged final model: $(du -h world_model_final.tar.gz | cut -f1)"
else
    echo "No final model directory found"
fi

if [ -d "world_model_output" ]; then
    # Only grab trainer_state and training_args, not full checkpoints.
    find world_model_output -name "trainer_state.json" -o -name "training_args.bin" \
        | tar czf training_logs.tar.gz -T -  2>/dev/null || true
    echo "Packaged training logs"
fi

section "Output files"
echo "Checkpoints:"
ls -lah world_model_output/ 2>/dev/null | head -20 || echo "  No checkpoint dir"
echo "Final model:"
ls -lah world_model_final/ 2>/dev/null || echo "  No final model dir"
echo "Packaged files:"
ls -lah *.tar.gz 2>/dev/null || echo "  No tar files"

section "Job finished"

# Final status file for HTCondor transfer
echo "exit_code=${TRAIN_EXIT}" > training_status.txt
echo "finished=$(date)" >> training_status.txt
echo "smoke_test=${SMOKE_TEST}" >> training_status.txt