#!/bin/bash
# run_eval.sh — install deps, load model + LoRA adapter, run inference on val set.
#
HF_TOKEN="${HF_TOKEN:?set HF_TOKEN in your environment; see .env.example}"
HF_REPO="ageppert/world-model-7b-lora"

# Limit samples for testing (leave empty for all 4210).
MAX_SAMPLES=""

set -e

# shellcheck source=common.sh
. ./common.sh

section "Job started"

# Create output files early so HTCondor transfer never fails.
touch eval_predictions.jsonl eval_stats.txt

install_packages_with_strip \
    "trl>=0.14,<0.16" \
    "peft>=0.12,<0.14" \
    "accelerate>=0.34,<1.0" \
    bitsandbytes \
    datasets \
    "huggingface_hub>=0.25,<1.0"

[ -n "${HF_TOKEN}" ] && export HF_TOKEN

# Build eval command
EVAL_CMD="python3 eval_world_model.py"
[ -n "${MAX_SAMPLES}" ] && EVAL_CMD="${EVAL_CMD} --max-samples ${MAX_SAMPLES}"
[ -n "${HF_REPO}" ] && [ -n "${HF_TOKEN}" ] && \
    EVAL_CMD="${EVAL_CMD} --upload --hf-repo ${HF_REPO}"

section "Starting evaluation"
echo "Command: ${EVAL_CMD}"
set +e
${EVAL_CMD}
EVAL_EXIT=$?
set -e
[ ${EVAL_EXIT} -ne 0 ] && echo "WARNING: Evaluation exited with code ${EVAL_EXIT}"

section "Output files"
ls -lah eval_predictions.jsonl eval_stats.txt 2>/dev/null || echo "No output files"

section "Job finished"