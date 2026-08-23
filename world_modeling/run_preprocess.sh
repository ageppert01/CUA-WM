#!/bin/bash
# run_preprocess.sh — download AgentNet, preprocess transitions, optionally upload.
#
HF_REPO="ageppert/world-model-transitions"
HF_TOKEN="${HF_TOKEN:?set HF_TOKEN in your environment; see .env.example}"

set -e

# shellcheck source=common.sh
. ./common.sh

section "Job started"
echo "Working directory: $(pwd)"

install_packages_simple huggingface_hub

download_agentnet

section "Running preprocessing"
if [ -n "${HF_REPO}" ] && [ -n "${HF_TOKEN}" ]; then
    export HF_TOKEN
    echo "Will upload to HuggingFace repo: ${HF_REPO}"
    python3 preprocess_transitions.py \
        --upload \
        --hf-repo "${HF_REPO}" \
        --hf-token "${HF_TOKEN}"
else
    echo "No HF_REPO/HF_TOKEN set, skipping upload."
    python3 preprocess_transitions.py
fi

section "Output files"
ls -lah transition_*.jsonl preprocessing_stats.txt 2>/dev/null || echo "No output files found"

section "Job finished"