#!/bin/bash
# run_inspect.sh — download AgentNet JSONL, run inspection.

set -e

# shellcheck source=common.sh
. ./common.sh

section "Job started"
echo "Working directory: $(pwd)"
python3 --version

install_packages_simple huggingface_hub

download_agentnet

echo ""
echo "=== Contents of working directory ==="
ls -lah *.jsonl 2>/dev/null || echo "No .jsonl files found at top level"

section "Running inspection script"
python3 inspect_agentnet.py

section "Job finished"