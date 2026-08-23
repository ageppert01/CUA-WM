#!/bin/bash
# common.sh — shared helpers for CUA-WM job scripts.
#
# Source from run_*.sh and startup.sh:
#     . ./common.sh
#
# Functions provided:
#   section <name>                   Print a banner with timestamp
#   print_gpu_info                   nvidia-smi + torch CUDA check
#   install_packages_with_strip ...  pip install to ./pip_packages, then
#                                    strip container-provided packages
#   install_packages_simple ...      pip install to ./pip_packages, no strip
#                                    (use only for pure-python helpers)
#   setup_ssh_key                    Place the tunnel SSH key in ~/.ssh
#                                    (CUA_WM_TUNNEL_KEY, default `tunnel_key`)
#   open_ssh_tunnel [user@host]      Reverse tunnel, loops forever. Host comes from
#                                    $1 or CUA_WM_TUNNEL_HOST; port from
#                                    CUA_WM_TUNNEL_PORT (default 9009)
#   download_agentnet                Fetch agentnet_ubuntu_5k.jsonl from HF
#   prepare_model_workspace          Copy /OpenCUA into scratch + fetch model

# Packages that must come from the container, not pip_packages.
# torch/nvidia/triton are CUDA-bound and must match the node's driver.
# transformers/tokenizers are pinned to 4.53.0 by the OpenCUA model code.
# accelerate is installed alongside but stripped to use the container's.
CONTAINER_PACKAGES="torch nvidia triton sympy mpmath transformers tokenizers numpy accelerate"

section() {
    echo ""
    echo "=== $* ($(date)) ==="
}

print_gpu_info() {
    section "GPU info"
    nvidia-smi 2>/dev/null || echo "nvidia-smi not available in container"
    python3 -c "
import torch
print(f'  torch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'  CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print(f'  Devices: {torch.cuda.device_count()}')
" 2>&1 || echo "GPU info check failed (non-fatal)"
}

install_packages_with_strip() {
    section "Installing packages (with container-strip): $*"
    pip install --target=./pip_packages --no-cache-dir "$@" 2>&1 | tail -15

    echo "Stripping container-provided packages from pip_packages..."
    for pkg in ${CONTAINER_PACKAGES}; do
        rm -rf pip_packages/${pkg}*
    done

    # Some pip resolvers ignore the < 1.0 pin on huggingface_hub.
    # transformers 4.53.0 requires huggingface_hub < 1.0.
    python3 -c "
import sys, os
sys.path.insert(0, 'pip_packages')
try:
    import huggingface_hub
    v = tuple(int(x) for x in huggingface_hub.__version__.split('.')[:2])
    if v >= (1, 0):
        print(f'  Removing incompatible huggingface_hub {huggingface_hub.__version__}')
        os.system('rm -rf pip_packages/huggingface_hub*')
    else:
        print(f'  huggingface_hub {huggingface_hub.__version__} is compatible')
except Exception:
    pass
"

    export PYTHONPATH="$(pwd)/pip_packages:${PYTHONPATH:-}"
    export PATH="$(pwd)/pip_packages/bin:${PATH}"
}

install_packages_simple() {
    section "Installing packages (no strip): $*"
    pip install --target=./pip_packages --no-cache-dir "$@" 2>&1 | tail -5

    export PYTHONPATH="$(pwd)/pip_packages:${PYTHONPATH:-}"
    export PATH="$(pwd)/pip_packages/bin:${PATH}"
}

setup_ssh_key() {
    local key="${CUA_WM_TUNNEL_KEY:-tunnel_key}"
    mkdir -p ~/.ssh
    cp "${key}" ~/.ssh/
    chmod 600 ~/.ssh/"${key}"
}

open_ssh_tunnel() {
    local remote_host="${1:-${CUA_WM_TUNNEL_HOST:?set CUA_WM_TUNNEL_HOST to user@host; see .env.example}}"
    local key="${CUA_WM_TUNNEL_KEY:-tunnel_key}"
    local port="${CUA_WM_TUNNEL_PORT:-9009}"
    section "Opening SSH reverse tunnel to ${remote_host}"
    while true; do
        echo "$(date): (Re)connecting SSH tunnel..."
        ssh -i ~/.ssh/"${key}" \
            -N \
            -R "${port}":localhost:"${port}" \
            -o StrictHostKeyChecking=no \
            -o ServerAliveInterval=60 \
            -o ServerAliveCountMax=3 \
            -o ExitOnForwardFailure=yes \
            "${remote_host}"
        echo "$(date): SSH tunnel dropped, reconnecting in 10s..."
        sleep 10
    done
}

download_agentnet() {
    section "Downloading agentnet_ubuntu_5k.jsonl"
    python3 -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id='xlangai/AgentNet',
    repo_type='dataset',
    filename='agentnet_ubuntu_5k.jsonl',
    local_dir='.'
)
print(f'Downloaded to: {path}')
"
}

prepare_model_workspace() {
    # Container filesystem is read-only; everything must run from scratch dir.
    cp -r /OpenCUA/* .
    section "Downloading OpenCUA-7B"
    python fetch_opencua-7B.py
}