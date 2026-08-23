from huggingface_hub import snapshot_download
import sys

try:
    snapshot_download(
        repo_id="xlangai/OpenCUA-7B",
        local_dir="OpenCUA-7B",                
        local_dir_use_symlinks=False  
    )
except Exception as e:
    print(f"Failed to download model: {e}")
    sys.exit(1)
