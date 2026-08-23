#!/usr/bin/env python3
"""
preprocess_transitions.py

Extracts transition training data from the AgentNet Ubuntu 5K dataset
for fine-tuning a world model (transition predictor).

Tier 1 only: uses steps that have non-empty reflections as ground truth
transition descriptions.

Filtering:
  - Reflection must be non-empty
  - last_step_correct != False (keep True and None)
  - Action must not be a terminate action
  - Step must not be the last step in the trajectory (needs a next state to exist)
  - last_step_redundant != True

Output: Chat JSONL in HuggingFace conversational format, split into
        transition_train.jsonl and transition_val.jsonl (90/10).

Usage:
  python preprocess_transitions.py [--upload --hf-repo <repo>]
"""

import json
import os
import sys
import random
import argparse
from collections import Counter

# Prompts come from core.prompts so the training data and the inference-time
# world-model call use byte-identical templates. If you change these, you change
# both halves; retrain the LoRA after any edit.
from core.prompts import (
    WORLD_MODEL_SYSTEM_PROMPT as SYSTEM_PROMPT,
    WORLD_MODEL_USER_TEMPLATE as USER_PROMPT_TEMPLATE,
)

# ===========================================================================
# Configuration
# ===========================================================================

JSONL_PATH = "agentnet_ubuntu_5k.jsonl"
OUTPUT_TRAIN = "transition_train.jsonl"
OUTPUT_VAL = "transition_val.jsonl"
OUTPUT_STATS = "preprocessing_stats.txt"
VAL_RATIO = 0.10
RANDOM_SEED = 42


def load_trajectories(path):
    """Load all trajectory entries from the JSONL file."""
    trajectories = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                trajectories.append(entry)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping malformed line {line_num}: {e}")
    return trajectories


def is_terminate_action(code):
    """Check if a code string is a terminate action."""
    return "terminate" in code.lower()


def extract_transitions(trajectories):
    """Extract all valid Tier 1 transition pairs."""
    transitions = []
    stats = Counter()

    for traj_idx, entry in enumerate(trajectories):
        traj = entry.get("traj", [])
        task_id = entry.get("task_id", "unknown")
        instruction = entry.get("instruction", "")

        for step_idx in range(len(traj)):
            step = traj[step_idx]
            val = step.get("value", {})
            stats["total_steps"] += 1

            observation = val.get("observation", "").strip()
            action = val.get("action", "").strip()
            code = val.get("code", "").strip()
            reflection = val.get("reflection", "").strip()
            correct = val.get("last_step_correct")
            redundant = val.get("last_step_redundant")

            # --- Apply filters ---

            # Must have non-empty reflection (Tier 1 requirement)
            if not reflection:
                stats["filtered_empty_reflection"] += 1
                continue

            # Must have non-empty observation and action
            if not observation:
                stats["filtered_empty_observation"] += 1
                continue
            if not action:
                stats["filtered_empty_action"] += 1
                continue

            # Skip incorrect steps
            if correct is False:
                stats["filtered_incorrect"] += 1
                continue

            # Skip redundant steps
            if redundant is True:
                stats["filtered_redundant"] += 1
                continue

            # Skip terminate actions
            if is_terminate_action(code):
                stats["filtered_terminate"] += 1
                continue

            # Must not be the last step (the transition needs a "next state"
            # to have existed for the reflection to describe a real change)
            if step_idx == len(traj) - 1:
                stats["filtered_last_step"] += 1
                continue

            # --- Build training example ---
            user_content = USER_PROMPT_TEMPLATE.format(
                observation=observation,
                action=action,
            )

            example = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": reflection},
                ],
                # Metadata (not used in training, useful for analysis)
                "metadata": {
                    "task_id": task_id,
                    "step_index": step.get("index", step_idx),
                    "instruction": instruction[:200],
                    "code": code,
                    "last_step_correct": correct,
                },
            }

            transitions.append(example)
            stats["kept"] += 1

    return transitions, stats


def split_train_val(transitions, val_ratio, seed):
    """Split transitions into train and val sets."""
    random.seed(seed)
    indices = list(range(len(transitions)))
    random.shuffle(indices)

    val_size = int(len(transitions) * val_ratio)
    val_indices = set(indices[:val_size])

    train = [transitions[i] for i in range(len(transitions)) if i not in val_indices]
    val = [transitions[i] for i in range(len(transitions)) if i in val_indices]

    return train, val


def write_jsonl(data, path):
    """Write a list of dicts to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_stats(stats, n_train, n_val, path):
    """Write preprocessing statistics to a text file."""
    with open(path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("TRANSITION DATA PREPROCESSING STATISTICS\n")
        f.write("=" * 60 + "\n\n")

        total = stats["total_steps"]
        f.write(f"Total steps scanned:          {total:>8d}\n\n")

        f.write("Filtered out:\n")
        for key in sorted(stats.keys()):
            if key.startswith("filtered_"):
                label = key.replace("filtered_", "  ")
                count = stats[key]
                pct = 100 * count / total if total > 0 else 0
                f.write(f"  {label:30s}: {count:>8d} ({pct:.1f}%)\n")

        kept = stats["kept"]
        pct_kept = 100 * kept / total if total > 0 else 0
        f.write(f"\nKept:                         {kept:>8d} ({pct_kept:.1f}%)\n")
        f.write(f"  Train split:                {n_train:>8d}\n")
        f.write(f"  Val split:                  {n_val:>8d}\n")

        # Sample length stats
        f.write(f"\nRandom seed: {RANDOM_SEED}\n")
        f.write(f"Val ratio: {VAL_RATIO}\n")

    # Also print to stdout
    with open(path, "r") as f:
        print(f.read())


def upload_to_huggingface(train_path, val_path, stats_path, repo_id):
    """Upload the processed dataset to HuggingFace."""
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("ERROR: huggingface_hub not available for upload.")
        return False

    api = HfApi()

    # Create the repo if it doesn't exist
    try:
        create_repo(repo_id, repo_type="dataset", exist_ok=True)
        print(f"Repository {repo_id} ready.")
    except Exception as e:
        print(f"Error creating repo: {e}")
        return False

    # Upload files
    for local_path, remote_name in [
        (train_path, "transition_train.jsonl"),
        (val_path, "transition_val.jsonl"),
        (stats_path, "preprocessing_stats.txt"),
    ]:
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=remote_name,
                repo_id=repo_id,
                repo_type="dataset",
            )
            print(f"  Uploaded {remote_name}")
        except Exception as e:
            print(f"  Error uploading {remote_name}: {e}")
            return False

    # Create a basic README
    readme = f"""---
language:
  - en
license: mit
tags:
  - world-model
  - computer-use
  - transition-prediction
---

# World Model Transition Dataset

Derived from the [AgentNet](https://huggingface.co/datasets/xlangai/AgentNet) Ubuntu 5K dataset
for training a world model (transition predictor) for computer-use agents.

## Format

Each example is a chat conversation:
- **System**: World model role description
- **User**: Current screen state observation + planned action
- **Assistant**: Predicted outcome (transition description)

## Source

Tier 1 extraction from AgentNet: steps with non-empty reflections,
correct actions, non-redundant, non-terminal.

See `preprocessing_stats.txt` for detailed filtering statistics.

## Citation

If you use this dataset, please cite the original OpenCUA paper:

```bibtex
@misc{{wang2025opencua,
    title={{OpenCUA: Open Foundations for Computer-Use Agents}},
    author={{Xinyuan Wang and others}},
    year={{2025}},
    eprint={{2508.09123}},
    archivePrefix={{arXiv}},
}}
```
"""
    try:
        api.upload_file(
            path_or_fileobj=readme.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"  Uploaded README.md")
    except Exception as e:
        print(f"  Error uploading README.md: {e}")

    print(f"\nDataset available at: https://huggingface.co/datasets/{repo_id}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Preprocess AgentNet for world model training")
    parser.add_argument("--upload", action="store_true", help="Upload to HuggingFace after processing")
    parser.add_argument("--hf-repo", type=str, default=None, help="HuggingFace repo ID (e.g., username/dataset-name)")
    parser.add_argument("--hf-token", type=str, default=None, help="HuggingFace write token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    # Set HF token if provided
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
    elif "HF_TOKEN" not in os.environ:
        # Try reading from token file
        token_path = os.path.expanduser("~/.huggingface/token")
        if os.path.exists(token_path):
            with open(token_path) as f:
                os.environ["HF_TOKEN"] = f.read().strip()

    if not os.path.exists(JSONL_PATH):
        print(f"ERROR: {JSONL_PATH} not found in {os.getcwd()}")
        sys.exit(1)

    print("Loading trajectories...")
    trajectories = load_trajectories(JSONL_PATH)
    print(f"Loaded {len(trajectories)} trajectories.\n")

    print("Extracting Tier 1 transition pairs...")
    transitions, stats = extract_transitions(trajectories)
    print(f"Extracted {len(transitions)} transition examples.\n")

    print("Splitting train/val...")
    train, val = split_train_val(transitions, VAL_RATIO, RANDOM_SEED)
    print(f"  Train: {len(train)}")
    print(f"  Val:   {len(val)}\n")

    print(f"Writing {OUTPUT_TRAIN}...")
    write_jsonl(train, OUTPUT_TRAIN)
    train_size = os.path.getsize(OUTPUT_TRAIN) / (1024 * 1024)
    print(f"  {train_size:.1f} MB\n")

    print(f"Writing {OUTPUT_VAL}...")
    write_jsonl(val, OUTPUT_VAL)
    val_size = os.path.getsize(OUTPUT_VAL) / (1024 * 1024)
    print(f"  {val_size:.1f} MB\n")

    print(f"Writing {OUTPUT_STATS}...")
    write_stats(stats, len(train), len(val), OUTPUT_STATS)

    # Print a few examples for sanity check
    print("\n" + "=" * 60)
    print("SAMPLE EXAMPLES (first 3)")
    print("=" * 60)
    for i, ex in enumerate(transitions[:3]):
        print(f"\n--- Example {i} ---")
        print(f"Task: {ex['metadata']['instruction'][:100]}")
        print(f"Step: {ex['metadata']['step_index']}")
        user_msg = ex["messages"][1]["content"]
        # Print first 300 chars of user message
        print(f"User (first 300 chars):\n  {user_msg[:300]}")
        assistant_msg = ex["messages"][2]["content"]
        print(f"Assistant (first 300 chars):\n  {assistant_msg[:300]}")

    if args.upload:
        if not args.hf_repo:
            print("\nERROR: --hf-repo required when using --upload")
            sys.exit(1)
        print(f"\nUploading to HuggingFace: {args.hf_repo}")
        upload_to_huggingface(OUTPUT_TRAIN, OUTPUT_VAL, OUTPUT_STATS, args.hf_repo)
    else:
        print("\nSkipping HuggingFace upload (use --upload --hf-repo <repo> to enable)")

    print("\nDone.")


if __name__ == "__main__":
    main()