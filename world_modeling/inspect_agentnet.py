#!/usr/bin/env python3
"""
inspect_agentnet.py

Inspects the AgentNet Ubuntu 5K JSONL file to understand the data structure
for building a world-model transition dataset.

Expected input: agentnet_ubuntu_5k.jsonl in the current working directory.
"""

import json
import sys
import os
from collections import Counter, defaultdict

JSONL_PATH = "agentnet_ubuntu_5k.jsonl"
NUM_DETAILED = 5        # number of trajectories to print in detail
NUM_TRANSITIONS = 3     # number of consecutive-observation pairs to print per trajectory

SEPARATOR = "=" * 80
SUB_SEPARATOR = "-" * 60


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


def print_schema(entry, prefix=""):
    """Print the top-level keys and types of a dictionary."""
    for key, value in entry.items():
        vtype = type(value).__name__
        if isinstance(value, list):
            if len(value) > 0:
                inner_type = type(value[0]).__name__
                print(f"  {prefix}{key}: list[{inner_type}] (len={len(value)})")
            else:
                print(f"  {prefix}{key}: list (empty)")
        elif isinstance(value, dict):
            print(f"  {prefix}{key}: dict with keys {list(value.keys())}")
        elif isinstance(value, str):
            preview = value[:80] + "..." if len(value) > 80 else value
            print(f"  {prefix}{key}: {vtype} = \"{preview}\"")
        else:
            print(f"  {prefix}{key}: {vtype} = {value}")


def analyze_step(step):
    """Extract key info from a single trajectory step."""
    val = step.get("value", {})
    return {
        "index": step.get("index"),
        "image": step.get("image"),
        "step_id": step.get("step_id"),
        "marks": step.get("marks"),
        "observation": val.get("observation", ""),
        "thought": val.get("thought", ""),
        "action": val.get("action", ""),
        "code": val.get("code", ""),
        "reflection": val.get("reflection", ""),
        "last_step_correct": val.get("last_step_correct"),
        "last_step_redundant": val.get("last_step_redundant"),
    }


def main():
    if not os.path.exists(JSONL_PATH):
        print(f"ERROR: {JSONL_PATH} not found in current directory ({os.getcwd()})")
        sys.exit(1)

    file_size_mb = os.path.getsize(JSONL_PATH) / (1024 * 1024)
    print(f"File: {JSONL_PATH} ({file_size_mb:.1f} MB)")
    print(f"Loading trajectories...")
    trajectories = load_trajectories(JSONL_PATH)
    print(f"Loaded {len(trajectories)} trajectories.\n")

    # =========================================================================
    # SECTION 1: Top-level schema of first entry
    # =========================================================================
    print(SEPARATOR)
    print("SECTION 1: TOP-LEVEL SCHEMA (first entry)")
    print(SEPARATOR)
    first = trajectories[0]
    print_schema(first)

    traj_steps = first.get("traj", [])
    if traj_steps:
        print(f"\n  First trajectory step schema:")
        print_schema(traj_steps[0], prefix="    ")
        val = traj_steps[0].get("value", {})
        if val:
            print(f"\n  Step 'value' sub-schema:")
            print_schema(val, prefix="      ")

    # =========================================================================
    # SECTION 2: Detailed view of first N trajectories
    # =========================================================================
    print(f"\n{SEPARATOR}")
    print(f"SECTION 2: DETAILED VIEW OF FIRST {NUM_DETAILED} TRAJECTORIES")
    print(SEPARATOR)

    for i, entry in enumerate(trajectories[:NUM_DETAILED]):
        traj = entry.get("traj", [])
        print(f"\n{'*' * 60}")
        print(f"Trajectory {i}: task_id={entry.get('task_id', 'N/A')}")
        print(f"  instruction: {entry.get('instruction', 'N/A')[:120]}")
        print(f"  task_completed: {entry.get('task_completed')}")
        print(f"  task_difficulty: {entry.get('task_difficulty')}")
        print(f"  alignment_score: {entry.get('alignment_score')}")
        print(f"  efficiency_score: {entry.get('efficiency_score')}")
        print(f"  num_steps: {len(traj)}")

        # Print first few steps in detail
        for j, step in enumerate(traj[:3]):
            info = analyze_step(step)
            print(f"\n  {SUB_SEPARATOR}")
            print(f"  Step {info['index']} (image: {info['image']})")
            print(f"    last_step_correct: {info['last_step_correct']}")
            print(f"    last_step_redundant: {info['last_step_redundant']}")
            print(f"    step_id: {info['step_id']}")
            print(f"    marks: {info['marks']}")
            print(f"    OBSERVATION ({len(info['observation'])} chars):")
            print(f"      {info['observation'][:500]}")
            print(f"    THOUGHT ({len(info['thought'])} chars):")
            print(f"      {info['thought'][:300]}")
            print(f"    ACTION ({len(info['action'])} chars):")
            print(f"      {info['action'][:300]}")
            print(f"    CODE:")
            print(f"      {info['code'][:200]}")
            print(f"    REFLECTION ({len(info['reflection'])} chars):")
            print(f"      {info['reflection'][:400]}")

    # =========================================================================
    # SECTION 3: Consecutive observation pairs (transition view)
    # =========================================================================
    print(f"\n{SEPARATOR}")
    print(f"SECTION 3: CONSECUTIVE OBSERVATION PAIRS (TRANSITION VIEW)")
    print(f"  Showing {NUM_TRANSITIONS} transition(s) from first {NUM_DETAILED} trajectories")
    print(SEPARATOR)

    for i, entry in enumerate(trajectories[:NUM_DETAILED]):
        traj = entry.get("traj", [])
        if len(traj) < 2:
            print(f"\nTrajectory {i}: only {len(traj)} step(s), skipping.")
            continue

        print(f"\nTrajectory {i}: \"{entry.get('instruction', '')[:80]}\"")
        for j in range(min(NUM_TRANSITIONS, len(traj) - 1)):
            step_a = analyze_step(traj[j])
            step_b = analyze_step(traj[j + 1])
            print(f"\n  Transition {j} -> {j+1}:")
            print(f"    STATE_t (observation @ step {j}):")
            print(f"      {step_a['observation'][:400]}")
            print(f"    ACTION_t (code @ step {j}):")
            print(f"      {step_a['code']}")
            print(f"    REFLECTION_t (reflection @ step {j}):")
            print(f"      {step_a['reflection'][:400]}")
            print(f"    STATE_t+1 (observation @ step {j+1}):")
            print(f"      {step_b['observation'][:400]}")

    # =========================================================================
    # SECTION 4: Dataset-wide statistics
    # =========================================================================
    print(f"\n{SEPARATOR}")
    print("SECTION 4: DATASET-WIDE STATISTICS")
    print(SEPARATOR)

    step_counts = []
    total_steps = 0
    correct_counts = Counter()      # True, False, None
    redundant_counts = Counter()
    obs_lengths = []
    thought_lengths = []
    action_lengths = []
    reflection_lengths = []
    code_patterns = Counter()
    task_completed_counts = Counter()
    difficulty_counts = Counter()
    empty_obs_count = 0
    empty_reflection_count = 0

    # Check which top-level keys exist across entries (schema consistency)
    all_keys = Counter()

    for entry in trajectories:
        for key in entry.keys():
            all_keys[key] += 1

        task_completed_counts[entry.get("task_completed")] += 1
        difficulty_counts[entry.get("task_difficulty")] += 1

        traj = entry.get("traj", [])
        step_counts.append(len(traj))
        total_steps += len(traj)

        for step in traj:
            info = analyze_step(step)
            correct_counts[info["last_step_correct"]] += 1
            redundant_counts[info["last_step_redundant"]] += 1
            obs_lengths.append(len(info["observation"]))
            thought_lengths.append(len(info["thought"]))
            action_lengths.append(len(info["action"]))
            reflection_lengths.append(len(info["reflection"]))

            if not info["observation"].strip():
                empty_obs_count += 1
            if not info["reflection"].strip():
                empty_reflection_count += 1

            # Extract the function name from code (e.g., pyautogui.click -> click)
            code = info["code"]
            if "pyautogui." in code:
                func = code.split("pyautogui.")[1].split("(")[0]
                code_patterns[func] += 1
            elif code.strip():
                # Non-pyautogui code
                code_patterns[code.strip()[:40]] += 1

    n = len(trajectories)
    print(f"\n  Total trajectories: {n}")
    print(f"  Total steps: {total_steps}")
    print(f"  Total transition pairs: {total_steps - n}")
    print(f"    (each trajectory of k steps yields k-1 transition pairs)")

    print(f"\n  Steps per trajectory:")
    print(f"    min:    {min(step_counts)}")
    print(f"    max:    {max(step_counts)}")
    print(f"    mean:   {sum(step_counts)/len(step_counts):.1f}")
    print(f"    median: {sorted(step_counts)[len(step_counts)//2]}")

    # Step count distribution (histogram buckets)
    print(f"\n  Step count distribution:")
    buckets = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 30), (31, 50), (51, 100), (101, 999)]
    for lo, hi in buckets:
        count = sum(1 for s in step_counts if lo <= s <= hi)
        if count > 0:
            print(f"    {lo:3d}-{hi:3d} steps: {count:5d} trajectories ({100*count/n:.1f}%)")

    print(f"\n  Task completed: {dict(task_completed_counts)}")
    print(f"  Task difficulty distribution: {dict(sorted(difficulty_counts.items(), key=lambda x: (x[0] is None, x[0] or 0)))}")

    print(f"\n  Step correctness:")
    print(f"    last_step_correct=True:  {correct_counts[True]:6d} ({100*correct_counts[True]/total_steps:.1f}%)")
    print(f"    last_step_correct=False: {correct_counts[False]:6d} ({100*correct_counts[False]/total_steps:.1f}%)")
    print(f"    last_step_correct=None:  {correct_counts[None]:6d} ({100*correct_counts[None]/total_steps:.1f}%)")

    print(f"\n  Step redundancy:")
    print(f"    last_step_redundant=True:  {redundant_counts[True]:6d} ({100*redundant_counts[True]/total_steps:.1f}%)")
    print(f"    last_step_redundant=False: {redundant_counts[False]:6d} ({100*redundant_counts[False]/total_steps:.1f}%)")
    print(f"    last_step_redundant=None:  {redundant_counts[None]:6d} ({100*redundant_counts[None]/total_steps:.1f}%)")

    print(f"\n  Empty fields:")
    print(f"    Empty observation:  {empty_obs_count:6d} ({100*empty_obs_count/total_steps:.1f}%)")
    print(f"    Empty reflection:   {empty_reflection_count:6d} ({100*empty_reflection_count/total_steps:.1f}%)")

    def percentiles(arr):
        s = sorted(arr)
        n = len(s)
        return {
            "min": s[0],
            "p25": s[n // 4],
            "median": s[n // 2],
            "p75": s[3 * n // 4],
            "max": s[-1],
            "mean": sum(s) / n,
        }

    print(f"\n  Field lengths (characters):")
    for name, arr in [("observation", obs_lengths), ("thought", thought_lengths),
                      ("action", action_lengths), ("reflection", reflection_lengths)]:
        if arr:
            p = percentiles(arr)
            print(f"    {name:12s}: min={p['min']:5d}  p25={p['p25']:5d}  "
                  f"median={p['median']:5d}  p75={p['p75']:5d}  max={p['max']:6d}  mean={p['mean']:.0f}")

    print(f"\n  Top 15 action types (from code field):")
    for func, count in code_patterns.most_common(15):
        print(f"    {func:25s}: {count:6d} ({100*count/total_steps:.1f}%)")

    # =========================================================================
    # SECTION 5: Schema consistency check
    # =========================================================================
    print(f"\n{SEPARATOR}")
    print("SECTION 5: SCHEMA CONSISTENCY")
    print(SEPARATOR)
    print(f"\n  Top-level keys and how many of {n} entries have them:")
    for key, count in sorted(all_keys.items(), key=lambda x: -x[1]):
        marker = " <-- INCONSISTENT" if count != n else ""
        print(f"    {key:30s}: {count:5d} / {n}{marker}")

    # =========================================================================
    # SECTION 6: Image path patterns
    # =========================================================================
    print(f"\n{SEPARATOR}")
    print("SECTION 6: IMAGE PATH PATTERNS")
    print(SEPARATOR)
    sample_images = []
    for entry in trajectories[:20]:
        traj = entry.get("traj", [])
        for step in traj[:2]:
            img = step.get("image", "")
            if img:
                sample_images.append(img)
    print(f"\n  First 10 image filenames:")
    for img in sample_images[:10]:
        print(f"    {img}")

    # Check if images are UUIDs or have path prefixes
    has_path_sep = sum(1 for img in sample_images if "/" in img or "\\" in img)
    print(f"\n  Images with path separators: {has_path_sep}/{len(sample_images)}")
    extensions = Counter(img.rsplit(".", 1)[-1] if "." in img else "none" for img in sample_images)
    print(f"  Image extensions: {dict(extensions)}")

    print(f"\n{SEPARATOR}")
    print("INSPECTION COMPLETE")
    print(SEPARATOR)


if __name__ == "__main__":
    main()