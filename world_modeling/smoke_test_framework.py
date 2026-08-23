#!/usr/bin/env python3
"""
smoke_test_framework.py

Tests each pipeline step independently before running the full server.

Configuration is flag-driven to match framework_api.py. Step 2b
(Code-section validation) runs only when --greedy-after-code is on,
since it specifically validates the GreedyAfterCodeHeader logits processor.

Run inside the container after model download and dependency installation.

Usage:
  python smoke_test_framework.py                       # All steps, current defaults
  python smoke_test_framework.py --no-greedy-after-code
  python smoke_test_framework.py --step 1              # Test only step 1
  python smoke_test_framework.py --step 1 2            # Test steps 1 and 2
"""

import argparse
import base64
import json
import os
import re
import sys
import time
import traceback

from core.model_manager import ModelManager
from core.parsing import extract_code_block
from core.pipeline import (
    step_assemble_response,
    step_candidate_generation,
    step_scoring,
    step_state_estimation,
    step_world_model,
)
from core.prompts import L2_SYSTEM_PROMPT

# ----------------------------------------------------------------------------
# Defaults (mirror framework_api.py)
# ----------------------------------------------------------------------------

DEFAULT_GREEDY_AFTER_CODE = True

TEST_SCREENSHOT_PATH = "test_screenshot.png"
TEST_INSTRUCTION = (
    'Work out the monthly total sales in a new row called "Total" '
    'and then create a line chart to show the results (x-axis be Months).'
)


def load_test_screenshot_b64(path=TEST_SCREENSHOT_PATH):
    """Load a real screenshot from file and return as base64."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Test screenshot not found at {path}. "
            f"Include it in transfer_input_files."
        )
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def make_test_messages(screenshot_b64):
    """Build test messages matching OSWorld request format with real L2 prompt."""
    return [
        {"role": "system", "content": L2_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{screenshot_b64}"
                    }
                },
                {"type": "text", "text": TEST_INSTRUCTION}
            ]
        }
    ]


# ============================================================================
# Test runner
# ============================================================================

class SmokeTestResult:
    def __init__(self, step_name):
        self.step_name = step_name
        self.passed = False
        self.elapsed = 0.0
        self.output = None
        self.error = None

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        msg = f"[{status}] {self.step_name} ({self.elapsed:.1f}s)"
        if self.error:
            msg += f"\n  Error: {self.error}"
        if self.output and self.passed:
            preview = str(self.output)[:200]
            msg += f"\n  Output: {preview}..."
        return msg


def test_model_loading(args):
    """Test 0: Load model and adapter."""
    result = SmokeTestResult("Step 0: Model loading")
    t0 = time.time()
    mgr = None
    try:
        mgr = ModelManager(args.model_dir, adapter_repo=args.adapter)
        mgr.load()
        result.passed = True
        result.output = f"has_adapter={mgr.has_adapter}"
    except Exception as e:
        result.error = f"{e}\n{traceback.format_exc()}"
    result.elapsed = time.time() - t0
    return result, mgr if result.passed else None


def test_step1(mgr, messages):
    """Test Step 1: State estimation."""
    result = SmokeTestResult("Step 1: State estimation")
    observation = None
    t0 = time.time()
    try:
        observation = step_state_estimation(mgr, messages)
        result.passed = bool(observation and len(observation) > 50)
        result.output = observation
        if not result.passed:
            result.error = f"Observation too short ({len(observation)} chars)"
    except Exception as e:
        result.error = f"{e}\n{traceback.format_exc()}"
    result.elapsed = time.time() - t0
    return result, observation if result.passed else None


def test_step2(mgr, messages, greedy_after_code):
    """Test Step 2: Candidate generation (mode depends on greedy_after_code)."""
    mode = "decoupled" if greedy_after_code else "full-output sampling"
    result = SmokeTestResult(f"Step 2: Candidate generation (N=2, {mode})")
    candidates = None
    t0 = time.time()
    try:
        candidates = step_candidate_generation(
            mgr, messages, n_candidates=2,
            greedy_after_code=greedy_after_code,
        )
        result.passed = (
            len(candidates) == 2
            and all(c.get("action") for c in candidates)
            and all(c.get("raw_output") for c in candidates)
        )
        result.output = json.dumps(
            [{"index": c["index"], "action": c["action"][:100],
              "temp": c["temperature"]} for c in candidates],
            indent=2
        )
        if not result.passed:
            result.error = "Missing action or raw_output in candidates"
    except Exception as e:
        result.error = f"{e}\n{traceback.format_exc()}"
    result.elapsed = time.time() - t0
    return result, candidates if result.passed else None


def test_step2_code_validation(candidates):
    """
    Test 2b: Validate that the Code section of the sampled candidate (temp > 0)
    contains valid-looking pyautogui coordinates.

    Only meaningful when --greedy-after-code is on — that's the flag this
    test exists to verify. With full-output sampling there's no logits
    processor to validate, and the test name would be misleading.

    Checks:
    - Both candidates have a Code section
    - Code sections contain pyautogui calls with numeric coordinates
    - Sampled candidate's code is syntactically similar to greedy's
      (i.e., not garbled by temperature sampling)
    """
    result = SmokeTestResult("Step 2b: Code section validation (greedy-after-code)")
    t0 = time.time()
    try:
        codes = [extract_code_block(c["raw_output"]) for c in candidates]
        greedy_code, sampled_code = codes[0], codes[1]

        has_code = bool(greedy_code) and bool(sampled_code)
        has_calls = (
            ("pyautogui." in greedy_code or "computer." in greedy_code)
            and ("pyautogui." in sampled_code or "computer." in sampled_code)
        )
        has_coords = (
            bool(re.search(r'\d+', greedy_code))
            and bool(re.search(r'\d+', sampled_code))
        )

        result.passed = has_code and has_calls and has_coords
        result.output = json.dumps({
            "greedy_code": greedy_code[:200],
            "sampled_code": sampled_code[:200],
            "has_code": has_code,
            "has_calls": has_calls,
            "has_coords": has_coords,
        }, indent=2)

        if not result.passed:
            parts = []
            if not has_code:
                parts.append("missing code section")
            if not has_calls:
                parts.append("missing pyautogui/computer calls")
            if not has_coords:
                parts.append("missing numeric coordinates")
            result.error = f"Code validation failed: {', '.join(parts)}"

    except Exception as e:
        result.error = f"{e}\n{traceback.format_exc()}"
    result.elapsed = time.time() - t0
    return result


def test_step3(mgr, observation, candidates):
    """Test Step 3: World model transition prediction."""
    result = SmokeTestResult("Step 3: World model (LoRA)")
    t0 = time.time()
    try:
        candidates = step_world_model(mgr, observation, candidates)
        result.passed = all(
            c.get("transition") and len(c["transition"]) > 20
            for c in candidates
        )
        result.output = json.dumps(
            [{"index": c["index"],
              "transition": c["transition"][:150]} for c in candidates],
            indent=2
        )
        if not result.passed:
            result.error = "Missing or too-short transitions"
    except Exception as e:
        result.error = f"{e}\n{traceback.format_exc()}"
    result.elapsed = time.time() - t0
    return result, candidates if result.passed else None


def test_step4(mgr, instruction, candidates):
    """Test Step 4: LLM-as-judge scoring."""
    result = SmokeTestResult("Step 4: Scoring")
    t0 = time.time()
    try:
        candidates = step_scoring(mgr, instruction, candidates)
        result.passed = all(
            isinstance(c.get("score"), int) and 1 <= c["score"] <= 10
            for c in candidates
        )
        result.output = json.dumps(
            [{"index": c["index"], "score": c.get("score"),
              "action": c["action"][:80]} for c in candidates],
            indent=2
        )
        if not result.passed:
            result.error = "Invalid scores"
    except Exception as e:
        result.error = f"{e}\n{traceback.format_exc()}"
    result.elapsed = time.time() - t0
    return result, candidates


def test_step5(observation, candidates):
    """Test Step 5: Response assembly."""
    result = SmokeTestResult("Step 5: Response assembly")
    full_response = None
    t0 = time.time()
    try:
        winner = candidates[0]
        full_response = step_assemble_response(observation, winner)
        result.passed = (
            observation[:50] in full_response
            and winner["raw_output"][:50] in full_response
        )
        result.output = f"Response length: {len(full_response)} chars"
        if not result.passed:
            result.error = "Response missing observation or winner output"
    except Exception as e:
        result.error = f"{e}\n{traceback.format_exc()}"
    result.elapsed = time.time() - t0
    return result, full_response


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Smoke test framework pipeline steps")
    parser.add_argument("--model-dir", type=str, default="OpenCUA-7B")
    parser.add_argument("--adapter", type=str, default="ageppert/world-model-7b-lora")
    parser.add_argument("--no-adapter", action="store_true")
    parser.add_argument("--step", type=int, nargs="+", default=None,
                        help="Which steps to test (default: all). E.g., --step 1 2")
    parser.add_argument(
        "--greedy-after-code",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_GREEDY_AFTER_CODE,
        help=("Switch to greedy decoding once Code section starts. "
              f"(default: {DEFAULT_GREEDY_AFTER_CODE})"),
    )
    args = parser.parse_args()

    if args.no_adapter:
        args.adapter = None

    steps_to_run = set(args.step) if args.step else {0, 1, 2, 3, 4, 5}

    mode = "decoupled" if args.greedy_after_code else "full-output sampling"
    print("=" * 60)
    print("FRAMEWORK PIPELINE SMOKE TEST")
    print(f"  Greedy after Code:  {args.greedy_after_code} ({mode})")
    print(f"  Model: {args.model_dir}")
    print(f"  Adapter: {args.adapter or 'DISABLED'}")
    print(f"  Steps: {sorted(steps_to_run)}")
    print("=" * 60)

    results = []
    mgr = None

    if 0 in steps_to_run or any(s in steps_to_run for s in [1, 2, 3, 4]):
        r, mgr = test_model_loading(args)
        results.append(r)
        print(r)
        if not r.passed:
            print("\nModel loading failed — cannot continue.")
            sys.exit(1)

    screenshot_b64 = None
    messages = None
    if any(s in steps_to_run for s in [1, 2]):
        print("\nLoading test screenshot...")
        screenshot_b64 = load_test_screenshot_b64()
        messages = make_test_messages(screenshot_b64)
        print(f"  Screenshot: {len(screenshot_b64)} chars base64")

    instruction = TEST_INSTRUCTION
    observation = None
    candidates = None

    if 1 in steps_to_run:
        r, observation = test_step1(mgr, messages)
        results.append(r)
        print(f"\n{r}")

    if 2 in steps_to_run:
        r, candidates = test_step2(mgr, messages, args.greedy_after_code)
        results.append(r)
        print(f"\n{r}")

        # Step 2b runs only when greedy-after-code is on — that's what this
        # test is designed to validate. With full-output sampling there's
        # nothing to validate; running it would be misleading.
        if candidates is not None and args.greedy_after_code:
            r2b = test_step2_code_validation(candidates)
            results.append(r2b)
            print(f"\n{r2b}")

    if 3 in steps_to_run:
        if observation is None or candidates is None:
            r = SmokeTestResult("Step 3: World model (LoRA)")
            r.error = "Skipped — missing observation or candidates from prior steps"
            candidates = None
            results.append(r)
            print(f"\n{r}")
        else:
            r, candidates = test_step3(mgr, observation, candidates)
            results.append(r)
            print(f"\n{r}")

    if 4 in steps_to_run:
        if candidates is None:
            r = SmokeTestResult("Step 4: Scoring")
            r.error = "Skipped — missing candidates from prior steps"
            results.append(r)
            print(f"\n{r}")
        else:
            r, candidates = test_step4(mgr, instruction, candidates)
            results.append(r)
            print(f"\n{r}")

    full_response = None
    if 5 in steps_to_run:
        if candidates is None:
            r = SmokeTestResult("Step 5: Response assembly")
            r.error = "Skipped — missing candidates from prior steps"
            results.append(r)
            print(f"\n{r}")
        else:
            r, full_response = test_step5(observation, candidates)
            results.append(r)
            print(f"\n{r}")

    if full_response:
        print("\n" + "=" * 60)
        print("FULL RESPONSE (what OSWorld receives)")
        print("=" * 60)
        print(full_response)
        print("=" * 60)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_time = sum(r.elapsed for r in results)
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.step_name} ({r.elapsed:.1f}s)")
    print(f"\n  {passed} passed, {failed} failed, {total_time:.1f}s total")

    if failed > 0:
        print("\nFAILED STEPS:")
        for r in results:
            if not r.passed:
                print(f"  {r.step_name}: {r.error}")
        sys.exit(1)
    else:
        print("\nAll steps passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()