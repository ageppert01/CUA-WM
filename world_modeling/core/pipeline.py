"""
core/pipeline.py

The five pipeline steps and the run_pipeline orchestrator.

Each step is a plain function taking an explicit ModelManager, so it can be
called and tested in isolation — smoke_test_framework.py exercises them one
at a time without going through the Flask layer.

    Step 1  step_state_estimation      observation from a truncated L3 call
    Step 2  step_candidate_generation  N candidate actions in L2 format
    Step 3  step_world_model           predicted transition per candidate (LoRA)
    Step 4  step_scoring               LLM-as-judge 1-10, sorted best-first
    Step 5  step_assemble_response     observation spliced into the winner

The adapter is enabled only in step 3; every other call passes
use_adapter=False, so one model instance serves as both policy and world model.
"""

import logging
import re
import time

from core.parsing import (
    extract_action_text,
    extract_instruction,
    extract_observation,
)
from core.prompts import (
    L3_SYSTEM_PROMPT,
    SCORING_SYSTEM_PROMPT,
    SCORING_USER_TEMPLATE,
    WORLD_MODEL_SYSTEM_PROMPT,
    WORLD_MODEL_USER_TEMPLATE,
)

log = logging.getLogger("core.pipeline")


# ============================================================================
# Pipeline steps (each independently testable)
# ============================================================================

def step_state_estimation(mgr, messages):
    """
    Step 1: truncated L3 call to extract the screen observation.

    Swaps the incoming system prompt for the L3 prompt and generates greedily,
    stopping at the first section header so only the Observation survives.
    This reuses OpenCUA's own screen comprehension instead of training a
    separate state estimator.

    Args:
        mgr: ModelManager instance
        messages: original chat messages from the request

    Returns:
        str: description of the current screen state, or "" if extraction failed
    """
    log.info("Step 1: State estimation (L3)...")
    t0 = time.time()

    l3_messages = []
    system_swapped = False
    for msg in messages:
        if msg["role"] == "system":
            l3_messages.append({"role": "system", "content": L3_SYSTEM_PROMPT})
            system_swapped = True
        else:
            l3_messages.append(msg)

    if not system_swapped:
        log.warning("  No system message found — prepending L3 system prompt.")
        l3_messages.insert(0, {"role": "system", "content": L3_SYSTEM_PROMPT})

    log.debug(f"  System swapped: {system_swapped}, "
              f"messages: {[m['role'] for m in l3_messages]}")

    raw = mgr.generate_vision(
        l3_messages,
        max_new_tokens=512,
        temperature=0.0,
        stop_at=["## Thought", "## Action", "## Code",
                 "\nThought:", "\nAction:", "\nCode:"],
        use_adapter=False,
    )

    observation = extract_observation(raw)
    elapsed = time.time() - t0
    if not observation:
        log.warning(f"  No observation extracted from {len(raw)} chars of L3 "
                    f"output; pipeline will fall back to raw L2.")
    log.info(f"  Observation ({len(observation)} chars, {elapsed:.1f}s): "
             f"{observation[:200]}...")
    return observation


def step_candidate_generation(mgr, messages, n_candidates,
                              temperature=0.7, greedy_after_code=True):
    """
    Step 2: generate N candidate actions using L2 (incoming messages as-is).

    Candidate 0 is always fully greedy, so it reproduces vanilla OpenCUA
    exactly. Later candidates sample at `temperature` for reasoning diversity.

    When greedy_after_code is True, sampled candidates use decoupled
    generation: temperature applies to Thought/Action, then decoding switches
    to greedy once the Code section begins, keeping pyautogui coordinates
    exact. Passing False reproduces the earlier behaviour where sampling ran
    over the whole output, including coordinate digits.

    Args:
        mgr: ModelManager instance
        messages: original chat messages (with the L2 system prompt)
        n_candidates: number of candidates to generate
        temperature: sampling temperature for non-greedy candidates
        greedy_after_code: enable decoupled generation for sampled candidates

    Returns:
        list[dict]: one dict per candidate with keys
            index, raw_output, action, temperature
    """
    log.info(f"Step 2: Generating {n_candidates} candidate(s) "
             f"(temp=0 + {temperature}, greedy_after_code={greedy_after_code})...")
    t0 = time.time()
    candidates = []

    for i in range(n_candidates):
        temp = 0.0 if i == 0 else temperature
        use_decoupled = temp > 0 and greedy_after_code
        log.info(f"  Candidate {i + 1}/{n_candidates} (temp={temp}, "
                 f"decoupled={use_decoupled})...")

        raw = mgr.generate_vision(
            messages,
            max_new_tokens=1024,
            temperature=temp,
            use_adapter=False,
            greedy_after_code=use_decoupled,
        )
        action = extract_action_text(raw)
        candidates.append({
            "index": i,
            "raw_output": raw,
            "action": action,
            "temperature": temp,
        })
        log.info(f"    Action: {action[:120]}...")

    elapsed = time.time() - t0
    log.info(f"  Generated {n_candidates} candidate(s) in {elapsed:.1f}s")
    return candidates


def step_world_model(mgr, observation, candidates):
    """
    Step 3: predict the transition for each candidate using the LoRA adapter.

    This is the only step that enables the adapter. It is text-only — the
    world model reasons over the natural-language state description rather
    than the screenshot, so it acts as a reasoning sandbox, not a visual
    simulator.

    Args:
        mgr: ModelManager instance
        observation: screen state description from step 1
        candidates: candidate dicts from step 2

    Returns:
        list[dict]: the same list, each candidate gaining a 'transition' key
    """
    log.info(f"Step 3: World model predictions for "
             f"{len(candidates)} candidate(s)...")
    t0 = time.time()

    for cand in candidates:
        wm_messages = [
            {"role": "system", "content": WORLD_MODEL_SYSTEM_PROMPT},
            {"role": "user", "content": WORLD_MODEL_USER_TEMPLATE.format(
                observation=observation,
                action=cand["action"],
            )},
        ]

        transition = mgr.generate_text_only(
            wm_messages,
            max_new_tokens=512,
            temperature=0.1,
            use_adapter=True,
        )
        cand["transition"] = transition
        log.info(f"  Candidate {cand['index']}: {transition[:150]}...")

    elapsed = time.time() - t0
    log.info(f"  World model done in {elapsed:.1f}s")
    return candidates


def step_scoring(mgr, instruction, candidates):
    """
    Step 4: LLM-as-judge scoring of each candidate's predicted transition.

    The base model (adapter disabled) rates how well each predicted transition
    advances the task, 1-10. Unparseable replies default to 5. Candidates are
    sorted best-first, with ties broken toward the lower index — the greedy
    candidate is the safer default when the judge cannot separate them.

    Args:
        mgr: ModelManager instance
        instruction: task goal text
        candidates: candidate dicts carrying a 'transition' key

    Returns:
        list[dict]: sorted by score descending, each gaining a 'score' key
    """
    log.info(f"Step 4: Scoring {len(candidates)} candidate(s)...")
    t0 = time.time()

    for cand in candidates:
        score_messages = [
            {"role": "system", "content": SCORING_SYSTEM_PROMPT},
            {"role": "user", "content": SCORING_USER_TEMPLATE.format(
                instruction=instruction,
                action=cand["action"],
                transition=cand["transition"],
            )},
        ]

        score_text = mgr.generate_text_only(
            score_messages,
            max_new_tokens=16,
            temperature=0.0,
            use_adapter=False,
        )

        try:
            score = int(re.search(r'\d+', score_text).group())
            score = max(1, min(10, score))
        except (AttributeError, ValueError):
            log.warning(f"  Could not parse score from: '{score_text}', "
                        f"defaulting to 5")
            score = 5

        cand["score"] = score
        log.info(f"  Candidate {cand['index']}: score={score} "
                 f"(action: {cand['action'][:80]}...)")

    elapsed = time.time() - t0
    log.info(f"  Scoring done in {elapsed:.1f}s")

    # Highest score first; ties prefer the greedy candidate at index 0.
    candidates.sort(key=lambda c: (-c["score"], c["index"]))
    return candidates


def step_assemble_response(observation, winner):
    """
    Step 5: splice the observation into the winning L2 output to form L3.

    L2 format:  # Step N: -> ## Thought: -> ## Action: -> ## Code:
    L3 format:  # Step N: -> ## Observation: -> ## Thought: -> ## Action: -> ## Code:

    If state estimation produced nothing, the raw L2 output is returned
    unchanged — the pipeline degrades to vanilla rather than emitting a
    malformed response.

    Args:
        observation: screen state description from step 1
        winner: winning candidate dict from step 4

    Returns:
        str: the response body to hand back to the client
    """
    raw = winner["raw_output"]

    if not observation:
        log.warning("No valid observation — returning raw L2 output.")
        return raw

    thought_match = re.search(r'\n(#{0,3}\s*Thought\s*:?)', raw)
    if thought_match:
        insert_pos = thought_match.start()
        observation_block = f"\n## Observation:\n{observation}\n"
        return raw[:insert_pos] + observation_block + raw[insert_pos:]

    # No Thought header found — prepend the observation instead.
    return f"## Observation:\n{observation}\n\n{raw}"


# ============================================================================
# Orchestrator
# ============================================================================

def run_pipeline(mgr, messages, n_cand, greedy_after_code=True):
    """
    Run the full world-model-augmented pipeline for one agent step.

    With n_cand == 1 there is nothing to choose between, so steps 3 and 4 are
    skipped and the result is an L3-style response from the base policy.

    Args:
        mgr: ModelManager instance
        messages: chat messages from the request
        n_cand: number of candidate actions to generate
        greedy_after_code: enable decoupled generation in step 2

    Returns:
        str: L3-formatted response for the client
    """
    pipeline_start = time.time()
    instruction = extract_instruction(messages)
    log.info(f"Pipeline start. Instruction: {instruction[:120]}...")

    observation = step_state_estimation(mgr, messages)

    candidates = step_candidate_generation(
        mgr, messages, n_cand, greedy_after_code=greedy_after_code,
    )

    if len(candidates) == 1:
        log.info("Single candidate — skipping world model and scoring.")
        winner = candidates[0]
    else:
        candidates = step_world_model(mgr, observation, candidates)
        candidates = step_scoring(mgr, instruction, candidates)
        winner = candidates[0]

    response_content = step_assemble_response(observation, winner)

    pipeline_elapsed = time.time() - pipeline_start
    log.info(f"Pipeline complete in {pipeline_elapsed:.1f}s")
    log.info(f"Winner: candidate {winner['index']} "
             f"(score={winner.get('score', 'N/A')}, "
             f"temp={winner['temperature']})")
    for cand in candidates:
        log.info(f"  [{cand['index']}] score={cand.get('score', 'N/A')} "
                 f"action={cand['action'][:100]}")

    return response_content
