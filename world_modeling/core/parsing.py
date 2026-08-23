"""
core/parsing.py

Regex helpers for extracting structured fields from model output and
messages.

These functions are intentionally lenient — OpenCUA's output sometimes
varies in header formatting (## vs bare, with/without colon, etc.) and
the regexes are tuned to match all observed variants from real runs.
"""

import re


def extract_instruction(messages):
    """
    Extract the task instruction text from a chat message history.

    Walks messages in reverse and returns the text of the most recent user
    message. Handles both string content and list-of-parts content (e.g.
    multimodal OpenAI-format messages with images interleaved).

    Returns:
        str: The instruction text, or empty string if no user text found.
    """
    for msg in reversed(messages):
        if msg["role"] != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    return item["text"]
        elif isinstance(content, str):
            return content
    return ""


def extract_observation(l3_output):
    """
    Extract the Observation section from L3-formatted output text.

    Handles both '## Observation:' and bare 'Observation:' formats.
    The regex stops at the next section header (Thought/Action/Code/Step)
    or at a code fence (```python) or end-of-string.

    Returns:
        str: The observation text, or empty string if no valid observation
             found (also returns empty if the matched text looks like code).
    """
    match = re.search(
        r'(?:^|\n)#{0,3}\s*Observation\s*:?\s*\n(.*?)(?=\n#{0,3}\s*(?:Thought|Action|Code|Step)\s*:?|```python|\Z)',
        l3_output, re.DOTALL
    )
    if match:
        obs = match.group(1).strip()
        if obs and not obs.startswith("pyautogui.") and not obs.startswith("```"):
            return obs
    return ""


def extract_action_text(l2_output):
    """
    Extract the descriptive Action text from L2 output for world model input.

    The world model is trained on textual action descriptions, not pyautogui
    code, so this strips any code blocks and returns just the natural-language
    description. Falls back to the full output if no Action section found.

    Returns:
        str: Action description text (no code), or the full L2 output stripped.
    """
    match = re.search(
        r'(?:^|\n)#{0,3}\s*Action\s*:\s*\n(.*?)(?=\n#{0,}\s*Code\s*:?|```|\Z)',
        l2_output, re.DOTALL
    )
    if match:
        action_text = match.group(1).strip()
        action_clean = re.sub(r'```[\s\S]*?```', '', action_text).strip()
        if action_clean and not action_clean.startswith("pyautogui."):
            return action_clean
    return l2_output.strip()


def extract_code_block(output):
    """
    Extract the Code section from model output.

    Tries fenced code blocks first (```python ... ``` or ``` ... ```),
    falls back to bare lines after a "Code:" header. Used primarily by
    smoke tests to validate that decoupled generation produces a clean
    Code section.

    Returns:
        str: The code text, or empty string if no Code section found.
    """
    # Match ```python ... ``` or ``` ... ```
    match = re.search(r'```(?:python)?\s*\n(.*?)```', output, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: look for bare pyautogui lines after Code: header
    match = re.search(
        r'(?:^|\n)#{0,3}\s*Code\s*:?\s*\n(.*)',
        output, re.DOTALL
    )
    if match:
        return match.group(1).strip()
    return ""