"""
core/decoupled.py

GreedyAfterCodeHeader: a HuggingFace LogitsProcessor that switches from
temperature sampling to greedy decoding once a Code section header appears
in the generated output.

Used by core.model_manager.generate_vision when `greedy_after_code=True`.
See docs/architecture.md for why this matters: temperature sampling perturbs
the digit tokens inside pyautogui coordinates, moving clicks onto the wrong
UI element.
"""

import logging

from transformers import LogitsProcessor

log = logging.getLogger("core.decoupled")


class GreedyAfterCodeHeader(LogitsProcessor):
    """
    Switch from temperature sampling to greedy decoding once the model
    outputs a Code section header.

    Monitors the generated text for markers like "Code:", "```python",
    or "```\\npyautogui". Once detected, forces greedy decoding for all
    subsequent tokens by zeroing out all logits except the top token.

    This preserves diversity in Thought/Action reasoning while ensuring
    precise coordinate grounding in the Code section.
    """

    # Markers that indicate the start of the Code section.
    # Checked in order; first match triggers greedy mode.
    CODE_MARKERS = [
        "Code:",
        "```python",
        "```\npyautogui",
    ]

    def __init__(self, tokenizer, prompt_len):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.greedy = False

    def __call__(self, input_ids, scores):
        if self.greedy:
            return self._force_greedy(scores)

        # Decode only the generated portion to check for code markers
        generated = input_ids[0, self.prompt_len:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)

        for marker in self.CODE_MARKERS:
            if marker in text:
                self.greedy = True
                log.debug(f"  GreedyAfterCodeHeader: triggered on '{marker}' "
                          f"at ~{len(text)} chars")
                return self._force_greedy(scores)

        return scores

    @staticmethod
    def _force_greedy(scores):
        """Zero out all logits except the argmax → deterministic selection."""
        top = scores.argmax(dim=-1, keepdim=True)
        scores.fill_(float("-inf"))
        scores.scatter_(1, top, 0.0)
        return scores