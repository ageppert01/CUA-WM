"""
core — shared building blocks for the world-model-augmented CUA framework.

Submodules:
  prompts        Prompt templates (single source of truth across all entry points)
  decoupled      GreedyAfterCodeHeader logits processor (decoupled generation)
  parsing        Regex helpers for extracting Observation / Action / Code sections
  model_manager  ModelManager class (load, image handling, generate_vision/text_only)
  pipeline       step_* functions and run_pipeline orchestrator

Used by: framework_api.py and smoke_test_framework.py. Both v1 (uniform
         temperature sampling) and v2 (decoupled generation) behaviour live
         here behind the `greedy_after_code` flag, rather than in forked
         files. preprocess_transitions.py imports from core.prompts only
         (no model dependency).

NOT used by: opencua_api.py (deliberately self-contained baseline).
"""

__version__ = "1.0.0"