#!/usr/bin/env python3
"""
framework_api.py

World-model-augmented CUA framework API.

Drop-in replacement for opencua_api.py: same port, same /v1/chat/completions
endpoint, same request/response format. Azure (OSWorld) sees no difference.

All pipeline logic lives in core/. This file is just argument parsing,
the Flask app, model loading, and request handling.

Configuration is flag-driven so ablations can be expressed as CLI args
without forking the file. Each flag toggles one orthogonal axis of behavior;
default values can change over time and each is documented at its argparse
definition.

Usage:
  python framework_api.py [--n-candidates 2] [--port 9009]
                          [--no-greedy-after-code]
"""

import argparse
import logging
import sys

from flask import Flask, jsonify, request

from core.model_manager import ModelManager
from core.pipeline import run_pipeline

# ----------------------------------------------------------------------------
# Defaults — these are CURRENT defaults, not "correct" answers.
# Changing any of these is a one-line edit; no other code should hardcode
# these values. /health reports the live values so the server always tells
# you what it's actually doing.
# ----------------------------------------------------------------------------

DEFAULT_PORT = 9009
DEFAULT_N_CANDIDATES = 2
DEFAULT_GREEDY_AFTER_CODE = True
BASE_MODEL_DIR = "OpenCUA-7B"
ADAPTER_REPO = "ageppert/world-model-7b-lora"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("framework")

app = Flask(__name__)

# Server-wide configuration, populated by main() and read by request handlers.
# Held as a plain dict rather than module-level globals so tests can override
# without monkey-patching imports.
config = {
    "mgr": None,
    "n_candidates": DEFAULT_N_CANDIDATES,
    "greedy_after_code": DEFAULT_GREEDY_AFTER_CODE,
}


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """
    Drop-in replacement for the OpenAI chat completions endpoint.
    Runs the world-model-augmented pipeline using the server's
    current configuration.

    Extra fields accepted in request JSON:
        n_candidates (int): Override default candidate count
        bypass_world_model (bool): If true, run vanilla generation only
    """
    try:
        data = request.get_json()
        messages = data.get("messages", [])
        req_n = data.get("n_candidates", config["n_candidates"])
        mgr = config["mgr"]

        if data.get("bypass_world_model", False):
            log.info("Bypass flag — running vanilla generation.")
            response_content = mgr.generate_vision(
                messages,
                max_new_tokens=data.get("max_tokens", 1024),
                temperature=data.get("temperature", 0.0),
                use_adapter=False,
            )
        else:
            response_content = run_pipeline(
                mgr, messages, req_n,
                greedy_after_code=config["greedy_after_code"],
            )

        return jsonify({
            "choices": [{
                "finish_reason": "stop",
                "message": {"content": response_content},
            }],
        }), 200

    except Exception as e:
        log.error(f"Error in chat_completions: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """Report live server config — always reflects what the server is doing right now."""
    mgr = config["mgr"]
    return jsonify({
        "status": "ok",
        "model": mgr.model_dir if mgr else "not loaded",
        "adapter": mgr.adapter_repo if mgr else "not loaded",
        "has_adapter": mgr.has_adapter if mgr else False,
        "n_candidates": config["n_candidates"],
        "greedy_after_code": config["greedy_after_code"],
    }), 200


@app.route("/shutdown", methods=["POST"])
def shutdown():
    log.info("Shutdown requested.")
    sys.exit(0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="World-model-augmented CUA framework API"
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--n-candidates", type=int, default=DEFAULT_N_CANDIDATES,
                        help="Default number of candidate actions")
    parser.add_argument("--model-dir", type=str, default=BASE_MODEL_DIR,
                        help="Path to base model directory")
    parser.add_argument("--adapter", type=str, default=ADAPTER_REPO,
                        help="HuggingFace repo for LoRA adapter")
    parser.add_argument("--no-adapter", action="store_true",
                        help="Run without world model adapter (vanilla mode)")

    # Pipeline behavior flags. Each is independently toggleable so future
    # axes (e.g. --scoring=pairwise) can be added without touching the others.
    parser.add_argument(
        "--greedy-after-code",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_GREEDY_AFTER_CODE,
        help=("During candidate generation, switch from temperature sampling "
              "to greedy decoding once the Code section header appears. "
              f"(default: {DEFAULT_GREEDY_AFTER_CODE})"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config["n_candidates"] = args.n_candidates
    config["greedy_after_code"] = args.greedy_after_code
    adapter = None if args.no_adapter else args.adapter

    log.info("=" * 60)
    log.info("WORLD-MODEL-AUGMENTED CUA FRAMEWORK")
    log.info(f"  Base model:         {args.model_dir}")
    log.info(f"  LoRA adapter:       {adapter or 'DISABLED'}")
    log.info(f"  N candidates:       {config['n_candidates']}")
    log.info(f"  Greedy after Code:  {config['greedy_after_code']}")
    log.info(f"  Port:               {args.port}")
    log.info("=" * 60)

    config["mgr"] = ModelManager(args.model_dir, adapter_repo=adapter)
    config["mgr"].load()

    log.info(f"Starting server on 0.0.0.0:{args.port}...")
    app.run(host="0.0.0.0", port=args.port, threaded=False)


if __name__ == "__main__":
    main()