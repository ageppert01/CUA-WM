#!/usr/bin/env python3
"""
eval_world_model.py

Runs inference on all examples in transition_val.jsonl using the
trained LoRA adapter. Saves predictions alongside ground truth
for qualitative and quantitative analysis.

Output: eval_predictions.jsonl with fields:
  - metadata (task_id, step_index, instruction, code)
  - ground_truth (the actual reflection)
  - prediction (model's predicted transition)
  - input_observation (the state description)
  - input_action (the action taken)

Usage:
  python eval_world_model.py
  python eval_world_model.py --max-samples 50  # quick test
"""

import argparse
import json
import os
import sys
import time
import torch
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate world model on val set")

    # Model
    parser.add_argument("--base-model", type=str, default="xlangai/OpenCUA-7B")
    parser.add_argument("--adapter", type=str, default="ageppert/world-model-7b-lora")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)

    # Dataset
    parser.add_argument("--dataset-repo", type=str,
                        default="ageppert/world-model-transitions")
    parser.add_argument("--val-file", type=str, default="transition_val.jsonl")

    # Generation
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Low temperature for more deterministic predictions")
    parser.add_argument("--top-p", type=float, default=0.9)

    # Scope
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples to evaluate (None = all)")

    # Output
    parser.add_argument("--output-file", type=str, default="eval_predictions.jsonl")
    parser.add_argument("--output-stats", type=str, default="eval_stats.txt")

    # Upload
    parser.add_argument("--upload", action="store_true", default=False)
    parser.add_argument("--hf-repo", type=str, default=None,
                        help="HuggingFace dataset repo to upload results")
    parser.add_argument("--hf-token", type=str, default=None)

    return parser.parse_args()


def load_model_and_tokenizer(args):
    """Load base model with LoRA adapter."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"Loading tokenizer from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model {args.base_model}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )

    print(f"Loading LoRA adapter from {args.adapter}...")
    model = PeftModel.from_pretrained(base_model, args.adapter)
    model.eval()

    print(f"Model loaded. Parameters: {base_model.num_parameters() / 1e9:.2f}B")
    return model, tokenizer


def extract_user_fields(user_content):
    """Extract observation and action from the user message."""
    observation = ""
    action = ""

    # Parse the structured user prompt
    lines = user_content.split("\n")
    current_field = None
    current_lines = []

    for line in lines:
        if line.strip() == "Current state:":
            if current_field == "observation":
                observation = "\n".join(current_lines).strip()
            current_field = "observation"
            current_lines = []
        elif line.strip() == "Action:":
            if current_field == "observation":
                observation = "\n".join(current_lines).strip()
            current_field = "action"
            current_lines = []
        elif line.strip().startswith("Given the current state"):
            if current_field == "action":
                action = "\n".join(current_lines).strip()
            current_field = None
        else:
            if current_field:
                current_lines.append(line)

    # Handle case where action is last before the prompt
    if current_field == "action":
        action = "\n".join(current_lines).strip()

    return observation, action


def generate_prediction(model, tokenizer, messages, args):
    """Generate a transition prediction from the model."""
    # Build prompt using only system + user (no assistant)
    prompt_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m["role"] in ("system", "user")
    ]

    prompt = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated tokens (not the prompt)
    prediction = tokenizer.decode(
        outputs[0][input_len:],
        skip_special_tokens=True,
    ).strip()

    return prediction


def main():
    args = parse_args()

    print("=" * 60)
    print("WORLD MODEL EVALUATION")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    # Set HF token
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    # =====================================================================
    # Load model
    # =====================================================================
    model, tokenizer = load_model_and_tokenizer(args)

    # =====================================================================
    # Load dataset
    # =====================================================================
    print(f"\n--- Loading validation set ---")
    from datasets import load_dataset

    dataset = load_dataset(
        args.dataset_repo,
        data_files={"validation": args.val_file},
    )
    val_data = dataset["validation"]

    if args.max_samples:
        val_data = val_data.select(range(min(args.max_samples, len(val_data))))

    print(f"Evaluating on {len(val_data)} examples")

    # =====================================================================
    # Run inference
    # =====================================================================
    print(f"\n--- Running inference ---")
    results = []
    start_time = time.time()

    for i, example in enumerate(val_data):
        messages = example["messages"]
        metadata = example.get("metadata", {})

        # Extract ground truth
        ground_truth = messages[2]["content"]  # assistant message

        # Extract observation and action from user message
        observation, action = extract_user_fields(messages[1]["content"])

        # Generate prediction
        prediction = generate_prediction(model, tokenizer, messages, args)

        result = {
            "index": i,
            "metadata": metadata,
            "input_observation": observation,
            "input_action": action,
            "ground_truth": ground_truth,
            "prediction": prediction,
        }
        results.append(result)

        # Progress logging
        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed
        eta = (len(val_data) - i - 1) / rate if rate > 0 else 0

        if (i + 1) % 10 == 0 or (i + 1) == len(val_data):
            print(f"  [{i+1}/{len(val_data)}] "
                  f"{rate:.2f} examples/sec, "
                  f"ETA: {eta/60:.1f} min")

        # Print first 3 examples for sanity check
        if i < 3:
            print(f"\n  --- Example {i} ---")
            print(f"  Action: {action[:100]}")
            print(f"  Ground truth: {ground_truth[:200]}")
            print(f"  Prediction:   {prediction[:200]}")

    total_time = time.time() - start_time

    # =====================================================================
    # Save predictions
    # =====================================================================
    print(f"\n--- Saving predictions to {args.output_file} ---")
    with open(args.output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    file_size = os.path.getsize(args.output_file) / (1024 * 1024)
    print(f"  {len(results)} predictions, {file_size:.1f} MB")

    # =====================================================================
    # Compute basic statistics
    # =====================================================================
    print(f"\n--- Computing statistics ---")
    pred_lengths = [len(r["prediction"]) for r in results]
    gt_lengths = [len(r["ground_truth"]) for r in results]
    empty_preds = sum(1 for r in results if not r["prediction"].strip())

    stats_lines = [
        "=" * 60,
        "WORLD MODEL EVALUATION STATISTICS",
        "=" * 60,
        "",
        f"Model: {args.base_model} + {args.adapter}",
        f"Validation examples: {len(results)}",
        f"Total inference time: {total_time:.1f}s ({total_time/60:.1f} min)",
        f"Average time per example: {total_time/len(results):.2f}s",
        "",
        "Prediction lengths (characters):",
        f"  min:    {min(pred_lengths)}",
        f"  median: {sorted(pred_lengths)[len(pred_lengths)//2]}",
        f"  mean:   {sum(pred_lengths)/len(pred_lengths):.0f}",
        f"  max:    {max(pred_lengths)}",
        "",
        "Ground truth lengths (characters):",
        f"  min:    {min(gt_lengths)}",
        f"  median: {sorted(gt_lengths)[len(gt_lengths)//2]}",
        f"  mean:   {sum(gt_lengths)/len(gt_lengths):.0f}",
        f"  max:    {max(gt_lengths)}",
        "",
        f"Empty predictions: {empty_preds} ({100*empty_preds/len(results):.1f}%)",
        "",
        f"Generation config:",
        f"  max_new_tokens: {args.max_new_tokens}",
        f"  temperature: {args.temperature}",
        f"  top_p: {args.top_p}",
    ]

    stats_text = "\n".join(stats_lines)
    print(stats_text)

    with open(args.output_stats, "w") as f:
        f.write(stats_text + "\n")

    # =====================================================================
    # Upload if requested
    # =====================================================================
    if args.upload and args.hf_repo:
        print(f"\n--- Uploading to {args.hf_repo} ---")
        from huggingface_hub import HfApi
        api = HfApi()
        for local_path, remote_name in [
            (args.output_file, "eval_predictions.jsonl"),
            (args.output_stats, "eval_stats.txt"),
        ]:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=remote_name,
                repo_id=args.hf_repo,
                repo_type="model",
            )
            print(f"  Uploaded {remote_name}")

    print(f"\n{'=' * 60}")
    print("EVALUATION COMPLETE")
    print(f"Predictions saved to: {args.output_file}")
    print(f"Finished: {datetime.now().isoformat()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()