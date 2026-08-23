#!/usr/bin/env python3
"""
train_world_model.py

LoRA fine-tuning of OpenCUA-7B for world model transition prediction
using TRL's SFTTrainer.

Usage:
  # Smoke test (10 steps, validates everything works)
  python train_world_model.py --smoke-test

  # Full training
  python train_world_model.py

  # Custom settings
  python train_world_model.py --epochs 3 --lr 1e-4 --lora-rank 32
"""

import argparse
import os
import json
import torch
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Train world model via LoRA SFT")

    # Model
    parser.add_argument("--base-model", type=str, default="xlangai/OpenCUA-7B",
                        help="Base model to fine-tune")
    parser.add_argument("--trust-remote-code", action="store_true", default=True,
                        help="Trust remote code for model loading")

    # Dataset
    parser.add_argument("--dataset-repo", type=str,
                        default="ageppert/world-model-transitions",
                        help="HuggingFace dataset repo")
    parser.add_argument("--train-file", type=str, default="transition_train.jsonl")
    parser.add_argument("--val-file", type=str, default="transition_val.jsonl")

    # LoRA
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", type=str, nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj"],
                        help="Modules to apply LoRA to")

    # Training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8,
                        help="Effective batch size = per_device * grad_accum * num_gpus")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)

    # Checkpointing
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=10)

    # Output
    parser.add_argument("--output-dir", type=str, default="./world_model_output")
    parser.add_argument("--final-model-dir", type=str, default="./world_model_final")

    # HuggingFace upload
    parser.add_argument("--push-to-hub", action="store_true", default=False)
    parser.add_argument("--hub-model-id", type=str, default=None,
                        help="HuggingFace model repo for upload")

    # Smoke test
    parser.add_argument("--smoke-test", action="store_true", default=False,
                        help="Run 10 training steps to validate setup")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("WORLD MODEL TRAINING")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    # Override settings for smoke test
    if args.smoke_test:
        print("\n*** SMOKE TEST MODE ***")
        print("Running 10 steps to validate setup.\n")
        args.epochs = 1
        args.save_steps = 5
        args.eval_steps = 5
        args.logging_steps = 1

    # =====================================================================
    # Step 1: Print environment info
    # =====================================================================
    print("\n--- Environment ---")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"bf16 support: {torch.cuda.is_bf16_supported() if torch.cuda.is_available() else 'N/A'}")

    # =====================================================================
    # Step 2: Load dataset
    # =====================================================================
    print("\n--- Loading dataset ---")
    from datasets import load_dataset

    dataset = load_dataset(
        args.dataset_repo,
        data_files={
            "train": args.train_file,
            "validation": args.val_file,
        },
    )

    if args.smoke_test:
        # Use tiny subsets for smoke test
        dataset["train"] = dataset["train"].select(range(min(50, len(dataset["train"]))))
        dataset["validation"] = dataset["validation"].select(range(min(10, len(dataset["validation"]))))

    print(f"Train examples: {len(dataset['train'])}")
    print(f"Val examples: {len(dataset['validation'])}")

    # Verify format
    sample = dataset["train"][0]
    assert "messages" in sample, f"Expected 'messages' key, got: {list(sample.keys())}"
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Message roles: {[m['role'] for m in sample['messages']]}")
    print(f"System prompt length: {len(sample['messages'][0]['content'])} chars")
    print(f"User prompt length: {len(sample['messages'][1]['content'])} chars")
    print(f"Assistant response length: {len(sample['messages'][2]['content'])} chars")

    # =====================================================================
    # Step 3: Load model and tokenizer
    # =====================================================================
    print("\n--- Loading model and tokenizer ---")
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

    # Load tokenizer/processor
    # OpenCUA-7B is based on Qwen2.5-VL which uses a processor
    try:
        tokenizer = AutoProcessor.from_pretrained(
            args.base_model,
            trust_remote_code=args.trust_remote_code,
        )
        print("Loaded AutoProcessor")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model,
            trust_remote_code=args.trust_remote_code,
        )
        print("Loaded AutoTokenizer")

    # Ensure pad token is set
    if hasattr(tokenizer, 'tokenizer'):
        # Processor wraps a tokenizer
        inner_tok = tokenizer.tokenizer
    else:
        inner_tok = tokenizer

    if inner_tok.pad_token is None:
        inner_tok.pad_token = inner_tok.eos_token
        print(f"Set pad_token to eos_token: {inner_tok.pad_token}")

    # Load model
    # OpenCUA-7B uses custom model code via trust_remote_code.
    # Flash attention is not supported by the custom code, so try sdpa then eager.
    print(f"Loading {args.base_model}...")

    model = None
    for attn_impl in ["sdpa", "eager"]:
        try:
            print(f"  Trying AutoModelForCausalLM with attn={attn_impl}...")
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                trust_remote_code=args.trust_remote_code,
                torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
                attn_implementation=attn_impl,
                device_map="auto",
            )
            print(f"  Success with {attn_impl}")
            break
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    if model is None:
        raise RuntimeError("Failed to load model")
    print(f"Model loaded. Parameters: {model.num_parameters() / 1e9:.2f}B")

    # =====================================================================
    # Step 4: Configure LoRA
    # =====================================================================
    print("\n--- Configuring LoRA ---")
    from peft import LoraConfig, TaskType, get_peft_model

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        model.enable_input_require_grads()

    # =====================================================================
    # Step 5: Configure training
    # =====================================================================
    print("\n--- Configuring trainer ---")
    from trl import SFTConfig, SFTTrainer

    import trl
    print(f"TRL version: {trl.__version__}")

    # Calculate total steps for smoke test max_steps
    total_train_samples = len(dataset["train"])
    steps_per_epoch = total_train_samples // (args.per_device_batch_size * args.gradient_accumulation_steps)

    training_args = SFTConfig(
        output_dir=args.output_dir,

        # Training duration
        num_train_epochs=args.epochs,
        max_steps=10 if args.smoke_test else -1,

        # Batch size
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        # Optimizer
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        optim="adamw_torch",
        lr_scheduler_type="cosine",

        # Precision
        bf16=args.bf16,

        # Sequence length
        max_seq_length=args.max_seq_length,

        # Gradient checkpointing
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # Logging
        logging_steps=args.logging_steps,
        logging_first_step=True,
        report_to="none",

        # Evaluation
        eval_strategy="steps",
        eval_steps=args.eval_steps,

        # Checkpointing
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Hub
        push_to_hub=False,  # We handle upload manually after training

        # Misc
        dataloader_num_workers=2,
        remove_unused_columns=True,
        seed=42,
    )

    effective_batch = (
        args.per_device_batch_size
        * args.gradient_accumulation_steps
        * max(1, torch.cuda.device_count() if torch.cuda.is_available() else 1)
    )
    print(f"Effective batch size: {effective_batch}")
    print(f"Steps per epoch: ~{steps_per_epoch}")
    print(f"Total epochs: {args.epochs}")
    if args.smoke_test:
        print(f"Smoke test: limited to 10 steps")

    # =====================================================================
    # Step 6: Format dataset for chat template
    # =====================================================================
    print("\n--- Formatting dataset ---")

    def format_chat(example):
        """Apply the chat template to the messages."""
        messages = example["messages"]
        # Only use the three main messages (system, user, assistant)
        # Strip metadata if present
        chat_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
        ]
        text = inner_tok.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    # Test formatting on one example
    test_formatted = format_chat(dataset["train"][0])
    print(f"Formatted text length: {len(test_formatted['text'])} chars")
    print(f"First 300 chars:\n  {test_formatted['text'][:300]}")

    # Apply formatting
    formatted_train = dataset["train"].map(format_chat, remove_columns=dataset["train"].column_names)
    formatted_val = dataset["validation"].map(format_chat, remove_columns=dataset["validation"].column_names)

    # Tokenize a sample to check length
    sample_tokens = inner_tok(test_formatted["text"], return_tensors="pt")
    print(f"Sample token count: {sample_tokens['input_ids'].shape[1]}")

    # =====================================================================
    # Step 7: Train
    # =====================================================================
    print("\n--- Starting training ---")
    print(f"Time: {datetime.now().isoformat()}")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_train,
        eval_dataset=formatted_val,
        tokenizer=inner_tok,
    )

    # Train
    train_result = trainer.train()

    # =====================================================================
    # Step 8: Save and report
    # =====================================================================
    print("\n--- Training complete ---")
    print(f"Time: {datetime.now().isoformat()}")

    # Log metrics
    metrics = train_result.metrics
    print(f"\nTraining metrics:")
    for key, value in sorted(metrics.items()):
        print(f"  {key}: {value}")

    # Run final eval
    print("\n--- Final evaluation ---")
    eval_metrics = trainer.evaluate()
    print(f"Eval metrics:")
    for key, value in sorted(eval_metrics.items()):
        print(f"  {key}: {value}")

    # Save the final model
    print(f"\n--- Saving model to {args.final_model_dir} ---")
    trainer.save_model(args.final_model_dir)
    inner_tok.save_pretrained(args.final_model_dir)

    # Save training config for reproducibility
    config_path = os.path.join(args.final_model_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved training config to {config_path}")

    # Push to hub if requested
    if args.push_to_hub and args.hub_model_id:
        print(f"\n--- Pushing LoRA adapter to HuggingFace: {args.hub_model_id} ---")
        from huggingface_hub import HfApi, create_repo

        # Create repo if needed
        create_repo(args.hub_model_id, exist_ok=True)

        # Push the LoRA adapter (not the full base model)
        model.push_to_hub(args.hub_model_id)
        inner_tok.push_to_hub(args.hub_model_id)

        # Upload training config and metrics
        api = HfApi()
        api.upload_file(
            path_or_fileobj=config_path,
            path_in_repo="training_config.json",
            repo_id=args.hub_model_id,
        )

        # Create model card
        model_card = f"""---
language:
  - en
license: mit
base_model: {args.base_model}
tags:
  - world-model
  - computer-use
  - transition-prediction
  - lora
  - peft
library_name: peft
---

# World Model LoRA Adapter

LoRA adapter fine-tuned on [{args.dataset_repo}](https://huggingface.co/datasets/{args.dataset_repo})
for predicting GUI state transitions in desktop computer-use tasks.

## Base Model
[{args.base_model}](https://huggingface.co/{args.base_model})

## Usage
```python
from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor

base_model = AutoModelForVision2Seq.from_pretrained("{args.base_model}", trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, "{args.hub_model_id}")
processor = AutoProcessor.from_pretrained("{args.hub_model_id}")
```

## Training
- LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}
- Target modules: {args.target_modules}
- Learning rate: {args.lr}
- Epochs: {args.epochs}
- Train loss: {metrics.get('train_loss', 'N/A')}
- Eval loss: {eval_metrics.get('eval_loss', 'N/A')}

## Citation
Based on OpenCUA ([arXiv:2508.09123](https://arxiv.org/abs/2508.09123)).
"""
        api.upload_file(
            path_or_fileobj=model_card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=args.hub_model_id,
        )
        print(f"Model uploaded to: https://huggingface.co/{args.hub_model_id}")

    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE")
    print(f"Final train loss: {metrics.get('train_loss', 'N/A')}")
    print(f"Final eval loss: {eval_metrics.get('eval_loss', 'N/A')}")
    print(f"Model saved to: {args.final_model_dir}")
    print(f"Finished: {datetime.now().isoformat()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()