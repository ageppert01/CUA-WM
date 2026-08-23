# Reproducing

Four stages, in order. Only stage 2 is expensive; stages 1 and 4 are cheap, and stage 3
is optional if you take the published adapter.

Every stage has both a shell script (runs anywhere) and an HTCondor submit file (runs on
a cluster). The scripts are the source of truth; the `.sub` files only wrap them.

## 0. Prerequisites

```bash
cp .env.example .env     # fill in HF_TOKEN at minimum
set -a; . ./.env; set +a
pip install -r requirements.txt
```

A GPU with ≥40 GB VRAM for anything that loads the model. Stage 1 is CPU-only.

## 1. Build the transition dataset

Downloads AgentNet Ubuntu 5K (5,000 trajectories, 82,448 steps) and extracts supervised
transition pairs.

```bash
cd world_modeling
./run_preprocess.sh                      # or: condor_submit preprocess_transitions.sub
```

A step becomes an example only if it has a non-empty reflection, a non-empty observation
and action, is not marked incorrect or redundant, is not a terminate action, and is not
the last step in its trajectory. Expect **42,103 examples**, split 90/10 into
37,893 train / 4,210 validation, written as `transition_*.jsonl` with a
`preprocessing_stats.txt` summary.

The script uploads to `$HF_REPO` when `HF_TOKEN` is set; drop the `--upload` flag to keep
everything local. Prompts come from `core/prompts.py`, the same constants used at
inference — training and serving cannot drift apart.

To look at the raw data first:

```bash
./run_inspect.sh                         # or: condor_submit inspect_agentnet.sub
```

## 2. Train the LoRA adapter

```bash
SMOKE_TEST=1 ./run_train.sh              # ~10 steps, validates the whole path
./run_train.sh                           # full run
# or: condor_submit train_world_model.sub
```

LoRA r=16, α=32, dropout 0.05 on the attention projections; TRL `SFTTrainer`, cosine
schedule peaking at 2e-4, effective batch 16, max seq len 2,048, bf16 with gradient
checkpointing, 3 epochs.

**Cost: ~11.7 hours on one NVIDIA L40 (48 GB), 7,107 steps.** Output is a 39 MB adapter.
Expect final validation loss ≈ 0.577, down from ≈ 0.737 and still decreasing — it does
not overfit within 3 epochs. Progress is mirrored to `training_status.txt` so the file
exists for HTCondor transfer even if the job dies.

**To skip this stage entirely**, use the published adapter — it is the default value of
`--adapter` everywhere:

```
ageppert/world-model-7b-lora
```

## 3. Evaluate the adapter

Runs the adapter over all 4,210 held-out examples.

```bash
./run_eval.sh                            # or: condor_submit eval_world_model.sub
```

Writes `eval_predictions.jsonl` and `eval_stats.txt`. The reference run produced zero
empty predictions at ~7.4 s per example. Cap the work with `--max-samples N` (or the
`MAX_SAMPLES` variable in the script) while checking the path works.

This measures the world model in isolation — whether it predicts plausible transitions.
It says nothing about end-to-end task success; that needs stage 4.

## 4. Serve the framework

```bash
python framework_api.py --n-candidates 2 --port 9009
curl localhost:9009/health
```

On a cluster, `condor_submit framework.sub`, which additionally opens the reverse SSH
tunnel described in [chtc-deployment.md](chtc-deployment.md).

`/health` reports the live configuration rather than the defaults in the source, so it is
the fastest way to confirm what a running server is actually doing:

```json
{"status":"ok","model":"OpenCUA-7B","adapter":"ageppert/world-model-7b-lora",
 "has_adapter":true,"n_candidates":2,"greedy_after_code":true}
```

### Configurations

The three configurations compared in the report:

```bash
# Vanilla OpenCUA-7B baseline — separate self-contained server
python opencua_api.py

# CUA-WM v1: temperature sampling across the whole output (the broken one)
python framework_api.py --n-candidates 2 --no-greedy-after-code

# CUA-WM v2: decoupled generation — the headline configuration
python framework_api.py --n-candidates 2
```

Two request-level overrides are also accepted, for A/B testing without a restart:
`n_candidates` (int) and `bypass_world_model` (bool).

### Checking it works, without OSWorld

```bash
python smoke_test_framework.py           # or: condor_submit smoke_test.sub
```

Exercises each pipeline step against `test_screenshot.png` and asserts, among other
things, that greedy and sampled candidates emit identical `pyautogui` code under
decoupled generation. This is the fast way to confirm a change did not break the
coordinate guarantee.

## Running OSWorld against the framework

**OSWorld is not vendored in this repository and was not modified.** It runs as its own
checkout, against its own VM infrastructure, pointed at this framework's endpoint.

The framework deliberately exposes the OpenAI-compatible shape vanilla OpenCUA already
speaks — `POST /v1/chat/completions`, same request (screenshot + instruction), same
response (chain-of-thought + PyAutoGUI code). Substitution is transparent: OSWorld cannot
tell it is talking to a five-step pipeline rather than a single model call.

So the wiring is just:

1. Stand up [OSWorld](https://github.com/xlang-ai/OSWorld) per its own instructions.
2. Start the framework (above) and make its port reachable from the OSWorld host —
   directly, or through the reverse tunnel if the GPU sits on a cluster node.
3. Point OSWorld's model endpoint at `http://<host>:9009/v1/chat/completions`.
4. Run the Ubuntu task set with a 30-step limit per task.

The reported numbers come from the full OSWorld Ubuntu set, ~359 tasks, one run per
configuration.

## Expected results

| Configuration | All tasks | Feasible | Infeasible |
|---|---|---|---|
| Vanilla OpenCUA-7B | 24.8% (89/359) | 23.2% (77/332) | 44.4% (12/27) |
| CUA-WM (N=2, decoupled) | 31.0% (109/352) | 26.2% (85/324) | 85.7% (24/28) |

Two caveats when comparing against your own run. Task totals differ between runs because
a few OSWorld tasks fail to initialize or time out during environment setup, independent
of the agent. And these are **single runs per configuration** — no seed variance is
reported, so treat small per-category differences as noise.

Per-step latency is roughly 48 s at N=2, against ~10 s for vanilla. A full benchmark
sweep is correspondingly long; budget for it.
