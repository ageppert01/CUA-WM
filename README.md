# CUA-WM: World-Model-Augmented Computer-Use Agents

Wrapping a **frozen** OpenCUA-7B computer-use agent in a learned world model, using
supervised fine-tuning only — no reinforcement learning, no policy retraining.

On [OSWorld](https://os-world.github.io/), this raises overall task success from
**24.8% → 31.0%** and infeasible-task recognition from **44.4% → 85.7%**.

| Configuration | All tasks | Feasible | Infeasible |
|---|---|---|---|
| Vanilla OpenCUA-7B | 24.8% (89/359) | 23.2% (77/332) | 44.4% (12/27) |
| **CUA-WM** | **31.0%** (109/352) | **26.2%** (85/324) | **85.7%** (24/28) |
| Δ | +6.2 pp | +3.0 pp | +41.3 pp |

Task counts differ slightly between runs because a few OSWorld tasks fail to
initialize or time out during environment setup, independently of the agent.

📄 **[Full write-up (12 pp.)](docs/report.pdf)** · 🤗 **[LoRA adapter](https://huggingface.co/ageppert/world-model-7b-lora)** · 🤗 **[Transition dataset](https://huggingface.co/datasets/ageppert/world-model-transitions)**

---

## The idea

World-model reasoning — estimate the current state, predict what each candidate action
would do, then pick the best — has improved agent performance across web, GUI, and game
environments. But the existing approaches (VAGEN, WMA, SPA) get there through
reinforcement learning or end-to-end co-training.

This project asks a narrower question: **can world-model augmentation be retrofitted
around an agent you cannot touch, using only SFT?** The base policy stays frozen. The
only trained component is a 39 MB LoRA adapter that predicts state transitions in
natural language.

The whole pipeline is a drop-in replacement for the model endpoint. It accepts the same
request (screenshot + instruction) and returns the same response (chain-of-thought +
PyAutoGUI code), so OSWorld cannot tell the difference.

## Pipeline

Five steps per action, all in [`world_modeling/core/pipeline.py`](world_modeling/core/pipeline.py).
A single OpenCUA-7B instance serves every role; the LoRA adapter is toggled on for
step 3 and off everywhere else, so only one copy of the model is ever in memory.

| # | Step | What it does | Adapter |
|---|---|---|---|
| 1 | **State estimation** | Re-prompts the model with OpenCUA's L3 format and stops generation at the `Thought:` header, harvesting just the `Observation` section. No separate state encoder is trained. | off |
| 2 | **Candidate generation** | Produces N action proposals in L2 format. Candidate 0 is greedy (identical to vanilla); the rest use τ=0.7. | off |
| 3 | **Transition prediction** | The world model. Text-only: given the state description and a candidate action, predict what changes on screen. | **on** |
| 4 | **Scoring** | LLM-as-judge rates each predicted transition 1–10 against the task goal. Ties break toward the greedy candidate. | off |
| 5 | **Response assembly** | Inserts the observation ahead of the winning candidate's `Thought:` section, yielding a well-formed L3 response. | off |

With `--n-candidates 1`, steps 3 and 4 are skipped entirely and the pipeline degrades
gracefully to vanilla L2 generation with an observation prepended.

## Decoupled generation

This is the part worth reading even if you don't care about world models.

OpenCUA emits actions as PyAutoGUI code with literal pixel coordinates —
`pyautogui.click(x=742, y=356)` — generated autoregressively **as digit tokens**.
Temperature sampling, applied uniformly, spreads probability mass onto neighbouring
digits. `x=742` becomes `x=748`, the click lands on the wrong UI element, and the task
fails. The first version of this framework scored **8.3%** against vanilla's 17.3% on a
20-task pilot for exactly this reason.

The fix is to sample where diversity helps and stop sampling where precision is
required. [`core/decoupled.py`](world_modeling/core/decoupled.py) is a ~60-line
`LogitsProcessor` that watches the decoded text for a `Code:` marker and, once it
appears, forces argmax selection for every remaining token:

```python
CODE_MARKERS = ["Code:", "```python", "```\npyautogui"]

def __call__(self, input_ids, scores):
    if self.greedy:
        return self._force_greedy(scores)
    text = self.tokenizer.decode(input_ids[0, self.prompt_len:], skip_special_tokens=True)
    if any(marker in text for marker in CODE_MARKERS):
        self.greedy = True
        return self._force_greedy(scores)
    return scores
```

Reasoning stays diverse; coordinates become deterministic. It attaches to a standard
`generate()` call, needs no change to the model, and costs nothing extra. Smoke tests
confirm greedy and sampled candidates emit identical code despite differing reasoning.

Toggle it with `--no-greedy-after-code` to reproduce the broken v1 behaviour.

## Results by category

Feasible tasks only. Full table including infeasible tasks in [the report](docs/report.pdf).

| Category | Vanilla | CUA-WM | Δ (pp) |
|---|---|---|---|
| Multi Apps | 8.7% | 13.1% | +4.4 |
| LibreOffice Impress | 31.9% | 27.7% | −4.3 |
| LibreOffice Calc | 2.2% | 8.7% | +6.5 |
| Chrome | 30.2% | 38.6% | +8.4 |
| LibreOffice Writer | 31.8% | 36.4% | +4.5 |
| OS | 31.6% | 33.3% | +1.8 |
| VS Code | 55.6% | 44.4% | −11.1 |
| GIMP | 37.5% | 43.8% | +6.3 |
| VLC | 20.0% | 33.3% | +13.3 |
| Thunderbird | 57.1% | 42.9% | −14.3 |
| **All feasible** | **23.2%** | **26.2%** | **+3.0** |

## Limitations

Stated plainly, because the aggregate number hides them:

- **It regresses where the base policy is already strong.** Thunderbird (−14.3 pp) and
  VS Code (−11.1 pp) were vanilla's two best categories. The gains cluster in categories
  vanilla handles poorly — Calc (2.2%), Multi Apps (8.7%), VLC (20.0%). The world model
  appears to help when the policy is lost and to add misselection risk when it isn't.
- **It is ~5× slower.** Roughly 48 s per step with N=2, against ~10 s for vanilla, from
  the extra state-estimation call, the second candidate, and two transition + two scoring
  passes. Steps 3 and 4 are text-only and comparatively cheap.
- **The infeasible-task gain was not a design goal.** 44.4% → 85.7% is the largest effect
  in the project and it was unplanned; the explanation in §5.3 of the report is a
  hypothesis, not a measurement.
- **Single run per configuration.** No seed variance is reported.

## Repository layout

```
world_modeling/
  core/
    pipeline.py        the five steps and run_pipeline orchestrator
    decoupled.py       GreedyAfterCodeHeader logits processor
    model_manager.py   model loading, adapter toggling, generation
    parsing.py         lenient regexes for Observation / Action / Code
    prompts.py         prompt templates, single source of truth
  framework_api.py         serves the full pipeline (OpenAI-compatible)
  opencua_api.py           vanilla baseline server, deliberately self-contained
  preprocess_transitions.py  AgentNet -> transition training pairs
  train_world_model.py       LoRA fine-tuning via TRL SFTTrainer
  eval_world_model.py        adapter evaluation on the held-out split
  smoke_test_framework.py    end-to-end checks, no OSWorld required
  startup.sh, common.sh, run_*.sh   container entry points and job helpers
  *.sub                    HTCondor submit files
docs/
  architecture.md      the five steps in depth
  chtc-deployment.md   HTCondor, the container, and the tunnel
  reproducing.md       preprocess -> train -> eval -> serve
  report.pdf           full write-up
```

## Getting started

```bash
git clone https://github.com/ageppert01/CUA-WM.git && cd CUA-WM
cp .env.example .env      # then fill in HF_TOKEN and CUA_WM_TUNNEL_HOST
pip install -r requirements.txt
```

Serving the framework needs one GPU with ≥40 GB VRAM (L40/L40S); OpenCUA-7B plus the
adapter occupies ~14 GB in bf16, and the headroom is for long multimodal contexts.

```bash
cd world_modeling
python framework_api.py --n-candidates 2 --port 9009
curl localhost:9009/health
```

`/health` reports the live configuration — model, adapter, candidate count, and whether
decoupled generation is on — so the server always tells you what it is actually doing.

To reproduce training and evaluation from scratch, see
**[docs/reproducing.md](docs/reproducing.md)**. For the cluster setup, see
**[docs/chtc-deployment.md](docs/chtc-deployment.md)**.

**OSWorld is not vendored here.** It runs unmodified, outside this repository, pointed
at this framework's `/v1/chat/completions` endpoint. See
[docs/reproducing.md](docs/reproducing.md#running-osworld-against-the-framework).

## Built on

- **[OpenCUA](https://github.com/xlang-ai/OpenCUA)** (Wang et al.) — the frozen base policy,
  `OpenCUA-7B`, and its L1/L2/L3 reasoning hierarchy.
- **[AgentNet](https://huggingface.co/datasets/xlangai/AgentNet)** — the Ubuntu 5K split,
  5,000 trajectories / 82,448 steps, from which 42,103 transition pairs were extracted.
- **[OSWorld](https://github.com/xlang-ai/OSWorld)** (Xie et al.) — the benchmark.
- Method lineage: **VAGEN** (Wang et al.) for the POMDP state/transition decomposition,
  **WMA-Agents** (Chae et al.) for transition-focused observation abstraction, and
  **SPA** (Chen et al.) for confining world-model learning to SFT.

Full references in [the report](docs/report.pdf).

## Citing

See [CITATION.cff](CITATION.cff).

## License

MIT — see [LICENSE](LICENSE). Note that OpenCUA-7B, AgentNet, and OSWorld carry their own
licenses; this repository does not redistribute any of them.
