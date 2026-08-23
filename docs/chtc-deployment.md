# Running on CHTC (HTCondor)

This project was developed on the UW–Madison [Center for High Throughput
Computing](https://chtc.cs.wisc.edu/) cluster. Nothing here is CHTC-specific in principle
— it is ordinary HTCondor container-universe work — but the constraints that shaped the
scripts are worth explaining, because most of them are not obvious.

## The three constraints

**1. The container filesystem is read-only.** Everything must run from the job's scratch
directory. `prepare_model_workspace` in `common.sh` copies `/OpenCUA/*` out of the image
into scratch before doing anything else.

**2. Packages cannot simply be `pip install`ed.** Execute nodes have no writable
site-packages, so `install_packages_with_strip` installs into a local `./pip_packages`
and puts it on `PYTHONPATH`. But pip will happily pull its own `torch` as a transitive
dependency, and a pip `torch` will not match the node's CUDA driver. So after installing,
the function *removes* anything that would shadow the container's copy:

```bash
CONTAINER_PACKAGES="torch nvidia triton sympy mpmath transformers tokenizers numpy accelerate"
```

`transformers` and `tokenizers` are on that list for a different reason: OpenCUA-7B ships
custom modeling code pinned to `transformers` 4.53.0, and a newer version breaks it.
There is also an explicit guard for `huggingface_hub`, because some pip resolvers ignore
the `<1.0` constraint that 4.53.0 requires:

```bash
if v >= (1, 0):
    os.system('rm -rf pip_packages/huggingface_hub*')
```

Use `install_packages_simple` instead for pure-Python helpers where none of this applies.

**3. Execute nodes have no inbound routing.** OSWorld runs elsewhere and needs to reach
the model API. The job therefore dials *out*, opening a reverse SSH tunnel that forwards
a port on a reachable host back to the port the API is bound to on the execute node.
`open_ssh_tunnel` runs this in a loop, reconnecting after drops, and never returns — it
is the job's foreground process, keeping the job alive while the API serves in the
background.

## Configuration

Nothing personal is hardcoded. Set these before submitting (see [`.env.example`](../.env.example)):

| Variable | Meaning | Default |
|---|---|---|
| `HF_TOKEN` | HuggingFace token, for AgentNet download and Hub uploads | *required* |
| `CUA_WM_TUNNEL_HOST` | `user@host` the tunnel dials out to | *required for serving* |
| `CUA_WM_TUNNEL_KEY` | private key file, staged alongside the job scripts | `tunnel_key` |
| `CUA_WM_TUNNEL_PORT` | port forwarded back, and bound by the API | `9009` |

The tunnel key must live next to the job scripts so HTCondor's `transfer_input_files`
stages it onto the execute node. It is gitignored. Generate a dedicated key for this —
do not reuse a personal one:

```bash
ssh-keygen -t ed25519 -f world_modeling/tunnel_key -N ""
ssh-copy-id -i world_modeling/tunnel_key.pub user@your-host
```

## The container

Submit files reference the published image directly:

```
universe = container
container_image = docker://ageppert01/chtc-cua:latest
```

HTCondor pulls and converts it per job. For repeated submissions, building a local `.sif`
once is much faster:

```bash
apptainer build chtc-cua.sif docker://ageppert01/chtc-cua:latest
# then point container_image at the local file
```

**The Dockerfile that produced this image is not in this repository.** The published
image is the reference. If you are rebuilding from scratch, the contract the scripts
assume is: a CUDA base with a driver-matched `torch`, `transformers==4.53.0`,
`tokenizers`, `numpy`, `accelerate`, and `flask` baked in; `/OpenCUA` present with the
model helper scripts; everything else installed at runtime into `./pip_packages`.

## Submit files

Each `.sub` pairs a mode with resources. The `startup.sh`-based jobs take the mode as an
argument; the `run_*.sh` jobs are one-shot batch work.

| Submit file | Runs | Purpose |
|---|---|---|
| `baseline.sub` | `startup.sh baseline` | vanilla OpenCUA API, the comparison point |
| `framework.sub` | `startup.sh framework` | full pipeline, current defaults |
| `framework_no_greedy_after_code.sub` | `startup.sh framework-no-greedy-after-code` | the v1 ablation |
| `smoke_test.sub` | `startup.sh smoke-test` | end-to-end checks, no OSWorld needed |
| `smoke_test_no_greedy_after_code.sub` | `startup.sh smoke-test-no-greedy-after-code` | same, v1 decoding |
| `preprocess_transitions.sub` | `run_preprocess.sh` | AgentNet → transition pairs |
| `train_world_model.sub` | `run_train.sh` | LoRA fine-tuning |
| `eval_world_model.sub` | `run_eval.sh` | adapter eval on the held-out split |
| `inspect_agentnet.sub` | `run_inspect.sh` | dataset inspection |

Adding an ablation axis is one new mode in `startup.sh` plus one `.sub` — the flags live
in `framework_api.py`, and no file is forked to add one.

> **Note on the baseline job.** `baseline.sub` does not stage `opencua_api.py`; the
> baseline server comes from the container instead, via the `cp -r /OpenCUA/* .` in
> `prepare_model_workspace`. That copy runs *after* HTCondor stages input files, so it
> would overwrite a transferred copy anyway. The consequence worth knowing: the
> `opencua_api.py` in this repository is the reference implementation of the baseline,
> but the bytes that actually executed on the cluster came from the image. The framework
> jobs do not share this quirk — they stage `framework_api.py` and `core/` explicitly,
> and those are the files that ran.

## Resources

The serving and training jobs ask for:

```
+WantGPULab = true
+GPUJobLength = "short"
gpus_minimum_memory = 48GB
+RequireExclusiveGPU = true
request_gpus = 1
request_cpus = 1
request_memory = 40GB
request_disk = 40GB
```

40 GB VRAM is the floor; in practice L40 or L40S. The model plus adapter is ~14 GB in
bf16 — the headroom absorbs long multimodal contexts. `+RequireExclusiveGPU` matters:
sharing the GPU makes per-step latency unpredictable enough to distort timing numbers.

Job logs are written under `logs/api_jobs/` and are gitignored.

## Submitting

```bash
cd world_modeling
condor_submit framework.sub
condor_q
condor_tail -f <cluster>.0        # stream_output is on
```

Then, from the tunnel host, confirm the API came up:

```bash
curl localhost:9009/health
```

## A note on what was removed

Earlier bring-up work — pool probes, container-runtime and privilege checks, nested-submit
experiments, and a dump of CHTC pool state — lived in `test_jobs/` and `test_jobs2/`.
It was scaffolding for learning the cluster, not part of the framework, and was removed
when this repository was made public.
