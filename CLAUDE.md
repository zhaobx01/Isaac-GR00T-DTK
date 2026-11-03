# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Isaac GR00T N1.7

## Project overview

Isaac GR00T N1.7 is an open vision-language-action (VLA) model for generalized humanoid robot skills.
The repo contains the model, training pipeline, evaluation harness, and deployment tooling.

- **Language:** Python 3.12
- **Package manager:** [uv](https://docs.astral.sh/uv/)
- **Build system:** setuptools (see `pyproject.toml`)

## Quick-start commands

```bash
# Install (dev mode with all extras)
uv sync --all-extras

# Lint and format (uses ruff via pre-commit)
pre-commit run --all-files

# Run CPU tests
python -m pytest tests/ -m "not gpu" -v --timeout=300

# Run GPU tests
python -m pytest tests/ -m gpu -v --timeout=300

# Build package
uv build

# Validate lockfile
uv lock --locked
```

## Code style

- Formatter: `ruff format` (double quotes, spaces, line-length 100)
- Linter: `ruff check` with rules E, F, I (ignores E501)
- Config lives in `pyproject.toml` under `[tool.ruff]`
- Run `pre-commit run --all-files` before committing

## Directory layout

```
gr00t/              # Main package
  configs/          #   Training, data, and model configs
  data/             #   Data loading, embodiment tags, dataset processing
  eval/             #   Evaluation (run_gr00t_server.py)
  experiment/       #   Training pipeline (launch_finetune.py, trainer.py)
  model/            #   Model architecture (N1.7, base, modules)
  policy/           #   Policy inference (Gr00tPolicy, server/client)
examples/           # Per-embodiment example configs and READMEs
scripts/            # Deployment, conversion, and utility scripts
  deployment/       #   Platform install scripts (dgpu, orin, thor, spark)
tests/              # pytest suite (markers: gpu, not gpu)
getting_started/    # User-facing guides and notebooks
```

## Architecture

**Model (`gr00t/model/`):** `Gr00tN1d7` (`gr00t/model/gr00t_n1d7/gr00t_n1d7.py:500`) composes `backbone` (`Qwen3Backbone`, selected via `get_backbone_cls` based on `config.model_name`), `action_head` (`Gr00tN1d7ActionHead`), and `collator`. Forward pass: `prepare_input` splits the batch into backbone/action inputs → `backbone()` produces features → `action_head()` runs flow-matching diffusion through a `DiT`/`AlternateVLDiT` (`gr00t/model/modules/dit.py`) and returns an MSE loss. Inference goes through `get_action` → `get_action_with_features`, which does iterative Euler integration over diffusion timesteps and supports real-time-chunking (RTC) for action inpainting across chunks. `EmbodimentTag` (`gr00t/data/embodiment_tags.py:29`) is the string-level robot identity (e.g. `"libero_sim"`); it's mapped to an integer projector slot via `EMBODIMENT_TAG_TO_PROJECTOR_INDEX` (`gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py:118`), which indexes the per-embodiment `CategorySpecificMLP` state/action encoders. `MODEL_REGISTRY` (`gr00t/model/registry.py`) dispatches a model config type to its `BasicPipeline` (`gr00t/model/base/model_pipeline.py`).

**Data (`gr00t/data/`):** `DatasetFactory.build` (`gr00t/data/dataset/factory.py:36`) builds a `ShardedSingleStepDataset` per dataset spec — it reads LeRobot-format episodes via `LeRobotEpisodeLoader` and shards them by `embodiment_tag` plus that tag's `ModalityConfig`. Multiple per-embodiment datasets combine into a `ShardedMixtureDataset`, weighted by `mix_ratio`/`ds_weights_alpha`, which applies the model's processor while iterating. `modality_configs` is `dict[embodiment_tag, dict[modality_name, ModalityConfig]]`, resolved from the pre-registered `MODALITY_CONFIGS` table (`gr00t/configs/data/embodiment_configs.py`) or a user-supplied config file. `BasicDataCollator` batches samples into tensors; `gr00t/data/stats.py` precomputes per-embodiment normalization statistics.

**Policy (`gr00t/policy/`):** `Gr00tPolicy` (`gr00t/policy/gr00t_policy.py:70`) loads the model and processor via `AutoModel`/`AutoProcessor`, selects the modality config for its embodiment tag, and in `_get_action` turns raw observations into `VLAStepData` → processor output → collated tensors → `model.get_action()` → unnormalized actions. `Gr00tSimPolicyWrapper` adapts flat sim-env observation keys into the nested dict `Gr00tPolicy` expects. The server/client split (`gr00t/policy/server_client.py`) uses ZeroMQ REQ/REP: `PolicyServer` wraps any `BasePolicy` and exposes `get_action`/`reset`/`get_modality_config` over `MsgSerializer` (msgpack+numpy, with an explicit `allow_pickle=False` boundary); `PolicyClient` proxies the same calls over the socket. `gr00t/eval/run_gr00t_server.py:main` is the CLI that picks a `Gr00tPolicy` (given `--model-path`) or `ReplayPolicy` (given `--dataset-path`) and starts the server.

**Training pipeline (`gr00t/experiment/`):** `launch_finetune.py` is the CLI entry — it parses `FinetuneConfig` (tyro), builds a full `Config` via `get_default_config().load_dict(...)`, copies `FinetuneConfig` fields onto `config.model`/`config.training`/`config.data`, then calls `experiment.run(config)`. `experiment.py:run` drives the actual training loop: sets up the distributed process group, resolves the `MODEL_REGISTRY` pipeline, optionally calls `install_model_compile` (`compilation.py`) to wrap submodules in `torch.compile`, builds `TrainingArguments` and the custom `Gr00tTrainer` (`trainer.py:222`), attaches callbacks (checkpointing, best-metric tracking, an optional profiler via `profiling.py`), then calls `trainer.train()`. `Gr00tTrainer` overrides `get_train_dataloader` (reseeds the mixture dataset correctly on resume) and `compute_loss` (adds token-accuracy and numerics-monitor logging). `compilation.py` installs lazy `torch.compile` wrappers onto specific submodules (DiT blocks, the whole DiT, Qwen3VL text decoder layers, VL self-attention blocks) with strict preflight checks and idempotency guards so repeated calls with the same config are no-ops. `profiling.py` builds a `torch.profiler.schedule` from the training config's `profiler_*` fields.

**Configs (`gr00t/configs/`):** `FinetuneConfig` is the flat, tyro-parsed CLI front door; `launch_finetune.py` fans its fields into the nested `Config` (model/data/training sub-configs). `TrainingConfig` (`gr00t/configs/training/training_config.py`) is the nested sub-config actually consumed by `experiment.run` to build `TrainingArguments`/DeepSpeed config/trainer callbacks. Some fields (e.g. `compile_dit_blocks`, `global_batch_size`) exist in both — `FinetuneConfig` values overwrite `TrainingConfig` defaults in `launch_finetune.py`.

## Key entry points

- **Fine-tune:** `bash examples/finetune.sh --base-model-path <path> --dataset-path <path> --embodiment-tag <tag> --output-dir <dir>`
- **Inference server:** `python gr00t/eval/run_gr00t_server.py --model-path <path> --embodiment-tag <tag>`
- **ONNX export:** `python scripts/deployment/export_onnx_n1d7.py`
- **TensorRT build:** `python scripts/deployment/build_trt_pipeline.py`
- **Benchmark:** `python scripts/deployment/benchmark_inference.py`

## Testing

- Test markers: `gpu` (requires GPU), default is CPU-safe
- Fixtures live in `tests/fixtures/` and `demo_data/`
- CI runs CPU and GPU tests in separate jobs with 300s timeout

## Deployment platforms

- **dGPU (H100, A100, RTX):** CUDA 12.8 — install via `scripts/deployment/dgpu/install_deps.sh`, container via top-level `docker/Dockerfile` (supports x86_64 and aarch64)
- **Jetson Orin:** JetPack 7.2 / CUDA 13.2 — install via `scripts/deployment/orin/install_deps.sh`, container via `scripts/deployment/orin/Dockerfile`
- **Jetson Thor:** JetPack 7.2 / CUDA 13.2 — install via `scripts/deployment/thor/install_deps.sh`, container via `scripts/deployment/thor/Dockerfile`
- **DGX Spark:** CUDA 13.0 — install via `scripts/deployment/spark/install_deps.sh`, container via `scripts/deployment/spark/Dockerfile`

Each Jetson/Spark platform ships an `activate_*.sh` helper (`scripts/activate_orin.sh`, `scripts/activate_spark.sh`, `scripts/activate_thor.sh`) that exports platform-specific library paths. For dGPU, the standard `source .venv/bin/activate` is sufficient.
