#!/usr/bin/env python

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
from pathlib import Path
import warnings

from omegaconf import OmegaConf
import torch
import torch.distributed as dist
from transformers import TrainingArguments, set_seed
from transformers.trainer_callback import ProgressCallback
from transformers.utils import is_torch_tf32_available

from gr00t.configs.base_config import Config
from gr00t.configs.training.training_config import check_resume_compatibility

# Use custom trainer that profiles data loading & forward times
from gr00t.experiment.trainer import (
    Gr00tTrainer,
    ProfCallback,
    SamplesPerSecondProgressCallback,
)
from gr00t.experiment.utils import BestMetricCheckpointCallback, CheckpointFormatCallback
from gr00t.model import MODEL_REGISTRY
from gr00t.utils.dist_utils import run_on_rank0, run_or_wait_on_rank0
from gr00t.utils.initial_actions import INITIAL_ACTIONS_FILENAME, save_initial_actions


def setup_logging(debug: bool = False):
    """Configure logging."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    # Reduce verbosity of some libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)


def _assert_num_gpus_matches_world_size(num_gpus: int) -> None:
    """``num_gpus`` must equal the launcher's data-parallel world size.

    ``num_gpus`` drives per-device batch size and the DeepSpeed gating, while
    HF ``Trainer`` and dataset sharding use the real ``WORLD_SIZE`` from
    torchrun. A mismatch silently rescales the effective batch (e.g. an 8-rank
    launch with ``num_gpus=1`` trains at 8× the intended batch), so reconcile
    at the launcher trust boundary.
    """
    world_size = os.environ.get("WORLD_SIZE")
    if world_size is not None and int(world_size) != num_gpus:
        raise ValueError(
            f"config.training.num_gpus={num_gpus} does not match the launcher "
            f"WORLD_SIZE={world_size}. num_gpus drives per-device batch size and "
            f"DeepSpeed gating, while HF Trainer and dataset sharding use "
            f"WORLD_SIZE; set num_gpus to {world_size}."
        )


def warn_configs(config: Config):
    # updates to batch size
    _assert_num_gpus_matches_world_size(config.training.num_gpus)
    assert config.training.global_batch_size % config.training.num_gpus == 0, (
        "global_batch_size must be divisible by num_gpus"
    )

    if config.training.gradient_accumulation_steps > 1:
        logging.info(
            "global_batch_size=%d × gradient_accumulation_steps=%d "
            "→ accumulated_batch_size=%d per optimizer step",
            config.training.global_batch_size,
            config.training.gradient_accumulation_steps,
            config.training.accumulated_batch_size,
        )

    if config.training.per_gpu_batch_size is not None:
        warnings.warn(
            "per_gpu_batch_size will be deprecated in the future, please use global_batch_size instead. For now, this will override global_batch_size."
        )

    if config.training.warmup_steps > 0:
        warnings.warn(
            "warmup_steps will be deprecated in the future, please use warmup_ratio instead. For now, this will override warmup_ratio."
        )

    if (
        hasattr(config.model, "backbone_trainable_params_fp32")
        and not config.model.backbone_trainable_params_fp32
    ):
        warnings.warn(
            "backbone_trainable_params_fp32 is not True. This will be deprecated in the future."
        )

    if (
        hasattr(config.model, "use_albumentations_transforms")
        and not config.model.use_albumentations_transforms
    ):
        warnings.warn(
            "use_albumentations_transforms is not True. This will be deprecated in the future."
        )

    if (
        hasattr(config.model, "image_crop_size")
        and hasattr(config.model, "image_target_size")
        and (config.model.image_crop_size is not None or config.model.image_target_size is not None)
    ):
        assert (
            config.model.image_crop_size is not None and config.model.image_target_size is not None
        ), "image_crop_size and image_target_size must be set together"
        warnings.warn(
            "image_crop_size and image_target_size will be deprecated in the future. Please use shortest_image_edge and crop_fraction instead."
        )
        if hasattr(config.model, "shortest_image_edge") and hasattr(config.model, "crop_fraction"):
            assert (
                config.model.shortest_image_edge is None and config.model.crop_fraction is None
            ), (
                "Do not set shortest_image_edge and crop_fraction together with image_crop_size and image_target_size"
            )

    if (
        hasattr(config.model, "shortest_image_edge")
        and hasattr(config.model, "crop_fraction")
        and (config.model.shortest_image_edge is not None or config.model.crop_fraction is not None)
    ):
        assert config.model.use_albumentations_transforms, (
            "use_albumentations_transforms must be True when shortest_image_edge and crop_fraction are set"
        )


def _init_distributed_process_group() -> int:
    """Init the NCCL process group with ``device_id=cuda:LOCAL_RANK`` so NCCL
    binds the communicator eagerly (the PyTorch >=2.4 recommended pattern).
    No-op when ``dist`` is already initialized or ``WORLD_SIZE`` is unset / 1.
    """
    if dist.is_initialized():
        return dist.get_rank()
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
        )
        return dist.get_rank()
    return 0


def save_run_config_artifacts(
    save_cfg_dir: Path, output_dir: Path, config: Config, experiment_name: str
):
    """Write ``config.yaml`` / ``conf.yaml`` / ``wandb_config.json``."""
    save_cfg_dir.mkdir(parents=True, exist_ok=True)
    config.save(save_cfg_dir / "config.yaml")
    omegaconf_config = OmegaConf.create(config.__dict__)
    omegaconf_config["max_steps"] = config.training.max_steps
    omegaconf_config["save_steps"] = config.training.save_steps
    OmegaConf.save(omegaconf_config, save_cfg_dir / "conf.yaml", resolve=True)
    wandb_config_file = output_dir / "wandb_config.json"
    with open(wandb_config_file, "w") as f:
        json.dump(
            {
                "project": config.training.wandb_project,
                "run_id": experiment_name,
            },
            f,
        )
    logging.info(f"Saved config to {save_cfg_dir}")


def save_initial_actions_artifact(train_dataset, save_cfg_dir: Path):
    """Write ``initial_actions.npz`` from ``train_dataset``; a falsy payload is a no-op."""
    initial_actions = train_dataset.get_initial_actions()
    if not initial_actions:
        return
    initial_actions_path = save_cfg_dir / INITIAL_ACTIONS_FILENAME
    save_initial_actions(initial_actions, initial_actions_path)
    logging.info(f"Saved {len(initial_actions)} initial actions to {initial_actions_path}")


def run(config: Config):
    """Main training function."""
    from gr00t.experiment.training_runtime import (
        resolve_trainer_class,
        resolve_compile_variant,
        resolve_legacy_compile_request,
        resolve_zero2_comm_switches,
        resolve_zero2_pack_timing,
    )

    trainer_class = resolve_trainer_class(config)
    compile_variant = resolve_compile_variant()
    legacy_compile_request = resolve_legacy_compile_request(config.training, compile_variant)
    from gr00t.Replacement.sequence_buckets import load_buckets, resolve_policy, validate_activation

    sequence_policy, sequence_bucket_path, sequence_diagnostics = resolve_policy()
    validate_activation(sequence_policy, sequence_diagnostics, compile_variant, legacy_compile_request)
    sequence_buckets = load_buckets(sequence_bucket_path) if sequence_policy == "dynamic-buckets" else None
    zero2_overlap, zero2_reduce_scatter = resolve_zero2_comm_switches()
    if zero2_overlap or zero2_reduce_scatter:
        from gr00t.Replacement.zero2_comm import validate_training_config as validate_zero2_comm_config

        validate_zero2_comm_config(config, zero2_overlap, zero2_reduce_scatter)
    if (config.training.check_numerics and config.training.num_gpus > 1 and
            not config.training.use_ddp and config.training.deepspeed_stage != 2):
        raise ValueError("Numerical diagnostics currently support DeepSpeed ZeRO-2 only")
    warn_configs(config)
    check_resume_compatibility(config.training)

    global_rank = _init_distributed_process_group()

    # Setup
    setup_logging()
    if global_rank != 0:
        logging.getLogger().setLevel(logging.WARNING)
    set_seed(config.data.seed)

    # Validate config
    config.validate()

    # Create output directory
    if config.training.experiment_name is None:
        output_dir = Path(config.training.output_dir)
        experiment_name = output_dir.name
    else:
        output_dir = Path(config.training.output_dir) / config.training.experiment_name
        experiment_name = config.training.experiment_name

    run_on_rank0(output_dir.mkdir, parents=True, exist_ok=True, label="output_dir.mkdir")

    save_cfg_dir = output_dir / "experiment_cfg"
    processor_dir = output_dir / "processor"

    # Rank-0-only write; wrapped so a rank-0 failure surfaces on every rank
    # instead of stranding peers at the next NCCL collective.
    run_on_rank0(
        save_run_config_artifacts,
        save_cfg_dir,
        output_dir,
        config,
        experiment_name,
    )

    # wandb.init does network I/O; wrap so a rank-0 failure can't strand peers.
    if config.training.use_wandb:
        import wandb
        with run_or_wait_on_rank0(label="wandb.init") as is_rank0:
            if is_rank0:
                config_dict = {
                    **config.__dict__,
                    "git_commit_hash": os.environ.get("GROOT_COMMIT_HASH", "unknown"),
                }

                wandb.init(
                    project=config.training.wandb_project,
                    name=experiment_name,
                    config=config_dict,
                    tags=[config.data.mode],
                )

    # Setup model training pipeline.
    pipeline = MODEL_REGISTRY.get(type(config.model))(config, save_cfg_dir)
    pipeline.setup()
    model = pipeline.return_model()
    compile_info = {"variant": compile_variant, "paths": []}
    if legacy_compile_request is not None:
        from gr00t.experiment.compilation import install_model_compile

        compiled_paths = install_model_compile(model, **legacy_compile_request)
        compile_info = {
            "variant": "legacy-cli", "paths": compiled_paths, "request": legacy_compile_request
        }
    elif compile_variant != "off":
        from gr00t.Replacement.model_compile import install, write_manifest

        compile_info = install(model, compile_variant)
        write_manifest(output_dir, compile_info)
    numerics_monitor = None
    if config.training.check_numerics:
        from gr00t.experiment.numerics import NumericalMonitor

        numerics_monitor = NumericalMonitor(output_dir)
        numerics_monitor.attach(model)
    train_dataset, eval_dataset = pipeline.return_dataset()
    data_collator = pipeline.return_collator()
    processor = pipeline.return_processor()
    # statistics.json here is read by Gr00tPolicy.from_pretrained at deploy time;
    # a torn write silently degrades inference, so gate + broadcast failures.
    run_on_rank0(processor.save_pretrained, processor_dir, label="processor.save_pretrained")

    # deepspeed config
    if config.training.num_gpus > 1 and not config.training.use_ddp:
        from gr00t.experiment.training_runtime import load_experiment_deepspeed

        deepspeed_config = load_experiment_deepspeed(config.get_deepspeed_config())
    else:
        deepspeed_config = None

    # for now we will let per_gpu_batch_size override global_batch_size, in future we will deprecate per_gpu_batch_size
    if config.training.per_gpu_batch_size is None:
        per_device_train_batch_size = config.training.global_batch_size // config.training.num_gpus
    else:
        per_device_train_batch_size = config.training.per_gpu_batch_size

    tf32 = config.training.tf32 and is_torch_tf32_available()
    if config.training.tf32 and not tf32:
        logging.info("TF32 requested but unsupported by this device; disabling TF32")

    # Create training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=config.training.max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=config.training.eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        lr_scheduler_type=config.training.lr_scheduler_type,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        warmup_steps=config.training.warmup_steps,
        max_grad_norm=config.training.max_grad_norm,
        logging_steps=config.training.logging_steps,
        logging_nan_inf_filter=not config.training.check_numerics,
        save_steps=config.training.save_steps,
        save_strategy=config.training.save_strategy,
        save_total_limit=config.training.save_total_limit,
        save_only_model=config.training.save_only_model,
        fp16=config.training.fp16,
        bf16=config.training.bf16,
        tf32=tf32,
        gradient_checkpointing=config.training.gradient_checkpointing,
        optim=config.training.optim,
        dataloader_num_workers=config.training.dataloader_num_workers,
        report_to="wandb" if config.training.use_wandb else "none",
        seed=config.data.seed,
        deepspeed=deepspeed_config,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=config.training.ddp_bucket_cap_mb,
        eval_strategy=config.training.eval_strategy,
        eval_steps=config.training.eval_steps,
        batch_eval_metrics=True,
        remove_unused_columns=config.training.remove_unused_columns,
        ignore_data_skip=True,
    )

    # Create trainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        multiprocessing_context=config.data.multiprocessing_context,
        numerics_monitor=numerics_monitor,
    )
    if compile_variant != "off":
        from gr00t.Replacement.model_compile import evidence_callback

        trainer.add_callback(evidence_callback(model, compile_info))
    if os.environ.get("GR00T_EXPERIMENT_CASE"):
        from gr00t.Replacement.experiment_manifest import ExperimentManifestCallback

        trainer.add_callback(ExperimentManifestCallback(trainer))
    trainer.remove_callback(ProgressCallback)
    trainer.add_callback(SamplesPerSecondProgressCallback())
    from gr00t.Replacement.zero2_comm import verify_switch_consistency

    verify_switch_consistency(zero2_overlap, zero2_reduce_scatter)
    zero2_callback = None
    if zero2_overlap or zero2_reduce_scatter:
        from gr00t.Replacement.zero2_comm import Zero2CommCallback

        zero2_callback = Zero2CommCallback(
            trainer, zero2_overlap, zero2_reduce_scatter, output_dir=output_dir,
            pack_timing=resolve_zero2_pack_timing(),
        )
        trainer.add_callback(zero2_callback)
    if config.training.log_step_metrics:
        from gr00t.experiment.trainer import StepMetricsCallback

        trainer.add_callback(
            StepMetricsCallback(
                Path(output_dir) / "step_metrics",
                run_info={
                    "zero2_allgather_pipeline": os.environ.get("GR00T_ZERO2_ALLGATHER_PIPELINE", "0") == "1",
                    "zero2_overlap": zero2_overlap,
                    "zero2_reduce_scatter": zero2_reduce_scatter,
                    "compile_variant": compile_info["variant"],
                    "compiled_paths": compile_info["paths"],
                    "legacy_compile_request": legacy_compile_request,
                    "global_batch_size": config.training.global_batch_size,
                    "num_gpus": config.training.num_gpus,
                },
            )
        )

    trainer.add_callback(
        CheckpointFormatCallback(
            run_name=experiment_name,
            exp_cfg_dir=save_cfg_dir,
            processor_dir=processor_dir,
        )
    )

    if config.training.save_best_eval_metric_name != "":
        trainer.add_callback(
            BestMetricCheckpointCallback(
                metric_name=config.training.save_best_eval_metric_name,
                greater_is_better=config.training.save_best_eval_metric_greater_is_better,
                exp_cfg_dir=save_cfg_dir,
                trainer=trainer,
            )
        )

    if hasattr(train_dataset, "get_initial_actions"):
        run_on_rank0(save_initial_actions_artifact, train_dataset, save_cfg_dir)

    # Train
    sequence_adapter = None
    sequence_trainer_diagnostics = None
    fixed_padding = None
    logging.info("🚀 Starting training...")
    try:
        if sequence_policy == "dynamic-buckets" or sequence_diagnostics:
            from gr00t.Replacement.sequence_buckets import (
                DiagnosticRecorder, SequenceAdapter, TrainerDiagnostics,
            )

            recorder = None
            if sequence_diagnostics:
                local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                visible = os.environ.get("HIP_VISIBLE_DEVICES", "").split(",")
                physical = visible[local_rank] if len(visible) > local_rank and visible[local_rank] else "unknown"
                recorder = DiagnosticRecorder(output_dir / "sequence-diagnostics", rank=global_rank,
                                              physical_device=physical)
                sequence_trainer_diagnostics = TrainerDiagnostics(trainer, recorder)
                sequence_trainer_diagnostics.install()
            targets = [("dit" if path == "action_head.model" else "vl", model.get_submodule(path), path)
                       for path in compile_info["paths"]]
            sequence_adapter = SequenceAdapter(targets, sequence_buckets, recorder=recorder)
            sequence_adapter.__enter__()
        if sequence_policy in ("pad-152-156", "pad-156"):
            from gr00t.Replacement.fixed_sequence_padding import FixedSequencePadding

            fixed_padding = FixedSequencePadding(
                trainer, sequence_policy, data_collator.processor.tokenizer.pad_token_id,
                output_dir, global_rank,
            ).install()
        if config.training.enable_profiling:
            from functools import partial

            from gr00t.experiment.profiling import make_profiler_schedule

            logging.info(f"{global_rank} Starting training with profiling...")

            def on_trace_ready_handler(trainer, profile_dir, prof):
                if global_rank == 0:
                    table = prof.key_averages().table(sort_by="self_cuda_time_total")
                    print(table)
                    (profile_dir / "rank0-key-averages.txt").write_text(table + "\n")
                output_path = (
                    profile_dir / f"trace_rank_{global_rank}_iter_{trainer.state.global_step}.json"
                )
                prof.export_chrome_trace(str(output_path))
                logging.info(f"Trace saved to {output_path}")

            profile_dir = output_dir / "profiling"
            run_on_rank0(profile_dir.mkdir, parents=True, exist_ok=True, label="profile_dir.mkdir")

            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=make_profiler_schedule(config.training),
                # profile_memory=True,
                with_stack=config.training.profiler_with_stack,
                # record_shapes=True,
                on_trace_ready=partial(on_trace_ready_handler, trainer, profile_dir),
            ) as prof:
                trainer.add_callback(ProfCallback(prof=prof))
                trainer.train(resume_from_checkpoint=config.training.resume_from_checkpoint)
        else:
            trainer.train(resume_from_checkpoint=config.training.resume_from_checkpoint)
    finally:
        if fixed_padding is not None:
            fixed_padding.close()
        if sequence_adapter is not None:
            sequence_adapter.close()
        if sequence_trainer_diagnostics is not None:
            sequence_trainer_diagnostics.close()
        if zero2_callback is not None:
            zero2_callback.close()

    # Save final model
    if config.training.save_final_model:
        trainer.save_model()
        logging.info(f"Model saved to {output_dir}")

    if config.training.assert_loss_less_than is not None:
        final_loss = trainer.loss
        if final_loss.item() > config.training.assert_loss_less_than:
            raise AssertionError(
                f"Loss too high: {final_loss.item()} vs {config.training.assert_loss_less_than})"
            )

    # # Cleanup
    if hasattr(train_dataset, "close"):
        train_dataset.close()
    if eval_dataset is not None and hasattr(eval_dataset, "close"):
        eval_dataset.close()
    logging.info("Training completed!")
