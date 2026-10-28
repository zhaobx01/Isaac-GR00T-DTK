# SPDX-License-Identifier: Apache-2.0
"""Record the actual engine settings for explicitly selected communication experiments."""
import json
import os
from pathlib import Path

from transformers import TrainerCallback


class ExperimentManifestCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        import torch.distributed as dist

        case = json.loads(os.environ['GR00T_EXPERIMENT_CASE'])
        engine = self.trainer.model_wrapped
        zero = engine.optimizer
        actual = {
            'rank': dist.get_rank(), 'case': case,
            'reduce_bucket_size': int(zero.reduce_bucket_size),
            'allgather_bucket_size': int(zero.allgather_bucket_size),
            'timer_sync': bool(engine.tput_timer.config.synchronized),
            'queues': os.environ.get('GPU_MAX_HW_QUEUES'),
            'logging_nan_inf_filter': args.logging_nan_inf_filter,
            'optimizer_groups': len(zero.bit16_groups),
            'partitions': [{'flat_elements': int(flat.numel()),
                            'local_elements': int(master.numel()),
                            'parameter_elements': sum(p.numel() for p in params)}
                           for flat, master, params in zip(zero.bit16_groups_flat,
                               zero.single_partition_of_fp32_groups, zero.bit16_groups)],
        }
        if actual['reduce_bucket_size'] != case['bucket'] or actual['timer_sync'] is not True:
            raise ValueError('Actual DeepSpeed bucket/timer differs from selected candidate')
        if actual['queues'] != (str(case['queues']) if case['queues'] is not None else None):
            raise ValueError('GPU_MAX_HW_QUEUES differs from selected candidate')
        path = Path(args.output_dir) / 'effective'
        path.mkdir(parents=True, exist_ok=True)
        (path / f'rank-{dist.get_rank():02d}.json').write_text(json.dumps(actual, indent=2)+'\n')
        return control
