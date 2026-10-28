"""Optional fixed left padding at the trainer input boundary."""
import json
from collections.abc import MutableMapping
import types
from pathlib import Path

import torch
import torch.nn.functional as F


class FixedSequencePadding:
    def __init__(self, trainer, policy, pad_token_id, output_dir, rank):
        if policy not in ("pad-152-156", "pad-156") or type(pad_token_id) is not int:
            raise ValueError("Fixed padding needs a supported policy and tokenizer pad token")
        self.trainer, self.policy, self.pad_token_id = trainer, policy, pad_token_id
        self.path = Path(output_dir) / "bucket-counts" / f"rank-{rank:02d}.json"
        self.counts = {"152": 0, "156": 0}
        self.original = None

    def install(self):
        self.original = self.trainer._prepare_inputs
        self.had_instance_method = "_prepare_inputs" in vars(self.trainer)
        original = self.original

        def prepare(_trainer, inputs):
            if not isinstance(inputs, MutableMapping) or set(inputs) != {"inputs"} or not isinstance(inputs["inputs"], MutableMapping):
                raise ValueError("Unsupported batch structure for fixed sequence padding")
            values = inputs["inputs"]
            allowed = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw", "state", "action", "action_mask", "state_mask", "embodiment_id"}
            unexpected = set(values) - allowed
            if unexpected:
                raise ValueError(f"Unsupported batch fields for fixed sequence padding: {sorted(unexpected)}")
            ids, mask = values.get("input_ids"), values.get("attention_mask")
            if not isinstance(ids, torch.Tensor) or not isinstance(mask, torch.Tensor) or ids.ndim != 2 or ids.shape != mask.shape or ids.device.type != "cpu" or mask.device.type != "cpu":
                raise ValueError("Expected matching CPU rank-2 input_ids and attention_mask")
            if not torch.all((mask == 0) | (mask == 1)) or torch.any(mask[:, 1:] < mask[:, :-1]):
                raise ValueError("Expected a binary left-padded attention mask")
            longest = int(mask.sum(dim=1).max().item())
            if longest > 156 or ids.shape[1] > 156:
                raise ValueError("Sequence exceeds fixed maximum 156")
            target = 152 if self.policy == "pad-152-156" and longest <= 152 else 156
            if ids.shape[1] > target:
                raise ValueError("Input width exceeds selected bucket; refusing to truncate")
            extra = target - ids.shape[1]
            values["input_ids"] = F.pad(ids, (extra, 0), value=self.pad_token_id)
            values["attention_mask"] = F.pad(mask, (extra, 0), value=0)
            self.counts[str(target)] += 1
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"policy": self.policy, "counts": self.counts}) + "\n")
            return original(inputs)

        self.trainer._prepare_inputs = types.MethodType(prepare, self.trainer)
        return self

    def close(self):
        if self.original is not None:
            if self.had_instance_method:
                self.trainer._prepare_inputs = self.original
            else:
                del self.trainer._prepare_inputs
            self.original = None
