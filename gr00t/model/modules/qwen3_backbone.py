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

import logging
import types

from huggingface_hub.errors import GatedRepoError
import torch
from transformers.feature_extraction_utils import BatchFeature


logger = logging.getLogger(__name__)


try:
    from transformers import Qwen3VLForConditionalGeneration

    _QWEN3VL_AVAILABLE = True
except ImportError:
    _QWEN3VL_AVAILABLE = False


_GATED_BACKBONE_HINT = (
    "Cannot download the VLM backbone '{model_name}', which is a gated Hugging "
    "Face repo. Every GR00T checkpoint (including the base nvidia/GR00T-N1.7-3B) "
    "loads this backbone, so both zero-shot inference and finetuning require "
    "access to it. Request access at https://huggingface.co/{model_name} and "
    "authenticate with `hf auth login` (or set the HF_TOKEN environment variable) "
    "before loading a GR00T model."
)


_GATED_MARKERS = ("gated repo", "is restricted", "access to model", "401 client error")


def _is_gated_repo_error(exc: BaseException) -> bool:
    """True if ``exc`` (or any exception it wraps) is a gated/forbidden HF download.

    transformers may raise ``GatedRepoError`` directly or wrap it (as
    ``__cause__``/``__context__``) inside an ``OSError`` whose own message omits
    the gated markers, so we walk the chain.
    """
    cur: BaseException | None = exc
    for _ in range(10):  # bounded walk guards against cyclic exception chains
        if cur is None:
            break
        if isinstance(cur, GatedRepoError) or any(m in str(cur).lower() for m in _GATED_MARKERS):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def _real_inference_device(module: torch.nn.Module) -> torch.device:
    """Best-effort real (non-meta) device backing a module, defaulting to CPU.

    RoPE buffers must be rebuilt on the device holding the loaded weights, never
    on the ``meta`` device that may be the active default during nested load.
    """
    for tensor in module.parameters():
        if tensor.device.type != "meta":
            return tensor.device
    for tensor in module.buffers():
        if tensor.device.type != "meta":
            return tensor.device
    return torch.device("cpu")


def recompute_vision_rotary_inv_freq(
    rotary: torch.nn.Module, head_dim_half: int, device: torch.device
) -> torch.Tensor:
    """Re-derive Qwen3-VL's vision RoPE ``inv_freq`` via the module's own class.

    Reusing ``type(rotary)(...)`` keeps the analytic formula owned by Transformers;
    the real-device context makes it correct even under a ``meta`` default device.
    """
    with torch.device(device):
        fresh = type(rotary)(head_dim_half)
    return fresh.inv_freq.detach().to(device=device, dtype=torch.float32)


def recompute_text_rotary_inv_freq(
    rotary: torch.nn.Module, config, device: torch.device
) -> tuple[torch.Tensor, float]:
    """Re-derive Qwen3-VL's text RoPE ``inv_freq`` via the module's own class.

    Delegates to the module's constructor so the configured ``rope_init_fn``
    (default / scaled / dynamic) stays the single source of truth; returns the
    FP32 ``inv_freq`` and its ``attention_scaling``.
    """
    with torch.device(device):
        fresh = type(rotary)(config=config, device=device)
    inv_freq = fresh.inv_freq.detach().to(device=device, dtype=torch.float32)
    attention_scaling = float(getattr(fresh, "attention_scaling", 1.0))
    return inv_freq, attention_scaling


def _assign_inv_freq(
    rotary: torch.nn.Module, name: str, value: torch.Tensor, *, persistent: bool
) -> bool:
    """Write ``value`` onto ``rotary.<name>`` if it differs; return whether it changed.

    Returns ``True`` when the buffer/attribute was (re)written, ``False`` when the
    existing value already matched (idempotent no-op). This boolean lets tests
    assert that a corrupted buffer is actually repaired rather than silently
    skipped.
    """
    current = getattr(rotary, name, None)
    if (
        isinstance(current, torch.Tensor)
        and current.device.type != "meta"
        and current.device == value.device
        and current.shape == value.shape
        and current.dtype == value.dtype
        and torch.equal(current, value)
    ):
        return False
    if name in rotary._buffers:
        rotary.register_buffer(name, value, persistent=persistent)
    else:
        setattr(rotary, name, value)
    return True


def _grid_thw_as_int_rows(grid_thw: torch.Tensor) -> list[tuple[int, int, int]]:
    """Materialize small vision-grid metadata on CPU with at most one DtoH copy."""
    grid_cpu = grid_thw.detach().to(device="cpu", dtype=torch.long)
    return [tuple(row) for row in grid_cpu.tolist()]


def _cpu_fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
    """Build interpolation metadata on CPU, then do the trainable lookup on-device.

    The Transformers implementation already builds ``idx_list`` and ``weight_list``
    on CPU. When ``grid_thw`` is on an accelerator, however, using its scalar values
    as ``linspace``/``repeat``/``view`` arguments performs many synchronous DtoH
    copies. Converting the tiny grid once keeps the same arithmetic and autograd path.
    """
    grid_rows = _grid_thw_as_int_rows(grid_thw)
    weight = self.pos_embed.weight
    cache_key = (tuple(grid_rows), weight.device, weight.dtype, self.num_grid_per_side)
    cache = getattr(self, "_gr00t_fast_pos_cache", None)
    if cache is not None and cache[0] == cache_key and not torch.jit.is_tracing():
        idx_tensor, weight_tensor = cache[1:]
    else:
        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]
        for _t, h, w in grid_rows:
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h, device="cpu")
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w, device="cpu")

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor
            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]
            for index in range(4):
                idx_list[index].extend(indices[index].tolist())
                weight_list[index].extend(weights[index].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=weight.device)
        weight_tensor = torch.tensor(weight_list, dtype=weight.dtype, device=weight.device)
        if not torch.jit.is_tracing():
            self._gr00t_fast_pos_cache = (cache_key, idx_tensor, weight_tensor)

    pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
    patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

    patch_pos_embeds = patch_pos_embeds.split([h * w for _t, h, w in grid_rows])
    merge_size = self.config.spatial_merge_size
    permuted = []
    for pos_embed, (t, h, w) in zip(patch_pos_embeds, grid_rows):
        pos_embed = pos_embed.repeat(t, 1)
        pos_embed = (
            pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
            .permute(0, 1, 3, 2, 4, 5)
            .flatten(0, 4)
        )
        permuted.append(pos_embed)
    return torch.cat(permuted)


def _cpu_rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
    """Build integer rotary coordinates on CPU and transfer them in one batch."""
    grid_rows = _grid_thw_as_int_rows(grid_thw)
    max_hw = max(max(height, width) for _t, height, width in grid_rows)
    freq_table = self.rotary_pos_emb(max_hw)
    merge_size = self.spatial_merge_size
    cache_key = (tuple(grid_rows), freq_table.device, merge_size)
    cache = getattr(self, "_gr00t_rot_pos_cache", None)
    if cache is not None and cache[0] == cache_key and not torch.jit.is_tracing():
        pos_ids_device = cache[1]
    else:
        pos_ids = []
        for num_frames, height, width in grid_rows:
            merged_h, merged_w = height // merge_size, width // merge_size
            block_rows = torch.arange(merged_h, device="cpu")
            block_cols = torch.arange(merged_w, device="cpu")
            intra_row = torch.arange(merge_size, device="cpu")
            intra_col = torch.arange(merge_size, device="cpu")
            row_idx = (
                block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            )
            col_idx = (
                block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]
            )
            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            coords = torch.stack((row_idx, col_idx), dim=-1)
            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)
            pos_ids.append(coords)

        pos_ids_device = torch.cat(pos_ids).to(freq_table.device)
        if not torch.jit.is_tracing():
            self._gr00t_rot_pos_cache = (cache_key, pos_ids_device)
    return freq_table[pos_ids_device].flatten(1)


def _cpu_get_rope_index(
    self,
    input_ids=None,
    image_grid_thw=None,
    video_grid_thw=None,
    attention_mask=None,
):
    """Compute multimodal mRoPE integer positions on CPU without scalar syncs."""
    if input_ids is None or (image_grid_thw is None and video_grid_thw is None):
        return self._gr00t_original_get_rope_index(
            input_ids, image_grid_thw, video_grid_thw, attention_mask=attention_mask
        )

    output_device = input_ids.device
    cpu_inputs = (
        input_ids.detach().to("cpu"),
        None if image_grid_thw is None else image_grid_thw.detach().to("cpu"),
        None if video_grid_thw is None else video_grid_thw.detach().to("cpu"),
    )
    attention_mask_cpu = (
        None if attention_mask is None else attention_mask.detach().to("cpu")
    )
    position_ids, rope_deltas = self._gr00t_original_get_rope_index(
        *cpu_inputs, attention_mask=attention_mask_cpu
    )
    return position_ids.to(output_device), rope_deltas.to(output_device)


def _compile_qwen3_vision_block_classes(visual: torch.nn.Module) -> int:
    """Apply the local Qwen3-VL VisionBlock ``torch.compile`` override."""
    compiled = 0
    block_classes = {type(block) for block in getattr(visual, "blocks", ())}
    for block_class in block_classes:
        block_call = getattr(block_class, "__call__", None)
        if block_call is None or hasattr(block_call, "_torchdynamo_orig_callable"):
            continue
        # This is equivalent to decorating Qwen3VLVisionBlock itself with
        # @torch.compile(...). torch.compile installs an optimized __call__ on the
        # class, so it also applies to instances that were already constructed.
        torch.compile(block_class, mode="max-autotune-no-cudagraphs")
        compiled += 1
    return compiled


def _disable_transformers_flash_attention_for_dynamo(
    flash_attention_module=None,
    attention_functions=None,
) -> bool:
    """Move the local ``@torch._dynamo.disable`` FlashAttention override in-repo."""
    if flash_attention_module is None or attention_functions is None:
        from transformers.integrations import flash_attention as flash_attention_module
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS as attention_functions

    original = flash_attention_module.flash_attention_forward
    already_disabled = hasattr(original, "_torchdynamo_disable")
    disabled = original if already_disabled else torch._dynamo.disable(original)
    flash_attention_module.flash_attention_forward = disabled

    # AttentionInterface stores function objects in a registry. Updating only the
    # integration module would leave Qwen3-VL calling the old reference.
    for implementation in ("flash_attention_2", "flash_attention_3"):
        try:
            registered = attention_functions[implementation]
        except KeyError:
            continue
        if registered is original:
            attention_functions.register(implementation, disabled)
    return not already_disabled


class Qwen3Backbone(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "nvidia/Cosmos-Reason2-2B",
        tune_llm: bool = False,
        tune_visual: bool = False,
        select_layer: int = -1,
        reproject_vision: bool = True,
        use_flash_attention: bool = False,
        projector_dim: int = -1,
        load_bf16: bool = False,
        tune_top_llm_layers: int = 0,
        trainable_params_fp32: bool = False,
        transformers_loading_kwargs: dict = {},
    ):
        """
        Qwen3Backbone is to generate n_queries to represent the future action hidden states.
        Args:
            model_name: nvidia/Cosmos-Reason2-2B
            tune_llm: whether to tune the LLM model (default: False)
            tune_visual: whether to tune the visual model (default: False)
        """
        if not _QWEN3VL_AVAILABLE:
            raise ImportError(
                "Qwen3VLForConditionalGeneration is not available. "
                "Please upgrade transformers to a version that supports Qwen3-VL: "
                "pip install transformers>=4.57.0"
            )

        super().__init__()

        # Add attention kwargs
        extra_kwargs = {}
        if use_flash_attention:
            try:
                import flash_attn  # noqa: F401

                extra_kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                logger.warning(
                    "flash_attn is not installed. Falling back to sdpa attention. "
                    "Install flash-attn for better performance: pip install flash-attn"
                )
                extra_kwargs["attn_implementation"] = "sdpa"
        if load_bf16:
            extra_kwargs["torch_dtype"] = torch.bfloat16

        try:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                **extra_kwargs,
                **transformers_loading_kwargs,
            ).eval()
        except Exception as exc:
            if _is_gated_repo_error(exc):
                raise RuntimeError(_GATED_BACKBONE_HINT.format(model_name=model_name)) from exc
            raise

        # needed since we don't use these layers. Also saves compute
        while len(self.model.language_model.layers) > select_layer:
            self.model.language_model.layers.pop(-1)

        self.select_layer = select_layer
        self.set_trainable_parameters(tune_llm, tune_visual, tune_top_llm_layers)
        if load_bf16 and trainable_params_fp32:
            # cast trainable parameters to fp32
            for n, p in self.named_parameters():
                if p.requires_grad:
                    p.data = p.data.to(torch.float32)
                    logger.debug(f"Casting trainable parameter {n} to fp32")

        # Repair Qwen3-VL's non-persistent RoPE buffers once, after weights are
        # loaded and trainable dtypes are finalized. See _reset_rotary_inv_freq.
        self._reset_rotary_inv_freq()

        # Qwen3-VL's position helpers mix accelerator scalars with Python control
        # flow. Materialize the small integer metadata once on CPU instead of doing
        # thousands of synchronous scalar DtoH copies per forward.
        self._apply_cpu_position_preprocessing()

        # Keep local performance/compiler workarounds independent of site-packages.
        self._apply_transformers_compilation_overrides()

        # Restore the fast vision patch-embed kernel on torch>=2.9. See method docstring.
        self._apply_vision_patch_embed_channels_last()

    def _apply_transformers_compilation_overrides(self) -> None:
        """Install repository-owned Qwen3-VL compile/Dynamo overrides."""
        compiled_blocks = _compile_qwen3_vision_block_classes(self.model.visual)
        flash_disabled = _disable_transformers_flash_attention_for_dynamo()
        logger.debug(
            "Applied Qwen3-VL compilation overrides (compiled_block_classes=%d, "
            "flash_attention_newly_disabled=%s).",
            compiled_blocks,
            flash_disabled,
        )

    def _apply_cpu_position_preprocessing(self) -> None:
        """Install sync-free CPU metadata builders on this Transformers instance."""
        visual = getattr(self.model, "visual", None)
        rope_model = self.model
        if not hasattr(rope_model, "get_rope_index"):
            rope_model = getattr(rope_model, "model", None)
        required_visual_methods = ("fast_pos_embed_interpolate", "rot_pos_emb")
        if visual is None or not all(hasattr(visual, name) for name in required_visual_methods):
            raise RuntimeError(
                "Qwen3-VL vision position helpers were not found. The Transformers layout "
                "may have changed; refusing to install the GR00T position preprocessing patch."
            )
        if rope_model is None or not hasattr(rope_model, "get_rope_index"):
            raise RuntimeError(
                "Qwen3-VL get_rope_index was not found. The Transformers layout may have "
                "changed; refusing to install the GR00T position preprocessing patch."
            )

        visual.fast_pos_embed_interpolate = types.MethodType(
            _cpu_fast_pos_embed_interpolate, visual
        )
        visual.rot_pos_emb = types.MethodType(_cpu_rot_pos_emb, visual)
        # Preserve the upstream path for text-only calls, which do not contain the
        # multimodal scalar-sync pattern targeted by this patch.
        if not hasattr(rope_model, "_gr00t_original_get_rope_index"):
            rope_model._gr00t_original_get_rope_index = rope_model.get_rope_index
        rope_model.get_rope_index = types.MethodType(_cpu_get_rope_index, rope_model)

    def _apply_vision_patch_embed_channels_last(self) -> None:
        """Force the Qwen3-VL vision patch-embed ``Conv3d`` to ``channels_last_3d``.

        On torch>=2.9 (cuDNN 9.10.x) the contiguous bf16 patch-embed ``Conv3d``
        dispatches to a pathologically slow kernel on sm_80; ``channels_last_3d``
        selects the fast one and is numerically equivalent (harmless elsewhere).

        Enabled by default; set ``GR00T_VISION_PATCH_EMBED_CHANNELS_LAST=0`` and restart
        to keep the default layout. See ``gr00t.Replacement.vision_patch_embed_layout``.
        """
        visual = getattr(self.model, "visual", None)
        if visual is None:
            return

        from gr00t.Replacement import vision_patch_embed_layout

        vision_patch_embed_layout.apply(visual)

    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool, tune_top_llm_layers: int):
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        for p in self.parameters():
            p.requires_grad = True
        if not tune_llm:
            self.model.language_model.requires_grad_(False)
        if not tune_visual:
            self.model.visual.requires_grad_(False)

        if tune_top_llm_layers > 0:
            for layer in self.model.language_model.layers[-tune_top_llm_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        logger.debug(f"Tune backbone llm: {self.tune_llm}")
        logger.debug(f"Tune backbone visual: {self.tune_visual}")
        # Check if any parameters are still trainable. If not, log a warning.
        for name, p in self.named_parameters():
            if p.requires_grad:
                logger.debug(f"Backbone trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            logger.warning("No backbone trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if self.model.language_model and not self.tune_llm:
                self.model.language_model.eval()
            if self.model.visual and not self.tune_visual:
                self.model.visual.eval()

    def _reset_rotary_inv_freq(self) -> None:
        """Re-derive Qwen3-VL's non-persistent RoPE ``inv_freq`` buffers once at load.

        These ``persistent=False`` buffers are not restored from a checkpoint and
        can be left uninitialized under the nested ``no_init_weights`` load, so we
        rebuild the analytic FP32 frequencies here (load time only, no per-forward
        cost). Fails closed (raises) when the layout/config needed to rebuild is
        missing, so a drifted transformers version cannot silently serve
        uninitialized buffers.
        """
        config = getattr(self.model, "config", None)
        vision_changed = self._reset_vision_rotary_inv_freq(config)
        text_changed = self._reset_language_rotary_inv_freq()
        logger.debug(
            "Qwen3-VL RoPE inv_freq reset (vision_rewritten=%s, text_rewritten=%s).",
            vision_changed,
            text_changed,
        )

    def _reset_vision_rotary_inv_freq(self, config) -> bool:
        visual = getattr(self.model, "visual", None)
        rotary = getattr(visual, "rotary_pos_emb", None)
        if rotary is None or not hasattr(rotary, "inv_freq"):
            raise RuntimeError(
                "Qwen3-VL visual rotary_pos_emb/inv_freq not found; cannot rebuild the "
                "non-persistent vision RoPE inv_freq. Continuing would leave it "
                "uninitialized and silently corrupt inference, so we fail the load. The "
                "transformers Qwen3-VL layout may have changed; pin a compatible version."
            )

        vision_config = getattr(config, "vision_config", None)
        if vision_config is None or not all(
            hasattr(vision_config, attr) for attr in ("hidden_size", "num_heads")
        ):
            raise RuntimeError(
                "Qwen3-VL vision_config missing hidden_size/num_heads; cannot rebuild the "
                "non-persistent vision RoPE inv_freq (continuing would leave it "
                "uninitialized and silently corrupt inference)."
            )

        head_dim = vision_config.hidden_size // vision_config.num_heads
        device = _real_inference_device(visual)
        inv_freq = recompute_vision_rotary_inv_freq(rotary, head_dim // 2, device)
        return _assign_inv_freq(rotary, "inv_freq", inv_freq, persistent=False)

    def _reset_language_rotary_inv_freq(self) -> bool:
        language_model = getattr(self.model, "language_model", None)
        rotary = getattr(language_model, "rotary_emb", None)
        text_config = getattr(rotary, "config", None) or getattr(language_model, "config", None)
        if rotary is None or not hasattr(rotary, "inv_freq") or text_config is None:
            raise RuntimeError(
                "Qwen3-VL language rotary_emb/inv_freq/config not found; cannot rebuild the "
                "non-persistent text RoPE inv_freq. Continuing would leave it uninitialized "
                "and silently corrupt inference, so we fail the load. The transformers "
                "Qwen3-VL layout may have changed; pin a compatible version."
            )

        device = _real_inference_device(language_model)
        inv_freq, _attention_scaling = recompute_text_rotary_inv_freq(rotary, text_config, device)
        changed = _assign_inv_freq(rotary, "inv_freq", inv_freq, persistent=False)
        # ``original_inv_freq`` is a plain attribute (not a buffer) that dynamic
        # RoPE updates restore from; keep it consistent with inv_freq.
        if hasattr(rotary, "original_inv_freq"):
            changed = (
                _assign_inv_freq(rotary, "original_inv_freq", inv_freq.clone(), persistent=False)
                or changed
            )
        return changed

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()
        # 0. Set frozen module to eval
        keys_to_use = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]
        vl_input = {k: vl_input[k] for k in keys_to_use}
        outputs = self.model(**vl_input, output_hidden_states=True)
        outputs = outputs.hidden_states[-1]
        image_mask = vl_input["input_ids"] == self.model.config.image_token_id
        attention_mask = vl_input["attention_mask"] == 1
        return BatchFeature(
            data={
                "backbone_features": outputs,
                "backbone_attention_mask": attention_mask,
                "image_mask": image_mask,
            }
        )  # [B, T2, hidden_size]
