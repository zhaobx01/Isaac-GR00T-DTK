"""Contracts for the opt-in BF16 pre-cast of ``pixel_values`` in the collator."""

import copy
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader
from transformers import BatchFeature

from gr00t.Replacement import pixel_values_bf16 as pv

ENV = pv.ENV_VAR


class _FakeCollator:
    """Stands in for Gr00tN1d7DataCollator: builds a float32 BatchFeature."""

    def __init__(self, pixel_dtype=torch.float32):
        self.processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=7))
        self.pixel_dtype = pixel_dtype

    def __call__(self, features):
        n = len(features)
        return BatchFeature(
            data={
                "inputs": {
                    "pixel_values": torch.randn(n * 4, 6, dtype=torch.float32).to(self.pixel_dtype),
                    "image_grid_thw": torch.ones(n, 3, dtype=torch.long),
                    "input_ids": torch.arange(n * 5).reshape(n, 5),
                    "attention_mask": torch.ones(n, 5, dtype=torch.long),
                    "state": torch.randn(n, 3, dtype=torch.float32),
                }
            }
        )


class _Seq(torch.utils.data.Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, i):
        return {"i": i}


def _config(bf16=True, fp16=False, workers=4):
    return SimpleNamespace(
        training=SimpleNamespace(bf16=bf16, fp16=fp16, dataloader_num_workers=workers)
    )


DS = {"zero_optimization": {"stage": 2}}


def test_switch_defaults_off_and_only_accepts_zero_or_one():
    assert pv.resolve_enabled({}) is False
    assert pv.resolve_enabled({ENV: "0"}) is False
    assert pv.resolve_enabled({ENV: "1"}) is True
    for bad in ("", "true", "2", "on"):
        with pytest.raises(ValueError, match=ENV):
            pv.resolve_enabled({ENV: bad})


def test_disabled_install_returns_the_same_object():
    collator = _FakeCollator()
    assert pv.install(collator, _config(), DS, {}) is collator
    # Disabled must not even validate the training config.
    assert pv.install(collator, _config(bf16=False), None, {ENV: "0"}) is collator


def test_enabled_casts_only_pixel_values_bitwise_equal_to_to_bfloat16():
    torch.manual_seed(0)
    wrapper = pv.install(_FakeCollator(), _config(), DS, {ENV: "1"})
    assert isinstance(wrapper, pv.Bf16PixelValuesCollator)
    torch.manual_seed(1)
    out = wrapper([{}, {}])
    torch.manual_seed(1)
    ref = _FakeCollator()([{}, {}])
    inputs, ref_inputs = out["inputs"], ref["inputs"]
    assert isinstance(out, BatchFeature)
    assert inputs["pixel_values"].dtype == torch.bfloat16
    assert torch.equal(inputs["pixel_values"], ref_inputs["pixel_values"].to(torch.bfloat16))
    assert inputs["pixel_values"].shape == ref_inputs["pixel_values"].shape
    for key in ("image_grid_thw", "input_ids", "attention_mask", "state"):
        assert inputs[key].dtype == ref_inputs[key].dtype
        assert torch.equal(inputs[key], ref_inputs[key])


def test_already_bfloat16_passes_through_unchanged():
    wrapper = pv.Bf16PixelValuesCollator(_FakeCollator(pixel_dtype=torch.bfloat16))
    assert wrapper([{}])["inputs"]["pixel_values"].dtype == torch.bfloat16


def test_install_does_not_wrap_twice():
    wrapper = pv.install(_FakeCollator(), _config(), DS, {ENV: "1"})
    assert pv.install(wrapper, _config(), DS, {ENV: "1"}) is wrapper


def test_forwards_wrapped_attributes_and_survives_pickle_and_copy():
    wrapper = pv.Bf16PixelValuesCollator(_FakeCollator())
    assert wrapper.processor.tokenizer.pad_token_id == 7
    with pytest.raises(AttributeError):
        wrapper.no_such_attribute
    for clone in (pickle.loads(pickle.dumps(wrapper)), copy.copy(wrapper), copy.deepcopy(wrapper)):
        assert clone([{}])["inputs"]["pixel_values"].dtype == torch.bfloat16


def test_dataloader_workers_return_bfloat16_pixel_values():
    wrapper = pv.Bf16PixelValuesCollator(_FakeCollator())
    loader = DataLoader(_Seq(), batch_size=4, num_workers=2, collate_fn=wrapper)
    batches = list(loader)
    assert len(batches) == 2
    assert all(b["inputs"]["pixel_values"].dtype == torch.bfloat16 for b in batches)


@pytest.mark.parametrize(
    "output",
    [
        {},
        {"inputs": None},
        {"inputs": {"input_ids": torch.zeros(1, 2)}},
        {"inputs": {"pixel_values": [1.0, 2.0]}},
        {"inputs": {"pixel_values": torch.zeros(2, dtype=torch.float16)}},
    ],
)
def test_unexpected_batch_structure_raises(output):
    wrapper = pv.Bf16PixelValuesCollator(lambda features: output)
    with pytest.raises(ValueError, match=ENV):
        wrapper([{}])


def test_non_cpu_pixel_values_raise():
    meta = torch.zeros(2, device="meta")
    wrapper = pv.Bf16PixelValuesCollator(lambda features: {"inputs": {"pixel_values": meta}})
    with pytest.raises(ValueError, match="CPU"):
        wrapper([{}])


@pytest.mark.parametrize(
    "config,ds,message",
    [
        (_config(), None, "DeepSpeed"),
        (_config(bf16=False), DS, "bf16"),
        (_config(fp16=True), DS, "bf16"),
    ],
)
def test_enabled_requires_deepspeed_and_bf16(config, ds, message):
    with pytest.raises(ValueError, match=message):
        pv.install(_FakeCollator(), config, ds, {ENV: "1"})


def test_zero_workers_warns_but_is_allowed(caplog):
    with caplog.at_level("WARNING", logger=pv.logger.name):
        wrapper = pv.install(_FakeCollator(), _config(workers=0), DS, {ENV: "1"})
    assert isinstance(wrapper, pv.Bf16PixelValuesCollator)
    assert "no benefit" in caplog.text


def test_experiment_installs_after_deepspeed_validation_and_before_trainer():
    root = Path(__file__).resolve().parents[3]
    source = (root / "gr00t/experiment/experiment.py").read_text()
    validate = source.index("validate_deepspeed_config(config.training.optim, deepspeed_config)")
    install = source.index("pixel_values_bf16.install(data_collator, config, deepspeed_config)")
    trainer = source.index("data_collator=data_collator,")
    assert validate < install < trainer


class _TrainerStub:
    """Provides what ``Trainer._prepare_input`` reads when DeepSpeed is enabled."""

    def __init__(self, device):
        from transformers import Trainer

        self._prepare_input = Trainer._prepare_input.__get__(self)
        self.args = SimpleNamespace(device=device)
        self.is_deepspeed_enabled = True
        plugin = SimpleNamespace(hf_ds_config=SimpleNamespace(dtype=lambda: torch.bfloat16))
        self.accelerator = SimpleNamespace(state=SimpleNamespace(deepspeed_plugin=plugin))


@pytest.mark.gpu
def test_precast_matches_hf_prepare_input_and_stays_pinned_through_dataloader():
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    device = torch.device("cuda", 0)
    stub = _TrainerStub(device)

    torch.manual_seed(0)
    old_batch = _FakeCollator()([{}, {}, {}])
    torch.manual_seed(0)
    new_batch = pv.Bf16PixelValuesCollator(_FakeCollator())([{}, {}, {}])
    old = stub._prepare_input(old_batch)
    new = stub._prepare_input(new_batch)
    for key in old["inputs"]:
        assert old["inputs"][key].dtype == new["inputs"][key].dtype
        assert torch.equal(old["inputs"][key], new["inputs"][key]), key
    assert new["inputs"]["pixel_values"].device.type == "cuda"
    assert new["inputs"]["pixel_values"].dtype == torch.bfloat16

    loader = DataLoader(
        _Seq(),
        batch_size=4,
        num_workers=2,
        pin_memory=True,
        collate_fn=pv.Bf16PixelValuesCollator(_FakeCollator()),
    )
    for batch in loader:
        pixel_values = batch["inputs"]["pixel_values"]
        assert pixel_values.dtype == torch.bfloat16
        assert pixel_values.is_pinned()
