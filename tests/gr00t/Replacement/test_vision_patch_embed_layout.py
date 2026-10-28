"""CPU contracts for the switchable vision patch-embed channels_last_3d layout."""

import types

import pytest
import torch

from gr00t.Replacement import vision_patch_embed_layout as layout

ENV = layout.ENV_VAR
CL3D = torch.channels_last_3d


def _visual(seed=0):
    torch.manual_seed(seed)
    # Same geometry as Qwen3-VL patch-embed: kernel == stride == (T, P, P).
    return torch.nn.Sequential(torch.nn.Conv3d(3, 8, kernel_size=(2, 4, 4), stride=(2, 4, 4)))


def _pair():
    base, cand = _visual(), _visual()
    cand.load_state_dict(base.state_dict())
    return base, cand


def test_switch_defaults_on_and_only_accepts_zero_or_one():
    assert layout.resolve_enabled({}) is True
    assert layout.resolve_enabled({ENV: "1"}) is True
    assert layout.resolve_enabled({ENV: "0"}) is False
    for bad in ("", "true", "2", "off"):
        with pytest.raises(ValueError, match=ENV):
            layout.resolve_enabled({ENV: bad})


def test_disabled_keeps_default_layout_and_installs_no_hook():
    visual = _visual()
    assert layout.apply(visual, {ENV: "0"}) == 0
    conv = visual[0]
    assert conv.weight.is_contiguous()
    assert not conv.weight.is_contiguous(memory_format=CL3D)
    assert len(conv._forward_pre_hooks) == 0
    seen = []
    conv.register_forward_pre_hook(lambda _m, inputs: seen.append(inputs[0].is_contiguous()))
    conv(torch.randn(5, 3, 2, 4, 4))
    assert seen == [True]


def test_enabled_converts_weight_once_and_feeds_conv_channels_last_3d():
    visual = _visual()
    conv = visual[0]
    param_id = id(conv.weight)
    assert layout.apply(visual, {}) == 1
    assert id(conv.weight) == param_id  # same Parameter object, optimizer refs stay valid
    assert conv.weight.is_contiguous(memory_format=CL3D)
    seen = []
    # Registered after ours, so it observes what the Conv3d actually receives.
    conv.register_forward_pre_hook(
        lambda _m, inputs: seen.append(inputs[0].is_contiguous(memory_format=CL3D))
    )
    conv(torch.randn(5, 3, 2, 4, 4))
    assert seen == [True]


def test_enabled_matches_baseline_forward_and_gradients():
    base, cand = _pair()
    layout.apply(cand, {})
    x = torch.randn(7, 3, 2, 4, 4)
    out_base, out_cand = base(x), cand(x.clone())
    torch.testing.assert_close(out_cand, out_base)
    out_base.square().sum().backward()
    out_cand.square().sum().backward()
    torch.testing.assert_close(cand[0].weight.grad, base[0].weight.grad)
    torch.testing.assert_close(cand[0].bias.grad, base[0].bias.grad)


def test_non_5d_input_is_left_untouched():
    base, cand = _pair()
    layout.apply(cand, {})
    x = torch.randn(3, 2, 4, 4)  # unbatched 4D input accepted by Conv3d
    torch.testing.assert_close(cand(x), base(x))


def test_hook_is_skipped_while_tracing(monkeypatch):
    visual = _visual()
    layout.apply(visual, {})
    conv = visual[0]
    seen = []
    conv.register_forward_pre_hook(lambda _m, inputs: seen.append(inputs[0].is_contiguous()))
    monkeypatch.setattr(torch.jit, "is_tracing", lambda: True)
    conv(torch.randn(5, 3, 2, 4, 4))
    assert seen == [True]


def test_apply_is_idempotent():
    visual = _visual()
    assert layout.apply(visual, {}) == 1
    assert layout.apply(visual, {}) == 0
    assert len(visual[0]._forward_pre_hooks) == 1


def test_only_conv3d_modules_are_touched():
    visual = torch.nn.Sequential(_visual(), torch.nn.Linear(4, 4), torch.nn.Conv2d(3, 3, 1))
    assert layout.apply(visual, {}) == 1
    assert visual[2].weight.is_contiguous()
    assert len(visual[2]._forward_pre_hooks) == 0


@pytest.mark.parametrize("value, expect_cl3d", [(None, True), ("1", True), ("0", False)])
def test_backbone_method_honours_environment(monkeypatch, value, expect_cl3d):
    backbone_mod = pytest.importorskip("gr00t.model.modules.qwen3_backbone")
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    visual = _visual()
    fake_self = types.SimpleNamespace(model=types.SimpleNamespace(visual=visual))
    backbone_mod.Qwen3Backbone._apply_vision_patch_embed_channels_last(fake_self)
    assert visual[0].weight.is_contiguous(memory_format=CL3D) is expect_cl3d
    assert (len(visual[0]._forward_pre_hooks) == 1) is expect_cl3d


def test_backbone_method_rejects_invalid_value(monkeypatch):
    backbone_mod = pytest.importorskip("gr00t.model.modules.qwen3_backbone")
    monkeypatch.setenv(ENV, "yes")
    fake_self = types.SimpleNamespace(model=types.SimpleNamespace(visual=_visual()))
    with pytest.raises(ValueError, match=ENV):
        backbone_mod.Qwen3Backbone._apply_vision_patch_embed_channels_last(fake_self)
