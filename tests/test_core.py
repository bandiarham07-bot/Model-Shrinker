import pytest
import torch
from torch import nn

from examples.test_models import SmallMLP, SmallVGG
from shrinker import CompressionError, compress, list_methods, register
from shrinker.registry import UnknownMethodError
from shrinker.utils import replace_module, slice_conv_out, slice_linear_out

X_VGG = torch.randn(2, 3, 32, 32)


def test_unknown_method_lists_available():
    with pytest.raises(UnknownMethodError, match="l1_pruning"):
        compress(SmallMLP(), "no_such_method")


def test_rejects_non_module():
    with pytest.raises(TypeError):
        compress("model.pth", "l1_pruning")


def test_original_untouched_even_if_method_mutates(restore_registry):
    @register("mutator")
    def apply(model):
        for p in model.parameters():
            p.data.fill_(7.0)
        return model

    model = SmallMLP()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    out = compress(model, "mutator")
    assert out is not model
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_method_must_return_model(restore_registry):
    @register("forgot_return")
    def apply(model):
        pass

    with pytest.raises(CompressionError, match="forget `return model`"):
        compress(SmallMLP(), "forgot_return")


def test_broken_cascade_is_caught(restore_registry):
    @register("half_pruning")
    def apply(model):
        replace_module(model, "features.0", slice_conv_out(model.features[0], [0, 1, 2]))
        return model

    with pytest.raises(CompressionError, match="crashes on example_input"):
        compress(SmallVGG(), "half_pruning", example_input=X_VGG)


def test_changed_output_shape_is_caught(restore_registry):
    @register("drop_classes")
    def apply(model):
        replace_module(model, "classifier.4", slice_linear_out(model.classifier[4], [0, 1]))
        return model

    with pytest.raises(CompressionError, match="changed the output shape"):
        compress(SmallVGG(), "drop_classes", example_input=X_VGG)


def test_bad_kwargs_give_clear_error():
    with pytest.raises(TypeError, match="It accepts: ratio"):
        compress(SmallMLP(), "l1_pruning", sparsity=0.3)


def test_missing_required_loader(restore_registry):
    @register("needs_data")
    def apply(model, loader, ratio=0.1):
        return model

    with pytest.raises(TypeError, match="loader"):
        compress(SmallMLP(), "needs_data")


def test_example_input_forwarded_only_if_accepted(restore_registry):
    got = {}

    @register("wants_input")
    def apply(model, example_input=None):
        got["x"] = example_input
        return model

    x = torch.randn(2, 1, 28, 28)
    compress(SmallMLP(), "wants_input", example_input=x)
    assert got["x"] is x
    compress(SmallMLP(), "l1_pruning", example_input=x)


def test_training_mode_preserved():
    for mode in (True, False):
        model = SmallVGG().train(mode)
        assert compress(model, "l1_pruning").training is mode
        assert model.training is mode


def test_tuple_example_input(restore_registry):
    class TwoInputs(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)

        def forward(self, a, b):
            return self.fc(a + b)

    @register("noop_test")
    def apply(model):
        return model

    x = (torch.randn(2, 4), torch.randn(2, 4))
    assert isinstance(compress(TwoInputs(), "noop_test", example_input=x), TwoInputs)


def test_public_api():
    import shrinker

    assert list_methods() == shrinker.list_methods()
    for name in ("compress", "compare", "finetune", "list_methods"):
        assert callable(getattr(shrinker, name))
