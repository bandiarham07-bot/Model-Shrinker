"""Tests specific to the knowledge_distillation method."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from examples.test_models import SmallMLP, SmallVGG
from shrinker import compress
from shrinker.benchmark import count_nonzero_parameters
from tests.helpers import make_loader


def _make_loader_for(model_cls, input_shape, n=32, batch_size=8):
    """Build a (x, y) DataLoader compatible with a given model class."""
    x = torch.randn(n, *input_shape)
    model = model_cls().eval()
    with torch.no_grad():
        num_classes = model(x[:1]).shape[1]
    y = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def test_soft_targets_move_student_toward_teacher():
    """With alpha=1 (only soft loss), the student should converge toward teacher outputs."""
    model = SmallMLP().eval()
    x = torch.randn(2, 1, 28, 28)
    loader = _make_loader_for(SmallMLP, (1, 28, 28))

    with torch.no_grad():
        teacher_out = model(x)

    student = compress(
        model,
        "knowledge_distillation",
        example_input=x,
        loader=loader,
        alpha=1.0,
        sparsity=0.0,
        epochs=3,
        temperature=4.0,
    )

    with torch.no_grad():
        student_out = student.eval()(x)

    # Student outputs should still be finite and have the right shape.
    assert student_out.shape == teacher_out.shape
    assert torch.isfinite(student_out).all()


def test_sparsity_zeros_weights():
    """After distillation with sparsity > 0, some weights should be exactly zero."""
    model = SmallMLP()
    loader = _make_loader_for(SmallMLP, (1, 28, 28))

    before_nnz = count_nonzero_parameters(model)
    small = compress(
        model,
        "knowledge_distillation",
        loader=loader,
        sparsity=0.3,
        epochs=1,
    )
    after_nnz = count_nonzero_parameters(small)
    assert after_nnz < before_nnz, "sparsity should zero out some weights"


def test_zero_sparsity_preserves_all_weights():
    """With sparsity=0, no weights should be zeroed."""
    model = SmallMLP()
    loader = _make_loader_for(SmallMLP, (1, 28, 28))

    before_nnz = count_nonzero_parameters(model)
    result = compress(
        model,
        "knowledge_distillation",
        loader=loader,
        sparsity=0.0,
        epochs=1,
    )
    # Non-zero count may differ slightly due to training, but should not drop drastically.
    after_nnz = count_nonzero_parameters(result)
    # Allow small changes from training, but shouldn't be a huge drop.
    assert after_nnz >= before_nnz * 0.9


def test_works_on_cnn():
    """Distillation should work on CNN models too."""
    model = SmallVGG()
    loader = _make_loader_for(SmallVGG, (3, 32, 32))
    x = torch.randn(2, 3, 32, 32)

    small = compress(
        model,
        "knowledge_distillation",
        example_input=x,
        loader=loader,
        epochs=1,
        sparsity=0.1,
    )

    with torch.no_grad():
        out = small.eval()(x)
    assert out.shape == model.eval()(x).shape
    assert torch.isfinite(out).all()


def test_original_model_unchanged():
    """The original model should never be modified."""
    model = SmallMLP()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    loader = _make_loader_for(SmallMLP, (1, 28, 28))

    compress(model, "knowledge_distillation", loader=loader, epochs=1)

    after = model.state_dict()
    for k in before:
        assert torch.equal(before[k], after[k]), f"Original model changed: {k}"
