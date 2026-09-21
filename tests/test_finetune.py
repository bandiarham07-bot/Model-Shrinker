import torch
from torch.utils.data import DataLoader, TensorDataset

from examples.test_models import SmallMLP
from shrinker import compress, finetune
from shrinker.benchmark import evaluate


def test_finetune_learns_after_pruning():
    x = torch.randn(256, 784)
    y = x[:, :10].argmax(1)
    loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)

    small = compress(SmallMLP(), "l1_pruning", ratio=0.5).eval()
    before = evaluate(small, loader)
    out = finetune(small, loader, epochs=5, lr=3e-3, verbose=False)
    assert out is small
    assert not small.training
    assert evaluate(small, loader) > before + 0.2
