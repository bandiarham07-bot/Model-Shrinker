import pytest
import torch
from torch import nn

from examples.test_models import SmallMLP, SmallVGG
from shrinker import compress


def test_ratio_sets_channel_counts():
    small = compress(SmallVGG(), "l1_pruning", ratio=0.5)
    convs = [m for m in small.features if isinstance(m, nn.Conv2d)]
    assert [c.out_channels for c in convs] == [16, 16, 32, 32, 64]
    assert small.classifier[1].in_features == 64 * 4 * 4
    assert small.classifier[1].out_features == 128
    assert small.classifier[4].out_features == 10


def test_keeps_filters_with_largest_l1_norm():
    model = SmallMLP(in_features=4, hidden=8)
    with torch.no_grad():
        model.net[1].weight.copy_(torch.arange(8.0)[:, None].repeat(1, 4))
    small = compress(model, "l1_pruning", ratio=0.5)
    assert torch.equal(small.net[1].weight[:, 0], torch.tensor([4.0, 5.0, 6.0, 7.0]))


def test_ratio_zero_changes_nothing():
    model = SmallVGG().eval()
    x = torch.randn(2, 3, 32, 32)
    small = compress(model, "l1_pruning", ratio=0.0).eval()
    with torch.no_grad():
        assert torch.allclose(small(x), model(x))


@pytest.mark.parametrize("ratio", [-0.1, 1.0, 1.5])
def test_invalid_ratio(ratio):
    with pytest.raises(ValueError, match="ratio"):
        compress(SmallMLP(), "l1_pruning", ratio=ratio)
