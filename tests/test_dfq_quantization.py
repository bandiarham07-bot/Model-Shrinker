import torch
from torch import nn

from shrinker import compress
from shrinker.methods.dfq_quantization import _relu_stats


class _BnChain(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Conv2d(3, 4, 3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.relu = nn.ReLU()
        self.second = nn.Conv2d(4, 2, 1)

    def forward(self, x):
        return self.second(self.relu(self.bn(self.first(x))))


def test_dfq_replaces_safe_consumer_with_quantized_kernel():
    small = compress(_BnChain().eval(), "dfq_quantization", example_input=torch.randn(2, 3, 8, 8))
    assert "quantized" in type(small.second.kernel).__module__
    assert small(torch.randn(2, 3, 8, 8)).shape == (2, 2, 8, 8)


def test_dfq_without_example_input_leaves_model_float():
    small = compress(_BnChain(), "dfq_quantization")
    assert type(small.second) is nn.Conv2d


def test_relu_statistics_match_zero_mean_unit_normal():
    bn = nn.BatchNorm1d(2)
    with torch.no_grad():
        bn.weight.fill_(1)
        bn.bias.zero_()
    stats = _relu_stats(bn)
    assert torch.allclose(stats.mean, torch.full((2,), 1.0 / (2.0 * torch.pi) ** 0.5))
    assert stats.scale > 0
