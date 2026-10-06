import copy
import logging

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from shrinker import analyze_sensitivity


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 2, bias=False)
        self.output = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.hidden.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 1.0]]))
            self.output.weight.copy_(torch.tensor([[1.0, -1.0], [-1.0, 1.0]]))

    def forward(self, x):
        return self.output(self.hidden(x))


def _loader():
    return DataLoader(TensorDataset(torch.eye(2), torch.tensor([0, 1])), batch_size=1)


def test_sensitivity_reports_accuracy_drop_without_mutating_model():
    model = TinyClassifier().train()
    before = copy.deepcopy(model.state_dict())

    result = analyze_sensitivity(model, _loader(), ratios=(0.5,))

    assert model.training
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
    assert result.baseline_accuracy == 1.0
    assert len(result.rows) == 1
    point = result.rows[0]
    assert point.layer == "hidden"
    assert point.ratio == 0.5
    assert point.accuracy == 0.5
    assert point.accuracy_drop == 0.5
    assert result.layers["hidden"][0.5] == point


def test_sensitivity_uses_custom_score_function():
    calls = []

    def score(layer):
        calls.append(layer)
        return torch.tensor([0.0, 1.0])

    result = analyze_sensitivity(TinyClassifier(), _loader(), ratios=(0.5,), score_fn=score)

    assert calls
    assert len(result.rows) == 1


def test_sensitivity_skips_bad_score_and_malformed_batches(caplog):
    caplog.set_level(logging.INFO, logger="shrinker.sensitivity")
    result = analyze_sensitivity(TinyClassifier(), [(torch.eye(2), torch.tensor([0, 1]))], ratios=(0.5,), score_fn=lambda _: torch.ones(1))
    assert result.baseline_accuracy == 1.0
    assert not result.rows
    assert "score_fn returned" in caplog.text

    malformed = analyze_sensitivity(TinyClassifier(), [torch.eye(2)])
    assert malformed.baseline_accuracy is None
    assert not malformed.rows


def test_sensitivity_respects_max_batches():
    loader = DataLoader(TensorDataset(torch.eye(2), torch.tensor([0, 0])), batch_size=1)
    result = analyze_sensitivity(TinyClassifier(), loader, ratios=(0.5,), max_batches=1)
    assert result.baseline_accuracy == 1.0
