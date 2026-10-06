"""Layer-by-layer validation-accuracy sensitivity analysis for channel pruning."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple

import torch
from torch import nn

from .utils import keep_indices, prunable_layers, prune_channels


logger = logging.getLogger(__name__)

ScoreFn = Callable[[nn.Module], torch.Tensor]


@dataclass(frozen=True)
class SensitivityPoint:
    """Accuracy measured after pruning one layer at one ratio."""

    layer: str
    ratio: float
    accuracy: float
    accuracy_drop: float


@dataclass(frozen=True)
class SensitivityResult:
    """Baseline and layer-wise pruning measurements.

    ``layers`` is indexed as ``layers[layer_name][ratio]``.  ``rows`` is a
    flat, table-friendly view of the same measurements.
    """

    baseline_accuracy: Optional[float]
    layers: Dict[str, Dict[float, SensitivityPoint]]
    rows: Tuple[SensitivityPoint, ...]


def _l1_scores(layer: nn.Module) -> torch.Tensor:
    return layer.weight.detach().abs().flatten(1).sum(dim=1)  # type: ignore[attr-defined]


def _model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        try:
            return next(model.buffers()).device
        except StopIteration:
            return torch.device("cpu")


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def _forward(model: nn.Module, inputs: Any) -> Any:
    if isinstance(inputs, tuple):
        return model(*inputs)
    if isinstance(inputs, dict):
        return model(**inputs)
    return model(inputs)


def _accuracy(model: nn.Module, loader: Iterable[Any], max_batches: Optional[int]) -> Optional[float]:
    """Return top-1 accuracy, or ``None`` when the data is not supported."""
    if max_batches is not None and max_batches <= 0:
        logger.info("Skipping sensitivity evaluation: max_batches must be positive")
        return None

    was_training = model.training
    device = _model_device(model)
    correct = total = 0
    try:
        model.eval()
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                    logger.info("Skipping sensitivity evaluation: loader must yield (inputs, labels)")
                    return None
                inputs, labels = _to_device(batch[0], device), _to_device(batch[1], device)
                if not isinstance(labels, torch.Tensor):
                    logger.info("Skipping sensitivity evaluation: labels must be tensors")
                    return None
                outputs = _forward(model, inputs)
                if not isinstance(outputs, torch.Tensor) or outputs.ndim != 2:
                    logger.info("Skipping sensitivity evaluation: model must return 2-D classification logits")
                    return None
                if outputs.shape[0] != labels.numel():
                    logger.info("Skipping sensitivity evaluation: logits and labels have incompatible batch sizes")
                    return None
                correct += (outputs.argmax(dim=1) == labels.reshape(-1)).sum().item()
                total += labels.numel()
    except Exception as exc:
        logger.info("Skipping sensitivity evaluation: %s", exc)
        return None
    finally:
        model.train(was_training)

    if total == 0:
        logger.info("Skipping sensitivity evaluation: loader yielded no examples")
        return None
    return correct / total


def analyze_sensitivity(
    model: nn.Module,
    loader: Iterable[Any],
    ratios: Sequence[float] = tuple(i / 10 for i in range(1, 10)),
    score_fn: Optional[ScoreFn] = None,
    max_batches: Optional[int] = None,
) -> SensitivityResult:
    """Measure validation-accuracy sensitivity to pruning each layer alone.

    ``loader`` must be re-iterable and yield ``(inputs, labels)`` classification
    batches.  A custom ``score_fn(layer)`` must return one score per output
    channel, where larger scores are retained.  Unsupported trials are logged
    and omitted from the returned result.
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")

    baseline = _accuracy(model, loader, max_batches)
    if baseline is None:
        return SensitivityResult(None, {}, ())

    scorer = score_fn or _l1_scores
    valid_ratios = []
    for ratio in ratios:
        try:
            ratio = float(ratio)
            keep_indices(torch.ones(1), ratio)
        except (TypeError, ValueError) as exc:
            logger.info("Skipping sensitivity ratio %r: %s", ratio, exc)
            continue
        valid_ratios.append(ratio)

    layer_names = [name for name, _ in prunable_layers(model)]
    layers: Dict[str, Dict[float, SensitivityPoint]] = {}
    rows = []
    for name in layer_names:
        points: Dict[float, SensitivityPoint] = {}
        for ratio in valid_ratios:
            try:
                trial = copy.deepcopy(model)
                layer = dict(prunable_layers(trial)).get(name)
                if layer is None:
                    logger.info("Skipping %s: no longer safely prunable", name)
                    break
                scores = torch.as_tensor(scorer(layer)).detach().flatten()
                out_channels = getattr(layer, "out_channels", getattr(layer, "out_features", None))
                if scores.numel() != out_channels:
                    logger.info("Skipping %s: score_fn returned %d scores for %d output channels", name, scores.numel(), out_channels)
                    break
                keep = keep_indices(scores, ratio)
                prune_channels(trial, name, keep)
            except Exception as exc:
                logger.info("Skipping %s at ratio %s: %s", name, ratio, exc)
                break

            accuracy = _accuracy(trial, loader, max_batches)
            if accuracy is None:
                logger.info("Skipping %s at ratio %s: unable to measure accuracy", name, ratio)
                continue
            point = SensitivityPoint(name, ratio, accuracy, baseline - accuracy)
            points[ratio] = point
            rows.append(point)
        if points:
            layers[name] = points
    return SensitivityResult(baseline, layers, tuple(rows))
