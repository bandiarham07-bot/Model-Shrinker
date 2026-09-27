"""Knowledge distillation

Trains the model to match the soft probability outputs of a frozen copy of itself
(the "teacher"), transferring dark knowledge about inter-class relationships.
After distillation training, a small fraction of the weakest weights are zeroed
to produce a sparser model.

"""

from __future__ import annotations

import copy
import logging
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from shrinker.registry import register

logger = logging.getLogger(__name__)


def _soft_target_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """KL-divergence between teacher and student soft distributions, scaled by T²."""
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    # KL(teacher || student) = sum(teacher * log(teacher / student))
    # F.kl_div expects log-probabilities as input, targets as probabilities
    return F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temperature ** 2)


def _zero_smallest_weights(model: nn.Module, sparsity: float) -> None:
    """Zero out the smallest *sparsity* fraction of weights across the whole model."""
    if sparsity <= 0.0:
        return
    # Collect all weight magnitudes into one flat tensor.
    all_abs = []
    weight_refs = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.dim() >= 2:  # skip biases / BN params
            all_abs.append(param.detach().abs().flatten())
            weight_refs.append(param)
    if not all_abs:
        return
    flat = torch.cat(all_abs)
    n_zero = max(1, int(flat.numel() * sparsity))
    if n_zero >= flat.numel():
        return
    threshold = torch.topk(flat, n_zero, largest=False).values[-1]
    with torch.no_grad():
        for param in weight_refs:
            mask = param.abs() > threshold
            param.mul_(mask)


@register(
    kind="other",
    paper="Hinton et al. 2015, Distilling the Knowledge in a Neural Network",
)
def apply(
    model: nn.Module,
    loader: DataLoader,
    temperature: float = 4.0,
    alpha: float = 0.7,
    epochs: int = 5,
    lr: float = 1e-3,
    sparsity: float = 0.1,
) -> nn.Module:
    """Train the model to mimic its own soft outputs (knowledge distillation), then sparsify."""

    # 1. Snapshot a frozen teacher 
    teacher = copy.deepcopy(model)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # 2. Distillation training 
    device = next(model.parameters()).device
    was_training = model.training
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        seen = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                teacher_logits = teacher(x)

            student_logits = model(x)

            soft_loss = _soft_target_loss(student_logits, teacher_logits, temperature)
            hard_loss = F.cross_entropy(student_logits, y)
            loss = alpha * soft_loss + (1.0 - alpha) * hard_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.numel()
            seen += y.numel()

        logger.info(
            "distillation epoch %d/%d: loss %.4f",
            epoch, epochs, total_loss / max(seen, 1),
        )

    model.train(was_training)

    # 3. Sparsify weakest weights 
    _zero_smallest_weights(model, sparsity)

    return model
