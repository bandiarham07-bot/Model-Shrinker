from __future__ import annotations

import torch
from torch import nn


def finetune(
    model: nn.Module,
    train_loader,
    epochs: int = 5,
    lr: float = 1e-3,
    device: str = "cpu",
    optimizer: str = "adam",
    verbose: bool = True,
) -> nn.Module:
    """Train ``model`` in place with cross-entropy to recover accuracy after compression."""
    if optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"optimizer must be 'adam' or 'sgd', got {optimizer!r}")

    was_training = model.training
    model.to(device).train()
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        total_loss = correct = seen = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * y.numel()
            correct += (out.argmax(1) == y).sum().item()
            seen += y.numel()
        if verbose:
            print(f"finetune epoch {epoch}/{epochs}: loss {total_loss / max(seen, 1):.4f}, "
                  f"train acc {correct / max(seen, 1) * 100:.2f}%")
    model.train(was_training)
    return model
