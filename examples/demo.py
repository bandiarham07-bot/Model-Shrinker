"""End-to-end demo: train SmallVGG, prune it, compare, finetune, compare again.

    python examples/demo.py                 # CIFAR-10 (downloads ~170 MB to ./data)
    python examples/demo.py --synthetic     # no download, runs in under a minute
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import shrinker  # noqa: E402
from examples.test_models import SmallVGG  # noqa: E402


def cifar10_loaders(data_dir, batch_size):
    import torchvision
    import torchvision.transforms as T

    norm = T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), norm])
    test_tf = T.Compose([T.ToTensor(), norm])
    train = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
    test = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2),
        DataLoader(test, batch_size=256, num_workers=2),
    )


def synthetic_loaders(batch_size, n_train=4000, n_test=1000):
    g = torch.Generator().manual_seed(0)
    templates = torch.randn(10, 3, 32, 32, generator=g)

    def make(n):
        y = torch.randint(0, 10, (n,), generator=g)
        x = 0.6 * templates[y] + torch.randn(n, 3, 32, 32, generator=g)
        return TensorDataset(x, y)

    return (
        DataLoader(make(n_train), batch_size=batch_size, shuffle=True),
        DataLoader(make(n_test), batch_size=256),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--synthetic", action="store_true", help="use synthetic data instead of CIFAR-10")
    p.add_argument("--epochs", type=int, default=5, help="training epochs for the big model")
    p.add_argument("--finetune-epochs", type=int, default=2)
    p.add_argument("--method", default="l1_pruning", choices=shrinker.list_methods())
    p.add_argument("--ratio", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--data", default="./data")
    args = p.parse_args()
    torch.manual_seed(0)

    print("Available methods:\n" + shrinker.describe_methods() + "\n")
    if args.synthetic:
        train_loader, val_loader = synthetic_loaders(args.batch_size)
    else:
        train_loader, val_loader = cifar10_loaders(args.data, args.batch_size)

    print(f"== 1. Train SmallVGG for {args.epochs} epochs")
    t0 = time.time()
    model = SmallVGG()
    shrinker.finetune(model, train_loader, epochs=args.epochs, lr=1e-3)
    print(f"   ({time.time() - t0:.0f}s)\n")

    print(f"== 2. Compress with {args.method}, ratio={args.ratio}")
    example = torch.randn(1, 3, 32, 32)
    small = shrinker.compress(model, method=args.method, ratio=args.ratio, example_input=example)
    shrinker.compare(model, small, example_input=example, val_loader=val_loader)

    print(f"\n== 3. Finetune the compressed model for {args.finetune_epochs} epochs")
    shrinker.finetune(small, train_loader, epochs=args.finetune_epochs, lr=5e-4)

    print("\n== 4. Compare again")
    shrinker.compare(model, small, example_input=example, val_loader=val_loader)

    print("\nBatch of 64 (throughput):")
    shrinker.compare(model, small, example_input=torch.randn(64, 3, 32, 32))


if __name__ == "__main__":
    main()
