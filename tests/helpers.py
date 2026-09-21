import torch
from torch.utils.data import DataLoader, TensorDataset


def make_loader(input_shape, n=32, num_classes=10, batch_size=8):
    x = torch.randn(n, *input_shape)
    y = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)
