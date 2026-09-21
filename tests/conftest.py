import pytest
import torch

from shrinker import registry


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture
def restore_registry():
    saved, saved_errors = dict(registry.REGISTRY), dict(registry.LOAD_ERRORS)
    yield
    registry.REGISTRY.clear()
    registry.REGISTRY.update(saved)
    registry.LOAD_ERRORS.clear()
    registry.LOAD_ERRORS.update(saved_errors)

