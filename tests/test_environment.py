"""Smoke tests for the reproducible local environment."""

import sys

import torch


def test_python_minor_version_is_pinned() -> None:
    """The project must use the Python minor version selected for compatibility."""
    assert sys.version_info[:2] == (3, 12)


def test_torch_cpu_tensor_arithmetic() -> None:
    """The installed PyTorch wheel must execute a deterministic tensor operation."""
    result = torch.tensor([1.0, 2.0]) + torch.tensor([3.0, 4.0])
    assert torch.equal(result, torch.tensor([4.0, 6.0]))
