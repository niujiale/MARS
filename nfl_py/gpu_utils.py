#!/usr/bin/env python3
"""
GPU Utilities for NFL-Py

Provides GPU acceleration support via PyTorch and CuPy.
Falls back to CPU/NumPy if GPU is not available.

Author: MARS Team
"""

import logging
from typing import Union, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Try to import GPU libraries
_TORCH_AVAILABLE = False
_CUPY_AVAILABLE = False
_GPU_DEVICE = None

try:
    import torch
    _TORCH_AVAILABLE = True
    if torch.cuda.is_available():
        _GPU_DEVICE = 'cuda'
        logger.info(f"PyTorch CUDA available: {torch.cuda.get_device_name(0)}")
except ImportError:
    pass

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
    logger.info("CuPy available for GPU acceleration")
except ImportError:
    pass


def is_gpu_available() -> bool:
    """
    Check if GPU acceleration is available.

    Returns:
        True if PyTorch CUDA or CuPy is available
    """
    if _TORCH_AVAILABLE:
        import torch
        return torch.cuda.is_available()
    return _CUPY_AVAILABLE


def get_device() -> str:
    """
    Get the best available device.

    Returns:
        'cuda' if GPU available, 'cpu' otherwise
    """
    if is_gpu_available():
        return 'cuda'
    return 'cpu'


def to_device(
    array: Union[np.ndarray, 'torch.Tensor'],
    device: Optional[str] = None
) -> Union[np.ndarray, 'torch.Tensor']:
    """
    Move array to specified device.

    Args:
        array: NumPy array or PyTorch tensor
        device: Target device ('cuda' or 'cpu'), auto-detect if None

    Returns:
        Array on the specified device
    """
    if device is None:
        device = get_device()

    if _TORCH_AVAILABLE:
        import torch
        if isinstance(array, np.ndarray):
            tensor = torch.from_numpy(array)
            return tensor.to(device)
        elif isinstance(array, torch.Tensor):
            return array.to(device)

    return array


def to_numpy(array: Union[np.ndarray, 'torch.Tensor']) -> np.ndarray:
    """
    Convert array to NumPy.

    Args:
        array: NumPy array or PyTorch tensor

    Returns:
        NumPy array
    """
    if _TORCH_AVAILABLE:
        import torch
        if isinstance(array, torch.Tensor):
            return array.cpu().numpy()

    if _CUPY_AVAILABLE:
        import cupy as cp
        if isinstance(array, cp.ndarray):
            return cp.asnumpy(array)

    return np.asarray(array)


def get_array_module(use_gpu: bool = True):
    """
    Get the appropriate array module (numpy or cupy).

    Args:
        use_gpu: Whether to use GPU if available

    Returns:
        numpy or cupy module
    """
    if use_gpu and _CUPY_AVAILABLE:
        import cupy as cp
        return cp
    return np


class GPUAccelerator:
    """
    Context manager for GPU-accelerated computations.

    Usage:
        with GPUAccelerator() as xp:
            result = xp.mean(xp.array(data))
    """

    def __init__(self, use_gpu: bool = True):
        """
        Initialize GPU accelerator.

        Args:
            use_gpu: Whether to use GPU if available
        """
        self.use_gpu = use_gpu and is_gpu_available()
        self.xp = get_array_module(self.use_gpu)

    def __enter__(self):
        return self.xp

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.use_gpu and _CUPY_AVAILABLE:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
        return False

    def to_device(self, array: np.ndarray):
        """Convert numpy array to device array."""
        if self.use_gpu and _CUPY_AVAILABLE:
            import cupy as cp
            return cp.asarray(array)
        return array

    def to_numpy(self, array) -> np.ndarray:
        """Convert device array to numpy."""
        if self.use_gpu and _CUPY_AVAILABLE:
            import cupy as cp
            if isinstance(array, cp.ndarray):
                return cp.asnumpy(array)
        return np.asarray(array)


def gpu_covariance(
    x_values: np.ndarray,
    y_values: np.ndarray,
    use_gpu: bool = True
) -> float:
    """
    GPU-accelerated covariance calculation.

    Args:
        x_values: First array of values
        y_values: Second array of values
        use_gpu: Whether to use GPU

    Returns:
        Covariance value
    """
    with GPUAccelerator(use_gpu) as xp:
        x = xp.asarray(x_values, dtype=xp.float32)
        y = xp.asarray(y_values, dtype=xp.float32)

        n = len(x)
        if n < 3:
            return 0.0

        mu_x = xp.mean(x)
        mu_y = xp.mean(y)

        cov = xp.sum((x - mu_x) * (y - mu_y)) / (n - 1)

        if _CUPY_AVAILABLE and use_gpu:
            return float(cov.get())
        return float(cov)


def gpu_coskewness(
    x_values: np.ndarray,
    y_values: np.ndarray,
    use_gpu: bool = True
) -> float:
    """
    GPU-accelerated pairwise coskewness calculation.

    Args:
        x_values: First array of values
        y_values: Second array of values
        use_gpu: Whether to use GPU

    Returns:
        Coskewness value
    """
    with GPUAccelerator(use_gpu) as xp:
        x = xp.asarray(x_values, dtype=xp.float32)
        y = xp.asarray(y_values, dtype=xp.float32)

        n = len(x)
        if n < 3:
            return 0.0

        mu_x = xp.mean(x)
        mu_y = xp.mean(y)
        sigma_x = xp.std(x, ddof=1)
        sigma_y = xp.std(y, ddof=1)

        if sigma_x == 0 or sigma_y == 0:
            return 0.0

        sk = xp.sum((x - mu_x)**2 * (y - mu_y)) / n
        result = sk / (sigma_x**2 * sigma_y)

        if xp.isnan(result):
            return 0.0

        if _CUPY_AVAILABLE and use_gpu:
            return float(result.get())
        return float(result)


def gpu_cokurtosis(
    x_values: np.ndarray,
    y_values: np.ndarray,
    use_gpu: bool = True
) -> float:
    """
    GPU-accelerated pairwise cokurtosis calculation.

    Args:
        x_values: First array of values
        y_values: Second array of values
        use_gpu: Whether to use GPU

    Returns:
        Cokurtosis value
    """
    with GPUAccelerator(use_gpu) as xp:
        x = xp.asarray(x_values, dtype=xp.float32)
        y = xp.asarray(y_values, dtype=xp.float32)

        n = len(x)
        if n < 3:
            return 0.0

        mu_x = xp.mean(x)
        mu_y = xp.mean(y)
        sigma_x = xp.std(x, ddof=1)
        sigma_y = xp.std(y, ddof=1)

        if sigma_x == 0 or sigma_y == 0:
            return 0.0

        kt = xp.sum((x - mu_x)**2 * (y - mu_y)**2) / n
        result = kt / (sigma_x**2 * sigma_y**2)

        if xp.isnan(result):
            return 0.0

        if _CUPY_AVAILABLE and use_gpu:
            return float(result.get())
        return float(result)


def gpu_batch_statistics(
    qv_data: np.ndarray,
    use_gpu: bool = True
) -> tuple:
    """
    Compute batch statistics (mean, covariance, coskewness, cokurtosis) on GPU.

    Args:
        qv_data: Quality value data matrix (n_positions x n_reads)
        use_gpu: Whether to use GPU

    Returns:
        Tuple of (means, covariance_matrix, coskewness_matrix, cokurtosis_matrix)
    """
    with GPUAccelerator(use_gpu) as xp:
        data = xp.asarray(qv_data, dtype=xp.float32)
        n_pos, n_reads = data.shape

        # Compute means
        means = xp.nanmean(data, axis=1)

        # Center data
        centered = data - means[:, xp.newaxis]

        # Compute standard deviations
        stds = xp.nanstd(data, axis=1, ddof=1)
        stds = xp.where(stds == 0, 1.0, stds)  # Avoid division by zero

        # Compute covariance matrix
        cov_matrix = xp.zeros((n_pos, n_pos), dtype=xp.float32)
        for i in range(n_pos):
            for j in range(n_pos):
                valid = ~(xp.isnan(data[i]) | xp.isnan(data[j]))
                if xp.sum(valid) >= 3:
                    cov_matrix[i, j] = xp.sum(
                        centered[i, valid] * centered[j, valid]
                    ) / (xp.sum(valid) - 1)

        # Compute coskewness matrix
        sk_matrix = xp.zeros((n_pos, n_pos), dtype=xp.float32)
        for i in range(n_pos):
            for j in range(n_pos):
                valid = ~(xp.isnan(data[i]) | xp.isnan(data[j]))
                n = xp.sum(valid)
                if n >= 3 and stds[i] > 0 and stds[j] > 0:
                    sk = xp.sum(centered[i, valid]**2 * centered[j, valid]) / n
                    sk_matrix[i, j] = sk / (stds[i]**2 * stds[j])

        # Compute cokurtosis matrix
        kt_matrix = xp.zeros((n_pos, n_pos), dtype=xp.float32)
        for i in range(n_pos):
            for j in range(n_pos):
                valid = ~(xp.isnan(data[i]) | xp.isnan(data[j]))
                n = xp.sum(valid)
                if n >= 3 and stds[i] > 0 and stds[j] > 0:
                    kt = xp.sum(centered[i, valid]**2 * centered[j, valid]**2) / n
                    kt_matrix[i, j] = kt / (stds[i]**2 * stds[j]**2)

        # Convert back to numpy
        if _CUPY_AVAILABLE and use_gpu:
            return (
                means.get(),
                cov_matrix.get(),
                sk_matrix.get(),
                kt_matrix.get()
            )
        return (
            np.asarray(means),
            np.asarray(cov_matrix),
            np.asarray(sk_matrix),
            np.asarray(kt_matrix)
        )
