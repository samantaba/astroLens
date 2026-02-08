"""
GPU Acceleration Utility for AstroLens

Provides device detection, management, and performance utilities
for PyTorch-based ML inference and training.

Supports:
- NVIDIA CUDA GPUs
- Apple Metal Performance Shaders (MPS)
- CPU fallback

Usage:
    from inference.gpu_utils import get_device, DeviceInfo
    
    device = get_device()         # Returns best available torch device
    info = DeviceInfo.detect()    # Returns full hardware info
"""

from __future__ import annotations

import logging
import os
import platform
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Cache device to avoid re-detection
_cached_device = None


@dataclass
class DeviceInfo:
    """Hardware information for ML acceleration."""
    device_type: str          # 'cuda', 'mps', 'cpu'
    device_name: str          # Human-readable name
    memory_gb: Optional[float] = None
    cuda_version: Optional[str] = None
    is_available: bool = True
    
    @classmethod
    def detect(cls) -> "DeviceInfo":
        """Detect the best available compute device."""
        try:
            import torch
        except ImportError:
            return cls(device_type="cpu", device_name="CPU (PyTorch not installed)")
        
        # Check CUDA (NVIDIA GPUs)
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            mem_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            cuda_ver = torch.version.cuda
            logger.info(f"GPU detected: {gpu_name} ({mem_gb:.1f} GB, CUDA {cuda_ver})")
            return cls(
                device_type="cuda",
                device_name=gpu_name,
                memory_gb=round(mem_gb, 1),
                cuda_version=cuda_ver,
            )
        
        # Check MPS (Apple Silicon)
        # torch.backends.mps.is_available() can return False on newer macOS
        # versions that PyTorch doesn't recognize yet, so also probe directly.
        mps_works = False
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_built():
            if torch.backends.mps.is_available():
                mps_works = True
            else:
                # Direct probe: try creating a tensor on MPS
                try:
                    _t = torch.tensor([1.0], device="mps")
                    del _t
                    mps_works = True
                    logger.info("MPS probe succeeded (is_available was False)")
                except Exception:
                    pass
        
        if mps_works:
            chip = _detect_apple_chip()
            logger.info(f"Apple MPS detected: {chip}")
            return cls(
                device_type="mps",
                device_name=chip,
            )
        
        # CPU fallback
        cpu_name = platform.processor() or "Unknown CPU"
        logger.info(f"Using CPU: {cpu_name}")
        return cls(
            device_type="cpu",
            device_name=f"CPU ({cpu_name})",
        )
    
    def to_dict(self) -> dict:
        return {
            "device_type": self.device_type,
            "device_name": self.device_name,
            "memory_gb": self.memory_gb,
            "cuda_version": self.cuda_version,
            "is_available": self.is_available,
        }


def _detect_apple_chip() -> str:
    """Detect Apple Silicon chip type."""
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() or "Apple Silicon"
    except Exception:
        return "Apple Silicon (MPS)"


def get_device(force: Optional[str] = None):
    """
    Get the best available PyTorch device.
    
    Args:
        force: Force a specific device ('cuda', 'mps', 'cpu')
    
    Returns:
        torch.device object
    """
    global _cached_device
    
    if force:
        import torch
        return torch.device(force)
    
    if _cached_device is not None:
        return _cached_device
    
    try:
        import torch
    except ImportError:
        logger.warning("PyTorch not installed, cannot use GPU acceleration")
        return None
    
    # Priority: CUDA > MPS > CPU
    if torch.cuda.is_available():
        _cached_device = torch.device("cuda")
    else:
        # Check MPS with direct probe fallback (macOS version detection bug)
        mps_ok = False
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_built():
            if torch.backends.mps.is_available():
                mps_ok = True
            else:
                try:
                    _t = torch.tensor([1.0], device="mps")
                    del _t
                    mps_ok = True
                except Exception:
                    pass
        _cached_device = torch.device("mps") if mps_ok else torch.device("cpu")
    
    logger.info(f"Using device: {_cached_device}")
    return _cached_device


def move_model_to_device(model, device=None):
    """
    Move a PyTorch model to the best device.
    
    Args:
        model: PyTorch model (nn.Module)
        device: Target device (auto-detected if None)
    
    Returns:
        Model on the target device
    """
    if device is None:
        device = get_device()
    
    if device is None:
        return model
    
    try:
        model = model.to(device)
        logger.info(f"Model moved to {device}")
    except Exception as e:
        logger.warning(f"Failed to move model to {device}: {e}, falling back to CPU")
        import torch
        model = model.to(torch.device("cpu"))
    
    return model


def move_tensor_to_device(tensor, device=None):
    """
    Move a tensor to the best device.
    
    Args:
        tensor: PyTorch tensor
        device: Target device (auto-detected if None)
    
    Returns:
        Tensor on the target device
    """
    if device is None:
        device = get_device()
    
    if device is None:
        return tensor
    
    try:
        return tensor.to(device)
    except Exception:
        return tensor


def get_device_summary() -> str:
    """Get a human-readable device summary string."""
    info = DeviceInfo.detect()
    
    if info.device_type == "cuda":
        return f"GPU: {info.device_name} ({info.memory_gb} GB, CUDA {info.cuda_version})"
    elif info.device_type == "mps":
        return f"GPU: {info.device_name} (Metal Performance Shaders)"
    else:
        return f"CPU: {info.device_name} (No GPU acceleration)"
