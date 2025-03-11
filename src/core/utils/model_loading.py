"""
Optimized Model Loading Utilities for AMPTALK.

This module provides state-of-the-art techniques for efficient model loading
and unloading, with a focus on transformer models like Whisper. It enables
memory-efficient loading, fast initialization, and proper resource management.

Author: AMPTALK Team
Date: 2024
"""

import os
import time
import logging
import gc
import threading
import weakref
from enum import Enum
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass
import json

import torch
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


class ModelLoadingStrategy(Enum):
    """Strategies for loading models into memory."""
    
    STANDARD = "standard"               # Standard PyTorch model loading
    LAZY = "lazy"                       # Lazy loading of weights (requires matching dtypes)
    SEQUENTIAL = "sequential"           # Load model components sequentially
    LOW_MEMORY = "low_memory"           # Load with minimal memory footprint
    MMAP = "mmap"                       # Memory-mapped loading (where supported)
    DEVICE_MAP = "device_map"           # Use Accelerate's device map (CPU/GPU distribution)


@dataclass
class ModelLoadingStats:
    """Statistics about model loading operations."""
    
    model_name: str
    strategy: ModelLoadingStrategy
    start_time: float
    end_time: Optional[float] = None
    peak_memory: Optional[int] = None
    current_memory: Optional[int] = None
    load_success: bool = False
    error_message: Optional[str] = None
    
    @property
    def loading_time(self) -> Optional[float]:
        """Calculate the total loading time in seconds."""
        if self.end_time is not None and self.start_time is not None:
            return self.end_time - self.start_time
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to a dictionary."""
        result = {
            "model_name": self.model_name,
            "strategy": self.strategy.value,
            "loading_time": self.loading_time,
            "peak_memory": self.peak_memory,
            "current_memory": self.current_memory,
            "load_success": self.load_success
        }
        if self.error_message:
            result["error_message"] = self.error_message
        return result


class ModelRegistry:
    """
    Registry for tracking loaded models and their usage.
    
    This class helps manage model references, track which models
    are currently in memory, and provide usage statistics.
    """
    
    def __init__(self):
        """Initialize the model registry."""
        self._models = weakref.WeakValueDictionary()  # Use weak refs to allow GC
        self._usage_stats = {}
        self._last_access = {}
        self._loading_stats = {}
        self._lock = threading.RLock()
    
    def register_model(self, model_id: str, model: Any, stats: ModelLoadingStats) -> None:
        """
        Register a loaded model with the registry.
        
        Args:
            model_id: Unique identifier for the model
            model: The model object
            stats: Statistics about the model loading
        """
        with self._lock:
            self._models[model_id] = model
            self._usage_stats[model_id] = 0
            self._last_access[model_id] = time.time()
            self._loading_stats[model_id] = stats
            
            logger.info(f"Model {model_id} registered (loaded in {stats.loading_time:.2f}s)")
    
    def get_model(self, model_id: str) -> Optional[Any]:
        """
        Get a model by ID and update usage statistics.
        
        Args:
            model_id: ID of the model to retrieve
            
        Returns:
            The model object or None if not found
        """
        with self._lock:
            model = self._models.get(model_id)
            if model is not None:
                self._usage_stats[model_id] += 1
                self._last_access[model_id] = time.time()
            return model
    
    def unregister_model(self, model_id: str) -> bool:
        """
        Remove a model from the registry.
        
        Args:
            model_id: ID of the model to unregister
            
        Returns:
            True if the model was unregistered, False if not found
        """
        with self._lock:
            if model_id in self._models:
                del self._models[model_id]
                if model_id in self._usage_stats:
                    del self._usage_stats[model_id]
                if model_id in self._last_access:
                    del self._last_access[model_id]
                if model_id in self._loading_stats:
                    del self._loading_stats[model_id]
                logger.info(f"Model {model_id} unregistered")
                return True
            return False
    
    def is_model_loaded(self, model_id: str) -> bool:
        """
        Check if a model is currently loaded.
        
        Args:
            model_id: ID of the model to check
            
        Returns:
            True if the model is loaded, False otherwise
        """
        with self._lock:
            return model_id in self._models
    
    def get_model_stats(self, model_id: str) -> Dict[str, Any]:
        """
        Get usage statistics for a model.
        
        Args:
            model_id: ID of the model to get stats for
            
        Returns:
            Dictionary with usage statistics
        """
        with self._lock:
            if model_id not in self._models:
                return {"error": "Model not found"}
            
            return {
                "usage_count": self._usage_stats.get(model_id, 0),
                "last_access": self._last_access.get(model_id),
                "loading_stats": self._loading_stats.get(model_id).to_dict() if model_id in self._loading_stats else None
            }
    
    def get_least_recently_used_model(self) -> Optional[str]:
        """
        Get the ID of the least recently used model.
        
        Returns:
            ID of the least recently used model, or None if no models are loaded
        """
        with self._lock:
            if not self._last_access:
                return None
            
            return min(self._last_access.items(), key=lambda x: x[1])[0]
    
    def get_all_models(self) -> List[str]:
        """
        Get a list of all loaded model IDs.
        
        Returns:
            List of model IDs
        """
        with self._lock:
            return list(self._models.keys())


# Global registry instance
_model_registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    """
    Get the global model registry.
    
    Returns:
        The global ModelRegistry instance
    """
    return _model_registry


def get_optimal_device() -> torch.device:
    """
    Determine the optimal device to load models onto.
    
    This function checks available hardware and returns the most
    suitable device (CUDA, MPS, or CPU) for model loading.
    
    Returns:
        torch.device: The optimal device for model loading
    """
    # Check for CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        # Get available GPU memory
        gpu_id = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
        allocated_memory = torch.cuda.memory_allocated(gpu_id)
        free_memory = total_memory - allocated_memory
        
        # If GPU has sufficient free memory (>2GB), use it
        if free_memory > 2 * 1024 * 1024 * 1024:  # 2GB in bytes
            return torch.device(f"cuda:{gpu_id}")
        else:
            logger.warning(f"GPU has limited free memory ({free_memory / 1024**3:.2f} GB), falling back to CPU")
    
    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            # Test MPS by creating a small tensor
            _ = torch.ones(1, device="mps")
            return torch.device("mps")
        except Exception as e:
            logger.warning(f"MPS is available but encountered an error: {e}")
    
    # Fall back to CPU
    return torch.device("cpu")


def calculate_memory_usage() -> Dict[str, Any]:
    """
    Calculate current memory usage statistics.
    
    Returns:
        Dictionary with memory usage statistics
    """
    stats = {}
    
    # CPU memory usage
    try:
        import psutil
        process = psutil.Process(os.getpid())
        cpu_memory = process.memory_info().rss
        stats["cpu_memory_bytes"] = cpu_memory
        stats["cpu_memory_gb"] = cpu_memory / (1024 ** 3)
    except ImportError:
        logger.debug("psutil not available for CPU memory tracking")
    
    # CUDA memory usage
    if torch.cuda.is_available():
        gpu_stats = {}
        for i in range(torch.cuda.device_count()):
            gpu_stats[i] = {
                "allocated_bytes": torch.cuda.memory_allocated(i),
                "allocated_gb": torch.cuda.memory_allocated(i) / (1024 ** 3),
                "cached_bytes": torch.cuda.memory_reserved(i),
                "cached_gb": torch.cuda.memory_reserved(i) / (1024 ** 3)
            }
        stats["gpu_memory"] = gpu_stats
    
    # MPS memory usage (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # No direct API for getting MPS memory usage in PyTorch currently
        stats["mps_memory"] = "Memory tracking not available for MPS"
    
    return stats


def get_dtype_for_model(model_name: str, default_dtype: torch.dtype = torch.float32) -> torch.dtype:
    """
    Determine the optimal dtype for loading a model.
    
    This function attempts to identify the native dtype of the model's weights
    to enable efficient loading by matching architecture and weight dtypes.
    
    Args:
        model_name: Name or path of the model
        default_dtype: Default dtype to use if detection fails
        
    Returns:
        The optimal dtype for the model
    """
    # Try to load model configuration to determine dtype
    try:
        if os.path.isdir(model_name):
            # Local model path - check for config.json
            config_path = os.path.join(model_name, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
                    
                    # Check for dtype info in config
                    torch_dtype = config.get("torch_dtype")
                    if torch_dtype:
                        if torch_dtype == "float16":
                            return torch.float16
                        elif torch_dtype == "float32":
                            return torch.float32
                        elif torch_dtype == "bfloat16":
                            return torch.bfloat16
                        elif torch_dtype == "float64":
                            return torch.float64
    except Exception as e:
        logger.warning(f"Failed to determine model dtype from config: {e}")
    
    # Default to safe option
    return default_dtype


def load_model_with_strategy(
    model_name_or_path: str, 
    model_class: Any,
    strategy: ModelLoadingStrategy = ModelLoadingStrategy.STANDARD,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    **kwargs
) -> Tuple[Any, ModelLoadingStats]:
    """
    Load a model using the specified loading strategy.
    
    Args:
        model_name_or_path: Name or path of the model to load
        model_class: Class of the model (e.g., WhisperForConditionalGeneration)
        strategy: Strategy to use for loading the model
        device: Device to load the model onto (auto-detected if None)
        dtype: Data type to use for the model (auto-detected if None)
        **kwargs: Additional arguments to pass to the model's from_pretrained method
        
    Returns:
        Tuple of (loaded model, loading statistics)
    """
    if device is None:
        device = get_optimal_device()
    
    if dtype is None:
        dtype = get_dtype_for_model(model_name_or_path)
    
    # Prepare stats object
    stats = ModelLoadingStats(
        model_name=model_name_or_path,
        strategy=strategy,
        start_time=time.time()
    )
    
    try:
        # Record initial memory usage
        initial_memory = calculate_memory_usage()
        peak_memory = initial_memory.get("cpu_memory_bytes", 0)
        
        if strategy == ModelLoadingStrategy.STANDARD:
            # Standard PyTorch loading
            model = model_class.from_pretrained(
                model_name_or_path,
                torch_dtype=dtype,
                **kwargs
            )
            model = model.to(device)
            
        elif strategy == ModelLoadingStrategy.LAZY:
            # Lazy loading - requires matching dtype for architecture and weights
            model = model_class.from_pretrained(
                model_name_or_path,
                torch_dtype=dtype,
                device_map="auto" if device.type == "cuda" else None,
                low_cpu_mem_usage=True,
                **kwargs
            )
            # Note: device_map auto will handle the device placement
            if device.type != "cuda":
                model = model.to(device)
                
        elif strategy == ModelLoadingStrategy.LOW_MEMORY:
            # Low memory loading - load with minimal memory footprint
            model = model_class.from_pretrained(
                model_name_or_path,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                **kwargs
            )
            model = model.to(device)
            
        elif strategy == ModelLoadingStrategy.MMAP:
            # Memory-mapped loading - use mmap for weights
            from safetensors.torch import load_file
            
            # Load the model architecture first
            model = model_class.from_pretrained(
                model_name_or_path,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                **kwargs
            )
            
            # Then use mmap to load weights (where supported)
            try:
                # Check for safetensors weights
                safetensors_path = os.path.join(model_name_or_path, "model.safetensors")
                if os.path.exists(safetensors_path):
                    state_dict = load_file(safetensors_path, device="cpu")
                    model.load_state_dict(state_dict, strict=False)
            except Exception as e:
                logger.warning(f"Memory-mapped loading failed, falling back to standard: {e}")
                
            model = model.to(device)
            
        elif strategy == ModelLoadingStrategy.SEQUENTIAL:
            # Sequential loading - load components one at a time
            from accelerate import init_empty_weights, load_checkpoint_and_dispatch
            
            # 1. Create empty model architecture
            with init_empty_weights():
                model = model_class.from_pretrained(
                    model_name_or_path,
                    torch_dtype=dtype,
                    **kwargs
                )
            
            # 2. Load weights sequentially
            model = load_checkpoint_and_dispatch(
                model, 
                model_name_or_path,
                device_map="auto" if device.type == "cuda" else {"": device},
                no_split_module_classes=["WhisperEncoderLayer", "WhisperDecoderLayer"]
            )
            
        elif strategy == ModelLoadingStrategy.DEVICE_MAP:
            # Device map - use Accelerate to distribute model across devices
            try:
                from accelerate import infer_auto_device_map, dispatch_model
                
                # 1. Load model on CPU with low memory
                model = model_class.from_pretrained(
                    model_name_or_path,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
                
                # 2. Infer device map
                if device.type == "cuda" and torch.cuda.device_count() > 1:
                    # Multiple GPUs available
                    device_map = infer_auto_device_map(
                        model,
                        max_memory={i: "auto" for i in range(torch.cuda.device_count())},
                        no_split_module_classes=["WhisperEncoderLayer", "WhisperDecoderLayer"]
                    )
                else:
                    # Single device
                    device_map = {"": device}
                
                # 3. Dispatch model according to device map
                model = dispatch_model(model, device_map=device_map)
            except ImportError:
                logger.warning("Accelerate not available, falling back to standard loading")
                model = model_class.from_pretrained(
                    model_name_or_path,
                    torch_dtype=dtype,
                    **kwargs
                )
                model = model.to(device)
        else:
            raise ValueError(f"Unknown loading strategy: {strategy}")
        
        # Set model to evaluation mode
        model.eval()
        
        # Record final memory and update stats
        current_memory = calculate_memory_usage().get("cpu_memory_bytes", 0)
        stats.peak_memory = max(peak_memory, current_memory)
        stats.current_memory = current_memory
        stats.end_time = time.time()
        stats.load_success = True
        
        return model, stats
        
    except Exception as e:
        # Record failure in stats
        stats.end_time = time.time()
        stats.load_success = False
        stats.error_message = str(e)
        
        logger.error(f"Failed to load model with {strategy.value} strategy: {e}")
        raise
        

def unload_model(model: Any, cleanup_cache: bool = True) -> None:
    """
    Unload a model and free associated memory.
    
    Args:
        model: The model to unload
        cleanup_cache: Whether to clear CUDA cache if available
    """
    # Record initial memory
    initial_memory = calculate_memory_usage()
    
    # Move model to CPU first (if it's on GPU)
    if hasattr(model, "to") and callable(model.to):
        try:
            model.to("cpu")
        except Exception as e:
            logger.warning(f"Failed to move model to CPU: {e}")
    
    # Delete model and clear reference
    del model
    
    # Collect garbage
    gc.collect()
    
    # Clear CUDA cache if available and requested
    if cleanup_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Log memory change
    final_memory = calculate_memory_usage()
    
    # Log CPU memory change
    initial_cpu = initial_memory.get("cpu_memory_gb", 0)
    final_cpu = final_memory.get("cpu_memory_gb", 0)
    if initial_cpu and final_cpu:
        logger.info(f"Memory freed: {initial_cpu - final_cpu:.2f} GB CPU")
    
    # Log GPU memory change if available
    if "gpu_memory" in initial_memory and "gpu_memory" in final_memory:
        for i in range(torch.cuda.device_count()):
            if i in initial_memory["gpu_memory"] and i in final_memory["gpu_memory"]:
                initial_gpu = initial_memory["gpu_memory"][i]["allocated_gb"]
                final_gpu = final_memory["gpu_memory"][i]["allocated_gb"]
                logger.info(f"Memory freed: {initial_gpu - final_gpu:.2f} GB on GPU {i}")


def register_model(model_id: str, model: Any, stats: ModelLoadingStats) -> None:
    """
    Register a model with the global registry.
    
    Args:
        model_id: Unique identifier for the model
        model: The model object
        stats: Loading statistics for the model
    """
    get_registry().register_model(model_id, model, stats)


def get_model(model_id: str) -> Optional[Any]:
    """
    Get a model from the global registry.
    
    Args:
        model_id: ID of the model to retrieve
        
    Returns:
        The model object or None if not found
    """
    return get_registry().get_model(model_id)


def unregister_model(model_id: str) -> bool:
    """
    Unregister a model from the global registry.
    
    Args:
        model_id: ID of the model to unregister
        
    Returns:
        True if successful, False otherwise
    """
    return get_registry().unregister_model(model_id)


def is_model_loaded(model_id: str) -> bool:
    """
    Check if a model is currently loaded.
    
    Args:
        model_id: ID of the model to check
        
    Returns:
        True if the model is loaded, False otherwise
    """
    return get_registry().is_model_loaded(model_id)


def get_model_stats(model_id: str) -> Dict[str, Any]:
    """
    Get usage statistics for a model.
    
    Args:
        model_id: ID of the model to get stats for
        
    Returns:
        Dictionary with usage statistics
    """
    return get_registry().get_model_stats(model_id)


def get_all_loaded_models() -> List[str]:
    """
    Get a list of all currently loaded model IDs.
    
    Returns:
        List of loaded model IDs
    """
    return get_registry().get_all_models()


def free_memory_if_needed(threshold_gb: float = 2.0) -> bool:
    """
    Free memory by unloading least recently used models if memory is low.
    
    Args:
        threshold_gb: Threshold in GB below which memory is considered low
        
    Returns:
        True if a model was unloaded, False otherwise
    """
    memory_info = calculate_memory_usage()
    
    # Check GPU memory if available
    if "gpu_memory" in memory_info and torch.cuda.is_available():
        for i, stats in memory_info["gpu_memory"].items():
            # Get total memory for this GPU
            total_memory = torch.cuda.get_device_properties(i).total_memory
            free_memory = total_memory - stats["allocated_bytes"]
            free_memory_gb = free_memory / (1024 ** 3)
            
            # If free memory is below threshold, try to free some
            if free_memory_gb < threshold_gb:
                logger.warning(f"Low GPU memory on device {i}: {free_memory_gb:.2f} GB free")
                model_id = get_registry().get_least_recently_used_model()
                
                if model_id:
                    model = get_registry().get_model(model_id)
                    logger.info(f"Unloading model {model_id} to free memory")
                    unload_model(model)
                    unregister_model(model_id)
                    return True
    
    return False


def determine_best_loading_strategy(
    model_name_or_path: str,
    available_ram_gb: Optional[float] = None,
    has_gpu: Optional[bool] = None
) -> ModelLoadingStrategy:
    """
    Determine the best loading strategy based on the model and available resources.
    
    Args:
        model_name_or_path: Name or path of the model
        available_ram_gb: Available RAM in GB (auto-detected if None)
        has_gpu: Whether a GPU is available (auto-detected if None)
        
    Returns:
        The recommended loading strategy
    """
    # Auto-detect system resources if not provided
    if available_ram_gb is None:
        try:
            import psutil
            available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
        except ImportError:
            # Default to conservative estimate
            available_ram_gb = 8.0
    
    if has_gpu is None:
        has_gpu = torch.cuda.is_available()
    
    # Try to determine model size
    model_size_gb = 0.0
    try:
        # If it's a local path, try to estimate model size from files
        if os.path.isdir(model_name_or_path):
            total_size = 0
            for root, _, files in os.walk(model_name_or_path):
                for file in files:
                    if file.endswith('.bin') or file.endswith('.pt') or file.endswith('.safetensors'):
                        file_path = os.path.join(root, file)
                        total_size += os.path.getsize(file_path)
            model_size_gb = total_size / (1024 ** 3)
    except Exception as e:
        logger.warning(f"Failed to determine model size: {e}")
    
    # Check if model contains 'large' or 'medium' in the name
    is_large_model = any(size in model_name_or_path.lower() for size in ['large', 'medium'])
    
    # Make recommendations based on available resources and model size
    
    # For very large models on systems with limited RAM
    if (model_size_gb > 10 or is_large_model) and available_ram_gb < 16:
        if has_gpu:
            return ModelLoadingStrategy.DEVICE_MAP
        else:
            return ModelLoadingStrategy.LOW_MEMORY
    
    # For medium to large models with sufficient RAM
    if model_size_gb > 5 or is_large_model:
        if has_gpu:
            return ModelLoadingStrategy.LAZY
        else:
            return ModelLoadingStrategy.MMAP
    
    # For small models or when model size can't be determined
    if has_gpu:
        return ModelLoadingStrategy.LAZY
    else:
        return ModelLoadingStrategy.STANDARD


def get_required_memory_for_model(model_name_or_path: str) -> Dict[str, float]:
    """
    Estimate the memory required to load a model.
    
    Args:
        model_name_or_path: Name or path of the model
        
    Returns:
        Dictionary with estimated memory requirements in GB
    """
    # Try to determine model size from files
    model_size_gb = 0.0
    try:
        if os.path.isdir(model_name_or_path):
            total_size = 0
            for root, _, files in os.walk(model_name_or_path):
                for file in files:
                    if file.endswith('.bin') or file.endswith('.pt') or file.endswith('.safetensors'):
                        file_path = os.path.join(root, file)
                        total_size += os.path.getsize(file_path)
            model_size_gb = total_size / (1024 ** 3)
    except Exception as e:
        logger.warning(f"Failed to determine model size from files: {e}")
        
        # If files can't be checked, try to estimate from model name
        if 'tiny' in model_name_or_path.lower():
            model_size_gb = 0.5
        elif 'base' in model_name_or_path.lower():
            model_size_gb = 1.5
        elif 'small' in model_name_or_path.lower():
            model_size_gb = 3.0
        elif 'medium' in model_name_or_path.lower():
            model_size_gb = 6.0
        elif 'large' in model_name_or_path.lower():
            model_size_gb = 12.0
        else:
            # Default estimate
            model_size_gb = 3.0
    
    # For loading, we typically need extra memory
    cpu_loading_estimate = model_size_gb * 2  # 2x model size for loading
    
    # For GPU, we need the model size plus some overhead
    gpu_estimate = model_size_gb * 1.2  # 1.2x model size on GPU
    
    return {
        "model_size_gb": model_size_gb,
        "cpu_loading_estimate_gb": cpu_loading_estimate,
        "gpu_estimate_gb": gpu_estimate
    } 