"""
Device Build Configuration System.

This module provides configuration management for device-specific builds,
supporting different architectures, platforms, and optimization levels.

Author: AMPTALK Team
Date: 2024
"""

import os
import json
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from src.core.utils.logging_config import get_logger

logger = get_logger(__name__)

class Architecture(Enum):
    """Supported CPU architectures."""
    X86_64 = "x86_64"
    ARM64 = "arm64"
    ARMV7 = "armv7"
    AARCH64 = "aarch64"

class Platform(Enum):
    """Supported platforms/operating systems."""
    LINUX = "linux"
    MACOS = "macos"
    WINDOWS = "windows"
    ANDROID = "android"
    IOS = "ios"

class AcceleratorType(Enum):
    """Supported hardware accelerators."""
    CPU = "cpu"
    GPU_NVIDIA = "gpu_nvidia"
    GPU_AMD = "gpu_amd"
    GPU_INTEL = "gpu_intel"
    TPU = "tpu"
    NPU = "npu"
    APPLE_SILICON = "apple_silicon"

@dataclass
class DeviceCapabilities:
    """Device hardware capabilities."""
    architecture: Architecture
    platform: Platform
    accelerators: List[AcceleratorType]
    ram_mb: int
    storage_mb: int
    compute_units: int
    supports_fp16: bool = False
    supports_int8: bool = False
    supports_int4: bool = False

@dataclass
class BuildConfig:
    """Build configuration for a specific device."""
    device_capabilities: DeviceCapabilities
    optimization_level: str = "O2"  # O0, O1, O2, O3
    enable_vectorization: bool = True
    enable_threading: bool = True
    enable_cuda: bool = False
    enable_metal: bool = False
    enable_opencl: bool = False
    target_batch_size: int = 1
    model_precision: str = "fp32"  # fp32, fp16, int8, int4
    max_workspace_size_mb: int = 1024
    fallback_devices: List[AcceleratorType] = None

class DeviceBuildManager:
    """
    Manages device-specific build configurations and optimizations.
    
    Features:
    - Device capability detection
    - Build configuration generation
    - Cross-compilation support
    - Optimization selection
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the build manager."""
        self.config_path = config_path or "configs/device_builds.json"
        self.configs: Dict[str, BuildConfig] = {}
        self._load_configs()
    
    def _load_configs(self) -> None:
        """Load build configurations from file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                    for device_id, config_data in data.items():
                        self.configs[device_id] = self._parse_config(config_data)
                logger.info(f"Loaded {len(self.configs)} device configurations")
        except Exception as e:
            logger.error(f"Error loading configurations: {e}")
    
    def _parse_config(self, config_data: Dict) -> BuildConfig:
        """Parse configuration data into BuildConfig object."""
        capabilities = DeviceCapabilities(
            architecture=Architecture[config_data["architecture"].upper()],
            platform=Platform[config_data["platform"].upper()],
            accelerators=[AcceleratorType[acc.upper()] for acc in config_data["accelerators"]],
            ram_mb=config_data["ram_mb"],
            storage_mb=config_data["storage_mb"],
            compute_units=config_data["compute_units"],
            supports_fp16=config_data.get("supports_fp16", False),
            supports_int8=config_data.get("supports_int8", False),
            supports_int4=config_data.get("supports_int4", False)
        )
        
        return BuildConfig(
            device_capabilities=capabilities,
            optimization_level=config_data.get("optimization_level", "O2"),
            enable_vectorization=config_data.get("enable_vectorization", True),
            enable_threading=config_data.get("enable_threading", True),
            enable_cuda=config_data.get("enable_cuda", False),
            enable_metal=config_data.get("enable_metal", False),
            enable_opencl=config_data.get("enable_opencl", False),
            target_batch_size=config_data.get("target_batch_size", 1),
            model_precision=config_data.get("model_precision", "fp32"),
            max_workspace_size_mb=config_data.get("max_workspace_size_mb", 1024),
            fallback_devices=[AcceleratorType[dev.upper()] 
                            for dev in config_data.get("fallback_devices", [])]
        )
    
    def detect_device_capabilities(self) -> DeviceCapabilities:
        """Detect capabilities of the current device."""
        import platform
        import psutil
        
        # Detect architecture
        arch_map = {
            "x86_64": Architecture.X86_64,
            "arm64": Architecture.ARM64,
            "aarch64": Architecture.AARCH64,
            "armv7l": Architecture.ARMV7
        }
        arch = arch_map.get(platform.machine().lower(), Architecture.X86_64)
        
        # Detect platform
        platform_map = {
            "Linux": Platform.LINUX,
            "Darwin": Platform.MACOS,
            "Windows": Platform.WINDOWS
        }
        os_platform = platform_map.get(platform.system(), Platform.LINUX)
        
        # Detect accelerators
        accelerators = [AcceleratorType.CPU]
        
        # Check for NVIDIA GPU
        try:
            import torch
            if torch.cuda.is_available():
                accelerators.append(AcceleratorType.GPU_NVIDIA)
        except ImportError:
            pass
        
        # Check for Apple Silicon
        if platform.machine() == "arm64" and platform.system() == "Darwin":
            accelerators.append(AcceleratorType.APPLE_SILICON)
        
        # Get system resources
        ram_mb = psutil.virtual_memory().total // (1024 * 1024)
        storage = psutil.disk_usage('/')
        storage_mb = storage.total // (1024 * 1024)
        compute_units = psutil.cpu_count()
        
        return DeviceCapabilities(
            architecture=arch,
            platform=os_platform,
            accelerators=accelerators,
            ram_mb=ram_mb,
            storage_mb=storage_mb,
            compute_units=compute_units,
            supports_fp16=True,  # Will be updated based on actual testing
            supports_int8=True,  # Will be updated based on actual testing
            supports_int4=False  # Will be updated based on actual testing
        )
    
    def get_build_config(self, device_id: Optional[str] = None) -> BuildConfig:
        """
        Get build configuration for a device.
        
        Args:
            device_id: Optional device identifier. If None, detect current device.
        
        Returns:
            BuildConfig for the device
        """
        if device_id and device_id in self.configs:
            return self.configs[device_id]
        
        # Detect current device capabilities
        capabilities = self.detect_device_capabilities()
        
        # Create optimal configuration based on capabilities
        config = BuildConfig(
            device_capabilities=capabilities,
            optimization_level="O2",
            enable_vectorization=True,
            enable_threading=True,
            enable_cuda=AcceleratorType.GPU_NVIDIA in capabilities.accelerators,
            enable_metal=AcceleratorType.APPLE_SILICON in capabilities.accelerators,
            enable_opencl=False,
            target_batch_size=1,
            model_precision="fp32",
            max_workspace_size_mb=min(capabilities.ram_mb // 4, 1024),
            fallback_devices=[AcceleratorType.CPU]
        )
        
        return config
    
    def save_config(self, device_id: str, config: BuildConfig) -> None:
        """Save a device configuration."""
        self.configs[device_id] = config
        self._save_configs()
    
    def _save_configs(self) -> None:
        """Save all configurations to file."""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                data = {
                    device_id: {
                        "architecture": config.device_capabilities.architecture.value,
                        "platform": config.device_capabilities.platform.value,
                        "accelerators": [acc.value for acc in config.device_capabilities.accelerators],
                        "ram_mb": config.device_capabilities.ram_mb,
                        "storage_mb": config.device_capabilities.storage_mb,
                        "compute_units": config.device_capabilities.compute_units,
                        "supports_fp16": config.device_capabilities.supports_fp16,
                        "supports_int8": config.device_capabilities.supports_int8,
                        "supports_int4": config.device_capabilities.supports_int4,
                        "optimization_level": config.optimization_level,
                        "enable_vectorization": config.enable_vectorization,
                        "enable_threading": config.enable_threading,
                        "enable_cuda": config.enable_cuda,
                        "enable_metal": config.enable_metal,
                        "enable_opencl": config.enable_opencl,
                        "target_batch_size": config.target_batch_size,
                        "model_precision": config.model_precision,
                        "max_workspace_size_mb": config.max_workspace_size_mb,
                        "fallback_devices": [dev.value for dev in (config.fallback_devices or [])]
                    }
                    for device_id, config in self.configs.items()
                }
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.configs)} device configurations")
        except Exception as e:
            logger.error(f"Error saving configurations: {e}") 