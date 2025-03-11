"""
Device-specific Build System.

This module handles the compilation and optimization of code for different devices,
ensuring optimal performance across various hardware configurations.

Author: AMPTALK Team
Date: 2024
"""

import os
import shutil
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from src.core.device_builds.config import (
    DeviceBuildManager,
    BuildConfig,
    Architecture,
    Platform,
    AcceleratorType
)
from src.core.utils.logging_config import get_logger

logger = get_logger(__name__)

@dataclass
class BuildTarget:
    """Build target configuration."""
    name: str
    platform: Platform
    architecture: Architecture
    python_version: str
    requirements: List[str]
    build_flags: Dict[str, str]

class DeviceBuilder:
    """
    Handles device-specific builds and optimizations.
    
    Features:
    - Cross-compilation support
    - Platform-specific optimizations
    - Dependency management
    - Build artifact management
    """
    
    def __init__(self, build_dir: str = "build"):
        """
        Initialize the device builder.
        
        Args:
            build_dir: Directory for build outputs
        """
        self.build_dir = build_dir
        self.build_manager = DeviceBuildManager()
        self._setup_build_environment()
    
    def _setup_build_environment(self) -> None:
        """Set up the build environment."""
        os.makedirs(self.build_dir, exist_ok=True)
        logger.info(f"Build environment set up in {self.build_dir}")
    
    def create_build_target(self, config: BuildConfig) -> BuildTarget:
        """
        Create a build target from configuration.
        
        Args:
            config: Device build configuration
        
        Returns:
            BuildTarget instance
        """
        # Base Python version
        python_version = "3.11"  # Default version
        
        # Base requirements
        requirements = [
            "torch",
            "numpy",
            "psutil",
            "opentelemetry-api",
            "opentelemetry-sdk"
        ]
        
        # Platform-specific requirements
        if config.device_capabilities.platform == Platform.ANDROID:
            requirements.extend([
                "android-ndk",
                "android-sdk"
            ])
        elif config.device_capabilities.platform == Platform.IOS:
            requirements.extend([
                "coremltools",
                "torch2ios"
            ])
        
        # Accelerator-specific requirements
        if AcceleratorType.GPU_NVIDIA in config.device_capabilities.accelerators:
            requirements.append("torch==2.2.0+cu121")  # CUDA 12.1
        elif AcceleratorType.APPLE_SILICON in config.device_capabilities.accelerators:
            requirements.append("torch==2.2.0")  # MPS support
        
        # Build flags
        build_flags = {
            "CFLAGS": "-O3 -march=native",
            "CXXFLAGS": "-O3 -march=native",
            "LDFLAGS": "-Wl,-O3"
        }
        
        # Platform-specific flags
        if config.device_capabilities.platform == Platform.ANDROID:
            build_flags.update({
                "ANDROID_NDK_HOME": os.environ.get("ANDROID_NDK_HOME", ""),
                "ANDROID_SDK_ROOT": os.environ.get("ANDROID_SDK_ROOT", "")
            })
        elif config.device_capabilities.platform == Platform.IOS:
            build_flags.update({
                "IPHONEOS_DEPLOYMENT_TARGET": "14.0",
                "MACOSX_DEPLOYMENT_TARGET": "11.0"
            })
        
        # Optimization flags
        if config.optimization_level == "O3":
            build_flags["CFLAGS"] += " -ffast-math -funroll-loops"
            build_flags["CXXFLAGS"] += " -ffast-math -funroll-loops"
        
        return BuildTarget(
            name=f"{config.device_capabilities.platform.value}_{config.device_capabilities.architecture.value}",
            platform=config.device_capabilities.platform,
            architecture=config.device_capabilities.architecture,
            python_version=python_version,
            requirements=requirements,
            build_flags=build_flags
        )
    
    def build_for_device(
        self,
        source_dir: str,
        device_id: Optional[str] = None,
        clean: bool = False
    ) -> str:
        """
        Build the project for a specific device.
        
        Args:
            source_dir: Source code directory
            device_id: Optional device identifier
            clean: Whether to clean before building
        
        Returns:
            Path to build artifacts
        """
        # Get device configuration
        config = self.build_manager.get_build_config(device_id)
        build_target = self.create_build_target(config)
        
        # Create build directory
        build_path = os.path.join(self.build_dir, build_target.name)
        if clean and os.path.exists(build_path):
            shutil.rmtree(build_path)
        os.makedirs(build_path, exist_ok=True)
        
        try:
            # Create virtual environment
            self._create_venv(build_path, build_target)
            
            # Install dependencies
            self._install_dependencies(build_path, build_target)
            
            # Copy source code
            self._copy_source(source_dir, build_path)
            
            # Apply optimizations
            self._apply_optimizations(build_path, config)
            
            # Create platform-specific package
            artifact_path = self._create_package(build_path, build_target)
            
            logger.info(f"Successfully built for {build_target.name}")
            return artifact_path
            
        except Exception as e:
            logger.error(f"Build failed for {build_target.name}: {e}")
            raise
    
    def _create_venv(self, build_path: str, target: BuildTarget) -> None:
        """Create a virtual environment for building."""
        venv_path = os.path.join(build_path, "venv")
        subprocess.run(
            ["python3", "-m", "venv", venv_path],
            check=True
        )
        logger.info(f"Created virtual environment at {venv_path}")
    
    def _install_dependencies(self, build_path: str, target: BuildTarget) -> None:
        """Install dependencies in the virtual environment."""
        venv_pip = os.path.join(build_path, "venv", "bin", "pip")
        
        # Upgrade pip
        subprocess.run(
            [venv_pip, "install", "--upgrade", "pip"],
            check=True,
            env={**os.environ, **target.build_flags}
        )
        
        # Install requirements
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt') as f:
            f.write('\n'.join(target.requirements))
            f.flush()
            subprocess.run(
                [venv_pip, "install", "-r", f.name],
                check=True,
                env={**os.environ, **target.build_flags}
            )
        
        logger.info(f"Installed dependencies for {target.name}")
    
    def _copy_source(self, source_dir: str, build_path: str) -> None:
        """Copy source code to build directory."""
        dest_dir = os.path.join(build_path, "src")
        shutil.copytree(source_dir, dest_dir, dirs_exist_ok=True)
        logger.info(f"Copied source code to {dest_dir}")
    
    def _apply_optimizations(self, build_path: str, config: BuildConfig) -> None:
        """Apply platform and device-specific optimizations."""
        src_dir = os.path.join(build_path, "src")
        
        # Apply CUDA optimizations if enabled
        if config.enable_cuda:
            self._optimize_cuda(src_dir, config)
        
        # Apply Metal optimizations if enabled
        if config.enable_metal:
            self._optimize_metal(src_dir, config)
        
        # Apply threading optimizations
        if config.enable_threading:
            self._optimize_threading(src_dir, config)
        
        logger.info(f"Applied optimizations for {config.device_capabilities.platform.value}")
    
    def _optimize_cuda(self, src_dir: str, config: BuildConfig) -> None:
        """Apply CUDA-specific optimizations."""
        # TODO: Implement CUDA optimizations
        pass
    
    def _optimize_metal(self, src_dir: str, config: BuildConfig) -> None:
        """Apply Metal-specific optimizations."""
        # TODO: Implement Metal optimizations
        pass
    
    def _optimize_threading(self, src_dir: str, config: BuildConfig) -> None:
        """Apply threading optimizations."""
        # TODO: Implement threading optimizations
        pass
    
    def _create_package(self, build_path: str, target: BuildTarget) -> str:
        """Create deployable package for the target platform."""
        if target.platform == Platform.ANDROID:
            return self._create_android_package(build_path, target)
        elif target.platform == Platform.IOS:
            return self._create_ios_package(build_path, target)
        else:
            return self._create_generic_package(build_path, target)
    
    def _create_android_package(self, build_path: str, target: BuildTarget) -> str:
        """Create Android APK package."""
        # TODO: Implement Android packaging
        pass
    
    def _create_ios_package(self, build_path: str, target: BuildTarget) -> str:
        """Create iOS package."""
        # TODO: Implement iOS packaging
        pass
    
    def _create_generic_package(self, build_path: str, target: BuildTarget) -> str:
        """Create generic Python package."""
        dist_dir = os.path.join(build_path, "dist")
        os.makedirs(dist_dir, exist_ok=True)
        
        # Create wheel package
        subprocess.run(
            ["python3", "setup.py", "bdist_wheel"],
            cwd=build_path,
            check=True,
            env={**os.environ, **target.build_flags}
        )
        
        # Find the created wheel
        wheels = [f for f in os.listdir(dist_dir) if f.endswith('.whl')]
        if not wheels:
            raise RuntimeError("No wheel package created")
        
        return os.path.join(dist_dir, wheels[0]) 