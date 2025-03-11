#!/usr/bin/env python3
"""
Device Build System Demo.

This script demonstrates how to use the device-specific build system
to create optimized builds for different target platforms.

Author: AMPTALK Team
Date: 2024
"""

import os
import sys
import argparse
import logging
from typing import Optional

from src.core.device_builds.config import (
    DeviceBuildManager,
    Platform,
    Architecture,
    AcceleratorType
)
from src.core.device_builds.builder import DeviceBuilder
from src.core.utils.logging_config import get_logger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = get_logger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Device Build Demo")
    parser.add_argument(
        "--platform",
        choices=[p.value for p in Platform],
        help="Target platform"
    )
    parser.add_argument(
        "--arch",
        choices=[a.value for a in Architecture],
        help="Target architecture"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean build directory before building"
    )
    parser.add_argument(
        "--output-dir",
        default="build",
        help="Output directory for builds"
    )
    return parser.parse_args()

def create_example_config(
    build_manager: DeviceBuildManager,
    platform: Optional[str] = None,
    arch: Optional[str] = None
) -> str:
    """Create an example device configuration."""
    # Use current device capabilities as base
    capabilities = build_manager.detect_device_capabilities()
    
    # Override platform if specified
    if platform:
        capabilities.platform = Platform[platform.upper()]
    
    # Override architecture if specified
    if arch:
        capabilities.architecture = Architecture[arch.upper()]
    
    # Create device ID
    device_id = f"{capabilities.platform.value}_{capabilities.architecture.value}"
    
    # Create configuration
    config = build_manager.get_build_config()
    config.device_capabilities = capabilities
    
    # Save configuration
    build_manager.save_config(device_id, config)
    
    return device_id

def main():
    """Main function."""
    args = parse_args()
    
    try:
        # Initialize managers
        build_manager = DeviceBuildManager()
        builder = DeviceBuilder(build_dir=args.output_dir)
        
        # Create example configuration
        device_id = create_example_config(
            build_manager,
            platform=args.platform,
            arch=args.arch
        )
        
        logger.info(f"Building for device: {device_id}")
        
        # Get source directory (assuming we're in examples/)
        source_dir = os.path.join(os.path.dirname(__file__), "..", "src")
        
        # Build for device
        artifact_path = builder.build_for_device(
            source_dir=source_dir,
            device_id=device_id,
            clean=args.clean
        )
        
        logger.info(f"Build successful! Artifact: {artifact_path}")
        
    except Exception as e:
        logger.error(f"Build failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 