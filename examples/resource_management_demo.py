#!/usr/bin/env python3
"""
Resource Management Demo for AMPTALK.

This script demonstrates how to use the resource management module
to monitor and optimize resource usage in edge deployments.

Usage:
    python examples/resource_management_demo.py --strategy adaptive

Author: AMPTALK Team
Date: 2024
"""

import os
import time
import argparse
import logging
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from src.core.resource_management import (
    OptimizationStrategy,
    ResourceLimits,
    ResourceMetrics,
    create_resource_manager
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("resource_management_demo")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Resource Management Demo")
    parser.add_argument(
        "--strategy",
        choices=[s.value for s in OptimizationStrategy],
        default="adaptive",
        help="Resource optimization strategy"
    )
    parser.add_argument(
        "--output-dir",
        default="output/resource_management",
        help="Directory to save metrics"
    )
    parser.add_argument(
        "--max-memory-mb",
        type=int,
        default=1000,
        help="Maximum memory usage in MB"
    )
    parser.add_argument(
        "--max-cpu-percent",
        type=float,
        default=80,
        help="Maximum CPU usage percentage"
    )
    parser.add_argument(
        "--max-gpu-memory-mb",
        type=int,
        default=1000,
        help="Maximum GPU memory usage in MB"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Duration to run demo in seconds"
    )
    parser.add_argument(
        "--simulate-load",
        action="store_true",
        help="Simulate varying workload"
    )
    return parser.parse_args()


class DummyModel(nn.Module):
    """Dummy model for simulating workload."""
    def __init__(self, size: int = 1000):
        super().__init__()
        self.fc1 = nn.Linear(size, size)
        self.fc2 = nn.Linear(size, size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def simulate_workload(model: nn.Module, device: torch.device):
    """Simulate varying workload."""
    size = 1000
    # Generate random input
    x = torch.randn(size, size).to(device)
    
    # Forward pass
    output = model(x)
    
    # Simulate computation
    for _ in range(10):
        output = torch.matmul(output, output.t())
    
    # Force sync if using GPU
    if device.type == "cuda":
        torch.cuda.synchronize()


def metrics_callback(metrics: ResourceMetrics):
    """Callback for resource metrics."""
    logger.info(
        f"Memory: {metrics.memory_used_mb:.1f} MB ({metrics.memory_percent:.1f}%), "
        f"CPU: {metrics.cpu_percent:.1f}%, "
        f"GPU Memory: {metrics.gpu_memory_mb:.1f} MB ({metrics.gpu_utilization:.1f}%), "
        f"Power: {metrics.power_watts:.1f}W, "
        f"Temp: {metrics.temperature_celsius:.1f}°C"
    )


def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create resource manager
    manager = create_resource_manager(
        max_memory_mb=args.max_memory_mb,
        max_cpu_percent=args.max_cpu_percent,
        max_gpu_memory_mb=args.max_gpu_memory_mb,
        strategy=args.strategy,
        monitoring_interval=1.0,
        callback=metrics_callback
    )
    
    # Create dummy model if simulating load
    if args.simulate_load:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DummyModel().to(device)
        logger.info(f"Created dummy model on {device}")
    
    try:
        logger.info(f"Starting resource monitoring with strategy: {args.strategy}")
        start_time = time.time()
        
        while time.time() - start_time < args.duration:
            if args.simulate_load:
                # Simulate varying workload
                simulate_workload(model, device)
                
                # Random sleep to vary load
                time.sleep(np.random.uniform(0.1, 1.0))
            else:
                time.sleep(1.0)
        
        # Save metrics
        metrics_path = os.path.join(args.output_dir, "resource_metrics.json")
        manager.save_metrics(metrics_path)
        logger.info(f"Saved metrics to {metrics_path}")
        
    except KeyboardInterrupt:
        logger.info("Stopping demo...")
    
    finally:
        # Clean up
        if args.simulate_load and torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main() 