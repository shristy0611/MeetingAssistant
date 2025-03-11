#!/usr/bin/env python3
"""
Model Compression Demo for AMPTALK.

This script demonstrates how to use the model compression module
to compress neural networks using various state-of-the-art techniques.

Usage:
    python examples/model_compression_demo.py --model-size tiny

Author: AMPTALK Team
Date: 2024
"""

import os
import argparse
import time
import logging
import json
import torch
import torch.nn as nn
from pathlib import Path

from src.models.model_compression import (
    CompressionType,
    SharingStrategy,
    CompressionConfig,
    ModelCompressor
)
from src.core.utils.edge_optimization import EdgeOptimizer, OptimizationType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("model_compression_demo")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Model Compression Demo")
    parser.add_argument(
        "--model-size",
        choices=["tiny", "base", "small", "medium", "large"],
        default="tiny",
        help="Whisper model size to use"
    )
    parser.add_argument(
        "--output-dir",
        default="output/model_compression",
        help="Directory to save compressed models"
    )
    parser.add_argument(
        "--compression-types",
        nargs="+",
        choices=[t.value for t in CompressionType],
        default=["weight_sharing", "low_rank"],
        help="Compression techniques to apply"
    )
    parser.add_argument(
        "--target-ratio",
        type=float,
        default=0.5,
        help="Target compression ratio"
    )
    parser.add_argument(
        "--sharing-strategy",
        choices=[s.value for s in SharingStrategy],
        default="cluster",
        help="Weight sharing strategy"
    )
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=256,
        help="Number of clusters for weight sharing"
    )
    parser.add_argument(
        "--rank-ratio",
        type=float,
        default=0.3,
        help="Ratio for low-rank approximation"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark after compression"
    )
    return parser.parse_args()


def create_onnx_model(model_size: str, output_dir: str) -> str:
    """Create an ONNX model for compression."""
    try:
        # Create optimizer
        optimizer = EdgeOptimizer(
            working_dir=output_dir
        )
        
        # Convert Whisper to ONNX
        result = optimizer.optimize_whisper(
            model_size=model_size,
            optimizations=[OptimizationType.ONNX_CONVERSION],
            output_dir=output_dir
        )
        
        return result["model_path"]
    except ImportError as e:
        logger.error(f"Error importing edge_optimization: {e}")
        raise


def apply_compression(model_path: str, args) -> str:
    """Apply compression to the model."""
    try:
        # Convert string arguments to enums
        compression_types = [CompressionType[t.upper()] for t in args.compression_types]
        sharing_strategy = SharingStrategy[args.sharing_strategy.upper()]
        
        # Create configuration
        config = CompressionConfig(
            compression_types=compression_types,
            target_ratio=args.target_ratio,
            sharing_strategy=sharing_strategy,
            num_clusters=args.num_clusters,
            rank_ratio=args.rank_ratio
        )
        
        # Create compressor
        logger.info(f"Applying compression with config: {config.__dict__}")
        compressor = ModelCompressor(model_path, config)
        
        # Apply compression
        compressed_model = compressor.compress()
        
        # Save compressed model
        output_path = os.path.join(
            args.output_dir,
            f"compressed_{os.path.basename(model_path)}"
        )
        compressor.save_compressed_model(output_path)
        
        return output_path
    except Exception as e:
        logger.error(f"Error applying compression: {e}")
        raise


def benchmark(original_model: str, compressed_model: str) -> Dict[str, float]:
    """Benchmark the original and compressed models."""
    logger.info("Running benchmark...")
    
    try:
        import onnxruntime as ort
        import numpy as np
        
        # Sample audio input shape for Whisper (80-dim log-mel spectrogram)
        input_shape = (1, 80, 3000)
        
        # Create sessions
        logger.info("Creating ONNX Runtime sessions...")
        original_session = ort.InferenceSession(original_model, providers=['CPUExecutionProvider'])
        compressed_session = ort.InferenceSession(compressed_model, providers=['CPUExecutionProvider'])
        
        # Get input name
        original_input_name = original_session.get_inputs()[0].name
        compressed_input_name = compressed_session.get_inputs()[0].name
        
        # Create random input
        logger.info("Creating test input...")
        input_data = np.random.rand(*input_shape).astype(np.float32)
        
        # Warm-up
        logger.info("Warming up...")
        for _ in range(5):
            original_session.run(None, {original_input_name: input_data})
            compressed_session.run(None, {compressed_input_name: input_data})
        
        # Benchmark original
        logger.info("Benchmarking original model...")
        start_time = time.time()
        for _ in range(20):
            original_session.run(None, {original_input_name: input_data})
        original_time = (time.time() - start_time) / 20
        
        # Benchmark compressed
        logger.info("Benchmarking compressed model...")
        start_time = time.time()
        for _ in range(20):
            compressed_session.run(None, {compressed_input_name: input_data})
        compressed_time = (time.time() - start_time) / 20
        
        # Calculate speedup
        speedup = original_time / compressed_time
        
        # Get model sizes
        original_size = os.path.getsize(original_model)
        compressed_size = os.path.getsize(compressed_model)
        compression_ratio = compressed_size / original_size
        
        # Report results
        logger.info(f"Original model inference time: {original_time:.4f} seconds")
        logger.info(f"Compressed model inference time: {compressed_time:.4f} seconds")
        logger.info(f"Speedup: {speedup:.2f}x")
        logger.info(f"Compression ratio: {compression_ratio:.3f}")
        
        return {
            "original_time": original_time,
            "compressed_time": compressed_time,
            "speedup": speedup,
            "original_size": original_size,
            "compressed_size": compressed_size,
            "compression_ratio": compression_ratio
        }
    except ImportError as e:
        logger.error(f"Error importing onnxruntime: {e}")
        return None


def display_model_info(model_paths: Dict[str, str]) -> None:
    """Display information about the models."""
    logger.info("Model information:")
    
    for name, path in model_paths.items():
        if not path or not os.path.exists(path):
            continue
            
        size_bytes = os.path.getsize(path)
        size_mb = size_bytes / (1024 * 1024)
        
        logger.info(f"{name}: {path}")
        logger.info(f"  Size: {size_mb:.2f} MB ({size_bytes:,} bytes)")
        
        # Try to get operator information using onnx
        try:
            import onnx
            model = onnx.load(path)
            op_types = set(node.op_type for node in model.graph.node)
            logger.info(f"  Operators: {len(model.graph.node)} nodes, {len(op_types)} unique types")
            logger.info(f"  Unique operator types: {', '.join(sorted(op_types))}")
        except ImportError:
            logger.warning("  ONNX not available, skipping operator information")
        except Exception as e:
            logger.warning(f"  Error loading model: {e}")


def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # First create an ONNX model
    logger.info(f"Creating ONNX model for Whisper {args.model_size}...")
    original_model = create_onnx_model(args.model_size, args.output_dir)
    
    # Create a dictionary to keep track of model paths
    model_paths = {
        "Original": original_model,
        "Compressed": None
    }
    
    # Apply compression
    logger.info(f"Applying compression...")
    compressed_model = apply_compression(original_model, args)
    model_paths["Compressed"] = compressed_model
    
    # Display information about the models
    display_model_info(model_paths)
    
    # Benchmark if requested
    if args.benchmark:
        benchmark_results = benchmark(original_model, compressed_model)
        
        # Save benchmark results
        if benchmark_results:
            benchmark_path = os.path.join(args.output_dir, "benchmark_results.json")
            with open(benchmark_path, 'w') as f:
                json.dump(benchmark_results, f, indent=2)
            logger.info(f"Benchmark results saved to {benchmark_path}")
    
    logger.info("Demo completed successfully.")


if __name__ == "__main__":
    main() 