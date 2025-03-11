#!/usr/bin/env python3
"""
Operator Optimization Demo for AMPTALK.

This script demonstrates how to use the operator optimization module
to optimize ONNX models for specific hardware targets.

Usage:
    python examples/operator_optimization_demo.py --model-size tiny

Author: AMPTALK Team
Date: 2024
"""

import os
import argparse
import time
import subprocess
from pathlib import Path
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("operator_optimization_demo")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Operator Optimization Demo")
    parser.add_argument(
        "--model-size", 
        choices=["tiny", "base", "small", "medium", "large"],
        default="tiny",
        help="Whisper model size to use"
    )
    parser.add_argument(
        "--output-dir",
        default="output/operator_optimization",
        help="Directory to save optimized models"
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Hardware target (auto-detected if not specified)"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark after optimization"
    )
    parser.add_argument(
        "--compare-fusion",
        action="store_true",
        help="Compare with layer fusion optimization"
    )
    return parser.parse_args()

def create_onnx_model(model_size, output_dir):
    """Create an ONNX model for optimization."""
    try:
        from src.core.utils.edge_optimization import EdgeOptimizer, OptimizationType, DeviceTarget
        
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

def apply_operator_optimization(model_path, output_dir, target=None):
    """Apply operator optimization to the model."""
    try:
        from src.models.operator_optimization import (
            OptimizationTarget, 
            OperatorType, 
            optimize_operators,
            detect_optimal_target
        )
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Auto-detect target if not specified
        if target is None:
            optimization_target = detect_optimal_target()
            logger.info(f"Auto-detected optimization target: {optimization_target.value}")
        else:
            # Find the matching target
            try:
                optimization_target = next(t for t in OptimizationTarget 
                                         if t.value.lower() == target.lower())
            except StopIteration:
                logger.warning(f"Invalid target '{target}', auto-detecting instead")
                optimization_target = detect_optimal_target()
        
        # Set output path
        output_path = os.path.join(output_dir, "model_op_optimized.onnx")
        
        # Apply optimization
        logger.info(f"Applying operator optimization for target {optimization_target.value}")
        
        # Configure optimization
        config = {
            "cpu_threads": os.cpu_count(),
            "enable_tensorrt": False
        }
        
        # Optimize the model
        start_time = time.time()
        
        result = optimize_operators(
            model_path=model_path,
            output_path=output_path,
            target=optimization_target,
            config=config
        )
        
        elapsed = time.time() - start_time
        logger.info(f"Operator optimization completed in {elapsed:.2f} seconds")
        logger.info(f"Optimized model saved to {result}")
        
        return result
    except ImportError as e:
        logger.error(f"Error importing operator_optimization: {e}")
        raise

def apply_layer_fusion(model_path, output_dir):
    """Apply layer fusion to the model for comparison."""
    try:
        from src.models.layer_fusion import LayerFusion, FusionBackend, FusionPattern
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Set fusion backend
        fusion_backend = FusionBackend.ONNX
        
        # Configure fusion patterns
        config = {
            "enabled_patterns": [
                FusionPattern.ATTENTION_QKV.value,
                FusionPattern.MLP_FUSION.value,
                FusionPattern.LAYER_NORM_FUSION.value,
                FusionPattern.GELU_FUSION.value,
                FusionPattern.ATTENTION_BLOCK.value,
                FusionPattern.CONV_LAYER_FUSION.value
            ]
        }
        
        # Set output path
        output_path = os.path.join(output_dir, "model_layer_fusion.onnx")
        
        # Create fusion optimizer
        start_time = time.time()
        
        fusion = LayerFusion(
            model=model_path,
            backend=fusion_backend,
            config=config
        )
        
        # Apply fusion
        result = fusion.apply_fusion(output_path)
        
        elapsed = time.time() - start_time
        logger.info(f"Layer fusion completed in {elapsed:.2f} seconds")
        logger.info(f"Fused model saved to {result}")
        
        return result
    except ImportError as e:
        logger.error(f"Error importing layer_fusion: {e}")
        raise

def benchmark(original_model, optimized_model, layer_fusion_model=None):
    """Benchmark the original and optimized models."""
    logger.info("Running benchmark...")
    
    try:
        import onnxruntime as ort
        import numpy as np
        
        # Sample audio input shape for Whisper (80-dim log-mel spectrogram)
        input_shape = (1, 80, 3000)
        
        # Create sessions
        logger.info("Creating ONNX Runtime sessions...")
        original_session = ort.InferenceSession(original_model, providers=['CPUExecutionProvider'])
        optimized_session = ort.InferenceSession(optimized_model, providers=['CPUExecutionProvider'])
        
        layer_fusion_session = None
        if layer_fusion_model:
            layer_fusion_session = ort.InferenceSession(layer_fusion_model, providers=['CPUExecutionProvider'])
        
        # Get input name
        original_input_name = original_session.get_inputs()[0].name
        optimized_input_name = optimized_session.get_inputs()[0].name
        
        layer_fusion_input_name = None
        if layer_fusion_session:
            layer_fusion_input_name = layer_fusion_session.get_inputs()[0].name
        
        # Create random input
        logger.info("Creating test input...")
        input_data = np.random.rand(*input_shape).astype(np.float32)
        
        # Warm-up
        logger.info("Warming up...")
        for _ in range(5):
            original_session.run(None, {original_input_name: input_data})
            optimized_session.run(None, {optimized_input_name: input_data})
            if layer_fusion_session:
                layer_fusion_session.run(None, {layer_fusion_input_name: input_data})
        
        # Benchmark original
        logger.info("Benchmarking original model...")
        start_time = time.time()
        for _ in range(20):
            original_session.run(None, {original_input_name: input_data})
        original_time = (time.time() - start_time) / 20
        
        # Benchmark optimized
        logger.info("Benchmarking operator-optimized model...")
        start_time = time.time()
        for _ in range(20):
            optimized_session.run(None, {optimized_input_name: input_data})
        optimized_time = (time.time() - start_time) / 20
        
        # Benchmark layer fusion if provided
        layer_fusion_time = None
        if layer_fusion_session:
            logger.info("Benchmarking layer-fused model...")
            start_time = time.time()
            for _ in range(20):
                layer_fusion_session.run(None, {layer_fusion_input_name: input_data})
            layer_fusion_time = (time.time() - start_time) / 20
        
        # Report results
        logger.info(f"Original model inference time: {original_time:.4f} seconds")
        logger.info(f"Operator-optimized model inference time: {optimized_time:.4f} seconds")
        logger.info(f"Operator optimization speedup: {original_time / optimized_time:.2f}x")
        
        if layer_fusion_time:
            logger.info(f"Layer-fused model inference time: {layer_fusion_time:.4f} seconds")
            logger.info(f"Layer fusion speedup: {original_time / layer_fusion_time:.2f}x")
            logger.info(f"Operator optimization vs Layer fusion: {layer_fusion_time / optimized_time:.2f}x")
            
            # Determine which is better
            if optimized_time < layer_fusion_time:
                logger.info("Operator optimization is faster than layer fusion")
            elif layer_fusion_time < optimized_time:
                logger.info("Layer fusion is faster than operator optimization")
            else:
                logger.info("Operator optimization and layer fusion have similar performance")
        
        # Save benchmark results
        benchmark_results = {
            "original_time": original_time,
            "optimized_time": optimized_time,
            "optimized_speedup": original_time / optimized_time
        }
        
        if layer_fusion_time:
            benchmark_results.update({
                "layer_fusion_time": layer_fusion_time,
                "layer_fusion_speedup": original_time / layer_fusion_time,
                "op_vs_fusion": layer_fusion_time / optimized_time
            })
            
        return benchmark_results
        
    except ImportError as e:
        logger.error(f"Error importing onnxruntime: {e}")
        return None

def display_model_info(model_paths):
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
        "Operator-Optimized": None,
        "Layer-Fused": None
    }
    
    # Apply operator optimization
    logger.info(f"Applying operator optimization...")
    optimized_model = apply_operator_optimization(original_model, args.output_dir, args.target)
    model_paths["Operator-Optimized"] = optimized_model
    
    # Apply layer fusion if requested
    layer_fusion_model = None
    if args.compare_fusion:
        logger.info(f"Applying layer fusion for comparison...")
        layer_fusion_model = apply_layer_fusion(original_model, args.output_dir)
        model_paths["Layer-Fused"] = layer_fusion_model
    
    # Display information about the models
    display_model_info(model_paths)
    
    # Benchmark if requested
    if args.benchmark:
        benchmark_results = benchmark(original_model, optimized_model, layer_fusion_model)
        
        # Save benchmark results
        if benchmark_results:
            benchmark_path = os.path.join(args.output_dir, "benchmark_results.json")
            with open(benchmark_path, 'w') as f:
                json.dump(benchmark_results, f, indent=2)
            logger.info(f"Benchmark results saved to {benchmark_path}")
    
    logger.info("Demo completed successfully.")

if __name__ == "__main__":
    main() 