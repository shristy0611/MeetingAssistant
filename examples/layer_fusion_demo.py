#!/usr/bin/env python3
"""
Layer Fusion Demo for AMPTALK.

This script demonstrates how to use the layer fusion module
to optimize a model for edge deployment.

Usage:
    python examples/layer_fusion_demo.py --model-size tiny

Author: AMPTALK Team
Date: 2024
"""

import os
import argparse
import time
import torch
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("layer_fusion_demo")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Layer Fusion Demo")
    parser.add_argument(
        "--model-size", 
        choices=["tiny", "base", "small", "medium", "large"],
        default="tiny",
        help="Whisper model size to use"
    )
    parser.add_argument(
        "--output-dir",
        default="output/fusion",
        help="Directory to save optimized models"
    )
    parser.add_argument(
        "--backend",
        choices=["onnx", "torch_jit"],
        default="onnx",
        help="Fusion backend to use"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark after fusion"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to use (cpu, cuda, mps)"
    )
    return parser.parse_args()

def create_onnx_model(model_size, output_dir):
    """Create an ONNX model for fusion."""
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

def apply_fusion(model_path, output_dir, backend="onnx", device="cpu"):
    """Apply fusion to the model."""
    try:
        from src.models.layer_fusion import LayerFusion, FusionBackend, FusionPattern
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Set fusion backend
        if backend == "onnx":
            fusion_backend = FusionBackend.ONNX
        else:
            fusion_backend = FusionBackend.TORCH_JIT
        
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
        
        # Create fusion optimizer
        fusion = LayerFusion(
            model=model_path,
            backend=fusion_backend,
            config=config,
            device=device
        )
        
        # Analyze fusion opportunities
        opportunities = fusion.analyze_fusion_opportunities()
        logger.info(f"Fusion opportunities: {opportunities}")
        
        # Apply fusion
        output_path = os.path.join(output_dir, "model_fused.onnx")
        result = fusion.apply_fusion(output_path)
        
        # Save metadata
        metadata_path = os.path.join(output_dir, "fusion_metadata.json")
        fusion.save_fusion_metadata(metadata_path)
        
        logger.info(f"Fusion completed. Optimized model saved to {result}")
        
        return result
    except ImportError as e:
        logger.error(f"Error importing layer_fusion: {e}")
        raise

def benchmark(original_model, fused_model, backend="onnx", device="cpu"):
    """Benchmark the original and fused models."""
    logger.info("Running benchmark...")
    
    # Sample audio input shape for Whisper (80-dim log-mel spectrogram)
    input_shape = (1, 80, 3000)
    
    if backend == "onnx":
        try:
            import onnxruntime as ort
            import numpy as np
            
            # Create sessions
            original_session = ort.InferenceSession(original_model, providers=['CPUExecutionProvider'])
            fused_session = ort.InferenceSession(fused_model, providers=['CPUExecutionProvider'])
            
            # Get input name
            original_input_name = original_session.get_inputs()[0].name
            fused_input_name = fused_session.get_inputs()[0].name
            
            # Create random input
            input_data = np.random.rand(*input_shape).astype(np.float32)
            
            # Warm-up
            for _ in range(5):
                original_session.run(None, {original_input_name: input_data})
                fused_session.run(None, {fused_input_name: input_data})
            
            # Benchmark original
            start_time = time.time()
            for _ in range(20):
                original_session.run(None, {original_input_name: input_data})
            original_time = (time.time() - start_time) / 20
            
            # Benchmark fused
            start_time = time.time()
            for _ in range(20):
                fused_session.run(None, {fused_input_name: input_data})
            fused_time = (time.time() - start_time) / 20
            
            # Report results
            logger.info(f"Original model inference time: {original_time:.4f} seconds")
            logger.info(f"Fused model inference time: {fused_time:.4f} seconds")
            logger.info(f"Speedup: {original_time / fused_time:.2f}x")
            
        except ImportError as e:
            logger.error(f"Error importing onnxruntime: {e}")
    else:
        logger.info("Benchmarking PyTorch JIT models not implemented yet")

def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # First create an ONNX model
    logger.info(f"Creating ONNX model for Whisper {args.model_size}...")
    original_model = create_onnx_model(args.model_size, args.output_dir)
    
    # Apply fusion
    logger.info(f"Applying fusion with backend {args.backend}...")
    fused_model = apply_fusion(original_model, args.output_dir, args.backend, args.device)
    
    # Benchmark if requested
    if args.benchmark:
        benchmark(original_model, fused_model, args.backend, args.device)
    
    logger.info("Demo completed successfully.")

if __name__ == "__main__":
    main() 