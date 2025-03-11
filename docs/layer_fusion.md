# Layer Fusion

## Overview

Layer fusion is a critical optimization technique for deploying deep learning models on edge devices. The AMPTALK Layer Fusion module implements state-of-the-art fusion patterns to combine multiple consecutive operations into a single optimized operation, significantly reducing memory transfers and computational overhead during inference.

## Key Features

- **Multiple Fusion Patterns**: Implements various fusion patterns tailored for transformer-based models
- **Multiple Backends**: Support for ONNX Runtime and PyTorch JIT fusion
- **Automatic Pattern Detection**: Automatically detects fusion opportunities in the model
- **Benchmarking Tools**: Built-in tools to measure performance improvements
- **Whisper-Optimized**: Specially designed fusion patterns for Whisper's architecture

## Fusion Patterns

The module implements the following fusion patterns:

### Basic Patterns

- **Attention QKV Fusion**: Fuses query, key, and value projections into a single operation
- **MLP Fusion**: Combines consecutive linear layers with activations
- **Layer Normalization Fusion**: Fuses add and layer normalization operations
- **GELU Fusion**: Optimizes GELU activation with adjacent operations

### Advanced Patterns

- **Attention Block Fusion**: Fuses entire self-attention blocks
- **Multi-Head Attention Fusion**: Optimizes multi-head attention operations
- **Encoder/Decoder Block Fusion**: Fuses complete encoder or decoder blocks

### Whisper-Specific Patterns

- **Convolution Layer Fusion**: Fuses convolution, activation, and batch normalization
- **Encoder-Decoder Bridge Fusion**: Optimizes the connection between encoder and decoder

## Usage

### Basic Example

```python
from src.models.layer_fusion import LayerFusion, FusionBackend, FusionPattern

# Initialize the fusion optimizer with a model
fusion = LayerFusion(
    model="path/to/model.onnx",  # Path to ONNX model or PyTorch model
    backend=FusionBackend.ONNX,  # Backend to use (ONNX or TORCH_JIT)
)

# Analyze fusion opportunities
opportunities = fusion.analyze_fusion_opportunities()
print(f"Fusion opportunities: {opportunities}")

# Apply fusion optimizations
optimized_model = fusion.apply_fusion("path/to/optimized_model.onnx")

# Save fusion metadata
fusion.save_fusion_metadata("path/to/metadata.json")
```

### Helper Function

```python
from src.models.layer_fusion import fuse_onnx_model, FusionPattern

# Apply fusion using the helper function
optimized_model = fuse_onnx_model(
    model_path="path/to/model.onnx",
    output_path="path/to/optimized_model.onnx",
    patterns=[
        FusionPattern.ATTENTION_QKV.value,
        FusionPattern.MLP_FUSION.value,
    ]
)
```

### Running the Demo

```bash
# Run the layer fusion demo with a tiny Whisper model
python examples/layer_fusion_demo.py --model-size tiny --benchmark

# Use PyTorch JIT backend
python examples/layer_fusion_demo.py --model-size tiny --backend torch_jit

# Specify output directory
python examples/layer_fusion_demo.py --model-size tiny --output-dir output/my_fusion
```

## Integration with Edge Optimization

The Layer Fusion module is integrated with the Edge Optimization pipeline in `src/core/utils/edge_optimization.py`. You can use the following code to apply layer fusion as part of the edge optimization process:

```python
from src.core.utils.edge_optimization import (
    EdgeOptimizer, 
    OptimizationType, 
    OptimizationLevel
)

# Create optimizer
optimizer = EdgeOptimizer(optimization_level=OptimizationLevel.HIGH)

# Apply optimizations including layer fusion
result = optimizer.optimize_whisper(
    model_size="tiny",
    optimizations=[
        OptimizationType.ONNX_CONVERSION,
        OptimizationType.LAYER_FUSION,
        OptimizationType.INT8_QUANTIZATION
    ]
)

print(f"Optimized model path: {result['model_path']}")
```

## Performance Improvements

Layer fusion can provide significant performance improvements, especially on edge devices with limited memory bandwidth. Typical improvements include:

- **Inference Speedup**: 1.5-3x faster inference
- **Memory Bandwidth Reduction**: 30-60% less memory transfers
- **Energy Efficiency**: 20-40% reduced power consumption

The actual improvements depend on the model architecture, fusion patterns applied, and target hardware.

## Implementation Details

### ONNX Runtime Fusion

For ONNX models, we leverage ONNX Runtime's built-in graph optimization capabilities and extend them with custom fusion patterns. The implementation:

1. Applies ONNX Runtime's graph-level optimizations
2. Implements custom pattern detection for transformer-specific patterns
3. Modifies the ONNX graph directly for advanced patterns

### PyTorch JIT Fusion

For PyTorch models, we use PyTorch's JIT compilation and freeze capabilities to apply fusion optimizations. The implementation:

1. Scripts the model using torch.jit.script
2. Applies optimization passes
3. Freezes the model for further optimizations

## Limitations

- Some fusion patterns are model-specific and may not work for all architectures
- ONNX graph modifications require in-depth knowledge of the model structure
- PyTorch JIT fusion may not be compatible with all custom operations

## References

- [ONNX Runtime Transformers Optimizer](https://onnxruntime.ai/docs/performance/transformers-optimization.html)
- [Efficient Whisper on Streaming Speech](https://arxiv.org/abs/2412.11272)
- [PyTorch TorchScript Optimization](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html) 