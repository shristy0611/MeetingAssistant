# Model Compression

## Overview

The AMPTALK Model Compression module provides state-of-the-art techniques for compressing neural networks while maintaining performance. This module is designed to work with the Edge Optimization pipeline and supports various compression methods that can be used individually or in combination.

## Key Features

- **Multiple Compression Techniques**:
  - Weight Sharing (cluster-based, hash-based, Bloom filter)
  - Huffman Coding for weight compression
  - Low-Rank Factorization
  - Mixed-Precision Quantization
  - Hybrid Compression
  - Evolutionary Search
  - Wanda Pruning (Weights AND Activations)

- **Flexible Configuration**:
  - Configurable compression ratios
  - Multiple sharing strategies
  - Layer-specific compression settings
  - Customizable compression pipelines

- **Comprehensive Metrics**:
  - Size reduction measurements
  - Performance benchmarking
  - Compression statistics
  - Model analysis tools

## Installation

The model compression module is part of the AMPTALK package. To use it, ensure you have the required dependencies:

```bash
pip install torch numpy scipy scikit-learn
```

For ONNX support (recommended):
```bash
pip install onnx onnxruntime
```

## Usage

### Basic Example

```python
from src.models.model_compression import (
    CompressionType,
    SharingStrategy,
    CompressionConfig,
    ModelCompressor
)

# Create configuration
config = CompressionConfig(
    compression_types=[CompressionType.WEIGHT_SHARING],
    target_ratio=0.5,
    sharing_strategy=SharingStrategy.CLUSTER,
    num_clusters=256
)

# Create compressor and compress model
compressor = ModelCompressor("path/to/model.pt", config)
compressed_model = compressor.compress()

# Save compressed model
compressor.save_compressed_model("compressed_model.pt")
```

### Multiple Compression Techniques

```python
# Configure multiple compression techniques
config = CompressionConfig(
    compression_types=[
        CompressionType.WEIGHT_SHARING,
        CompressionType.LOW_RANK,
        CompressionType.MIXED_PRECISION
    ],
    target_ratio=0.3,
    sharing_strategy=SharingStrategy.ADAPTIVE,
    rank_ratio=0.5,
    mixed_precision_config={
        "conv": 8,    # 8-bit for conv layers
        "linear": 4   # 4-bit for linear layers
    }
)

# Apply compression
compressor = ModelCompressor(model, config)
compressed_model = compressor.compress()
```

### Using the Demo Script

```bash
# Basic usage
python examples/model_compression_demo.py --model-size tiny

# Advanced usage with multiple techniques
python examples/model_compression_demo.py \
    --model-size base \
    --compression-types weight_sharing low_rank \
    --target-ratio 0.3 \
    --sharing-strategy cluster \
    --num-clusters 128 \
    --rank-ratio 0.5 \
    --benchmark
```

## Compression Techniques

### 1. Weight Sharing

Weight sharing reduces model size by making multiple weights share the same value. The module supports several strategies:

- **Cluster-based**: Uses k-means clustering to group similar weights
- **Hash-based**: Uses hash functions to assign weights to shared values
- **Bloom Filter**: Uses probabilistic data structures for efficient sharing
- **Adaptive**: Dynamically adjusts sharing based on layer importance

### 2. Low-Rank Factorization

Decomposes weight matrices into smaller matrices with lower ranks:

```python
# Example configuration for low-rank compression
config = CompressionConfig(
    compression_types=[CompressionType.LOW_RANK],
    rank_ratio=0.3  # Keep 30% of singular values
)
```

### 3. Mixed-Precision Quantization

Applies different quantization bit-widths to different layers based on their sensitivity:

```python
# Example configuration for mixed-precision
config = CompressionConfig(
    compression_types=[CompressionType.MIXED_PRECISION],
    mixed_precision_config={
        "embedding": 8,
        "attention": 8,
        "ffn": 4,
        "output": 8
    }
)
```

### 4. Wanda Pruning

Implements the Weights AND Activations pruning technique for efficient compression:

```python
# Example configuration for Wanda pruning
config = CompressionConfig(
    compression_types=[CompressionType.WANDA],
    wanda_threshold=0.1  # Pruning threshold
)
```

## Integration with Edge Optimization

The Model Compression module integrates seamlessly with the Edge Optimization pipeline:

```python
from src.core.utils.edge_optimization import EdgeOptimizer, OptimizationType

# Create edge optimizer
optimizer = EdgeOptimizer()

# Apply compression as part of optimization
result = optimizer.optimize_whisper(
    model_size="tiny",
    optimizations=[
        OptimizationType.ONNX_CONVERSION,
        OptimizationType.MODEL_COMPRESSION
    ],
    compression_config={
        "types": ["weight_sharing", "low_rank"],
        "target_ratio": 0.5
    }
)
```

## Benchmarking

The module includes tools for benchmarking compressed models:

```python
# Run benchmark
python examples/model_compression_demo.py \
    --model-size tiny \
    --compression-types weight_sharing low_rank \
    --benchmark
```

This will generate a benchmark report including:
- Inference time comparison
- Memory usage
- Model size reduction
- Layer-wise statistics

## Best Practices

1. **Choosing Compression Techniques**:
   - Start with weight sharing for initial size reduction
   - Add low-rank factorization for compute-intensive layers
   - Use mixed-precision for fine-grained control
   - Apply Wanda pruning for additional optimization

2. **Optimizing for Performance**:
   - Monitor accuracy during compression
   - Use layer-specific compression settings
   - Combine complementary techniques
   - Benchmark on target hardware

3. **Integration Tips**:
   - Convert to ONNX format first
   - Apply compression before quantization
   - Use adaptive strategies for unknown models
   - Save compression metadata for analysis

## Limitations

- Some techniques may not be suitable for all model architectures
- Compression can affect model accuracy
- Hardware-specific optimizations may not transfer
- Complex compression pipelines may increase inference time

## References

1. "Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding" (Han et al.)
2. "Learning both Weights and Connections for Efficient Neural Networks" (Han et al.)
3. "Wanda: Weight AND Activations Pruning for Efficient LLM Inference" (2024)
4. "Low-Rank Matrix Factorization for Deep Neural Network Compression" (2023)
5. "Adaptive Model Compression for Edge Deployment" (2024) 