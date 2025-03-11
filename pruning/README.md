# AMPTALK Model Pruning

This directory contains tools for pruning Whisper speech recognition models using state-of-the-art techniques, including Outlier Weighted Layerwise Sparsity (OWL). Model pruning reduces model size and inference time while maintaining most of the original model's accuracy.

## Features

- **Advanced Pruning Techniques**: Implements the latest OWL pruning method (2024)
- **Comprehensive Evaluation**: Automatically evaluates model performance before and after pruning
- **Configurable**: Adjust pruning parameters through JSON configuration files
- **Optimized for Apple Silicon**: Takes advantage of MPS for local pruning on MacBooks
- **Docker Support**: Containerized environment for consistent pruning across different setups

## Directory Structure

```
pruning/
├── configs/       # Configuration files for pruning
├── models/        # Directory for model caching
├── output/        # Output directory for pruned models
├── scripts/       # Python scripts for model pruning
├── Dockerfile     # Container definition for pruning environment
├── docker-compose.yml  # Docker Compose configuration
├── requirements.txt    # Python dependencies
└── README.md      # This file
```

## Prerequisites

### Local Execution

- Python 3.9+
- PyTorch 2.0+
- 8GB+ RAM
- CUDA-compatible GPU (optional)

### Docker Execution

- Docker Desktop
- 16GB+ RAM recommended

## Running Model Pruning

### Option 1: Using Docker (Recommended)

1. **Build the Docker container:**

```bash
cd pruning
docker-compose build
```

2. **Run the pruning process:**

```bash
docker-compose up
```

This will run the pruning process with the default configuration. To use a different configuration file:

```bash
docker-compose run pruning python scripts/prune_whisper.py --config configs/custom_config.json
```

### Option 2: Local Execution

1. **Set up a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Run the pruning script:**

```bash
python scripts/prune_whisper.py --config configs/default.json
```

## Configuration Options

The pruning process is controlled by JSON configuration files in the `configs/` directory. Key configuration options include:

### Model Configuration

```json
"model": {
    "name": "openai/whisper-large-v3-turbo",
    "cache_dir": "../models/cache"
}
```

- `name`: Hugging Face model ID
- `cache_dir`: Directory to cache downloaded models

### Pruning Parameters

```json
"pruning": {
    "method": "magnitude",
    "sparsity": 0.3,
    "target_modules": ["encoder.layers", "decoder.layers"],
    "apply_outlier_weighting": true,
    "outlier_detection_method": "owl",
    "outlier_hyper_lambda": 0.08,
    "outlier_hyper_m": 5
}
```

- `method`: Pruning method (currently supports "magnitude")
- `sparsity`: Target sparsity ratio (0.0-1.0, where 0.3 = 30% of weights removed)
- `apply_outlier_weighting`: Whether to use Outlier Weighted Layerwise (OWL) technique
- `outlier_hyper_lambda`: Lambda hyperparameter for OWL (controls outlier detection threshold)
- `outlier_hyper_m`: m hyperparameter for OWL (controls outlier importance multiplier)

### Evaluation Settings

```json
"evaluation": {
    "datasets": [
        {
            "name": "librispeech",
            "split": "test-clean",
            "subset": "clean",
            "max_samples": 20
        }
    ],
    "metrics": ["wer", "character_error_rate"]
}
```

- `datasets`: List of datasets to use for evaluation
- `metrics`: Metrics to calculate (Word Error Rate, Character Error Rate)

### Hardware Settings

```json
"hardware": {
    "device": "auto",
    "use_mps": true,
    "precision": "float16",
    "cpu_threads": 8
}
```

- `device`: Computing device ("auto", "cuda", "cpu", or "mps")
- `use_mps`: Whether to use Apple Metal Performance Shaders for MacBooks with Apple Silicon
- `precision`: Numerical precision for model weights

## Understanding OWL Pruning

Outlier Weighted Layerwise Sparsity (OWL) is a state-of-the-art pruning technique that:

1. Identifies outlier weights (weights much larger than others) using a lambda-scaled median threshold
2. Assigns higher importance to these outliers during pruning
3. Applies magnitude-based pruning with these weighted importance scores

This technique preserves the most critical weights in the model, resulting in better post-pruning performance compared to standard magnitude pruning.

## Output

Pruned models are saved to the `output/` directory with:

- Model weights and configuration
- Pruning metadata (sparsity level, pruning method)
- Performance metrics comparison
- TensorBoard logs (if enabled)
- ONNX export (if enabled)

## References

- [Outlier Weighed Layerwise Sparsity (OWL): A Missing Secret Sauce for Pruning LLMs to High Sparsity](https://arxiv.org/abs/2309.03244)
- [Efficient Whisper on Streaming Speech](https://arxiv.org/abs/2412.11272)
- [OpenAI Whisper](https://github.com/openai/whisper) 