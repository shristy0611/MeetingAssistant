# AMPTALK: Multi-Agent AI Framework for Meeting Transcription

AMPTALK is a privacy-focused multi-agent system designed for meeting transcription and analysis. It enables real-time transcription, summarization, and insight extraction from audio with minimal data exposure.

## Features

- **Privacy-First Design**: Process all data locally without sending it to external servers
- **Multi-Agent Architecture**: Specialized agents work together to handle different tasks
- **Real-Time Processing**: Process audio streams in real-time with low latency
- **Optimized for Edge**: Designed to run efficiently on local devices
- **Modular and Extensible**: Easy to add new agents or functionality

### Core Framework
- **Robust Agent Architecture**: Flexible and extensible agent-based system
- **Asynchronous Execution**: Efficient processing of concurrent tasks
- **Error Recovery**: Sophisticated error handling with retry mechanisms
- **Memory Management**: Smart caching and memory optimization
- **State Persistence**: Persistent agent state across sessions

### Whisper Integration
- **Transcription Agent**: Specialized agent for audio transcription
- **Model Caching**: Efficient model loading and unloading
- **Edge Optimization**: Deployment optimizations for resource-constrained devices
  - ONNX conversion
  - Quantization (INT8, FP16, INT4)
  - Layer Fusion: Combines consecutive operations for faster inference
  - Operator fusion
  - Knowledge distillation
- **Mobile Framework Export**: Deploy models to mobile platforms
  - TensorFlow Lite export for Android
  - Core ML export for iOS/macOS
  - Model size optimization
  - Mobile-specific optimizations

### Monitoring and Performance
- **Real-time Metrics**: Comprehensive performance monitoring
- **OpenTelemetry Integration**: Industry-standard observability
- **Prometheus Exporting**: Metrics collection for dashboards

## System Architecture

AMPTALK uses a multi-agent architecture where specialized agents communicate through a message-passing system:

1. **Audio Processing Agent**: Handles audio input, preprocessing, and segmentation
2. **Transcription Agent**: Converts audio to text using optimized Whisper models
3. **NLP Processing Agent**: Performs language analysis, entity detection, and intent recognition
4. **Summarization Agent**: Generates concise summaries of meeting content
5. **Orchestrator**: Coordinates the agents and manages system resources

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/amptalk.git
cd amptalk

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Demo

```bash
# Run the basic demo
python src/run_demo.py

# Run with a custom configuration
python src/run_demo.py --config path/to/config.json
```

### Example Configuration

Create a JSON configuration file to customize the system:

```json
{
  "audio_agent": {
    "audio": {
      "sample_rate": 16000,
      "chunk_duration_ms": 1000,
      "vad_threshold": 0.3
    }
  },
  "transcription_agent": {
    "whisper": {
      "model_size": "large-v3-turbo",
      "device": "cuda",
      "language": "en"
    }
  }
}
```

## Model Pruning

AMPTALK includes a model pruning toolkit in the `pruning/` directory to optimize models for edge deployment:

```bash
# Run the pruning script
cd pruning
./prune.sh --model large-v3 --target-sparsity 0.6
```

## Development

### Project Structure

```
amptalk/
├── src/
│   ├── agents/              # Specialized agent implementations
│   ├── core/
│   │   ├── framework/       # Core multi-agent framework
│   │   └── utils/           # Common utilities
│   └── run_demo.py          # Demo application
├── pruning/                 # Model pruning tools
│   ├── scripts/             # Pruning scripts
│   ├── configs/             # Pruning configurations
│   └── models/              # Model storage
├── requirements.txt         # Dependencies
└── README.md                # This file
```

### Creating a New Agent

Agents inherit from the `Agent` base class in `src/core/framework/agent.py`:

```python
from src.core.framework.agent import Agent

class MyCustomAgent(Agent):
    def __init__(self, agent_id=None, name="MyCustomAgent", config=None):
        super().__init__(agent_id, name)
        # Initialize your agent
        
    async def process_message(self, message):
        # Implement your message handling logic
        pass
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for the Whisper model
- The open-source community for various audio processing tools
- All contributors to this project

## Getting Started

### Edge Optimization Example

For optimizing and running Whisper models on edge devices:

```bash
# Run the edge optimization demo
python examples/edge_optimization_demo.py --model-size tiny --optimization-level MEDIUM --compare
```

This will:
1. Optimize a Whisper model using ONNX and quantization
2. Compare performance with the non-optimized model
3. Run a transcription using both models

For more details, see the [Edge Optimization Documentation](docs/edge_optimization.md).

### INT4 Quantization Example

For ultra-low precision quantization with AWQ:

```bash
# Run the INT4 quantization demo
python examples/int4_quantization_demo.py --model-size tiny
```

This will:
1. Quantize a Whisper model to INT4 precision using AWQ
2. Quantize the same model to INT8 precision for comparison
3. Compare transcription performance and accuracy between original, INT4, and INT8 models

INT4 quantization can reduce model size by up to 8x while maintaining good transcription quality.

### Mobile Framework Export Example

For exporting Whisper models to mobile frameworks:

```bash
# Run the mobile optimization demo
python examples/mobile_optimization_demo.py
```

This will:
1. Export a Whisper model to TensorFlow Lite format (for Android)
2. Export a Whisper model to Core ML format (for iOS/macOS, only on macOS systems)
3. Demonstrate transcription using the exported models

For detailed information on mobile deployment, see the [Mobile Deployment Guide](docs/mobile_deployment.md). 