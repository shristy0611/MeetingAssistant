# Whisper Model Integration in AMPTALK

This document provides an overview of the Whisper speech recognition model integration in the AMPTALK framework, including implementation details, optimization techniques, and usage examples.

## Overview

OpenAI's Whisper is a robust, multilingual automatic speech recognition (ASR) system that has been integrated into the AMPTALK framework to provide high-quality speech transcription capabilities. The integration uses the `faster-whisper` library, which is an optimized version of Whisper that leverages CTranslate2 for better performance.

## Features

- **Multiple Model Sizes**: Support for all Whisper model sizes from tiny to large-v3-turbo
- **Performance Optimization**: Optimized implementation using faster-whisper for up to 4x faster inference
- **Hardware Acceleration**: Support for CPU, CUDA, and MPS (Metal Performance Shaders) acceleration
- **Quantization Support**: Options for fp16, int8, and dynamic precision settings
- **Model Caching**: Smart model caching system to avoid redundant model loading
- **Memory Management**: Automatic memory tracking and efficient resource utilization
- **Multilingual Support**: Full support for all 99 languages supported by Whisper
- **Word Timestamps**: Optional word-level timestamp generation

## Architecture

The Whisper integration in AMPTALK consists of the following key components:

1. **TranscriptionAgent**: The main agent responsible for transcribing audio data using Whisper
2. **WhisperModelConfig**: Configuration class for the Whisper model parameters
3. **ModelCache**: Singleton class for efficient model management

The architecture follows a modular design where the TranscriptionAgent receives processed audio segments from the AudioProcessingAgent and transcribes them using the Whisper model. The results are then sent back as messages in the AMPTALK framework.

## Configuration Options

The Whisper integration can be configured through the `whisper` section in the agent configuration:

```python
config = {
    "whisper": {
        "model_size": "large-v3-turbo",  # Model size: tiny, base, small, medium, large-v3, large-v3-turbo
        "device": "cuda",                # Device: cpu, cuda, mps, auto
        "compute_type": "float16",       # Precision: float16, int8, auto
        "language": "en",                # Optional language code (e.g., en, ja, fr)
        "beam_size": 5,                  # Beam size for beam search
        "word_timestamps": True          # Whether to include word-level timestamps
    }
}

# Create agent with configuration
transcription_agent = TranscriptionAgent(config=config)
```

## Model Caching System

The AMPTALK framework includes a sophisticated model caching system that optimizes the loading and management of Whisper models:

- **Singleton Pattern**: A global cache is shared across the application
- **LRU Eviction**: Least Recently Used models are evicted when the cache is full
- **Memory Tracking**: The system tracks memory usage and can enforce memory limits
- **Automatic Cleanup**: Orphaned models are automatically cleaned up
- **Owner References**: Models retain references to their owners for proper cleanup

The caching system can be configured through environment variables:

- `AMPTALK_MODEL_CACHE_SIZE`: Maximum number of models to keep in cache (default: 5)
- `AMPTALK_MODEL_CACHE_TTL`: Time-to-live for cached models in seconds (default: 3600)
- `AMPTALK_MODEL_CACHE_MEMORY_LIMIT`: Memory limit for the cache in MB (default: 0, unlimited)

## Edge Device Optimizations

For deployment on edge devices, the following optimizations are available:

1. **Model Quantization**: Using int8 precision significantly reduces memory requirements
2. **Smaller Model Variants**: Tiny and base models are suitable for edge deployment
3. **Reduced Beam Size**: Setting beam_size=1 reduces computation at the cost of accuracy
4. **Batch Processing**: Processing audio in larger chunks can improve throughput

## Performance Benchmarks

Performance varies significantly depending on the model size, hardware, and configuration:

| Model Size | Precision | Device | Real-Time Factor | Memory Usage |
|------------|-----------|--------|------------------|--------------|
| tiny       | int8      | CPU    | ~2-3x            | ~125MB       |
| base       | int8      | CPU    | ~3-4x            | ~250MB       |
| small      | int8      | CPU    | ~5-7x            | ~580MB       |
| medium     | int8      | CPU    | ~8-12x           | ~1.5GB       |
| large-v3   | int8      | CPU    | ~15-20x          | ~3GB         |
| tiny       | float16   | CUDA   | ~0.1-0.2x        | ~250MB       |
| base       | float16   | CUDA   | ~0.2-0.3x        | ~500MB       |
| small      | float16   | CUDA   | ~0.3-0.5x        | ~1GB         |
| medium     | float16   | CUDA   | ~0.5-0.8x        | ~2.5GB       |
| large-v3   | float16   | CUDA   | ~0.8-1.2x        | ~5GB         |

*Real-time factor represents how many seconds of processing time are needed per second of audio. Values < 1 indicate faster-than-real-time processing.*

## Usage Example

Here's a simple example of how to use the Whisper integration in AMPTALK:

```python
import asyncio
from src.core.framework.message import Message, MessageType
from src.core.framework.orchestrator import Orchestrator
from src.agents.audio_processing_agent import AudioProcessingAgent
from src.agents.transcription_agent import TranscriptionAgent

async def main():
    # Create orchestrator
    orchestrator = Orchestrator()
    
    # Create and configure agents
    audio_agent = AudioProcessingAgent()
    transcription_agent = TranscriptionAgent(
        config={"whisper": {"model_size": "medium", "device": "cuda"}}
    )
    
    # Register and connect agents
    orchestrator.register_agent(audio_agent)
    orchestrator.register_agent(transcription_agent)
    orchestrator.connect_agents(audio_agent.agent_id, transcription_agent.agent_id)
    
    # Start the orchestrator
    await orchestrator.start()
    
    # Create audio input message (with your audio data)
    audio_message = Message(
        message_type=MessageType.AUDIO_INPUT,
        source_agent_id="user",
        target_agent_id=audio_agent.agent_id,
        payload={"audio_data": your_audio_data, "sample_rate": 16000}
    )
    
    # Send the message
    await orchestrator.send_message_to_agent(audio_message)
    
    # In a real application, you would register handlers for transcription results

if __name__ == "__main__":
    asyncio.run(main())
```

For a more comprehensive example, see `examples/whisper_integration_demo.py`.

## Troubleshooting

Common issues and their solutions:

1. **ImportError for faster-whisper**: Install the required dependencies with `pip install faster-whisper`
2. **CUDA errors**: Ensure you have compatible CUDA libraries installed (see requirements.txt)
3. **High memory usage**: Use a smaller model size or enable int8 quantization
4. **Slow transcription**: Enable GPU acceleration or use a smaller model
5. **Model download issues**: Set `WHISPER_MODEL_DIR` environment variable to a writable directory

## Future Enhancements

Planned improvements for the Whisper integration:

1. **Streaming Transcription**: Support for real-time streaming transcription
2. **ONNX Export**: Export to ONNX format for wider hardware compatibility
3. **Fine-tuning**: Support for fine-tuned models on specific domains
4. **Distilled Models**: Integration of distilled/smaller models for edge devices
5. **WebAssembly Support**: Browser-based transcription capabilities 