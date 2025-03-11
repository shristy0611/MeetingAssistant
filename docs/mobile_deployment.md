# Mobile Deployment Guide

This guide explains how to use the mobile deployment capabilities of the AMPTALK framework to export Whisper models to formats suitable for mobile platforms.

## Overview

The AMPTALK framework supports exporting optimized Whisper models to standard mobile deployment formats:

- **TensorFlow Lite**: For Android and cross-platform applications
- **Core ML**: For iOS and macOS applications

These optimized models can be significantly smaller and faster than the original models, making them suitable for running on mobile devices with limited computational resources.

## Prerequisites

To use the mobile export capabilities, you'll need the following dependencies:

- For TensorFlow Lite export:
  - `tensorflow>=2.14.0`
  - `tensorflow-hub>=0.15.0`

- For Core ML export:
  - `coremltools>=7.0.0`
  - A macOS system (Core ML tools only work on macOS)

These dependencies are automatically checked when you attempt to use the export functionality, and appropriate warnings will be displayed if they're missing.

## Exporting Models

### Using the EdgeOptimizer Directly

You can use the `EdgeOptimizer` class directly to export models:

```python
from src.core.utils.edge_optimization import (
    EdgeOptimizer,
    OptimizationLevel,
    DeviceTarget,
    MobileFramework
)

# Initialize the optimizer
optimizer = EdgeOptimizer(
    optimization_level=OptimizationLevel.MEDIUM,
    target_device=DeviceTarget.CPU,
    cache_dir="./models/optimized"
)

# Export to TensorFlow Lite
tflite_result = optimizer.optimize_whisper(
    model_size="tiny",
    language=None,  # Auto-detect language
    mobile_export=MobileFramework.TFLITE
)

# The result contains the path to the TFLite model
tflite_model_path = tflite_result["tflite_model_path"]

# Export to Core ML (macOS only)
coreml_result = optimizer.optimize_whisper(
    model_size="tiny",
    language=None,  # Auto-detect language
    mobile_export=MobileFramework.COREML
)

# The result contains the path to the Core ML models directory
coreml_dir = coreml_result["coreml_model_path"]
```

### Using the TranscriptionAgent

You can also use the `TranscriptionAgent` class with mobile-optimized models:

```python
from src.agents.transcription_agent import TranscriptionAgent, ModelType, WhisperModelSize

# Create an agent with a TFLite model
tflite_agent = TranscriptionAgent(
    agent_id="tflite_agent",
    model_size=WhisperModelSize.TINY,
    model_type=ModelType.WHISPER_TFLITE,
    model_path="/path/to/whisper_tiny.tflite"  # Optional, will be auto-generated if not provided
)

# Create an agent with a Core ML model
coreml_agent = TranscriptionAgent(
    agent_id="coreml_agent",
    model_size=WhisperModelSize.TINY,
    model_type=ModelType.WHISPER_COREML,
    model_path="/path/to/coreml_models_dir"  # Optional, will be auto-generated if not provided
)
```

## Output Model Formats

### TensorFlow Lite

The TensorFlow Lite export produces a single `.tflite` file containing the entire Whisper model, with these characteristics:

- Quantized to INT8 or other precision depending on optimization level
- Optimized for inference on mobile CPUs
- Includes necessary metadata for TensorFlow Lite runtime
- Saved with serialized signatures when possible for easier API usage

### Core ML

The Core ML export produces:

1. A directory containing:
   - `WhisperEncoder.mlpackage`: The encoder model
   - `WhisperDecoder.mlpackage`: The decoder model
   - `whisper_config.json`: Configuration information
   - `tokenizer/`: Directory containing the tokenizer files

The split encoder/decoder architecture follows the Whisper model's design and allows for efficient inference on iOS/macOS devices.

## Integration in Mobile Applications

### Android (TensorFlow Lite)

To use the exported TensorFlow Lite model in an Android application:

1. Copy the `.tflite` file to the `assets` folder of your Android project
2. Load the model using the TensorFlow Lite Android API:

```kotlin
// Load model from assets
val modelFile = "whisper_tiny.tflite"
val tfliteModel = FileUtil.loadMappedFile(context, modelFile)
val options = Interpreter.Options()
val interpreter = Interpreter(tfliteModel, options)

// Or with GPU delegation
val gpuDelegate = GpuDelegate()
options.addDelegate(gpuDelegate)
val interpreter = Interpreter(tfliteModel, options)

// Create input/output buffers
val inputBuffer = ByteBuffer.allocateDirect(...)
val outputBuffer = ByteBuffer.allocateDirect(...)

// Run inference
interpreter.run(inputBuffer, outputBuffer)
```

### iOS (Core ML)

To use the exported Core ML models in an iOS application:

1. Add the `.mlpackage` files to your Xcode project
2. Use the Core ML API to load and run the models:

```swift
import CoreML

// Load the models
let encoderURL = Bundle.main.url(forResource: "WhisperEncoder", withExtension: "mlpackage")!
let decoderURL = Bundle.main.url(forResource: "WhisperDecoder", withExtension: "mlpackage")!

let encoderModel = try MLModel(contentsOf: encoderURL)
let decoderModel = try MLModel(contentsOf: decoderURL)

// Create model wrappers (assuming you have model classes generated by Xcode)
let encoder = try WhisperEncoder(model: encoderModel)
let decoder = try WhisperDecoder(model: decoderModel)

// Run inference
let encoderInput = WhisperEncoderInput(input_features: features)
let encoderOutput = try encoder.prediction(input: encoderInput)

// Use encoder output for decoder input
// ...
```

## Performance Considerations

1. **Memory Usage**: Mobile-optimized models are significantly smaller, but still require careful memory management
2. **Battery Impact**: Running inference on-device consumes battery - consider batching operations
3. **Accuracy vs Speed**: Higher optimization levels increase speed but may decrease accuracy
4. **Temperature Management**: Extended inference can heat up mobile devices - consider cooling intervals
5. **Background Processing**: For long transcriptions, use background processing to avoid UI freezes

## Example Code

See the `examples/mobile_optimization_demo.py` script for a complete example of exporting and using mobile-optimized models.

## Troubleshooting

### Common Issues with TensorFlow Lite

1. **Compatibility Errors**: TFLite models are version-specific. Ensure your TF version on mobile matches the version used for export.
2. **Memory Errors**: If you encounter OOM errors, try reducing the input size or using a smaller model.
3. **Slow Inference**: Enable delegate options for hardware acceleration when available.

### Common Issues with Core ML

1. **iOS Version Compatibility**: Core ML models may require specific iOS versions.
2. **Model Loading Failures**: Check that the model files are properly integrated into your app bundle.
3. **Input Shape Mismatches**: Ensure your input tensors match exactly what the model expects.

## Limitations

1. Core ML export is only available on macOS systems.
2. Not all Whisper model optimizations are compatible with mobile export.
3. Extremely large Whisper models (large, large-v2) may not be practical for mobile deployment.
4. Very small devices may still struggle with even the optimized models - testing is recommended. 