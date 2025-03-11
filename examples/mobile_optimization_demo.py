#!/usr/bin/env python3
"""
Mobile Optimization Demo

This script demonstrates how to use the mobile framework export capabilities
of the AMPTALK framework to optimize Whisper models for mobile deployment.

It shows:
1. How to export a Whisper model to TensorFlow Lite format
2. How to export a Whisper model to Core ML format (macOS only)
3. How to use the optimized models with the TranscriptionAgent

Usage:
    python examples/mobile_optimization_demo.py
"""

import os
import asyncio
import logging
import time
import platform
from typing import Dict, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mobile_optimization_demo")

# Add parent directory to path to import AMPTALK modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import AMPTALK modules
from src.core.utils.edge_optimization import (
    EdgeOptimizer, 
    OptimizationLevel, 
    DeviceTarget,
    MobileFramework
)
from src.agents.transcription_agent import TranscriptionAgent, ModelType, WhisperModelSize

# Check if TensorFlow and coremltools are available
TENSORFLOW_AVAILABLE = False
COREML_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
    logger.info("TensorFlow is available.")
except ImportError:
    logger.warning("TensorFlow is not available. TFLite demos will be skipped.")

if platform.system() == "Darwin":  # macOS
    try:
        import coremltools
        COREML_AVAILABLE = True
        logger.info("coremltools is available.")
    except ImportError:
        logger.warning("coremltools is not available. Core ML demos will be skipped.")
else:
    logger.warning("This is not a macOS system. Core ML demos will be skipped.")


async def export_model_to_tflite() -> Dict[str, Any]:
    """
    Export a Whisper model to TensorFlow Lite format.
    
    Returns:
        Dictionary containing optimization results
    """
    logger.info("Exporting Whisper tiny model to TensorFlow Lite...")
    
    # Initialize the edge optimizer
    optimizer = EdgeOptimizer(
        optimization_level=OptimizationLevel.MEDIUM,
        target_device=DeviceTarget.CPU,
        cache_dir=os.path.join(os.getcwd(), "models", "optimized")
    )
    
    # Optimize the model with TFLite export
    start_time = time.time()
    result = optimizer.optimize_whisper(
        model_size=WhisperModelSize.TINY,  # Use tiny model for speed in this demo
        language=None,  # Auto-detect language
        mobile_export=MobileFramework.TFLITE  # Export to TFLite
    )
    
    duration = time.time() - start_time
    logger.info(f"TFLite export completed in {duration:.1f} seconds.")
    
    # Log the result paths
    if "error" not in result:
        logger.info(f"TFLite model saved to: {result['tflite_model_path']}")
    else:
        logger.error(f"TFLite export failed: {result['error']}")
    
    return result


async def export_model_to_coreml() -> Optional[Dict[str, Any]]:
    """
    Export a Whisper model to Core ML format.
    
    Only works on macOS systems with coremltools installed.
    
    Returns:
        Dictionary containing optimization results or None if not applicable
    """
    if platform.system() != "Darwin":
        logger.warning("Core ML export is only supported on macOS.")
        return None
    
    logger.info("Exporting Whisper tiny model to Core ML...")
    
    # Initialize the edge optimizer
    optimizer = EdgeOptimizer(
        optimization_level=OptimizationLevel.MEDIUM,
        target_device=DeviceTarget.CPU,
        cache_dir=os.path.join(os.getcwd(), "models", "optimized")
    )
    
    # Optimize the model with Core ML export
    start_time = time.time()
    result = optimizer.optimize_whisper(
        model_size=WhisperModelSize.TINY,  # Use tiny model for speed in this demo
        language=None,  # Auto-detect language
        mobile_export=MobileFramework.COREML  # Export to Core ML
    )
    
    duration = time.time() - start_time
    logger.info(f"Core ML export completed in {duration:.1f} seconds.")
    
    # Log the result paths
    if "error" not in result:
        logger.info(f"Core ML models saved to: {result['coreml_model_path']}")
    else:
        logger.error(f"Core ML export failed: {result['error']}")
    
    return result


async def transcribe_with_tflite(tflite_model_path: str, audio_file: str) -> None:
    """
    Transcribe an audio file using a TensorFlow Lite model.
    
    Args:
        tflite_model_path: Path to the TFLite model file
        audio_file: Path to the audio file to transcribe
    """
    logger.info(f"Transcribing {audio_file} with TFLite model...")
    
    # Create a TranscriptionAgent with TFLite model
    agent = TranscriptionAgent(
        agent_id="tflite_demo_agent",
        model_size=WhisperModelSize.TINY,
        model_type=ModelType.WHISPER_TFLITE,
        model_path=tflite_model_path
    )
    
    # Initialize the agent
    await agent.initialize()
    
    # Transcribe the audio file
    start_time = time.time()
    result = await agent.transcribe(audio_file)
    duration = time.time() - start_time
    
    logger.info(f"Transcription completed in {duration:.1f} seconds.")
    logger.info(f"Transcription result: {result}")
    
    # Clean up
    await agent.shutdown()


async def transcribe_with_coreml(coreml_dir: str, audio_file: str) -> None:
    """
    Transcribe an audio file using a Core ML model.
    
    Note: This is a demonstration only. Core ML models can only be
    executed in Swift/Objective-C on macOS/iOS.
    
    Args:
        coreml_dir: Directory containing the Core ML models
        audio_file: Path to the audio file to transcribe
    """
    if platform.system() != "Darwin":
        logger.warning("Core ML is only supported on macOS.")
        return
    
    logger.info(f"Attempting to transcribe {audio_file} with Core ML model...")
    logger.info("Note: This is a demonstration only. Core ML models require an Objective-C/Swift runtime.")
    
    # Create a TranscriptionAgent with Core ML model
    agent = TranscriptionAgent(
        agent_id="coreml_demo_agent",
        model_size=WhisperModelSize.TINY,
        model_type=ModelType.WHISPER_COREML,
        model_path=coreml_dir
    )
    
    # Initialize the agent
    await agent.initialize()
    
    # Transcribe the audio file (will fall back to transformers in Python environment)
    start_time = time.time()
    result = await agent.transcribe(audio_file)
    duration = time.time() - start_time
    
    logger.info(f"Transcription completed in {duration:.1f} seconds.")
    logger.info(f"Transcription result: {result}")
    
    # Clean up
    await agent.shutdown()


async def main():
    """Run the mobile optimization demo."""
    logger.info("Starting Mobile Optimization Demo")
    
    # Ensure sample audio file exists
    sample_audio = os.path.join(os.getcwd(), "examples", "assets", "sample.mp3")
    if not os.path.exists(sample_audio):
        # Create examples/assets directory if it doesn't exist
        os.makedirs(os.path.dirname(sample_audio), exist_ok=True)
        
        logger.warning(f"Sample audio file {sample_audio} not found.")
        logger.info("Please place a sample audio file at this location to test transcription.")
        
        # Create a placeholder audio path for demonstration
        sample_audio = "path/to/your/audio/file.mp3"
    
    # Demo 1: Export to TensorFlow Lite
    if TENSORFLOW_AVAILABLE:
        try:
            tflite_result = await export_model_to_tflite()
            
            if "error" not in tflite_result:
                # Demo TFLite transcription
                await transcribe_with_tflite(
                    tflite_model_path=tflite_result["tflite_model_path"],
                    audio_file=sample_audio
                )
        except Exception as e:
            logger.error(f"TensorFlow Lite demo failed: {e}")
    
    # Demo 2: Export to Core ML (macOS only)
    if COREML_AVAILABLE:
        try:
            coreml_result = await export_model_to_coreml()
            
            if coreml_result and "error" not in coreml_result:
                # Demo Core ML transcription
                await transcribe_with_coreml(
                    coreml_dir=coreml_result["coreml_model_path"],
                    audio_file=sample_audio
                )
        except Exception as e:
            logger.error(f"Core ML demo failed: {e}")
    
    logger.info("Mobile Optimization Demo completed.")


if __name__ == "__main__":
    asyncio.run(main()) 