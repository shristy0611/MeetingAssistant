#!/usr/bin/env python3
"""
INT4 Quantization Demo

This script demonstrates how to use the INT4 quantization capabilities
of the AMPTALK framework to create ultra-low precision models for edge deployment.

It shows:
1. How to quantize a Whisper model to INT4 precision using AWQ
2. How to use the quantized models with the TranscriptionAgent
3. How to compare the performance and accuracy of different quantization methods

Usage:
    python examples/int4_quantization_demo.py [--model-size tiny] [--language en]
"""

import os
import sys
import time
import argparse
import asyncio
import logging
import tempfile
from typing import Dict, Optional, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("int4_quantization_demo")

# Add parent directory to path to import AMPTALK modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import AMPTALK modules
from src.core.utils.edge_optimization import (
    EdgeOptimizer, 
    OptimizationLevel, 
    DeviceTarget,
    OptimizationType
)
from src.agents.transcription_agent import (
    TranscriptionAgent, 
    ModelType, 
    WhisperModelSize
)

# Check if AWQ dependencies are available
AWQ_AVAILABLE = False
try:
    # Try to import AutoAWQ
    import autoawq
    AWQ_AVAILABLE = True
    logger.info("AutoAWQ is available.")
except ImportError:
    try:
        # Try to import LLM-AWQ as an alternative
        import awq
        AWQ_AVAILABLE = True
        logger.info("LLM-AWQ is available.")
    except ImportError:
        logger.warning("Neither AutoAWQ nor LLM-AWQ is available. INT4 quantization will be simulated.")


async def quantize_with_int4(
    model_size: str, 
    language: Optional[str] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Quantize a Whisper model to INT4 precision using AWQ.
    
    Args:
        model_size: Size of the Whisper model (tiny, base, small, etc.)
        language: Optional language code (e.g., 'en' for English)
        output_dir: Directory to save the quantized model
        
    Returns:
        Dictionary with the optimization results
    """
    logger.info(f"Quantizing Whisper {model_size} model to INT4 precision")
    
    # Initialize the edge optimizer
    optimizer = EdgeOptimizer(
        optimization_level=OptimizationLevel.MEDIUM,
        target_device=DeviceTarget.CPU,
        cache_dir=os.path.join(os.getcwd(), "models", "optimized")
    )
    
    # If output_dir is not specified, create one
    if output_dir is None:
        output_dir = os.path.join(
            os.getcwd(), 
            "models", 
            "optimized", 
            f"whisper-{model_size}{'-' + language if language else ''}-int4"
        )
        os.makedirs(output_dir, exist_ok=True)
    
    # Set specific optimization to INT4_QUANTIZATION
    optimizations = [OptimizationType.INT4_QUANTIZATION]
    
    # Optimize the model with INT4 quantization
    start_time = time.time()
    result = optimizer.optimize_whisper(
        model_size=model_size,
        language=language,
        compute_type="int4",  # Ensure INT4 is selected
        optimizations=optimizations,
        output_dir=output_dir
    )
    
    duration = time.time() - start_time
    logger.info(f"INT4 quantization completed in {duration:.1f} seconds")
    
    # Log the result
    if "error" not in result:
        logger.info(f"Quantized model saved to: {result['model_path']}")
        logger.info(f"Quantization type: {result.get('quant_type', 'int4')}")
    else:
        logger.error(f"INT4 quantization failed: {result['error']}")
    
    return result


async def quantize_with_int8(
    model_size: str, 
    language: Optional[str] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Quantize a Whisper model to INT8 precision using ONNX.
    
    Args:
        model_size: Size of the Whisper model (tiny, base, small, etc.)
        language: Optional language code (e.g., 'en' for English)
        output_dir: Directory to save the quantized model
        
    Returns:
        Dictionary with the optimization results
    """
    logger.info(f"Quantizing Whisper {model_size} model to INT8 precision")
    
    # Initialize the edge optimizer
    optimizer = EdgeOptimizer(
        optimization_level=OptimizationLevel.MEDIUM,
        target_device=DeviceTarget.CPU,
        cache_dir=os.path.join(os.getcwd(), "models", "optimized")
    )
    
    # If output_dir is not specified, create one
    if output_dir is None:
        output_dir = os.path.join(
            os.getcwd(), 
            "models", 
            "optimized", 
            f"whisper-{model_size}{'-' + language if language else ''}-int8"
        )
        os.makedirs(output_dir, exist_ok=True)
    
    # Set optimizations to ONNX + INT8
    optimizations = [
        OptimizationType.ONNX_CONVERSION,
        OptimizationType.INT8_QUANTIZATION
    ]
    
    # Optimize the model
    start_time = time.time()
    result = optimizer.optimize_whisper(
        model_size=model_size,
        language=language,
        compute_type="int8",
        optimizations=optimizations,
        output_dir=output_dir
    )
    
    duration = time.time() - start_time
    logger.info(f"INT8 quantization completed in {duration:.1f} seconds")
    
    # Log the result
    if "error" not in result:
        logger.info(f"Quantized model saved to: {result['model_path']}")
        logger.info(f"Quantization type: {result.get('quant_type', 'int8')}")
    else:
        logger.error(f"INT8 quantization failed: {result['error']}")
    
    return result


async def transcribe_and_compare(
    model_size: str,
    language: Optional[str] = None,
    audio_file: str = None,
    int4_model_path: Optional[str] = None,
    int8_model_path: Optional[str] = None
) -> None:
    """
    Transcribe an audio file using both INT4 and INT8 quantized models and compare results.
    
    Args:
        model_size: Size of the Whisper model
        language: Optional language code
        audio_file: Path to the audio file to transcribe
        int4_model_path: Path to the INT4 quantized model
        int8_model_path: Path to the INT8 quantized model
    """
    logger.info(f"Comparing transcription with different quantization levels")
    
    # Determine the audio file path
    if audio_file is None or not os.path.exists(audio_file):
        # Create a temporary audio file with silence if needed
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_file = tmp.name
            logger.warning(f"No valid audio file provided, using temporary file: {audio_file}")
            # We'd normally create a real audio file here, but for demo purposes we'll just show the flow
    else:
        logger.info(f"Using audio file: {audio_file}")
    
    # Track transcription times and results
    results = {}
    
    # 1. First, transcribe with the original HF model for reference
    logger.info("Transcribing with original Hugging Face model...")
    
    try:
        original_agent = TranscriptionAgent(
            agent_id="original_model",
            model_size=model_size,
            model_type=ModelType.WHISPER_TRANSFORMERS,
            language=language
        )
        
        await original_agent.initialize()
        
        start_time = time.time()
        original_result = await original_agent.transcribe(audio_file)
        original_time = time.time() - start_time
        
        logger.info(f"Original model transcription completed in {original_time:.2f} seconds")
        logger.info(f"Original result: {original_result}")
        
        results["original"] = {
            "time": original_time,
            "result": original_result
        }
        
        await original_agent.shutdown()
    except Exception as e:
        logger.error(f"Error transcribing with original model: {e}")
    
    # 2. Transcribe with INT4 model if available
    if int4_model_path and AWQ_AVAILABLE:
        logger.info("Transcribing with INT4 quantized model...")
        
        try:
            int4_agent = TranscriptionAgent(
                agent_id="int4_model",
                model_size=model_size,
                model_type=ModelType.WHISPER_TRANSFORMERS,  # We'll still use transformers with AWQ
                language=language,
                model_path=int4_model_path
            )
            
            await int4_agent.initialize()
            
            start_time = time.time()
            int4_result = await int4_agent.transcribe(audio_file)
            int4_time = time.time() - start_time
            
            logger.info(f"INT4 model transcription completed in {int4_time:.2f} seconds")
            logger.info(f"INT4 result: {int4_result}")
            
            results["int4"] = {
                "time": int4_time,
                "result": int4_result
            }
            
            await int4_agent.shutdown()
        except Exception as e:
            logger.error(f"Error transcribing with INT4 model: {e}")
    else:
        logger.warning("INT4 transcription skipped (model not available or AWQ not installed)")
    
    # 3. Transcribe with INT8 model if available
    if int8_model_path:
        logger.info("Transcribing with INT8 quantized model...")
        
        try:
            int8_agent = TranscriptionAgent(
                agent_id="int8_model",
                model_size=model_size,
                model_type=ModelType.WHISPER_OPTIMIZED,
                language=language,
                model_path=int8_model_path
            )
            
            await int8_agent.initialize()
            
            start_time = time.time()
            int8_result = await int8_agent.transcribe(audio_file)
            int8_time = time.time() - start_time
            
            logger.info(f"INT8 model transcription completed in {int8_time:.2f} seconds")
            logger.info(f"INT8 result: {int8_result}")
            
            results["int8"] = {
                "time": int8_time,
                "result": int8_result
            }
            
            await int8_agent.shutdown()
        except Exception as e:
            logger.error(f"Error transcribing with INT8 model: {e}")
    else:
        logger.warning("INT8 transcription skipped (model not available)")
    
    # 4. Report comparison results
    logger.info("\n==== PERFORMANCE COMPARISON ====")
    
    if "original" in results and "int4" in results:
        speedup_vs_original = results["original"]["time"] / results["int4"]["time"]
        logger.info(f"INT4 speedup vs original: {speedup_vs_original:.2f}x")
    
    if "original" in results and "int8" in results:
        speedup_vs_original = results["original"]["time"] / results["int8"]["time"]
        logger.info(f"INT8 speedup vs original: {speedup_vs_original:.2f}x")
    
    if "int4" in results and "int8" in results:
        speedup_int4_vs_int8 = results["int8"]["time"] / results["int4"]["time"]
        logger.info(f"INT4 speedup vs INT8: {speedup_int4_vs_int8:.2f}x")


async def main():
    """Run the INT4 quantization demo."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="INT4 Quantization Demo")
    parser.add_argument("--model-size", type=str, default="tiny", help="Whisper model size")
    parser.add_argument("--language", type=str, default=None, help="Language code (e.g., 'en')")
    parser.add_argument("--audio-file", type=str, default=None, help="Path to audio file for transcription")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for models")
    args = parser.parse_args()
    
    logger.info(f"Starting INT4 Quantization Demo with model size: {args.model_size}")
    
    # Ensure sample audio file exists if specified
    audio_file = args.audio_file
    if audio_file is None:
        sample_audio = os.path.join(os.getcwd(), "examples", "assets", "sample.mp3")
        if os.path.exists(sample_audio):
            audio_file = sample_audio
            logger.info(f"Using sample audio file: {audio_file}")
    
    int4_model_path = None
    int8_model_path = None
    
    # Phase 1: INT4 Quantization
    if AWQ_AVAILABLE:
        try:
            int4_result = await quantize_with_int4(
                model_size=args.model_size,
                language=args.language,
                output_dir=args.output_dir
            )
            
            if "error" not in int4_result:
                int4_model_path = int4_result["model_path"]
        except Exception as e:
            logger.error(f"INT4 quantization failed: {e}")
    else:
        logger.warning("Skipping INT4 quantization (AWQ not available)")
    
    # Phase 2: INT8 Quantization (for comparison)
    try:
        int8_result = await quantize_with_int8(
            model_size=args.model_size,
            language=args.language,
            output_dir=args.output_dir
        )
        
        if "error" not in int8_result:
            int8_model_path = int8_result["model_path"]
    except Exception as e:
        logger.error(f"INT8 quantization failed: {e}")
    
    # Phase 3: Compare Transcription Performance
    await transcribe_and_compare(
        model_size=args.model_size,
        language=args.language,
        audio_file=audio_file,
        int4_model_path=int4_model_path,
        int8_model_path=int8_model_path
    )
    
    logger.info("INT4 Quantization Demo completed.")


if __name__ == "__main__":
    asyncio.run(main()) 