#!/usr/bin/env python
"""
Edge Optimization Demo for AMPTALK Framework.

This example script demonstrates the edge optimization capabilities
for running Whisper models on resource-constrained devices.

It shows:
1. How to optimize a Whisper model using ONNX
2. How to measure performance improvements
3. How to use the optimized model for transcription

Usage:
    python edge_optimization_demo.py [OPTIONS]

Options:
    --model-size TEXT              Whisper model size (tiny, base, small, medium)
    --optimization-level TEXT      Optimization level (NONE, BASIC, MEDIUM, HIGH, EXTREME)
    --target-device TEXT           Target device (cpu, gpu, mobile)
    --audio-file PATH              Path to audio file for transcription
    --language TEXT                Language code (e.g., en, es, fr)
    --cache-dir PATH               Directory to cache optimized models
    --compare                      Compare performance with non-optimized model
    --all-in-one                  Use all-in-one model approach (encoder + decoder + beam search in one model)

Example:
    python edge_optimization_demo.py --model-size tiny --optimization-level MEDIUM --audio-file sample.mp3

Author: AMPTALK Team
Date: 2024
"""

import os
import sys
import time
import argparse
import tempfile
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import traceback

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import AMPTALK modules
from src.core.utils.edge_optimization import (
    EdgeOptimizer, OptimizationLevel, DeviceTarget, 
    optimize_whisper_model
)
from src.core.utils.logging_config import get_logger, configure_logging

# Configure logging
configure_logging()
logger = get_logger("edge_optimization_demo")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="AMPTALK Edge Optimization Demo")
    
    parser.add_argument(
        "--model-size",
        type=str,
        default="tiny",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size to use"
    )
    
    parser.add_argument(
        "--optimization-level",
        type=str,
        default="MEDIUM",
        choices=["NONE", "BASIC", "MEDIUM", "HIGH", "EXTREME"],
        help="Level of optimization to apply"
    )
    
    parser.add_argument(
        "--target-device",
        type=str,
        default="cpu",
        choices=["cpu", "gpu", "npu", "dsp", "mobile", "browser"],
        help="Target device for optimization"
    )
    
    parser.add_argument(
        "--audio-file",
        type=str,
        help="Path to audio file for transcription"
    )
    
    parser.add_argument(
        "--language",
        type=str,
        help="Language code (e.g., en, es, fr)"
    )
    
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=os.path.join(os.getcwd(), "models", "optimized"),
        help="Directory to cache optimized models"
    )
    
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare performance with non-optimized model"
    )
    
    parser.add_argument(
        "--all-in-one",
        action="store_true",
        help="Use all-in-one model approach (encoder + decoder + beam search in one model)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def optimize_model(
    model_size: str,
    optimization_level: str,
    target_device: str,
    language: Optional[str] = None,
    cache_dir: Optional[str] = None,
    verbose: bool = False,
    all_in_one: bool = False
) -> Dict[str, Any]:
    """
    Optimize a Whisper model for edge deployment.
    
    Args:
        model_size: Size of the Whisper model
        optimization_level: Level of optimization to apply
        target_device: Target device for optimization
        language: Optional language code
        cache_dir: Directory to cache optimized models
        verbose: Whether to enable verbose logging
        all_in_one: Whether to create an all-in-one model
        
    Returns:
        Dictionary with optimization results
    """
    logger.info(f"Optimizing Whisper model: {model_size} with {optimization_level} optimization")
    logger.info(f"Model type: {'All-in-one' if all_in_one else 'Separate encoder/decoder'}")
    
    # Convert string parameters to enums
    opt_level = getattr(OptimizationLevel, optimization_level, OptimizationLevel.MEDIUM)
    device = getattr(DeviceTarget, target_device.upper(), DeviceTarget.CPU)
    
    # Create progress callback for verbose mode
    progress_callback = None
    if verbose:
        def _progress_callback(message, progress):
            print(f"{message} ({progress:.1%})")
        progress_callback = _progress_callback
    
    # Create optimizer
    optimizer = EdgeOptimizer(
        optimization_level=opt_level,
        target_device=device,
        cache_dir=cache_dir,
        progress_callback=progress_callback
    )
    
    # Run optimization
    result = optimizer.optimize_whisper(
        model_size=model_size,
        language=language,
        all_in_one=all_in_one
    )
    
    return result


def transcribe_with_original_model(
    audio_file: str,
    model_size: str,
    language: Optional[str] = None
) -> Dict[str, Any]:
    """
    Transcribe audio using the original (non-optimized) Whisper model.
    
    Args:
        audio_file: Path to the audio file
        model_size: Size of the Whisper model
        language: Optional language code
        
    Returns:
        Dictionary with transcription results and metrics
    """
    try:
        import torch
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        import librosa
        import numpy as np
        
        logger.info(f"Transcribing with original model: {model_size}")
        
        # Prepare model ID
        lang_suffix = f".{language}" if language else ""
        model_id = f"openai/whisper-{model_size}{lang_suffix}"
        
        # Load processor and model
        load_start = time.time()
        processor = WhisperProcessor.from_pretrained(model_id)
        model = WhisperForConditionalGeneration.from_pretrained(model_id)
        load_time = time.time() - load_start
        
        # Preprocess audio
        preprocess_start = time.time()
        audio_array, sampling_rate = librosa.load(audio_file, sr=16000)
        inputs = processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt")
        preprocess_time = time.time() - preprocess_start
        
        # Run inference
        inference_start = time.time()
        with torch.no_grad():
            generated_ids = model.generate(inputs.input_features)
        inference_time = time.time() - inference_start
        
        # Decode the output
        decode_start = time.time()
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        decode_time = time.time() - decode_start
        
        total_time = load_time + preprocess_time + inference_time + decode_time
        
        # Return results
        return {
            "transcription": transcription,
            "metrics": {
                "load_time_s": load_time,
                "preprocess_time_s": preprocess_time,
                "inference_time_s": inference_time,
                "decode_time_s": decode_time,
                "total_time_s": total_time
            }
        }
        
    except Exception as e:
        logger.error(f"Error transcribing with original model: {e}")
        return {
            "error": str(e)
        }


def transcribe_with_optimized_model(
    audio_file: str,
    model_path: str,
    language: Optional[str] = None,
    all_in_one: bool = False
) -> Dict[str, Any]:
    """
    Transcribe audio using an optimized ONNX model.
    
    Args:
        audio_file: Path to the audio file
        model_path: Path to the optimized model
        language: Optional language code
        all_in_one: Whether this is an all-in-one model
        
    Returns:
        Dictionary with transcription results and metrics
    """
    try:
        import librosa
        import numpy as np
        import onnxruntime as ort
        from transformers import WhisperProcessor
        
        logger.info(f"Transcribing with optimized model: {model_path}")
        
        # Determine if this is an ONNX model
        if not model_path.endswith(".onnx"):
            logger.warning("Not an ONNX model, falling back to original model transcription")
            model_id = model_path  # Assume it's a HuggingFace model ID
            model_size = model_id.split("/")[-1].split("-")[1]
            return transcribe_with_original_model(audio_file, model_size, language)
        
        # Extract model directory and size
        model_dir = os.path.dirname(model_path)
        model_base = os.path.basename(model_dir)
        model_size = "tiny"  # Default
        
        # Try to extract model size from directory name
        for size in ["tiny", "base", "small", "medium", "large-v3"]:
            if size in model_base:
                model_size = size
                break
        
        # Load processor
        lang_suffix = f".{language}" if language else ""
        processor_id = f"openai/whisper-{model_size}{lang_suffix}"
        
        load_start = time.time()
        processor = WhisperProcessor.from_pretrained(processor_id)
        
        # Create session options
        session_options = ort.SessionOptions()
        
        # Check if we need to use onnxruntime extensions for all-in-one models
        if all_in_one:
            try:
                from onnxruntime_extensions import get_library_path
                session_options.register_custom_ops_library(get_library_path())
                logger.info("Using onnxruntime_extensions for all-in-one model")
            except ImportError:
                logger.warning("onnxruntime_extensions not found, some models may not work")
        
        # Create ONNX session
        session = ort.InferenceSession(
            model_path, 
            sess_options=session_options,
            providers=["CPUExecutionProvider"]
        )
        load_time = time.time() - load_start
        
        # Get model inputs
        model_inputs = session.get_inputs()
        model_outputs = session.get_outputs()
        
        logger.info(f"Model loaded with {len(model_inputs)} inputs and {len(model_outputs)} outputs")
        
        # Process audio differently based on model type
        if all_in_one:
            # All-in-one model processing
            preprocess_start = time.time()
            
            # Check if model expects raw audio or features
            if any(input.name == "audio_stream" for input in model_inputs):
                # Model expects raw audio bytes
                with open(audio_file, "rb") as f:
                    audio_data = np.asarray(list(f.read()), dtype=np.uint8)
                
                inputs = {
                    "audio_stream": np.array([audio_data]),
                    "max_length": np.array([448], dtype=np.int32),
                    "min_length": np.array([1], dtype=np.int32),
                    "num_beams": np.array([5], dtype=np.int32),
                    "num_return_sequences": np.array([1], dtype=np.int32),
                    "length_penalty": np.array([1.0], dtype=np.float32),
                    "repetition_penalty": np.array([1.0], dtype=np.float32),
                }
                
                # Add attention mask if needed
                if any(input.name == "attention_mask" for input in model_inputs):
                    inputs["attention_mask"] = np.zeros((1, 80, 3000), dtype=np.int32)
                
            else:
                # Model expects preprocessed features
                audio_array, sampling_rate = librosa.load(audio_file, sr=16000)
                feature_extractor = processor.feature_extractor
                features = feature_extractor(
                    audio_array, 
                    sampling_rate=sampling_rate, 
                    return_tensors="np"
                ).input_features
                
                # Add basic inputs for the encoder model
                inputs = {"input_features": features}
            
            preprocess_time = time.time() - preprocess_start
            
            # Run inference
            inference_start = time.time()
            outputs = session.run(None, inputs)
            inference_time = time.time() - inference_start
            
            # Process outputs - in all-in-one models this should be directly usable
            decode_start = time.time()
            
            if outputs[0].dtype == np.int64:
                # If output is token IDs, decode them
                token_ids = outputs[0]
                transcription = processor.batch_decode(token_ids, skip_special_tokens=True)[0]
            elif outputs[0].dtype == np.dtype('S1'):
                # If output is bytes, decode to string
                try:
                    transcription = outputs[0][0].decode('utf-8')
                except:
                    transcription = str(outputs[0])
            else:
                # If we can't interpret the output, return a placeholder
                transcription = f"[All-in-one model output not interpretable]"
            
            decode_time = time.time() - decode_start
            
        else:
            # Separate encoder/decoder processing (simplified - encoder only)
            preprocess_start = time.time()
            audio_array, sampling_rate = librosa.load(audio_file, sr=16000)
            feature_extractor = processor.feature_extractor
            features = feature_extractor(
                audio_array, 
                sampling_rate=sampling_rate, 
                return_tensors="np"
            ).input_features
            preprocess_time = time.time() - preprocess_start
            
            # Run inference on encoder only
            inference_start = time.time()
            input_name = session.get_inputs()[0].name
            encoder_outputs = session.run(None, {input_name: features})
            inference_time = time.time() - inference_start
            
            # Note: This is a simplified implementation (encoder-only)
            # A complete implementation would also run the decoder autoregressive loop
            decode_time = 0.0
            transcription = "[Encoder-only transcription - decoding not implemented in demo]"
        
        total_time = load_time + preprocess_time + inference_time + decode_time
        
        # Return results
        return {
            "transcription": transcription,
            "note": "Results from ONNX model" + (" (all-in-one)" if all_in_one else " (encoder-only)"),
            "metrics": {
                "load_time_s": load_time,
                "preprocess_time_s": preprocess_time,
                "inference_time_s": inference_time,
                "decode_time_s": decode_time,
                "total_time_s": total_time
            }
        }
        
    except Exception as e:
        logger.error(f"Error transcribing with optimized model: {e}")
        logger.debug(traceback.format_exc())
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def download_sample_audio_if_needed() -> str:
    """
    Download a sample audio file if none is provided.
    
    Returns:
        Path to the sample audio file
    """
    import urllib.request
    
    # Create directory for sample data
    samples_dir = os.path.join(os.getcwd(), "samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    # Sample file path
    sample_path = os.path.join(samples_dir, "sample.mp3")
    
    # Check if file already exists
    if os.path.exists(sample_path):
        logger.info(f"Using existing sample file: {sample_path}")
        return sample_path
    
    # Download from a sample repository
    logger.info("Downloading sample audio file...")
    url = "https://github.com/openai/whisper/raw/main/tests/jfk.flac"
    try:
        urllib.request.urlretrieve(url, sample_path)
        logger.info(f"Downloaded sample file to: {sample_path}")
        return sample_path
    except Exception as e:
        logger.error(f"Failed to download sample file: {e}")
        raise


def compare_models(
    original_results: Dict[str, Any],
    optimized_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare performance between original and optimized models.
    
    Args:
        original_results: Results from original model
        optimized_results: Results from optimized model
        
    Returns:
        Dictionary with comparison metrics
    """
    # Check for errors in results
    if "error" in original_results:
        return {"error": f"Original model error: {original_results['error']}"}
    
    if "error" in optimized_results:
        return {"error": f"Optimized model error: {optimized_results['error']}"}
    
    # Extract metrics
    original_metrics = original_results.get("metrics", {})
    optimized_metrics = optimized_results.get("metrics", {})
    
    # Calculate improvements
    comparison = {}
    
    # Calculate overall speedup
    original_total = original_metrics.get("total_time_s", 0)
    optimized_total = optimized_metrics.get("total_time_s", 0)
    
    if original_total > 0 and optimized_total > 0:
        speedup = original_total / optimized_total
        improvement_pct = ((original_total - optimized_total) / original_total) * 100
        
        comparison["total_speedup"] = speedup
        comparison["total_improvement_pct"] = improvement_pct
    
    # Calculate per-phase improvements
    for phase in ["load_time_s", "preprocess_time_s", "inference_time_s", "decode_time_s"]:
        original_time = original_metrics.get(phase, 0)
        optimized_time = optimized_metrics.get(phase, 0)
        
        if original_time > 0 and optimized_time > 0:
            phase_speedup = original_time / optimized_time
            phase_improvement_pct = ((original_time - optimized_time) / original_time) * 100
            
            comparison[f"{phase}_speedup"] = phase_speedup
            comparison[f"{phase}_improvement_pct"] = phase_improvement_pct
    
    return comparison


def print_results(
    original_results: Optional[Dict[str, Any]],
    optimized_results: Dict[str, Any],
    optimization_info: Dict[str, Any],
    comparison: Optional[Dict[str, Any]] = None
) -> None:
    """
    Print results in a readable format.
    
    Args:
        original_results: Results from original model
        optimized_results: Results from optimized model
        optimization_info: Information about the optimization
        comparison: Optional comparison metrics
    """
    print("\n" + "="*80)
    print(f"{'AMPTALK EDGE OPTIMIZATION RESULTS':^80}")
    print("="*80)
    
    # Print optimization info
    print("\nOPTIMIZATION INFORMATION:")
    print(f"  Model ID:           {optimization_info.get('model_id', 'Unknown')}")
    print(f"  Optimization Level: {OptimizationLevel(optimization_info.get('optimization_level', 0)).name}")
    print(f"  Target Device:      {optimization_info.get('target_device', 'cpu')}")
    print(f"  Optimizations:      {', '.join(optimization_info.get('optimizations_applied', []))}")
    
    size_reduction = optimization_info.get('size_reduction', 0) * 100
    print(f"  Size Reduction:     {size_reduction:.1f}%")
    
    speed_improvement = optimization_info.get('speed_improvement', 0) * 100
    print(f"  Speed Improvement:  {speed_improvement:.1f}%")
    
    # Print original model results if available
    if original_results and "error" not in original_results:
        print("\nORIGINAL MODEL PERFORMANCE:")
        metrics = original_results.get("metrics", {})
        print(f"  Load Time:          {metrics.get('load_time_s', 0):.4f}s")
        print(f"  Preprocess Time:    {metrics.get('preprocess_time_s', 0):.4f}s")
        print(f"  Inference Time:     {metrics.get('inference_time_s', 0):.4f}s")
        print(f"  Decode Time:        {metrics.get('decode_time_s', 0):.4f}s")
        print(f"  Total Time:         {metrics.get('total_time_s', 0):.4f}s")
        
        print("\n  Transcription Sample:")
        transcription = original_results.get("transcription", "")
        print(f"    \"{transcription[:100]}{'...' if len(transcription) > 100 else ''}\"")
    
    # Print optimized model results
    print("\nOPTIMIZED MODEL PERFORMANCE:")
    if "error" in optimized_results:
        print(f"  Error: {optimized_results['error']}")
    else:
        metrics = optimized_results.get("metrics", {})
        print(f"  Load Time:          {metrics.get('load_time_s', 0):.4f}s")
        print(f"  Preprocess Time:    {metrics.get('preprocess_time_s', 0):.4f}s")
        print(f"  Inference Time:     {metrics.get('inference_time_s', 0):.4f}s")
        print(f"  Decode Time:        {metrics.get('decode_time_s', 0):.4f}s")
        print(f"  Total Time:         {metrics.get('total_time_s', 0):.4f}s")
        
        if "note" in optimized_results:
            print(f"\n  Note: {optimized_results['note']}")
        
        print("\n  Transcription Sample:")
        transcription = optimized_results.get("transcription", "")
        print(f"    \"{transcription[:100]}{'...' if len(transcription) > 100 else ''}\"")
    
    # Print comparison if available
    if comparison:
        print("\nPERFORMANCE COMPARISON:")
        if "error" in comparison:
            print(f"  Error: {comparison['error']}")
        else:
            total_speedup = comparison.get("total_speedup", 0)
            total_improvement = comparison.get("total_improvement_pct", 0)
            print(f"  Overall Speedup:     {total_speedup:.2f}x ({total_improvement:.1f}% faster)")
            
            inference_speedup = comparison.get("inference_time_s_speedup", 0)
            inference_improvement = comparison.get("inference_time_s_improvement_pct", 0)
            if inference_speedup > 0:
                print(f"  Inference Speedup:   {inference_speedup:.2f}x ({inference_improvement:.1f}% faster)")
    
    print("\n" + "="*80)


def main():
    """Run the edge optimization demo."""
    args = parse_args()
    
    # Ensure cache directory exists
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # Get audio file
    audio_file = args.audio_file
    if not audio_file or not os.path.exists(audio_file):
        audio_file = download_sample_audio_if_needed()
    
    # Optimize model
    optimization_result = optimize_model(
        model_size=args.model_size,
        optimization_level=args.optimization_level,
        target_device=args.target_device,
        language=args.language,
        cache_dir=args.cache_dir,
        verbose=args.verbose,
        all_in_one=args.all_in_one
    )
    
    if "error" in optimization_result:
        logger.error(f"Optimization failed: {optimization_result['error']}")
        sys.exit(1)
    
    # Get model path
    model_path = optimization_result["model_path"]
    logger.info(f"Optimized model path: {model_path}")
    
    # Transcribe with optimized model
    optimized_results = transcribe_with_optimized_model(
        audio_file=audio_file,
        model_path=model_path,
        language=args.language,
        all_in_one=args.all_in_one
    )
    
    # Compare with original model if requested
    original_results = None
    comparison = None
    
    if args.compare:
        original_results = transcribe_with_original_model(
            audio_file=audio_file,
            model_size=args.model_size,
            language=args.language
        )
        
        comparison = compare_models(original_results, optimized_results)
    
    # Print results
    print_results(
        original_results=original_results,
        optimized_results=optimized_results,
        optimization_info=optimization_result,
        comparison=comparison
    )


if __name__ == "__main__":
    main() 