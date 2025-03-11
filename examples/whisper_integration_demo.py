#!/usr/bin/env python
"""
Whisper Integration Demo for AMPTALK

This script demonstrates the use of the TranscriptionAgent with the integrated
Whisper model to transcribe audio files.

Author: AMPTALK Team
Date: 2024
"""

import asyncio
import logging
import argparse
import os
import time
import numpy as np
import soundfile as sf
from pathlib import Path

from src.core.framework.message import (
    Message, MessageType, MessagePriority
)
from src.core.framework.orchestrator import Orchestrator
from src.agents.audio_processing_agent import AudioProcessingAgent
from src.agents.transcription_agent import TranscriptionAgent
from src.core.utils.logging_config import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger("amptalk.examples.whisper_demo")


async def transcribe_audio_file(
    audio_file_path: str,
    model_size: str = "small",
    device: str = "auto",
    compute_type: str = "auto",
    language: str = None
):
    """
    Transcribe an audio file using the TranscriptionAgent with Whisper.
    
    Args:
        audio_file_path: Path to the audio file to transcribe
        model_size: Size of the Whisper model to use
        device: Device to run inference on ('cpu', 'cuda', 'mps', 'auto')
        compute_type: Computation type ('float16', 'int8', 'auto')
        language: Language code for transcription (e.g., 'en', 'ja', etc.)
    """
    logger.info(f"Starting transcription of {audio_file_path}")
    
    # Load audio file
    try:
        audio_data, sample_rate = sf.read(audio_file_path)
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Normalize audio
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        elif audio_data.dtype == np.uint8:
            audio_data = (audio_data.astype(np.float32) - 128) / 128.0
        
        # Ensure data is in the range [-1, 1]
        max_val = np.max(np.abs(audio_data))
        if max_val > 1.0:
            audio_data = audio_data / max_val
        
        duration = len(audio_data) / sample_rate
        logger.info(f"Loaded audio file: {audio_file_path}")
        logger.info(f"Duration: {duration:.2f}s, Sample rate: {sample_rate}Hz")
    except Exception as e:
        logger.error(f"Failed to load audio file: {e}")
        return
    
    # Create orchestrator
    orchestrator = Orchestrator(name="TranscriptionOrchestrator")
    
    # Create and configure agents
    audio_agent = AudioProcessingAgent(name="AudioAgent")
    transcription_agent = TranscriptionAgent(
        name="WhisperAgent",
        config={
            "whisper": {
                "model_size": model_size,
                "device": device,
                "compute_type": compute_type,
                "language": language,
                "word_timestamps": True
            }
        }
    )
    
    # Register agents with orchestrator
    orchestrator.register_agent(audio_agent)
    orchestrator.register_agent(transcription_agent)
    
    # Connect agents (bidirectional)
    orchestrator.connect_agents(
        audio_agent.agent_id, 
        transcription_agent.agent_id, 
        bidirectional=True
    )
    
    # Start the orchestrator and agents
    await orchestrator.start()
    
    try:
        # Create a message collector to gather transcription results
        transcription_results = []
        
        async def collect_transcription(message):
            if message.message_type == MessageType.TRANSCRIPTION_RESULT:
                transcription_results.append(message.payload)
                logger.info(f"Received transcription: {message.payload['text']}")
        
        # Register message handler
        audio_agent.register_message_handler(
            MessageType.TRANSCRIPTION_RESULT,
            collect_transcription
        )
        
        # Create audio input message
        audio_message = Message(
            message_type=MessageType.AUDIO_INPUT,
            source_agent_id="demo",
            target_agent_id=audio_agent.agent_id,
            priority=MessagePriority.NORMAL,
            payload={
                "audio_data": audio_data,
                "sample_rate": sample_rate,
                "channels": 1,
                "filepath": audio_file_path,
                "format": "float32"
            }
        )
        
        # Send the message
        logger.info("Sending audio for transcription...")
        start_time = time.time()
        await orchestrator.send_message_to_agent(audio_message)
        
        # Wait for processing to complete
        # In a real system, you would typically use a callback or event mechanism
        # Here we'll just wait a bit longer than the audio duration
        wait_time = max(5.0, duration * 2)
        logger.info(f"Waiting for transcription to complete (timeout: {wait_time:.1f}s)...")
        await asyncio.sleep(wait_time)
        
        # Calculate overall metrics
        end_time = time.time()
        total_time = end_time - start_time
        
        # Print results
        logger.info("="*50)
        logger.info("Transcription Results:")
        logger.info("="*50)
        
        if transcription_results:
            # Sort results by start time
            sorted_results = sorted(transcription_results, key=lambda x: x.get("start_time", 0))
            
            full_text = ""
            for i, result in enumerate(sorted_results):
                logger.info(f"Segment {i+1}:")
                logger.info(f"  Text: {result['text']}")
                logger.info(f"  Confidence: {result['confidence']:.2f}")
                logger.info(f"  Language: {result['language']}")
                logger.info(f"  Processing time: {result['processing_time']:.2f}s")
                logger.info(f"  Real-time factor: {result['real_time_factor']:.2f}x")
                logger.info("-"*50)
                
                full_text += result['text'] + " "
            
            logger.info("Complete Transcription:")
            logger.info(full_text.strip())
            
            logger.info("="*50)
            logger.info(f"Total audio duration: {duration:.2f}s")
            logger.info(f"Total processing time: {total_time:.2f}s")
            logger.info(f"Overall real-time factor: {total_time/duration:.2f}x")
        else:
            logger.warning("No transcription results received")
    
    finally:
        # Stop the orchestrator and agents
        await orchestrator.stop()


async def main():
    """Main entry point for the demo."""
    parser = argparse.ArgumentParser(description="Whisper Transcription Demo")
    parser.add_argument(
        "audio_file", 
        type=str, 
        help="Path to the audio file to transcribe"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="small",
        choices=["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"],
        help="Size of the Whisper model to use"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        choices=["cpu", "cuda", "mps", "auto"],
        help="Device to run inference on"
    )
    parser.add_argument(
        "--compute", 
        type=str, 
        default="auto",
        choices=["float16", "int8", "auto"],
        help="Computation type"
    )
    parser.add_argument(
        "--language", 
        type=str, 
        default=None,
        help="Language code (e.g., 'en', 'ja', etc.)"
    )
    
    args = parser.parse_args()
    
    # Verify audio file exists
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        return
    
    await transcribe_audio_file(
        str(audio_path),
        model_size=args.model,
        device=args.device,
        compute_type=args.compute,
        language=args.language
    )


if __name__ == "__main__":
    asyncio.run(main()) 