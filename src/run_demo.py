#!/usr/bin/env python3
"""
AMPTALK Demo Application.

This script demonstrates the functionality of the AMPTALK multi-agent system
by setting up a simple pipeline for audio processing and transcription.

Author: AMPTALK Team
Date: 2024
"""

import os
import asyncio
import logging
import argparse
import json
from typing import Dict, Any, List, Optional

from src.core.utils.logging_config import configure_logging, get_logger
from src.core.framework.orchestrator import Orchestrator
from src.core.framework.message import (
    Message, MessageType, MessagePriority, create_audio_input_message
)
from src.agents.audio_processing_agent import AudioProcessingAgent
from src.agents.transcription_agent import TranscriptionAgent

# Configure logging
configure_logging(
    app_name="amptalk_demo",
    log_level="INFO",
    console_output=True,
    file_output=True
)

# Get logger for this module
logger = get_logger("amptalk.demo")


async def run_demo(config_file: Optional[str] = None) -> None:
    """
    Run the AMPTALK demo.
    
    Args:
        config_file: Optional path to a configuration file
    """
    logger.info("Starting AMPTALK demo")
    
    # Load configuration if provided
    config = {}
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_file}")
    
    # Create the orchestrator
    orchestrator = Orchestrator(name="AMPTALK Demo Orchestrator")
    
    # Create agents
    audio_agent = AudioProcessingAgent(
        name="AudioProcessor",
        config=config.get('audio_agent', {})
    )
    
    transcription_agent = TranscriptionAgent(
        name="Transcriber",
        config=config.get('transcription_agent', {})
    )
    
    # Register agents with the orchestrator
    orchestrator.register_agent(audio_agent, groups=["input"])
    orchestrator.register_agent(transcription_agent, groups=["processing"])
    
    # Connect agents
    orchestrator.connect_agents(
        audio_agent.agent_id, 
        transcription_agent.agent_id, 
        bidirectional=True
    )
    
    # Start the system
    await orchestrator.start()
    logger.info("System started")
    
    try:
        # Send initialization messages to each agent
        init_message = Message(
            message_type=MessageType.INITIALIZE,
            source_agent_id="system",
            target_agent_id=audio_agent.agent_id,
            priority=MessagePriority.HIGH,
            payload={"config": config.get('audio_agent', {})}
        )
        
        await audio_agent.enqueue_message(init_message)
        
        init_message = Message(
            message_type=MessageType.INITIALIZE,
            source_agent_id="system",
            target_agent_id=transcription_agent.agent_id,
            priority=MessagePriority.HIGH,
            payload={"config": config.get('transcription_agent', {})}
        )
        
        await transcription_agent.enqueue_message(init_message)
        
        # Wait for initialization to complete
        await asyncio.sleep(2)
        
        # Simulate audio input
        for i in range(3):
            # Create dummy audio data (in a real app, this would be actual audio)
            dummy_audio_data = b'\0' * 32000  # 1 second of silence at 16kHz
            
            # Create an audio input message
            audio_message = create_audio_input_message(
                source_id="system",
                target_id=audio_agent.agent_id,
                audio_data=dummy_audio_data,
                sample_rate=16000
            )
            
            # Send the message
            await audio_agent.enqueue_message(audio_message)
            logger.info(f"Sent audio segment {i+1}/3")
            
            # Wait for processing
            await asyncio.sleep(3)
        
        # Wait for all processing to complete
        await asyncio.sleep(5)
        
        # Get system status
        status = await orchestrator.get_system_status()
        print("\nSystem Status:")
        print(json.dumps(status, indent=2))
        
    finally:
        # Shutdown the system
        logger.info("Shutting down system")
        await orchestrator.stop()


def main():
    """Parse command line arguments and run the demo."""
    parser = argparse.ArgumentParser(description="AMPTALK Demo Application")
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # Run the demo
    asyncio.run(run_demo(args.config))


if __name__ == "__main__":
    main() 