"""
Integration test for the audio processing and transcription pipeline.

This module contains tests to verify that the AudioProcessingAgent and
TranscriptionAgent work together correctly in a pipeline configuration.

Author: AMPTALK Team
Date: 2024
"""

import pytest
import asyncio
from typing import List, Dict, Any

from src.core.framework.orchestrator import Orchestrator
from src.agents.audio_processing_agent import AudioProcessingAgent
from src.agents.transcription_agent import TranscriptionAgent
from src.core.framework.message import (
    Message, MessageType, MessagePriority, create_audio_input_message
)


@pytest.fixture
async def pipeline_orchestrator():
    """Create a test orchestrator with a configured pipeline."""
    # Create orchestrator
    orchestrator = Orchestrator(name="Pipeline Test Orchestrator")
    
    # Create agents with minimal configurations for fast testing
    audio_agent = AudioProcessingAgent(
        name="AudioProcessor",
        config={
            'audio': {
                'chunk_duration_ms': 100,  # Small for faster tests
            }
        }
    )
    
    transcription_agent = TranscriptionAgent(
        name="Transcriber",
        config={
            'whisper': {
                'model_size': 'tiny',  # Use smallest model for tests
                'device': 'cpu'
            }
        }
    )
    
    # Register agents
    orchestrator.register_agent(audio_agent, groups=["input"])
    orchestrator.register_agent(transcription_agent, groups=["processing"])
    
    # Set up pipeline
    pipeline_config = [
        {
            "agent_id": audio_agent.agent_id,
            "connects_to": [transcription_agent.agent_id]
        },
        {
            "agent_id": transcription_agent.agent_id,
            "connects_to": []
        }
    ]
    
    orchestrator.create_agent_pipeline(pipeline_config)
    
    # Start the system
    await orchestrator.start()
    
    # Allow agents to initialize
    await asyncio.sleep(0.2)
    
    # Return orchestrator and agents for test use
    yield {
        "orchestrator": orchestrator,
        "audio_agent": audio_agent,
        "transcription_agent": transcription_agent
    }
    
    # Cleanup
    await orchestrator.stop()


class MessageCollector:
    """Helper class to collect and track messages for testing."""
    
    def __init__(self):
        self.messages: List[Message] = []
        self.message_types: Dict[MessageType, int] = {}
    
    async def collect_message(self, message: Message) -> List[Message]:
        """Collect a message and track its type."""
        self.messages.append(message)
        
        msg_type = message.message_type
        self.message_types[msg_type] = self.message_types.get(msg_type, 0) + 1
        
        # Return empty list as we're just collecting
        return []
    
    def get_messages_by_type(self, message_type: MessageType) -> List[Message]:
        """Filter collected messages by type."""
        return [m for m in self.messages if m.message_type == message_type]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_audio_transcription_pipeline(pipeline_orchestrator):
    """Test the complete audio processing and transcription pipeline."""
    audio_agent = pipeline_orchestrator["audio_agent"]
    transcription_agent = pipeline_orchestrator["transcription_agent"]
    
    # Create a message collector to intercept and record messages
    collector = MessageCollector()
    
    # Add the collector as a message handler for the transcription agent
    transcription_agent.register_message_handler(
        MessageType.TRANSCRIPTION_RESULT,
        collector.collect_message
    )
    
    # Create dummy audio data (1 second of silence at 16kHz)
    dummy_audio_data = b'\0' * 32000
    
    # Create an audio input message
    audio_message = create_audio_input_message(
        source_id="test_system",
        target_id=audio_agent.agent_id,
        audio_data=dummy_audio_data,
        sample_rate=16000
    )
    
    # Send the message to the audio agent
    await audio_agent.enqueue_message(audio_message)
    
    # Wait for the message to be processed through the pipeline
    # This needs sufficient time for:
    # 1. Audio agent to process the audio
    # 2. Audio agent to send result to transcription agent
    # 3. Transcription agent to load model and process
    # 4. Transcription agent to generate and send result
    await asyncio.sleep(1.0)
    
    # Verify that we received at least one TRANSCRIPTION_RESULT message
    transcription_results = collector.get_messages_by_type(MessageType.TRANSCRIPTION_RESULT)
    
    assert len(transcription_results) >= 1, "No transcription results were produced"
    
    # Verify the content of the transcription result
    result = transcription_results[0]
    assert result.message_type == MessageType.TRANSCRIPTION_RESULT
    assert result.source_agent_id == transcription_agent.agent_id
    
    # Check payload
    payload = result.payload
    assert "text" in payload
    assert isinstance(payload["text"], str)
    assert "confidence" in payload
    assert "processing_time" in payload
    assert "real_time_factor" in payload


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pipeline_error_handling(pipeline_orchestrator, monkeypatch):
    """Test error handling in the pipeline."""
    audio_agent = pipeline_orchestrator["audio_agent"]
    
    # Create a message collector to intercept and record messages
    collector = MessageCollector()
    
    # Add the collector as a message handler for the audio agent
    audio_agent.register_message_handler(
        MessageType.AUDIO_ERROR,
        collector.collect_message
    )
    
    # Force an error in the audio agent by monkeypatching
    async def process_audio_error(self, message):
        raise ValueError("Forced error for testing pipeline error handling")
    
    monkeypatch.setattr(audio_agent, "_process_audio_input", process_audio_error)
    
    # Create dummy audio data
    dummy_audio_data = b'\0' * 32000
    
    # Create an audio input message
    audio_message = create_audio_input_message(
        source_id="test_system",
        target_id=audio_agent.agent_id,
        audio_data=dummy_audio_data,
        sample_rate=16000
    )
    
    # Send the message to the audio agent
    await audio_agent.enqueue_message(audio_message)
    
    # Wait for the message to be processed
    await asyncio.sleep(0.5)
    
    # Verify that we received an error message
    error_messages = collector.get_messages_by_type(MessageType.AUDIO_ERROR)
    
    assert len(error_messages) >= 1, "No error messages were produced"
    
    # Verify the content of the error message
    error = error_messages[0]
    assert error.message_type == MessageType.AUDIO_ERROR
    assert error.source_agent_id == audio_agent.agent_id
    assert "Forced error" in error.payload.get("error", "") 