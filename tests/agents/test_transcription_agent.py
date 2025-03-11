"""
Tests for the TranscriptionAgent.

This module contains unit tests for the TranscriptionAgent, which is responsible
for converting audio segments into text transcriptions.

Author: AMPTALK Team
Date: 2024
"""

import pytest
import asyncio
import time
from unittest.mock import patch, AsyncMock

from src.agents.transcription_agent import TranscriptionAgent, WhisperModelConfig
from src.core.framework.message import (
    Message, MessageType, MessagePriority, create_status_request
)


@pytest.fixture
async def transcription_agent():
    """Fixture providing a started TranscriptionAgent for testing."""
    agent = TranscriptionAgent()
    await agent.start()
    yield agent
    await agent.stop()


def create_audio_processed_message(source_id, target_id):
    """Helper to create a mock AUDIO_PROCESSED message."""
    return Message(
        message_type=MessageType.AUDIO_PROCESSED,
        source_agent_id=source_id,
        target_agent_id=target_id,
        priority=MessagePriority.NORMAL,
        payload={
            "segment_id": "test_segment_123",
            "sample_rate": 16000,
            "channels": 1,
            "start_time": 0.0,
            "end_time": 1.0,
            "duration": 1.0,
            "audio_data": "[PROCESSED_AUDIO_DATA]"
        }
    )


@pytest.mark.asyncio
async def test_transcription_agent_initialization():
    """Test TranscriptionAgent initialization with default and custom settings."""
    # Test default initialization
    default_agent = TranscriptionAgent()
    assert default_agent.model_config.model_size == "large-v3-turbo"
    assert default_agent.model_config.device == "auto"
    assert not default_agent.model_loaded
    
    # Test custom initialization
    custom_config = {
        'whisper': {
            'model_size': 'small',
            'device': 'cpu',
            'language': 'fr'
        }
    }
    custom_agent = TranscriptionAgent(config=custom_config)
    assert custom_agent.model_config.model_size == "small"
    assert custom_agent.model_config.device == "cpu"
    assert custom_agent.model_config.language == "fr"


@pytest.mark.asyncio
async def test_transcription_processing(transcription_agent):
    """Test processing of audio segments."""
    # Create an audio processed message
    message = create_audio_processed_message(
        "audio_agent", 
        transcription_agent.agent_id
    )
    
    # Process the message
    result = await transcription_agent.process_message(message)
    
    # Check response
    assert isinstance(result, list)
    assert len(result) == 1
    transcription = result[0]
    
    # Verify message properties
    assert transcription.message_type == MessageType.TRANSCRIPTION_RESULT
    assert transcription.source_agent_id == transcription_agent.agent_id
    assert transcription.target_agent_id == "audio_agent"  # Should return to sender
    
    # Check payload
    payload = transcription.payload
    assert "text" in payload
    assert isinstance(payload["text"], str)
    assert len(payload["text"]) > 0
    assert "confidence" in payload
    assert payload["confidence"] > 0.9  # Default is 0.95
    assert payload["start_time"] == 0.0
    assert payload["end_time"] == 1.0
    assert "processing_time" in payload
    assert "real_time_factor" in payload


@pytest.mark.asyncio
async def test_model_loading(monkeypatch):
    """Test model loading functionality."""
    agent = TranscriptionAgent()
    
    # Mock sleep to avoid actual delays
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    
    # Load the model
    await agent._load_model()
    
    # Verify model is loaded
    assert agent.model_loaded
    assert agent.model is not None
    
    # Test that loading model again doesn't do anything (coverage)
    await agent._load_model()
    
    # Unload the model
    agent._unload_model()
    
    # Verify model is unloaded
    assert not agent.model_loaded
    assert agent.model is None
    
    # Test that unloading already unloaded model doesn't error
    agent._unload_model()


@pytest.mark.asyncio
async def test_status_request_handling(transcription_agent):
    """Test handling of status requests."""
    # Create a status request message
    message = create_status_request("system", transcription_agent.agent_id)
    
    # Process the message
    result = await transcription_agent.process_message(message)
    
    # Verify a status response is returned
    assert isinstance(result, list)
    assert len(result) == 1
    status_message = result[0]
    
    # Check message properties
    assert status_message.message_type == MessageType.STATUS_RESPONSE
    assert status_message.target_agent_id == "system"
    
    # Check status contains expected fields
    status = status_message.payload.get("status", {})
    assert "is_transcribing" in status
    assert "model_loaded" in status
    assert "model_size" in status
    assert "device" in status
    assert status["model_size"] == transcription_agent.model_config.model_size


@pytest.mark.asyncio
async def test_initialization_message_handling():
    """Test handling of initialization messages."""
    agent = TranscriptionAgent()
    
    # Mock the load_model method to avoid actual loading
    with patch.object(agent, '_load_model', new_callable=AsyncMock) as mock_load:
        # Create an initialization message with custom config
        init_config = {
            'whisper': {
                'model_size': 'tiny',
                'language': 'es'
            }
        }
        
        init_message = Message(
            message_type=MessageType.INITIALIZE,
            source_agent_id="system",
            target_agent_id=agent.agent_id,
            priority=MessagePriority.HIGH,
            payload={"config": init_config}
        )
        
        # Process the message
        result = await agent.process_message(init_message)
        
        # Verify a status response is returned
        assert isinstance(result, list)
        assert len(result) == 1
        response = result[0]
        
        # Check response properties
        assert response.message_type == MessageType.STATUS_RESPONSE
        assert response.payload.get("status") == "initialized"
        assert response.payload.get("success") is True
        
        # Verify configuration was applied
        assert agent.model_config.model_size == "tiny"
        assert agent.model_config.language == "es"
        
        # Verify model was loaded
        mock_load.assert_called_once()


@pytest.mark.asyncio
async def test_shutdown_message_handling():
    """Test handling of shutdown messages."""
    agent = TranscriptionAgent()
    
    # Mock the unload_model method to verify it's called
    with patch.object(agent, '_unload_model') as mock_unload:
        # Create a shutdown message
        shutdown_message = Message(
            message_type=MessageType.SHUTDOWN,
            source_agent_id="system",
            target_agent_id=agent.agent_id,
            priority=MessagePriority.HIGH
        )
        
        # Process the message
        result = await agent.process_message(shutdown_message)
        
        # Verify a status response is returned
        assert isinstance(result, list)
        assert len(result) == 1
        response = result[0]
        
        # Check response properties
        assert response.message_type == MessageType.STATUS_RESPONSE
        assert response.payload.get("status") == "shutdown"
        assert response.payload.get("success") is True
        
        # Verify model was unloaded
        mock_unload.assert_called_once()


@pytest.mark.asyncio
async def test_transcription_error_handling(monkeypatch, transcription_agent):
    """Test error handling during transcription."""
    # Create a message to process
    message = create_audio_processed_message(
        "audio_agent", 
        transcription_agent.agent_id
    )
    
    # Monkeypatch _generate_sample_text to raise an error
    def mock_generate_text(*args, **kwargs):
        raise ValueError("Test error in text generation")
    
    monkeypatch.setattr(transcription_agent, '_generate_sample_text', mock_generate_text)
    
    # Process the message
    result = await transcription_agent.process_message(message)
    
    # Verify an error message is returned
    assert isinstance(result, list)
    assert len(result) == 1
    error_message = result[0]
    
    # Check error message properties
    assert error_message.message_type == MessageType.TRANSCRIPTION_ERROR
    assert error_message.target_agent_id == "audio_agent"
    assert "Test error in text generation" in error_message.payload.get("error", "")


@pytest.mark.asyncio
async def test_sample_text_generation():
    """Test the sample text generation functionality."""
    agent = TranscriptionAgent()
    
    # Test with different durations
    short_text = agent._generate_sample_text(1.0)
    assert isinstance(short_text, str)
    assert len(short_text) > 0
    
    medium_text = agent._generate_sample_text(5.0)
    assert len(medium_text) > len(short_text)
    
    long_text = agent._generate_sample_text(10.0)
    assert len(long_text) > len(medium_text)
    
    # Test text formatting
    assert short_text[0].isupper()  # First letter capitalized
    assert short_text[-1] == "."  # Ends with period
    assert "," in short_text or len(short_text) < 10  # Has commas if long enough


@pytest.mark.asyncio
async def test_unsupported_message_type(transcription_agent):
    """Test handling of unsupported message types."""
    message = Message(
        message_type=MessageType.AUDIO_ERROR,  # Unsupported type
        source_agent_id="system",
        target_agent_id=transcription_agent.agent_id
    )
    
    # Process the message
    result = await transcription_agent.process_message(message)
    
    # Should return None for unsupported message types
    assert result is None 