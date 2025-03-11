import pytest
import asyncio
import time
import numpy as np
from unittest.mock import patch

from src.agents.audio_processing_agent import AudioProcessingAgent, AudioProcessingConfig
from src.core.framework.message import (
    Message, MessageType, MessagePriority,
    create_audio_input_message, create_status_request
)


@pytest.fixture
async def audio_agent():
    """Fixture providing a started AudioProcessingAgent for testing."""
    agent = AudioProcessingAgent()
    await agent.start()
    yield agent
    await agent.stop()


@pytest.mark.asyncio
async def test_audio_processing_success(audio_agent):
    """Test AudioProcessingAgent processes audio input correctly."""
    # Create dummy audio data: 1 second of silence at 16kHz
    dummy_audio = b'\0' * 32000
    
    # Create an audio input message
    message = create_audio_input_message(
        source_id="system",
        target_id=audio_agent.agent_id,
        audio_data=dummy_audio,
        sample_rate=16000
    )
    
    # Process the message directly (bypassing the queue)
    result = await audio_agent.process_message(message)
    
    # Verify a response message is returned
    assert isinstance(result, list)
    assert len(result) == 1
    processed_message = result[0]
    
    # Check that the message type is AUDIO_PROCESSED
    assert processed_message.message_type == MessageType.AUDIO_PROCESSED
    
    # Check payload properties
    payload = processed_message.payload
    assert payload.get("sample_rate") == 16000
    # Duration should be approximately 1.0 second
    duration = payload.get("duration")
    assert duration is not None
    assert abs(duration - 1.0) < 0.05
    
    # Check segment_id is non-empty
    assert payload.get("segment_id")
    
    # Check audio_data placeholder
    assert payload.get("audio_data") == "[PROCESSED_AUDIO_DATA]"


@pytest.mark.asyncio
async def test_audio_processing_error_handling(monkeypatch):
    """Test AudioProcessingAgent error handling by forcing an error in processing."""
    agent = AudioProcessingAgent()
    await agent.start()
    
    # Monkeypatch asyncio.sleep to raise an error
    async def fake_sleep(duration):
        raise RuntimeError("Forced error for testing")

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    
    try:
        dummy_audio = b'\0' * 32000
        message = create_audio_input_message(
            source_id="system",
            target_id=agent.agent_id,
            audio_data=dummy_audio,
            sample_rate=16000
        )
        
        result = await agent.process_message(message)
        # Should return an error message
        assert isinstance(result, list)
        assert len(result) == 1
        error_message = result[0]
        assert error_message.message_type == MessageType.AUDIO_ERROR
        assert "Forced error for testing" in error_message.payload.get("error", "")
    finally:
        await agent.stop()


@pytest.mark.asyncio
async def test_agent_configuration():
    """Test AudioProcessingAgent custom configuration."""
    custom_config = {
        'audio': {
            'sample_rate': 44100,
            'chunk_duration_ms': 500,
            'vad_threshold': 0.5,
            'silence_duration_ms': 300,
            'max_segment_duration_ms': 15000,
            'channels': 2
        }
    }
    
    agent = AudioProcessingAgent(config=custom_config)
    
    # Verify configuration was applied
    assert agent.audio_config.sample_rate == 44100
    assert agent.audio_config.chunk_duration_ms == 500
    assert agent.audio_config.vad_threshold == 0.5
    assert agent.audio_config.silence_duration_ms == 300
    assert agent.audio_config.max_segment_duration_ms == 15000
    assert agent.audio_config.channels == 2
    
    # Check derived values
    assert agent.audio_config.chunk_size == 22050  # 44100 * 500 / 1000
    assert agent.audio_config.silence_samples == 13230  # 44100 * 300 / 1000
    assert agent.audio_config.max_segment_samples == 661500  # 44100 * 15000 / 1000


@pytest.mark.asyncio
async def test_status_request_handling(audio_agent):
    """Test handling of status requests."""
    # Create a status request message
    message = create_status_request(
        source_id="system",
        target_id=audio_agent.agent_id
    )
    
    # Process the message
    result = await audio_agent.process_message(message)
    
    # Verify a status response is returned
    assert isinstance(result, list)
    assert len(result) == 1
    status_message = result[0]
    
    # Check message properties
    assert status_message.message_type == MessageType.STATUS_RESPONSE
    assert status_message.target_agent_id == "system"
    assert status_message.source_agent_id == audio_agent.agent_id
    
    # Check status contains expected fields
    status = status_message.payload.get("status", {})
    assert "is_processing" in status
    assert "sample_rate" in status
    assert "channels" in status
    assert status["sample_rate"] == audio_agent.audio_config.sample_rate


@pytest.mark.asyncio
async def test_initialization_message_handling():
    """Test handling of initialization messages."""
    agent = AudioProcessingAgent()
    
    # Create an initialization message with custom config
    init_config = {
        'audio': {
            'sample_rate': 22050,
            'vad_threshold': 0.4
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
    assert agent.audio_config.sample_rate == 22050
    assert agent.audio_config.vad_threshold == 0.4


@pytest.mark.asyncio
async def test_shutdown_message_handling():
    """Test handling of shutdown messages."""
    agent = AudioProcessingAgent()
    await agent.start()
    
    # Set some state to verify it's cleared
    agent.is_processing = True
    
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
    
    # Verify state was cleared
    assert not agent.is_processing
    assert agent.current_segment is None


@pytest.mark.asyncio
async def test_voice_activity_detection():
    """Test voice activity detection method."""
    agent = AudioProcessingAgent()
    
    # Create audio with silence (zeros)
    silent_audio = np.zeros(1000)
    assert not agent._detect_voice_activity(silent_audio)
    
    # Create audio with speech (non-zero values above threshold)
    speech_audio = np.ones(1000) * 0.5
    assert agent._detect_voice_activity(speech_audio)
    
    # Create audio with low-level noise (below threshold)
    noise_audio = np.ones(1000) * 0.1
    assert not agent._detect_voice_activity(noise_audio)


@pytest.mark.asyncio
async def test_audio_normalization():
    """Test audio normalization method."""
    agent = AudioProcessingAgent()
    
    # Create test audio
    audio = np.array([0.1, 0.2, -0.5, 0.3])
    normalized = agent._normalize_audio(audio)
    
    # Verify maximum absolute value is 1.0
    assert np.max(np.abs(normalized)) == 1.0
    assert np.max(normalized) == 0.6  # 0.3 / 0.5 = 0.6
    assert np.min(normalized) == -1.0  # -0.5 / 0.5 = -1.0
    
    # Test with zero array
    zero_audio = np.zeros(10)
    normalized = agent._normalize_audio(zero_audio)
    assert np.array_equal(zero_audio, normalized)


@pytest.mark.asyncio
async def test_unsupported_message_type(audio_agent):
    """Test handling of unsupported message types."""
    message = Message(
        message_type=MessageType.TRANSCRIPTION_RESULT,  # Unsupported type
        source_agent_id="system",
        target_agent_id=audio_agent.agent_id
    )
    
    # Process the message
    result = await audio_agent.process_message(message)
    
    # Should return None for unsupported message types
    assert result is None 