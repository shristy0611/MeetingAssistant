"""
Tests for the Message module.

This module contains unit tests for the Message class and related functionality
in the AMPTALK multi-agent framework.

Author: AMPTALK Team
Date: 2024
"""

import pytest
import json
import time
from src.core.framework.message import (
    Message, MessageType, MessagePriority, MessageMetadata,
    create_audio_input_message, create_status_request
)


def test_message_creation():
    """Test basic message creation and attributes."""
    message = Message(
        message_type=MessageType.STATUS_REQUEST,
        source_agent_id="agent1",
        target_agent_id="agent2",
        priority=MessagePriority.HIGH,
        payload={"key": "value"}
    )
    
    assert message.message_type == MessageType.STATUS_REQUEST
    assert message.source_agent_id == "agent1"
    assert message.target_agent_id == "agent2"
    assert message.priority == MessagePriority.HIGH
    assert message.payload == {"key": "value"}
    assert isinstance(message.message_id, str)
    assert isinstance(message.metadata, MessageMetadata)


def test_message_serialization():
    """Test message serialization to and from JSON."""
    original = Message(
        message_type=MessageType.AUDIO_INPUT,
        source_agent_id="audio_agent",
        target_agent_id="transcription_agent",
        priority=MessagePriority.NORMAL,
        payload={"sample_rate": 16000}
    )
    
    # Convert to JSON
    json_str = original.to_json()
    assert isinstance(json_str, str)
    
    # Parse JSON back to message
    parsed = Message.from_json(json_str)
    assert parsed.message_type == original.message_type
    assert parsed.source_agent_id == original.source_agent_id
    assert parsed.target_agent_id == original.target_agent_id
    assert parsed.priority == original.priority
    assert parsed.payload == original.payload
    assert parsed.message_id == original.message_id


def test_message_metadata():
    """Test message metadata functionality."""
    message = Message(
        message_type=MessageType.TRANSCRIPTION_REQUEST,
        source_agent_id="test",
        target_agent_id="test"
    )
    
    # Test processing time tracking
    message.metadata.mark_processing_start()
    time.sleep(0.1)  # Simulate processing
    message.metadata.mark_processing_complete()
    
    assert message.metadata.processing_time is not None
    assert message.metadata.processing_time >= 0.1
    
    # Test hop count
    assert message.metadata.hop_count == 0
    message.metadata.increment_hop()
    assert message.metadata.hop_count == 1
    
    # Test retry count
    assert message.metadata.retry_count == 0
    message.metadata.increment_retry()
    assert message.metadata.retry_count == 1
    
    # Test error handling
    assert message.metadata.error is None
    message.metadata.set_error("Test error")
    assert message.metadata.error == "Test error"


def test_message_convenience_functions():
    """Test convenience functions for creating common message types."""
    # Test status request creation
    status_req = create_status_request("source", "target")
    assert status_req.message_type == MessageType.STATUS_REQUEST
    assert status_req.source_agent_id == "source"
    assert status_req.target_agent_id == "target"
    
    # Test audio input message creation
    audio_data = b'\0' * 1000
    audio_msg = create_audio_input_message(
        source_id="audio_agent",
        target_id="transcription_agent",
        audio_data=audio_data,
        sample_rate=16000
    )
    assert audio_msg.message_type == MessageType.AUDIO_INPUT
    assert audio_msg.source_agent_id == "audio_agent"
    assert audio_msg.target_agent_id == "transcription_agent"
    assert audio_msg.priority == MessagePriority.HIGH
    assert audio_msg.payload["sample_rate"] == 16000
    assert audio_msg.payload["audio_data_length"] == 1000


def test_message_expiration():
    """Test message TTL and expiration functionality."""
    message = Message(
        message_type=MessageType.STATUS_REQUEST,
        source_agent_id="test",
        target_agent_id="test"
    )
    
    # Message should not be expired initially
    assert not message.is_expired()
    
    # Increment hop count to TTL limit
    for _ in range(message.metadata.ttl):
        message.metadata.increment_hop()
    
    # Message should now be expired
    assert message.is_expired()


def test_message_retry_limits():
    """Test message retry functionality and limits."""
    message = Message(
        message_type=MessageType.STATUS_REQUEST,
        source_agent_id="test",
        target_agent_id="test"
    )
    
    # Should be able to retry initially
    assert message.can_retry()
    
    # Increment retries to limit
    for _ in range(message.metadata.max_retries):
        message.metadata.increment_retry()
    
    # Should not be able to retry anymore
    assert not message.can_retry()


def test_message_latency_tracking():
    """Test message latency tracking functionality."""
    message = Message(
        message_type=MessageType.STATUS_REQUEST,
        source_agent_id="test",
        target_agent_id="test"
    )
    
    # Initially, latency should be None
    assert message.get_latency() is None
    
    # Simulate processing
    message.metadata.mark_processing_start()
    time.sleep(0.1)
    message.metadata.mark_processing_complete()
    
    # Latency should now be measurable
    latency = message.get_latency()
    assert latency is not None
    assert latency >= 0.1 