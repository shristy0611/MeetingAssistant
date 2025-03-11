"""
Tests for the Agent base class.

This module contains unit tests for the Agent class, which is the foundation
for all specialized agents in the AMPTALK system.

Author: AMPTALK Team
Date: 2024
"""

import pytest
from unittest.mock import AsyncMock, patch
import asyncio
from typing import List, Optional

from src.core.framework.agent import Agent, SimpleAgent
from src.core.framework.message import (
    Message, MessageType, MessagePriority, create_status_request
)


@pytest.mark.asyncio
async def test_agent_initialization():
    """Test agent initialization and basic attributes."""
    # Test with default parameters
    agent = SimpleAgent()
    assert agent.agent_id is not None
    assert agent.name == "SimpleAgent"
    assert not agent.is_running
    assert len(agent.connected_agents) == 0
    assert isinstance(agent.input_queue, asyncio.Queue)
    
    # Test with custom parameters
    custom_agent = SimpleAgent(
        agent_id="test_agent_1",
        name="TestAgent"
    )
    assert custom_agent.agent_id == "test_agent_1"
    assert custom_agent.name == "TestAgent"


@pytest.mark.asyncio
async def test_agent_lifecycle():
    """Test agent start and stop functionality."""
    agent = SimpleAgent()
    
    # Test start
    await agent.start()
    assert agent.is_running
    assert len(agent.tasks) == 1  # Should have message processing task
    
    # Test double start (should not create additional tasks)
    await agent.start()
    assert len(agent.tasks) == 1
    
    # Test stop
    await agent.stop()
    assert not agent.is_running
    assert len(agent.tasks) == 0
    
    # Test double stop (should not raise errors)
    await agent.stop()
    assert not agent.is_running


@pytest.mark.asyncio
async def test_agent_message_processing():
    """Test agent message processing functionality."""
    agent = SimpleAgent()
    await agent.start()
    
    try:
        # Create test message
        message = create_status_request("test_source", agent.agent_id)
        
        # Send message to agent
        await agent.enqueue_message(message)
        
        # Allow time for processing
        await asyncio.sleep(0.1)
        
        # Verify message was processed
        assert agent.stats.messages_received == 1
        assert agent.stats.messages_processed == 1
        assert MessageType.STATUS_REQUEST in agent.stats.message_types_processed
        
    finally:
        await agent.stop()


@pytest.mark.asyncio
async def test_agent_message_routing():
    """Test message routing between agents."""
    agent1 = SimpleAgent(name="Agent1")
    agent2 = SimpleAgent(name="Agent2")
    
    await agent1.start()
    await agent2.start()
    
    try:
        # Connect agents
        agent1.connect(agent2)
        
        # Create and send message
        message = Message(
            message_type=MessageType.STATUS_REQUEST,
            source_agent_id=agent1.agent_id,
            target_agent_id=agent2.agent_id,
            priority=MessagePriority.NORMAL
        )
        
        success = await agent1.send_message(message)
        assert success
        
        # Allow time for processing
        await asyncio.sleep(0.1)
        
        # Verify message was received and processed
        assert agent2.stats.messages_received == 1
        assert message.metadata.hop_count == 1
        
    finally:
        await agent1.stop()
        await agent2.stop()


@pytest.mark.asyncio
async def test_agent_error_handling():
    """Test agent error handling during message processing."""
    class ErrorAgent(SimpleAgent):
        async def process_message(self, message: Message) -> Optional[List[Message]]:
            raise ValueError("Test error")
    
    agent = ErrorAgent()
    await agent.start()
    
    try:
        # Send test message
        message = create_status_request("test_source", agent.agent_id)
        await agent.enqueue_message(message)
        
        # Allow time for processing
        await asyncio.sleep(0.1)
        
        # Verify error was handled
        assert agent.stats.messages_failed == 1
        assert agent.stats.messages_processed == 0
        assert message.metadata.error is not None
        
    finally:
        await agent.stop()


@pytest.mark.asyncio
async def test_agent_configuration():
    """Test agent configuration functionality."""
    agent = SimpleAgent()
    
    # Test initial configuration
    assert isinstance(agent.config, dict)
    assert len(agent.config) == 0
    
    # Test configuration update
    test_config = {
        "param1": "value1",
        "param2": 42,
        "nested": {
            "key": "value"
        }
    }
    
    agent.configure(test_config)
    assert agent.config == test_config
    
    # Test configuration merge
    additional_config = {
        "param3": "value3",
        "nested": {
            "new_key": "new_value"
        }
    }
    
    agent.configure(additional_config)
    assert agent.config["param1"] == "value1"
    assert agent.config["param3"] == "value3"
    assert agent.config["nested"]["new_key"] == "new_value"


@pytest.mark.asyncio
async def test_agent_status():
    """Test agent status reporting."""
    agent = SimpleAgent()
    await agent.start()
    
    try:
        # Get initial status
        status = await agent.get_status()
        assert status["agent_id"] == agent.agent_id
        assert status["name"] == agent.name
        assert status["is_running"]
        assert status["queue_size"] == 0
        assert isinstance(status["stats"], dict)
        
        # Send a test message
        message = create_status_request("test_source", agent.agent_id)
        await agent.enqueue_message(message)
        
        # Allow time for processing
        await asyncio.sleep(0.1)
        
        # Get updated status
        status = await agent.get_status()
        assert status["stats"]["messages"]["received"] == 1
        assert status["stats"]["messages"]["processed"] == 1
        
    finally:
        await agent.stop()


@pytest.mark.asyncio
async def test_agent_message_expiration():
    """Test message TTL and expiration handling."""
    agent1 = SimpleAgent(name="Agent1")
    agent2 = SimpleAgent(name="Agent2")
    
    await agent1.start()
    await agent2.start()
    
    try:
        # Connect agents
        agent1.connect(agent2)
        
        # Create message with TTL of 1
        message = Message(
            message_type=MessageType.STATUS_REQUEST,
            source_agent_id=agent1.agent_id,
            target_agent_id=agent2.agent_id,
            priority=MessagePriority.NORMAL
        )
        message.metadata.ttl = 1
        
        # First hop should succeed
        success = await agent1.send_message(message)
        assert success
        
        # Allow time for processing
        await asyncio.sleep(0.1)
        
        # Second hop should fail due to TTL
        response = agent2.received_messages[0]
        success = await agent2.send_message(response)
        assert not success
        
    finally:
        await agent1.stop()
        await agent2.stop()


@pytest.mark.asyncio
async def test_agent_performance_tracking():
    """Test agent performance statistics tracking."""
    agent = SimpleAgent()
    await agent.start()
    
    try:
        # Send multiple messages
        for _ in range(3):
            message = create_status_request("test_source", agent.agent_id)
            await agent.enqueue_message(message)
        
        # Allow time for processing
        await asyncio.sleep(0.3)
        
        # Verify statistics
        assert agent.stats.messages_received == 3
        assert agent.stats.messages_processed == 3
        assert agent.stats.messages_sent == 3
        assert agent.stats.total_processing_time > 0
        assert agent.stats.avg_processing_time > 0
        assert agent.stats.get_success_rate() == 1.0
        
    finally:
        await agent.stop() 