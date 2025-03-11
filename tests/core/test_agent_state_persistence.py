"""
Tests for agent state persistence functionality.

These tests verify that the Agent class can correctly persist and restore
its state using the StateManager.

Author: AMPTALK Team
Date: 2024
"""

import pytest
import asyncio
import tempfile
import os
from typing import Dict, Any, List

from src.core.framework.agent import Agent, SimpleAgent
from src.core.framework.message import Message, MessageType
from src.core.utils.state_manager import StorageType


@pytest.fixture
def temp_dir():
    """Create a temporary directory for state files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


class TestAgent(SimpleAgent):
    """Test agent class with custom state handling."""
    
    def __init__(self, agent_id=None, name=None):
        """Initialize the test agent."""
        super().__init__(agent_id=agent_id, name=name)
        
        # Define a message handler
        self.register_message_handler(
            MessageType.STATUS_REQUEST,
            self.handle_status_request
        )
    
    async def handle_status_request(self, message: Message) -> List[Message]:
        """Handle a status request message."""
        # Use state data in the response
        counter = await self.get_state_value("counter", 0)
        await self.set_state_value("counter", counter + 1)
        
        # Create a response with state data
        response = Message(
            message_type=MessageType.STATUS_RESPONSE,
            source_agent_id=self.agent_id,
            target_agent_id=message.source_agent_id,
            payload={
                "counter": counter + 1,
                "status": "ok"
            }
        )
        
        return [response]


@pytest.mark.asyncio
async def test_agent_in_memory_state_persistence():
    """Test that an agent can persist and restore state in memory."""
    # Create an agent with in-memory state persistence
    agent = TestAgent(name="TestAgent")
    agent.state_persistence_enabled = True
    
    # Start the agent
    await agent.start()
    
    try:
        # Set some state values
        await agent.set_state_value("counter", 42)
        await agent.set_state_value("test_key", "test_value")
        
        # Manually save state
        await agent._save_state()
        
        # Clear state to verify restoration
        agent.state.clear()
        assert "counter" not in agent.state
        assert "test_key" not in agent.state
        
        # Load state
        await agent._load_state()
        
        # Verify state was restored
        assert await agent.get_state_value("counter") == 42
        assert await agent.get_state_value("test_key") == "test_value"
    finally:
        # Stop the agent
        await agent.stop()


@pytest.mark.asyncio
async def test_agent_file_state_persistence(temp_dir):
    """Test that an agent can persist and restore state from files."""
    # Create an agent with file-based state persistence
    agent = TestAgent(name="FileTestAgent")
    agent.state_persistence_enabled = True
    agent.state_persistence_path = temp_dir
    
    # Start the agent
    await agent.start()
    
    try:
        # Verify the state manager was initialized with file storage
        assert agent.state_manager is not None
        assert agent.state_manager.storage_type == StorageType.FILE
        
        # Set some state values
        await agent.set_state_value("counter", 100)
        await agent.set_state_value("complex_data", {"nested": {"value": [1, 2, 3]}})
        
        # Manually save state
        await agent._save_state()
        
        # Verify the state file was created
        state_files = [f for f in os.listdir(temp_dir) if f.endswith(".state")]
        assert len(state_files) == 1
        
        # Create a new agent with the same ID to test restoration
        agent2 = TestAgent(agent_id=agent.agent_id, name="FileTestAgent2")
        agent2.state_persistence_enabled = True
        agent2.state_persistence_path = temp_dir
        
        # Start the new agent
        await agent2.start()
        
        try:
            # Verify state was restored
            assert await agent2.get_state_value("counter") == 100
            complex_data = await agent2.get_state_value("complex_data")
            assert complex_data["nested"]["value"] == [1, 2, 3]
        finally:
            # Stop the second agent
            await agent2.stop()
    finally:
        # Stop the first agent
        await agent.stop()


@pytest.mark.asyncio
async def test_agent_message_handling_with_state():
    """Test that an agent's message handling can use persisted state."""
    # Create an agent with state persistence
    agent = TestAgent(name="StateHandlingAgent")
    agent.state_persistence_enabled = True
    
    # Start the agent
    await agent.start()
    
    try:
        # Send a status request message to increment counter
        status_request = Message(
            message_type=MessageType.STATUS_REQUEST,
            source_agent_id="test",
            target_agent_id=agent.agent_id
        )
        
        # Process the message
        responses = await agent.process_message(status_request)
        
        # Verify response
        assert len(responses) == 1
        assert responses[0].message_type == MessageType.STATUS_RESPONSE
        assert responses[0].payload["counter"] == 1
        
        # Send another request to check counter increment
        responses = await agent.process_message(status_request)
        
        # Verify counter was incremented
        assert responses[0].payload["counter"] == 2
        
        # Manually save state
        await agent._save_state()
        
        # Reset agent state
        agent.state.clear()
        
        # Load state
        await agent._load_state()
        
        # Send another request to check persistence
        responses = await agent.process_message(status_request)
        
        # Verify counter continued from last value
        assert responses[0].payload["counter"] == 3
    finally:
        # Stop the agent
        await agent.stop()


@pytest.mark.asyncio
async def test_agent_state_persistence_interval():
    """Test that agent state is persisted at regular intervals."""
    # Create an agent with a short persistence interval
    agent = TestAgent(name="IntervalAgent")
    agent.state_persistence_enabled = True
    agent.state_persistence_interval = 0.1  # 100ms interval
    
    # Start the agent
    await agent.start()
    
    try:
        # Set a state value
        await agent.set_state_value("test_key", "initial_value")
        
        # Wait for at least one persistence cycle
        await asyncio.sleep(0.2)
        
        # Change the state value
        await agent.set_state_value("test_key", "updated_value")
        
        # Create a new agent to verify persistence
        agent2 = TestAgent(agent_id=agent.agent_id, name="IntervalAgent2")
        agent2.state_persistence_enabled = True
        
        # Start the new agent
        await agent2.start()
        
        try:
            # Verify the state was persisted
            assert await agent2.get_state_value("test_key") == "updated_value"
        finally:
            # Stop the second agent
            await agent2.stop()
    finally:
        # Stop the first agent
        await agent.stop()


@pytest.mark.asyncio
async def test_agent_critical_state_immediate_persistence():
    """Test that critical state values are persisted immediately."""
    # Create an agent with state persistence
    agent = TestAgent(name="CriticalStateAgent")
    agent.state_persistence_enabled = True
    
    # Start the agent
    await agent.start()
    
    try:
        # Set a normal state value
        await agent.set_state_value("normal_key", "normal_value")
        
        # Set a critical state value
        await agent.set_state_value("critical_data", "critical_value")
        
        # Create a new agent to verify immediate persistence of critical data
        agent2 = TestAgent(agent_id=agent.agent_id, name="CriticalStateAgent2")
        agent2.state_persistence_enabled = True
        
        # Start the new agent
        await agent2.start()
        
        try:
            # Verify the critical state was persisted immediately
            assert await agent2.get_state_value("critical_data") == "critical_value"
            
            # The normal key might not be persisted yet due to the interval
            # so we don't check for it
        finally:
            # Stop the second agent
            await agent2.stop()
    finally:
        # Stop the first agent
        await agent.stop()


@pytest.mark.asyncio
async def test_agent_clear_state():
    """Test that an agent can clear its state."""
    # Create an agent with state persistence
    agent = TestAgent(name="ClearStateAgent")
    agent.state_persistence_enabled = True
    
    # Start the agent
    await agent.start()
    
    try:
        # Set some state values
        await agent.set_state_value("key1", "value1")
        await agent.set_state_value("key2", "value2")
        
        # Manually save state
        await agent._save_state()
        
        # Clear state
        await agent.clear_state()
        
        # Verify state is cleared in memory
        assert "key1" not in agent.state
        assert "key2" not in agent.state
        
        # Create a new agent to verify persistence was cleared
        agent2 = TestAgent(agent_id=agent.agent_id, name="ClearStateAgent2")
        agent2.state_persistence_enabled = True
        
        # Start the new agent
        await agent2.start()
        
        try:
            # Verify state was not restored
            assert await agent2.get_state_value("key1") is None
            assert await agent2.get_state_value("key2") is None
        finally:
            # Stop the second agent
            await agent2.stop()
    finally:
        # Stop the first agent
        await agent.stop() 