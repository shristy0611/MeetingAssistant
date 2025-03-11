"""
Tests for the Orchestrator module.

This module contains unit tests for the Orchestrator class, which manages
the multi-agent system's lifecycle and communication.

Author: AMPTALK Team
Date: 2024
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from typing import List, Dict, Any

from src.core.framework.orchestrator import Orchestrator
from src.core.framework.agent import Agent, SimpleAgent
from src.core.framework.message import (
    Message, MessageType, MessagePriority, create_status_request
)


@pytest.fixture
async def test_orchestrator():
    """Create a test orchestrator instance."""
    orchestrator = Orchestrator(name="TestOrchestrator")
    yield orchestrator
    await orchestrator.stop()


@pytest.fixture
async def test_agents():
    """Create a set of test agents."""
    agents = [
        SimpleAgent(name=f"TestAgent{i}")
        for i in range(3)
    ]
    for agent in agents:
        await agent.start()
    
    yield agents
    
    for agent in agents:
        await agent.stop()


@pytest.mark.asyncio
async def test_orchestrator_initialization(test_orchestrator):
    """Test orchestrator initialization and basic attributes."""
    assert test_orchestrator.name == "TestOrchestrator"
    assert test_orchestrator.orchestrator_id is not None
    assert not test_orchestrator.is_running
    assert len(test_orchestrator.agents) == 0
    assert len(test_orchestrator.agent_groups) == 0
    assert isinstance(test_orchestrator.tasks, list)


@pytest.mark.asyncio
async def test_agent_registration(test_orchestrator, test_agents):
    """Test agent registration and group assignment."""
    # Register agents with groups
    test_orchestrator.register_agent(test_agents[0], groups=["group1"])
    test_orchestrator.register_agent(test_agents[1], groups=["group1", "group2"])
    test_orchestrator.register_agent(test_agents[2], groups=["group2"])
    
    # Verify agent registration
    assert len(test_orchestrator.agents) == 3
    assert test_agents[0].agent_id in test_orchestrator.agents
    
    # Verify group assignments
    assert len(test_orchestrator.agent_groups["group1"]) == 2
    assert len(test_orchestrator.agent_groups["group2"]) == 2
    assert test_agents[0].agent_id in test_orchestrator.agent_groups["group1"]
    assert test_agents[1].agent_id in test_orchestrator.agent_groups["group2"]
    
    # Test duplicate registration
    test_orchestrator.register_agent(test_agents[0])
    assert len(test_orchestrator.agents) == 3  # Should not increase


@pytest.mark.asyncio
async def test_agent_unregistration(test_orchestrator, test_agents):
    """Test agent unregistration and cleanup."""
    # Register agents
    for agent in test_agents:
        test_orchestrator.register_agent(agent, groups=["test_group"])
    
    # Verify initial state
    assert len(test_orchestrator.agents) == 3
    assert len(test_orchestrator.agent_groups["test_group"]) == 3
    
    # Unregister one agent
    test_orchestrator.unregister_agent(test_agents[0].agent_id)
    
    # Verify agent removal
    assert len(test_orchestrator.agents) == 2
    assert test_agents[0].agent_id not in test_orchestrator.agents
    assert test_agents[0].agent_id not in test_orchestrator.agent_groups["test_group"]
    
    # Test unregistering non-existent agent
    test_orchestrator.unregister_agent("non_existent_id")  # Should not raise error


@pytest.mark.asyncio
async def test_group_management(test_orchestrator, test_agents):
    """Test agent group management functionality."""
    agent = test_agents[0]
    
    # Add agent to group
    test_orchestrator.register_agent(agent)
    test_orchestrator.add_agent_to_group(agent.agent_id, "test_group")
    assert agent.agent_id in test_orchestrator.agent_groups["test_group"]
    
    # Add to another group
    test_orchestrator.add_agent_to_group(agent.agent_id, "another_group")
    assert agent.agent_id in test_orchestrator.agent_groups["another_group"]
    
    # Remove from group
    test_orchestrator.remove_agent_from_group(agent.agent_id, "test_group")
    assert "test_group" in test_orchestrator.agent_groups
    assert agent.agent_id not in test_orchestrator.agent_groups["test_group"]
    
    # Remove last agent from group (should remove group)
    test_orchestrator.remove_agent_from_group(agent.agent_id, "another_group")
    assert "another_group" not in test_orchestrator.agent_groups


@pytest.mark.asyncio
async def test_agent_connections(test_orchestrator, test_agents):
    """Test agent connection management."""
    # Register agents
    for agent in test_agents:
        test_orchestrator.register_agent(agent)
    
    # Connect two agents
    test_orchestrator.connect_agents(
        test_agents[0].agent_id,
        test_agents[1].agent_id,
        bidirectional=True
    )
    
    # Verify connections
    assert test_agents[1].agent_id in test_agents[0].connected_agents
    assert test_agents[0].agent_id in test_agents[1].connected_agents
    
    # Test group connections
    test_orchestrator.add_agent_to_group(test_agents[0].agent_id, "group1")
    test_orchestrator.add_agent_to_group(test_agents[1].agent_id, "group2")
    
    test_orchestrator.connect_groups("group1", "group2")
    assert test_agents[1].agent_id in test_agents[0].connected_agents


@pytest.mark.asyncio
async def test_orchestrator_lifecycle(test_orchestrator, test_agents):
    """Test orchestrator start/stop functionality."""
    # Register agents
    for agent in test_agents:
        test_orchestrator.register_agent(agent)
    
    # Start orchestrator
    await test_orchestrator.start()
    assert test_orchestrator.is_running
    assert len(test_orchestrator.tasks) > 0  # Should have monitoring task
    
    # Verify agents are running
    for agent in test_orchestrator.agents.values():
        assert agent.is_running
    
    # Stop orchestrator
    await test_orchestrator.stop()
    assert not test_orchestrator.is_running
    assert len(test_orchestrator.tasks) == 0
    
    # Verify agents are stopped
    for agent in test_orchestrator.agents.values():
        assert not agent.is_running


@pytest.mark.asyncio
async def test_message_broadcasting(test_orchestrator, test_agents):
    """Test message broadcasting functionality."""
    # Register agents in a group
    for agent in test_agents:
        test_orchestrator.register_agent(agent, groups=["broadcast_group"])
    
    await test_orchestrator.start()
    
    try:
        # Create broadcast message
        message = Message(
            message_type=MessageType.STATUS_REQUEST,
            source_agent_id="system",
            target_agent_id="broadcast",
            priority=MessagePriority.NORMAL
        )
        
        # Broadcast to group
        await test_orchestrator.broadcast_message(
            message,
            group_name="broadcast_group"
        )
        
        # Allow time for processing
        await asyncio.sleep(0.1)
        
        # Verify all agents received the message
        for agent in test_agents:
            assert agent.stats.messages_received == 1
        
    finally:
        await test_orchestrator.stop()


@pytest.mark.asyncio
async def test_event_hooks(test_orchestrator, test_agents):
    """Test event hook functionality."""
    events_triggered = []
    
    async def test_hook(*args, **kwargs):
        events_triggered.append(args[0] if args else None)
    
    # Register hooks
    test_orchestrator.register_event_hook("agent_added", test_hook)
    test_orchestrator.register_event_hook("agent_removed", test_hook)
    test_orchestrator.register_event_hook("system_started", test_hook)
    test_orchestrator.register_event_hook("system_stopped", test_hook)
    
    # Trigger events
    test_orchestrator.register_agent(test_agents[0])
    await test_orchestrator.start()
    test_orchestrator.unregister_agent(test_agents[0].agent_id)
    await test_orchestrator.stop()
    
    # Verify events were triggered
    assert len(events_triggered) == 4
    assert test_agents[0] in events_triggered  # agent_added event


@pytest.mark.asyncio
async def test_system_status(test_orchestrator, test_agents):
    """Test system status reporting."""
    # Register agents
    for agent in test_agents:
        test_orchestrator.register_agent(agent, groups=["status_group"])
    
    await test_orchestrator.start()
    
    try:
        # Get system status
        status = await test_orchestrator.get_system_status()
        
        # Verify status content
        assert status["orchestrator"]["name"] == test_orchestrator.name
        assert status["orchestrator"]["is_running"]
        assert status["orchestrator"]["agent_count"] == len(test_agents)
        assert "status_group" in status["groups"]
        assert len(status["agents"]) == len(test_agents)
        
        # Verify agent statuses
        for agent_id, agent_status in status["agents"].items():
            assert agent_status["is_running"]
            assert isinstance(agent_status["stats"], dict)
        
    finally:
        await test_orchestrator.stop()


@pytest.mark.asyncio
async def test_error_handling(test_orchestrator):
    """Test error handling in the orchestrator."""
    class ErrorAgent(SimpleAgent):
        async def start(self):
            raise ValueError("Test error")
    
    error_agent = ErrorAgent()
    test_orchestrator.register_agent(error_agent)
    
    # Start should not fail even if agent fails
    await test_orchestrator.start()
    assert test_orchestrator.is_running
    
    # Error hook should be triggered
    events = []
    async def error_hook(source: str, error: str):
        events.append((source, error))
    
    test_orchestrator.register_event_hook("error", error_hook)
    
    # Trigger an error
    await test_orchestrator._trigger_event("error", "test", "error message")
    assert len(events) == 1
    assert events[0] == ("test", "error message")


@pytest.mark.asyncio
async def test_agent_pipeline(test_orchestrator, test_agents):
    """Test agent pipeline creation and management."""
    # Register agents
    for agent in test_agents:
        test_orchestrator.register_agent(agent)
    
    # Create pipeline configuration
    pipeline_config = [
        {
            "agent_id": test_agents[0].agent_id,
            "connects_to": [test_agents[1].agent_id]
        },
        {
            "agent_id": test_agents[1].agent_id,
            "connects_to": [test_agents[2].agent_id]
        },
        {
            "agent_id": test_agents[2].agent_id,
            "connects_to": []
        }
    ]
    
    # Create pipeline
    test_orchestrator.create_agent_pipeline(pipeline_config)
    
    # Verify connections
    assert test_agents[1].agent_id in test_agents[0].connected_agents
    assert test_agents[2].agent_id in test_agents[1].connected_agents
    assert len(test_agents[2].connected_agents) == 0  # Last in pipeline
    
    # Test message flow through pipeline
    await test_orchestrator.start()
    
    try:
        # Send message to first agent
        message = create_status_request("system", test_agents[0].agent_id)
        await test_agents[0].enqueue_message(message)
        
        # Allow time for processing
        await asyncio.sleep(0.3)
        
        # Verify message propagation
        assert test_agents[0].stats.messages_processed == 1
        assert test_agents[1].stats.messages_received == 1
        assert test_agents[2].stats.messages_received == 1
        
    finally:
        await test_orchestrator.stop() 