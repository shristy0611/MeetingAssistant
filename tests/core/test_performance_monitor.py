"""
Tests for the performance monitoring module.

This module contains tests to verify the functionality of the AMPTALK
performance monitoring system.

Author: AMPTALK Team
Date: 2024
"""

import pytest
import asyncio
import time
from typing import Dict, List, Any

import psutil
from opentelemetry import metrics

from src.core.framework.agent import Agent, SimpleAgent
from src.core.framework.message import Message, MessageType, MessagePriority
from src.core.framework.orchestrator import Orchestrator
from src.core.utils.performance_monitor import (
    PerformanceMonitor, get_monitor, setup_performance_monitoring
)


class TestAgent(SimpleAgent):
    """Test agent that generates predictable performance metrics."""
    
    def __init__(self, processing_time: float = 0.1, *args, **kwargs):
        """Initialize with configurable processing time."""
        super().__init__(*args, **kwargs)
        self.processing_time = processing_time
        self.message_count = 0
        
        # Register message handlers
        self.register_message_handler(
            MessageType.STATUS_REQUEST,
            self.handle_status_request
        )
        self.register_message_handler(
            MessageType.INITIALIZE,
            self.handle_initialize
        )
    
    async def handle_status_request(self, message: Message) -> List[Message]:
        """Handle a status request with simulated processing time."""
        # Simulate some processing work
        await asyncio.sleep(self.processing_time)
        
        # Increment counter
        self.message_count += 1
        
        # Return a response
        response = Message(
            message_type=MessageType.STATUS_RESPONSE,
            source_agent_id=self.agent_id,
            target_agent_id=message.source_agent_id,
            payload={"count": self.message_count}
        )
        
        return [response]
    
    async def handle_initialize(self, message: Message) -> List[Message]:
        """Handle an initialization message with variable processing time."""
        # Get processing time from message if provided
        if "processing_time" in message.payload:
            self.processing_time = message.payload["processing_time"]
        
        # Simulate initialization work
        await asyncio.sleep(self.processing_time * 2)
        
        # Return a response
        response = Message(
            message_type=MessageType.STATUS_RESPONSE,
            source_agent_id=self.agent_id,
            target_agent_id=message.source_agent_id,
            payload={"status": "initialized", "processing_time": self.processing_time}
        )
        
        return [response]


@pytest.fixture
async def monitor():
    """Create a PerformanceMonitor instance for testing."""
    # Use a different port to avoid conflicts
    monitor = PerformanceMonitor(
        name="test_monitor",
        enable_prometheus=True,
        prometheus_port=8001,
        collection_interval_seconds=0.1  # Fast collection for testing
    )
    
    # Start collecting metrics
    await monitor.start_collecting()
    
    yield monitor
    
    # Clean up
    await monitor.stop_collecting()


@pytest.fixture
async def test_agent():
    """Create a test agent for monitoring."""
    agent = TestAgent(name="TestAgent", processing_time=0.05)
    await agent.start()
    
    yield agent
    
    # Clean up
    await agent.stop()


@pytest.mark.asyncio
async def test_system_metrics_collection(monitor):
    """Test collection of system-level metrics."""
    # Allow time for collection
    await asyncio.sleep(0.2)
    
    # System metrics should be collected
    # We can't assert exact values, but we can check they're reasonable
    
    # CPU usage should be between 0 and 100
    cpu_attrs = {}
    monitor.system_cpu_usage.observe(lambda result: cpu_attrs.update(result))
    assert 0 <= float(cpu_attrs["value"]) <= 100, "CPU usage should be between 0 and 100%"
    
    # Memory usage should be between 0 and 100
    mem_attrs = {}
    monitor.system_memory_usage.observe(lambda result: mem_attrs.update(result))
    assert 0 <= float(mem_attrs["value"]) <= 100, "Memory usage should be between 0 and 100%"
    
    # Process memory should be positive
    proc_mem_attrs = {}
    monitor.process_memory_usage.observe(lambda result: proc_mem_attrs.update(result))
    assert float(proc_mem_attrs["value"]) > 0, "Process memory usage should be positive"


@pytest.mark.asyncio
async def test_agent_metrics_collection(monitor, test_agent):
    """Test collection of agent-specific metrics."""
    # Add agent to monitor
    monitor.add_agent(test_agent)
    
    # Create and send a test message
    message = Message(
        message_type=MessageType.STATUS_REQUEST,
        source_agent_id="test",
        target_agent_id=test_agent.agent_id
    )
    
    # Process the message and wait for metrics to be collected
    await test_agent.process_message(message)
    await asyncio.sleep(0.2)  # Allow time for collection
    
    # Check that agent metrics were collected
    queue_attrs = {}
    monitor.agent_queue_size.observe(lambda result: queue_attrs.update(result))
    assert "agent_id" in queue_attrs["attributes"]
    assert queue_attrs["attributes"]["agent_name"] == test_agent.name
    
    # Verify agent is marked as running
    running_attrs = {}
    monitor.agent_is_running.observe(lambda result: running_attrs.update(result))
    assert running_attrs["value"] == 1, "Agent should be marked as running"
    
    # Process more messages to generate processing time metrics
    for _ in range(5):
        await test_agent.process_message(message)
    
    await asyncio.sleep(0.2)  # Allow time for collection
    
    # Clean up
    monitor.remove_agent(test_agent.agent_id)


@pytest.mark.asyncio
async def test_performance_monitor_with_orchestrator():
    """Test performance monitoring integration with the Orchestrator."""
    # Create an orchestrator with monitoring enabled
    orchestrator = Orchestrator(name="TestOrchestrator")
    
    # Create and register test agents
    agents = []
    for i in range(3):
        agent = TestAgent(name=f"TestAgent_{i}", processing_time=(i+1)*0.05)
        orchestrator.register_agent(agent)
        agents.append(agent)
    
    # Start the orchestrator (and agents)
    await orchestrator.start()
    
    # Allow time for initialization
    await asyncio.sleep(0.5)
    
    # Send messages to all agents
    for agent in agents:
        message = Message(
            message_type=MessageType.STATUS_REQUEST,
            source_agent_id="test",
            target_agent_id=agent.agent_id
        )
        await orchestrator.send_message_to_agent(message)
    
    # Allow processing time
    await asyncio.sleep(0.5)
    
    # Check system status
    status = await orchestrator.get_system_status()
    
    # Verify performance monitoring is active
    assert status["performance_monitoring"]["enabled"]
    assert status["performance_monitoring"]["active"]
    
    # Clean up
    await orchestrator.stop()


@pytest.mark.asyncio
async def test_global_monitor_instance():
    """Test the global monitor instance functionality."""
    # Get the global monitor (should create one)
    monitor1 = get_monitor(
        create_if_none=True,
        enable_prometheus=True,
        prometheus_port=8002  # Different port to avoid conflicts
    )
    
    # Get again, should return the same instance
    monitor2 = get_monitor(create_if_none=False)
    
    # Should be the same instance
    assert monitor1 is monitor2
    
    # Create an agent and add to the monitor
    agent = TestAgent(name="GlobalTestAgent")
    await agent.start()
    
    monitor1.add_agent(agent)
    
    # Verify agent is in both monitors (since they're the same instance)
    assert agent.agent_id in monitor1.monitored_agents
    assert agent.agent_id in monitor2.monitored_agents
    
    # Clean up
    await agent.stop()
    monitor1.remove_agent(agent.agent_id)
    await monitor1.stop_collecting()


@pytest.mark.asyncio
async def test_measure_execution_time_context_manager(monitor, test_agent):
    """Test the measure_execution_time context manager."""
    # Add agent to monitor
    monitor.add_agent(test_agent)
    
    # Use the context manager
    with monitor.measure_execution_time(test_agent.agent_id, "test_operation"):
        # Simulate some work
        await asyncio.sleep(0.1)
    
    # Allow time for metrics to be updated
    await asyncio.sleep(0.2)
    
    # Clean up
    monitor.remove_agent(test_agent.agent_id)


@pytest.mark.asyncio
async def test_setup_performance_monitoring():
    """Test the setup_performance_monitoring utility function."""
    # Set up monitoring
    monitor = await setup_performance_monitoring(
        enable_prometheus=True,
        prometheus_port=8003  # Different port to avoid conflicts
    )
    
    # Verify monitor is collecting
    assert monitor.is_collecting
    
    # Clean up
    await monitor.stop_collecting()


@pytest.mark.asyncio
async def test_monitor_error_handling(monitor):
    """Test error handling in the monitor."""
    # Create an agent that will cause errors
    agent = SimpleAgent(name="ErrorAgent")
    
    # Add agent but don't start it
    monitor.add_agent(agent)
    
    # Allow collection attempt
    await asyncio.sleep(0.2)
    
    # No errors should be raised, even though the agent isn't started
    
    # Clean up
    monitor.remove_agent(agent.agent_id)


@pytest.mark.asyncio
async def test_state_manager_metrics(monitor, test_agent):
    """Test state manager metrics collection."""
    # Enable state persistence for the agent
    test_agent.state_persistence_enabled = True
    test_agent.state_persistence_path = "/tmp/test_state"
    
    # Restart the agent to initialize state manager
    await test_agent.stop()
    await test_agent.start()
    
    # Add agent to monitor
    monitor.add_agent(test_agent)
    
    # Set some state values
    await test_agent.set_state_value("test_key1", "test_value1")
    await test_agent.set_state_value("test_key2", "test_value2")
    
    # Force a state save
    await test_agent._save_state()
    
    # Allow time for metrics collection
    await asyncio.sleep(0.2)
    
    # Clean up
    monitor.remove_agent(test_agent.agent_id)
    await test_agent.stop() 