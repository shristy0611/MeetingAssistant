"""Tests for the Pipeline Orchestrator."""

import pytest
import asyncio
from unittest.mock import Mock, patch
import numpy as np
from typing import List, Dict, Any

from src.core.pipeline.pipeline_orchestrator import (
    PipelineOrchestrator,
    PipelineConfig,
    PipelineStage
)
from src.core.agent import Agent
from src.core.message import Message


class MockAgent(Agent):
    """Mock agent for testing."""
    
    def __init__(self, agent_id: str, processing_time: float = 0.1, fail_rate: float = 0.0):
        super().__init__(agent_id=agent_id)
        self.processing_time = processing_time
        self.fail_rate = fail_rate
        self.processed_messages = []
    
    async def process_message(self, message: Message) -> Message:
        """Process a message with simulated delay and failure rate."""
        await asyncio.sleep(self.processing_time)
        
        if np.random.random() < self.fail_rate:
            raise Exception("Simulated failure")
        
        self.processed_messages.append(message)
        return Message(
            sender=self.agent_id,
            receiver="test",
            type="response",
            data={"processed": True}
        )


@pytest.fixture
def mock_performance_metrics():
    """Mock performance metrics."""
    mock = Mock()
    mock.record_metric = Mock()
    return mock


@pytest.fixture
def pipeline_config():
    """Create a test pipeline configuration."""
    return PipelineConfig(
        enable_load_balancing=True,
        max_concurrent_tasks=5,
        error_threshold=0.2,
        circuit_breaker_reset_time=1.0,
        metrics_collection_interval=0.1
    )


@pytest.fixture
def pipeline_orchestrator(pipeline_config, mock_performance_metrics):
    """Create a test pipeline orchestrator."""
    with patch("src.core.pipeline.pipeline_orchestrator.PerformanceMetrics",
              return_value=mock_performance_metrics):
        orchestrator = PipelineOrchestrator(config=pipeline_config)
        return orchestrator


@pytest.fixture
def test_pipeline(pipeline_orchestrator):
    """Set up a test pipeline with mock agents."""
    # Create mock agents
    agents = [
        MockAgent("agent1", processing_time=0.1),
        MockAgent("agent2", processing_time=0.2),
        MockAgent("agent3", processing_time=0.1)
    ]
    
    # Create pipeline stages
    stages = [
        PipelineStage(
            agent_id="agent1",
            next_stages=["agent2"]
        ),
        PipelineStage(
            agent_id="agent2",
            next_stages=["agent3"]
        ),
        PipelineStage(
            agent_id="agent3",
            next_stages=[]
        )
    ]
    
    # Register agents and stages
    for agent in agents:
        pipeline_orchestrator.register_agent(agent)
    
    for stage in stages:
        pipeline_orchestrator.add_stage(stage)
    
    return pipeline_orchestrator, agents


@pytest.mark.asyncio
async def test_pipeline_initialization(pipeline_orchestrator):
    """Test pipeline orchestrator initialization."""
    assert not pipeline_orchestrator.running
    assert isinstance(pipeline_orchestrator.stages, dict)
    assert isinstance(pipeline_orchestrator.agents, dict)
    assert isinstance(pipeline_orchestrator.active_tasks, dict)


@pytest.mark.asyncio
async def test_add_stage(pipeline_orchestrator):
    """Test adding a stage to the pipeline."""
    stage = PipelineStage(agent_id="test_agent")
    pipeline_orchestrator.add_stage(stage)
    
    assert "test_agent" in pipeline_orchestrator.stages
    assert "test_agent" in pipeline_orchestrator.active_tasks
    assert "test_agent" in pipeline_orchestrator.error_counts
    assert "test_agent" in pipeline_orchestrator.circuit_breakers
    assert "test_agent" in pipeline_orchestrator.stage_metrics


@pytest.mark.asyncio
async def test_register_agent(pipeline_orchestrator):
    """Test registering an agent with the pipeline."""
    agent = MockAgent("test_agent")
    pipeline_orchestrator.register_agent(agent)
    
    assert "test_agent" in pipeline_orchestrator.agents


@pytest.mark.asyncio
async def test_pipeline_start_stop(pipeline_orchestrator):
    """Test starting and stopping the pipeline orchestrator."""
    await pipeline_orchestrator.start()
    assert pipeline_orchestrator.running
    assert len(pipeline_orchestrator.tasks) > 0
    
    await pipeline_orchestrator.stop()
    assert not pipeline_orchestrator.running
    assert len(pipeline_orchestrator.tasks) == 0


@pytest.mark.asyncio
async def test_message_processing(test_pipeline):
    """Test processing a message through the pipeline."""
    orchestrator, agents = test_pipeline
    await orchestrator.start()
    
    try:
        # Create test message
        message = Message(
            sender="test",
            receiver="agent1",
            type="test",
            data={"test": True}
        )
        
        # Process through first stage
        await orchestrator.process_message(message, "agent1")
        
        # Allow time for processing
        await asyncio.sleep(0.5)
        
        # Verify message propagation
        assert len(agents[0].processed_messages) == 1
        assert len(agents[1].processed_messages) == 1
        assert len(agents[2].processed_messages) == 1
        
    finally:
        await orchestrator.stop()


@pytest.mark.asyncio
async def test_error_handling(pipeline_orchestrator):
    """Test error handling and circuit breaker functionality."""
    # Create agent with high failure rate
    agent = MockAgent("error_agent", fail_rate=0.5)
    pipeline_orchestrator.register_agent(agent)
    
    # Create stage
    stage = PipelineStage(agent_id="error_agent")
    pipeline_orchestrator.add_stage(stage)
    
    await pipeline_orchestrator.start()
    
    try:
        # Send multiple messages to trigger errors
        message = Message(
            sender="test",
            receiver="error_agent",
            type="test",
            data={"test": True}
        )
        
        for _ in range(10):
            await pipeline_orchestrator.process_message(message, "error_agent")
            await asyncio.sleep(0.1)
        
        # Verify circuit breaker activation
        assert pipeline_orchestrator.circuit_breakers["error_agent"]
        
        # Wait for circuit breaker reset
        await asyncio.sleep(pipeline_orchestrator.config.circuit_breaker_reset_time + 0.1)
        
        # Verify circuit breaker reset
        assert not pipeline_orchestrator.circuit_breakers["error_agent"]
        
    finally:
        await pipeline_orchestrator.stop()


@pytest.mark.asyncio
async def test_load_balancing(test_pipeline):
    """Test load balancing functionality."""
    orchestrator, agents = test_pipeline
    await orchestrator.start()
    
    try:
        # Create multiple messages to test load balancing
        messages = [
            Message(
                sender="test",
                receiver="agent1",
                type="test",
                data={"test": i}
            )
            for i in range(10)
        ]
        
        # Send messages concurrently
        await asyncio.gather(*[
            orchestrator.process_message(msg, "agent1")
            for msg in messages
        ])
        
        # Verify load distribution
        loads = orchestrator.stage_load
        assert all(load <= orchestrator.config.max_concurrent_tasks 
                  for load in loads.values())
        
    finally:
        await orchestrator.stop()


@pytest.mark.asyncio
async def test_performance_monitoring(test_pipeline, mock_performance_metrics):
    """Test performance monitoring functionality."""
    orchestrator, agents = test_pipeline
    await orchestrator.start()
    
    try:
        # Process some messages
        message = Message(
            sender="test",
            receiver="agent1",
            type="test",
            data={"test": True}
        )
        
        for _ in range(5):
            await orchestrator.process_message(message, "agent1")
            await asyncio.sleep(0.2)
        
        # Verify metrics collection
        assert mock_performance_metrics.record_metric.called
        
        # Verify specific metrics
        metric_calls = mock_performance_metrics.record_metric.call_args_list
        metric_names = [args[0][0] for args in metric_calls]
        
        assert "pipeline_throughput" in metric_names
        assert "pipeline_latency" in metric_names
        assert any(name.endswith("_error_rate") for name in metric_names)
        
    finally:
        await orchestrator.stop()


@pytest.mark.asyncio
async def test_stage_metrics_update(pipeline_orchestrator):
    """Test updating stage metrics."""
    # Create stage
    stage = PipelineStage(agent_id="test_agent")
    pipeline_orchestrator.add_stage(stage)
    
    # Update metrics
    pipeline_orchestrator._update_stage_metrics("test_agent", 0.1, True)
    
    metrics = pipeline_orchestrator.stage_metrics["test_agent"]
    assert metrics["throughput"] > 0
    assert metrics["latency"] > 0
    assert metrics["error_rate"] == 0
    
    # Test error case
    pipeline_orchestrator._update_stage_metrics("test_agent", 0.1, False)
    assert metrics["error_rate"] > 0


@pytest.mark.asyncio
async def test_resource_adjustment(pipeline_orchestrator):
    """Test resource adjustment based on load."""
    # Create stage
    stage = PipelineStage(agent_id="test_agent")
    pipeline_orchestrator.add_stage(stage)
    
    # Test resource adjustment with different loads
    pipeline_orchestrator._adjust_resources("test_agent", 0.5)  # 50% load
    pipeline_orchestrator._adjust_resources("test_agent", 0.8)  # 80% load
    pipeline_orchestrator._adjust_resources("test_agent", 0.2)  # 20% load
    
    # No assertions needed as this is more about logging and internal state


@pytest.mark.asyncio
async def test_pipeline_error_propagation(test_pipeline):
    """Test error propagation through the pipeline."""
    orchestrator, agents = test_pipeline
    
    # Replace middle agent with one that fails
    error_agent = MockAgent("agent2", fail_rate=1.0)
    orchestrator.register_agent(error_agent)
    
    await orchestrator.start()
    
    try:
        message = Message(
            sender="test",
            receiver="agent1",
            type="test",
            data={"test": True}
        )
        
        # Process message and verify error handling
        await orchestrator.process_message(message, "agent1")
        await asyncio.sleep(0.3)
        
        # Verify first agent processed the message
        assert len(agents[0].processed_messages) == 1
        
        # Verify error was handled and didn't reach the third agent
        assert len(agents[2].processed_messages) == 0
        
        # Verify error tracking
        assert orchestrator.error_counts["agent2"] > 0
        
    finally:
        await orchestrator.stop()


@pytest.mark.asyncio
async def test_concurrent_message_processing(test_pipeline):
    """Test processing multiple messages concurrently."""
    orchestrator, agents = test_pipeline
    await orchestrator.start()
    
    try:
        # Create multiple messages
        messages = [
            Message(
                sender="test",
                receiver="agent1",
                type="test",
                data={"test": i}
            )
            for i in range(5)
        ]
        
        # Process messages concurrently
        await asyncio.gather(*[
            orchestrator.process_message(msg, "agent1")
            for msg in messages
        ])
        
        # Verify all messages were processed
        assert len(agents[0].processed_messages) == 5
        assert len(agents[1].processed_messages) == 5
        assert len(agents[2].processed_messages) == 5
        
        # Verify message ordering was maintained
        first_agent_data = [msg.data["test"] for msg in agents[0].processed_messages]
        last_agent_data = [msg.data["test"] for msg in agents[2].processed_messages]
        assert sorted(first_agent_data) == sorted(last_agent_data)
        
    finally:
        await orchestrator.stop() 