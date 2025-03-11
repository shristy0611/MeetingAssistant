"""
Tests for the enhanced communication system.

This module tests the enhanced inter-agent communication features,
including different serialization formats, compression types, and
communication modes.

Author: AMPTALK Team
Date: 2024
"""

import pytest
import asyncio
import time
import json
import zlib
import random
import string
import sys
from typing import Dict, List, Any, Optional

from src.core.framework.agent import Agent, SimpleAgent
from src.core.framework.message import Message, MessageType, MessagePriority
from src.core.framework.orchestrator import Orchestrator
from src.core.framework.communication import (
    CommunicationConfig, CommunicationManager, CommunicationMode,
    SerializationType, CompressionType, MessageSerializer, MessageTracker
)


@pytest.fixture
def large_payload() -> Dict[str, Any]:
    """Create a large message payload for testing compression."""
    # Generate a payload with ~100KB of data
    payload = {
        "text": "".join(random.choice(string.ascii_letters) for _ in range(20000)),
        "numbers": [random.randint(1, 1000) for _ in range(5000)],
        "nested": {
            "data": [{"name": f"item_{i}", "value": random.random()} for i in range(500)]
        }
    }
    return payload


@pytest.fixture
def standard_payload() -> Dict[str, Any]:
    """Create a standard message payload for testing."""
    return {
        "text": "This is a test message with standard content",
        "timestamp": time.time(),
        "values": [1, 2, 3, 4, 5],
        "metadata": {
            "version": "1.0",
            "priority": "normal"
        }
    }


@pytest.fixture
async def test_agents():
    """Create test agents for communication testing."""
    # Create a list of agents with different communication configs
    agents = []
    
    # Default local communication
    agent1 = SimpleAgent(name="LocalAgent")
    
    # JSON serialization with ZLIB compression
    config2 = CommunicationConfig(
        mode=CommunicationMode.LOCAL,
        serialization=SerializationType.JSON,
        compression=CompressionType.ZLIB,
        compression_threshold=1024
    )
    agent2 = SimpleAgent(name="CompressedAgent", communication_config=config2)
    
    # MessagePack serialization
    try:
        import msgpack
        config3 = CommunicationConfig(
            mode=CommunicationMode.LOCAL,
            serialization=SerializationType.MSGPACK,
            compression=CompressionType.NONE
        )
        agent3 = SimpleAgent(name="MsgPackAgent", communication_config=config3)
        agents.extend([agent1, agent2, agent3])
    except ImportError:
        # MessagePack not available, skip this agent
        agents.extend([agent1, agent2])
        
    # Start all agents
    for agent in agents:
        await agent.start()
    
    # Create an orchestrator and register agents
    orchestrator = Orchestrator(name="TestOrchestrator")
    for agent in agents:
        orchestrator.add_agent(agent)
    
    # Connect agents
    for i, source_agent in enumerate(agents):
        for target_agent in agents[i+1:]:
            source_agent.connect(target_agent)
            target_agent.connect(source_agent)
    
    # Return the test environment
    yield {
        "agents": agents,
        "orchestrator": orchestrator
    }
    
    # Cleanup - stop all agents
    for agent in agents:
        await agent.stop()


class TestMessageSerializer:
    """Tests for the MessageSerializer class."""
    
    def test_json_serialization(self, standard_payload):
        """Test JSON serialization with no compression."""
        # Create config and serializer
        config = CommunicationConfig(
            serialization=SerializationType.JSON,
            compression=CompressionType.NONE
        )
        serializer = MessageSerializer(config)
        
        # Create message
        message = Message(
            message_type=MessageType.STATUS_REQUEST,
            source_agent_id="test_source",
            target_agent_id="test_target",
            payload=standard_payload
        )
        
        # Serialize
        serialized, _ = serializer.serialize(message)
        
        # Verify result is bytes
        assert isinstance(serialized, bytes)
        
        # Deserialize
        deserialized = serializer.deserialize(serialized)
        
        # Verify round-trip
        assert deserialized.message_id == message.message_id
        assert deserialized.message_type == message.message_type
        assert deserialized.source_agent_id == message.source_agent_id
        assert deserialized.target_agent_id == message.target_agent_id
        assert deserialized.payload == message.payload
    
    def test_json_with_zlib_compression(self, large_payload):
        """Test JSON serialization with ZLIB compression."""
        # Create config and serializer
        config = CommunicationConfig(
            serialization=SerializationType.JSON,
            compression=CompressionType.ZLIB,
            compression_threshold=1024  # Ensure compression is applied
        )
        serializer = MessageSerializer(config)
        
        # Create message with large payload
        message = Message(
            message_type=MessageType.STATUS_REQUEST,
            source_agent_id="test_source",
            target_agent_id="test_target",
            payload=large_payload
        )
        
        # Serialize
        serialized, _ = serializer.serialize(message)
        
        # Verify result is bytes
        assert isinstance(serialized, bytes)
        
        # Verify compression occurred (compressed should be smaller)
        json_bytes = json.dumps(message.to_dict()).encode('utf-8')
        assert len(serialized) < len(json_bytes)
        
        # Deserialize
        deserialized = serializer.deserialize(serialized)
        
        # Verify round-trip
        assert deserialized.message_id == message.message_id
        assert deserialized.message_type == message.message_type
        assert deserialized.payload == message.payload
    
    @pytest.mark.skipif(
        sys.modules.get("msgpack") is None,
        reason="MessagePack not installed"
    )
    def test_msgpack_serialization(self, standard_payload):
        """Test MessagePack serialization."""
        # Create config and serializer
        config = CommunicationConfig(
            serialization=SerializationType.MSGPACK,
            compression=CompressionType.NONE
        )
        serializer = MessageSerializer(config)
        
        # Create message
        message = Message(
            message_type=MessageType.STATUS_REQUEST,
            source_agent_id="test_source",
            target_agent_id="test_target",
            payload=standard_payload
        )
        
        # Serialize
        serialized, _ = serializer.serialize(message)
        
        # Verify result is bytes
        assert isinstance(serialized, bytes)
        
        # Deserialize
        deserialized = serializer.deserialize(serialized)
        
        # Verify round-trip
        assert deserialized.message_id == message.message_id
        assert deserialized.message_type == message.message_type
        assert deserialized.payload == message.payload


class TestMessageTracker:
    """Tests for the MessageTracker class."""
    
    def test_message_tracking(self):
        """Test basic message tracking functionality."""
        # Create config and tracker
        config = CommunicationConfig(enable_message_tracking=True)
        tracker = MessageTracker(config)
        
        # Create a message
        message = Message(
            message_type=MessageType.STATUS_REQUEST,
            source_agent_id="test_source",
            target_agent_id="test_target"
        )
        
        # Track the message
        tracker.track_message(message, 100)
        
        # Verify message is tracked
        assert tracker.total_messages_sent == 1
        assert tracker.total_bytes_sent == 100
        assert message.message_id in tracker.pending_messages
        
        # Mark as delivered
        tracker.mark_delivered(message.message_id, 0.05)
        
        # Verify message is now in delivered
        assert message.message_id not in tracker.pending_messages
        assert message.message_id in tracker.delivered_messages
        assert tracker.total_messages_delivered == 1
        assert tracker.get_average_delivery_time() == 0.05
        
        # Create another message and mark as failed
        message2 = Message(
            message_type=MessageType.STATUS_REQUEST,
            source_agent_id="test_source",
            target_agent_id="test_target"
        )
        
        # Track and fail the message
        tracker.track_message(message2, 200)
        tracker.mark_failed(message2.message_id, "Test error")
        
        # Verify metrics
        assert tracker.total_messages_sent == 2
        assert tracker.total_bytes_sent == 300
        assert tracker.total_messages_delivered == 1
        assert tracker.total_messages_failed == 1
        assert message2.message_id in tracker.failed_messages
        assert "Test error" == tracker.failed_messages[message2.message_id]["error"]
    
    def test_metrics_generation(self):
        """Test generation of performance metrics."""
        # Create config and tracker
        config = CommunicationConfig(enable_message_tracking=True)
        tracker = MessageTracker(config)
        
        # Add some data
        for i in range(5):
            message = Message(
                message_type=MessageType.STATUS_REQUEST,
                source_agent_id="test_source",
                target_agent_id="test_target"
            )
            tracker.track_message(message, 100)
            tracker.record_serialization_time(0.01)
            
            # Mark as delivered with different latencies
            tracker.mark_delivered(message.message_id, 0.05 + i*0.01)
        
        # Get metrics
        metrics = tracker.get_metrics()
        
        # Verify metrics
        assert metrics["total_messages_sent"] == 5
        assert metrics["total_messages_delivered"] == 5
        assert metrics["total_bytes_sent"] == 500
        assert metrics["delivery_success_rate"] == 1.0
        assert metrics["average_message_size"] == 100.0
        assert 0.05 < metrics["average_delivery_time"] < 0.1
        assert metrics["average_serialization_time"] == 0.01


class TestCommunicationManager:
    """Tests for the CommunicationManager class."""
    
    @pytest.mark.asyncio
    async def test_local_delivery(self, standard_payload):
        """Test local message delivery."""
        # Create config and manager
        config = CommunicationConfig(
            mode=CommunicationMode.LOCAL,
            enable_message_tracking=True
        )
        manager = CommunicationManager(config)
        
        # Create source and target agents
        source_agent = SimpleAgent(name="SourceAgent")
        target_agent = SimpleAgent(name="TargetAgent")
        
        # Start agents
        await source_agent.start()
        await target_agent.start()
        
        # Connect agents
        source_agent.connect(target_agent)
        
        # Create message
        message = Message(
            message_type=MessageType.STATUS_REQUEST,
            source_agent_id=source_agent.agent_id,
            target_agent_id=target_agent.agent_id,
            payload=standard_payload
        )
        
        # Set up a handler to signal receipt
        received_message = None
        receipt_event = asyncio.Event()
        
        async def message_handler(msg):
            nonlocal received_message
            received_message = msg
            receipt_event.set()
            return []
        
        target_agent.register_message_handler(MessageType.STATUS_REQUEST, message_handler)
        
        # Send message
        success = await manager.send_message(message, target_agent)
        
        # Wait for receipt or timeout
        try:
            await asyncio.wait_for(receipt_event.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            pytest.fail("Message was not received within timeout")
        
        # Verify
        assert success
        assert received_message is not None
        assert received_message.message_id == message.message_id
        assert received_message.payload == message.payload
        
        # Check metrics
        metrics = manager.get_metrics()
        assert metrics["total_messages_sent"] == 1
        assert metrics["total_messages_delivered"] == 1
        
        # Cleanup
        await source_agent.stop()
        await target_agent.stop()
    
    @pytest.mark.asyncio
    async def test_batched_delivery(self, standard_payload):
        """Test batched message delivery."""
        # Create config with batching enabled
        config = CommunicationConfig(
            mode=CommunicationMode.LOCAL,
            batch_messages=True,
            max_batch_size=5,
            batch_timeout=0.1,
            enable_message_tracking=True
        )
        manager = CommunicationManager(config)
        
        # Create source and target agents
        source_agent = SimpleAgent(name="SourceAgent")
        target_agent = SimpleAgent(name="TargetAgent")
        
        # Start agents
        await source_agent.start()
        await target_agent.start()
        
        # Connect agents
        source_agent.connect(target_agent)
        
        # Track received messages
        received_messages = []
        receipt_count = 0
        expected_count = 10
        receipt_event = asyncio.Event()
        
        async def message_handler(msg):
            nonlocal received_messages, receipt_count
            received_messages.append(msg)
            receipt_count += 1
            if receipt_count == expected_count:
                receipt_event.set()
            return []
        
        target_agent.register_message_handler(MessageType.STATUS_REQUEST, message_handler)
        
        # Send multiple messages
        tasks = []
        for i in range(expected_count):
            message = Message(
                message_type=MessageType.STATUS_REQUEST,
                source_agent_id=source_agent.agent_id,
                target_agent_id=target_agent.agent_id,
                payload={**standard_payload, "sequence": i}
            )
            tasks.append(manager.send_message(message, target_agent))
        
        # Wait for all sends to complete
        results = await asyncio.gather(*tasks)
        assert all(results)
        
        # Wait for all messages to be received
        try:
            await asyncio.wait_for(receipt_event.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pytest.fail(f"Only received {receipt_count}/{expected_count} messages within timeout")
        
        # Verify all messages were received
        assert len(received_messages) == expected_count
        
        # Verify sequence preservation
        sequence_numbers = sorted([msg.payload["sequence"] for msg in received_messages])
        assert sequence_numbers == list(range(expected_count))
        
        # Check metrics
        metrics = manager.get_metrics()
        assert metrics["total_messages_sent"] == expected_count
        assert metrics["total_messages_delivered"] == expected_count
        
        # Cleanup
        await source_agent.stop()
        await target_agent.stop()


@pytest.mark.asyncio
async def test_agent_communication(test_agents):
    """Test communication between agents with different configurations."""
    agents = test_agents["agents"]
    assert len(agents) >= 2
    
    # Create a test message for each source agent
    test_message_count = 10
    receipt_count = 0
    expected_count = test_message_count * len(agents) * (len(agents) - 1)
    receipt_event = asyncio.Event()
    
    # Register message handler for all agents
    async def message_handler(message):
        nonlocal receipt_count
        receipt_count += 1
        if receipt_count == expected_count:
            receipt_event.set()
        return []
    
    for agent in agents:
        agent.register_message_handler(MessageType.STATUS_REQUEST, message_handler)
    
    # Send messages between all agent pairs
    start_time = time.time()
    for source_agent in agents:
        for target_agent in agents:
            if source_agent.agent_id == target_agent.agent_id:
                continue
                
            for i in range(test_message_count):
                message = Message(
                    message_type=MessageType.STATUS_REQUEST,
                    source_agent_id=source_agent.agent_id,
                    target_agent_id=target_agent.agent_id,
                    payload={
                        "text": f"Test message {i} from {source_agent.name} to {target_agent.name}",
                        "timestamp": time.time()
                    }
                )
                await source_agent.send_message(message)
    
    # Wait for all messages to be received
    try:
        await asyncio.wait_for(receipt_event.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        pytest.fail(f"Only received {receipt_count}/{expected_count} messages within timeout")
    
    # Calculate throughput
    elapsed = time.time() - start_time
    messages_per_second = expected_count / elapsed
    
    print(f"\nMessage throughput: {messages_per_second:.2f} msgs/sec")
    print(f"Total messages: {expected_count}")
    print(f"Time elapsed: {elapsed:.4f} seconds")
    
    # Verify all messages were received
    assert receipt_count == expected_count


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_serialization_performance(large_payload):
    """Benchmark different serialization and compression options."""
    # Options to test
    options = [
        ("JSON", SerializationType.JSON, CompressionType.NONE),
        ("JSON+ZLIB", SerializationType.JSON, CompressionType.ZLIB),
    ]
    
    # Add MessagePack if available
    try:
        import msgpack
        options.append(("MessagePack", SerializationType.MSGPACK, CompressionType.NONE))
        options.append(("MessagePack+ZLIB", SerializationType.MSGPACK, CompressionType.ZLIB))
    except ImportError:
        pass
    
    # Test message
    message = Message(
        message_type=MessageType.STATUS_REQUEST,
        source_agent_id="benchmark_source",
        target_agent_id="benchmark_target",
        payload=large_payload
    )
    
    results = {}
    
    # Run benchmarks
    for name, serialization_type, compression_type in options:
        config = CommunicationConfig(
            serialization=serialization_type,
            compression=compression_type,
            compression_threshold=1024
        )
        serializer = MessageSerializer(config)
        
        # Measure time for multiple iterations
        iterations = 100
        serialize_times = []
        deserialize_times = []
        serialized_sizes = []
        
        for _ in range(iterations):
            # Serialize
            start_time = time.time()
            serialized, _ = serializer.serialize(message)
            serialize_time = time.time() - start_time
            serialize_times.append(serialize_time)
            serialized_sizes.append(len(serialized))
            
            # Deserialize
            start_time = time.time()
            serializer.deserialize(serialized)
            deserialize_time = time.time() - start_time
            deserialize_times.append(deserialize_time)
        
        # Calculate averages
        avg_serialize = sum(serialize_times) / iterations
        avg_deserialize = sum(deserialize_times) / iterations
        avg_size = sum(serialized_sizes) / iterations
        
        results[name] = {
            "avg_serialize_time": avg_serialize,
            "avg_deserialize_time": avg_deserialize,
            "avg_size": avg_size
        }
    
    # Print results
    print("\nSerialization Benchmark Results:")
    print(f"{'Method':<20} {'Avg Size':<12} {'Serialize Time':<18} {'Deserialize Time':<18}")
    print("-" * 70)
    
    for name, data in results.items():
        print(f"{name:<20} {data['avg_size']:<12.2f} {data['avg_serialize_time']:<18.6f} {data['avg_deserialize_time']:<18.6f}")
    
    # Make sure benchmark is valid
    assert len(results) > 0 