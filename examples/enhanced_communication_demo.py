"""
Enhanced Communication Demo for AMPTALK.

This script demonstrates the enhanced communication features of the AMPTALK
framework, including different serialization formats, compression, and
message delivery optimizations.

Usage:
    python examples/enhanced_communication_demo.py

Author: AMPTALK Team
Date: 2024
"""

import asyncio
import logging
import time
import json
import argparse
from typing import Dict, List, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.framework.agent import SimpleAgent
from src.core.framework.message import Message, MessageType, MessagePriority
from src.core.framework.orchestrator import Orchestrator
from src.core.framework.communication import (
    CommunicationConfig, CommunicationMode, 
    SerializationType, CompressionType
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("enhanced_communication_demo")


def create_large_payload(size_kb: int = 100) -> Dict[str, Any]:
    """Create a message payload of approximately the specified size in KB."""
    # Generate random text (each char is ~1 byte)
    text_size = size_kb * 1024
    
    # Create a dictionary with various field types to simulate real message content
    payload = {
        "text": "A" * (text_size // 2),
        "numbers": list(range(1000)),
        "nested": {
            "data": [{"id": i, "value": f"item_{i}"} for i in range(100)]
        }
    }
    
    # Convert to JSON and back to ensure consistent structure
    json_data = json.dumps(payload)
    actual_size = len(json_data)
    
    logger.info(f"Created test payload of {actual_size} bytes")
    return json.loads(json_data)


async def run_demo(message_count: int = 100, 
                  payload_size_kb: int = 10, 
                  batch_size: int = 10,
                  use_compression: bool = True):
    """
    Run the enhanced communication demo.
    
    Args:
        message_count: Number of messages to send
        payload_size_kb: Size of each message payload in KB
        batch_size: Number of messages to batch together
        use_compression: Whether to enable compression
    """
    logger.info("Starting Enhanced Communication Demo")
    logger.info(f"Message count: {message_count}")
    logger.info(f"Payload size: {payload_size_kb} KB")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Compression: {'Enabled' if use_compression else 'Disabled'}")
    
    # Create different communication configurations
    default_config = CommunicationConfig(
        mode=CommunicationMode.LOCAL,
        serialization=SerializationType.JSON,
        compression=CompressionType.NONE
    )
    
    msgpack_config = None
    try:
        import msgpack
        msgpack_config = CommunicationConfig(
            mode=CommunicationMode.LOCAL,
            serialization=SerializationType.MSGPACK,
            compression=CompressionType.NONE
        )
        logger.info("MessagePack serialization available")
    except ImportError:
        logger.warning("MessagePack serialization not available")
    
    compressed_config = CommunicationConfig(
        mode=CommunicationMode.LOCAL,
        serialization=SerializationType.JSON,
        compression=CompressionType.ZLIB if use_compression else CompressionType.NONE,
        compression_threshold=1024
    )
    
    batched_config = CommunicationConfig(
        mode=CommunicationMode.LOCAL,
        serialization=SerializationType.JSON,
        compression=CompressionType.ZLIB if use_compression else CompressionType.NONE,
        compression_threshold=1024,
        batch_messages=True,
        max_batch_size=batch_size
    )
    
    # Create test agents with different configurations
    agents = []
    agent_configs = [
        ("DefaultAgent", default_config),
        ("CompressedAgent", compressed_config),
        ("BatchedAgent", batched_config)
    ]
    
    if msgpack_config:
        agent_configs.append(("MsgPackAgent", msgpack_config))
    
    # Create and start agents
    for name, config in agent_configs:
        agent = SimpleAgent(name=name, communication_config=config)
        await agent.start()
        agents.append(agent)
    
    # Create orchestrator
    orchestrator = Orchestrator(name="DemoOrchestrator", communication_config=batched_config)
    
    # Register all agents with the orchestrator
    for agent in agents:
        orchestrator.add_agent(agent)
    
    # Connect all agents for direct communication
    orchestrator.connect_all_agents()
    
    # Create a test payload
    payload = create_large_payload(payload_size_kb)
    
    # Set up message receipt tracking
    received_messages = {agent.agent_id: 0 for agent in agents}
    receipt_count = 0
    expected_count = message_count * len(agents) * (len(agents) - 1)
    
    # Define a message handler that tracks receipts
    async def message_handler(message):
        nonlocal receipt_count
        agent_id = message.target_agent_id
        received_messages[agent_id] += 1
        receipt_count += 1
        if receipt_count % 100 == 0:
            logger.info(f"Processed {receipt_count}/{expected_count} messages")
        return []
    
    # Register the handler with all agents
    for agent in agents:
        agent.register_message_handler(MessageType.STATUS_REQUEST, message_handler)
    
    logger.info("Starting communication test...")
    
    # Measure message throughput
    start_time = time.time()
    
    # Send messages from each agent to all other agents
    tasks = []
    for source_agent in agents:
        for target_agent in agents:
            if source_agent.agent_id == target_agent.agent_id:
                continue
                
            for i in range(message_count):
                message = Message(
                    message_type=MessageType.STATUS_REQUEST,
                    source_agent_id=source_agent.agent_id,
                    target_agent_id=target_agent.agent_id,
                    priority=MessagePriority.NORMAL,
                    payload={
                        **payload,
                        "sequence": i,
                        "source": source_agent.name,
                        "target": target_agent.name,
                        "timestamp": time.time()
                    }
                )
                tasks.append(source_agent.send_message(message))
    
    # Execute all send tasks
    send_results = await asyncio.gather(*tasks)
    send_time = time.time() - start_time
    
    logger.info(f"All messages sent in {send_time:.4f} seconds")
    logger.info(f"Send throughput: {len(tasks) / send_time:.2f} msgs/sec")
    
    # Wait for all messages to be processed
    processed_time_start = time.time()
    timeout = 30.0
    while receipt_count < expected_count and (time.time() - processed_time_start) < timeout:
        logger.info(f"Waiting for message processing: {receipt_count}/{expected_count} received")
        await asyncio.sleep(0.5)
    
    total_time = time.time() - start_time
    
    if receipt_count < expected_count:
        logger.warning(f"Timeout reached. Only {receipt_count}/{expected_count} messages were processed")
    else:
        logger.info(f"All {expected_count} messages processed successfully")
    
    # Calculate and display results
    logger.info("\n--- Communication Results ---")
    logger.info(f"Total messages: {expected_count}")
    logger.info(f"Total time: {total_time:.4f} seconds")
    logger.info(f"Overall throughput: {expected_count / total_time:.2f} msgs/sec")
    
    # Display agent-specific stats
    logger.info("\n--- Agent Statistics ---")
    for agent in agents:
        # Get communication metrics
        metrics = await agent.get_communication_metrics()
        
        logger.info(f"\n{agent.name} ({agent.agent_id}):")
        logger.info(f"  Messages received: {received_messages[agent.agent_id]}")
        logger.info(f"  Messages sent: {metrics['messages_sent']}")
        
        # Display communication manager metrics if available
        if 'communication' in metrics:
            comm = metrics['communication']
            logger.info(f"  Communication mode: {comm.get('communication_mode', 'unknown')}")
            logger.info(f"  Serialization: {comm.get('serialization_type', 'unknown')}")
            logger.info(f"  Compression: {comm.get('compression_type', 'unknown')}")
            if 'average_message_size' in comm:
                logger.info(f"  Average message size: {comm['average_message_size']:.2f} bytes")
            if 'average_serialization_time' in comm:
                logger.info(f"  Average serialization time: {comm['average_serialization_time']*1000:.4f} ms")
            if 'average_delivery_time' in comm:
                logger.info(f"  Average delivery time: {comm['average_delivery_time']*1000:.4f} ms")
    
    # Stop all agents
    for agent in agents:
        await agent.stop()
    
    logger.info("Enhanced Communication Demo completed")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Enhanced Communication Demo for AMPTALK")
    parser.add_argument("--messages", type=int, default=100, help="Number of messages per agent pair")
    parser.add_argument("--size", type=int, default=10, help="Message payload size in KB")
    parser.add_argument("--batch", type=int, default=10, help="Batch size for batched messaging")
    parser.add_argument("--no-compression", action="store_true", help="Disable compression")
    args = parser.parse_args()
    
    # Run the demo
    asyncio.run(run_demo(
        message_count=args.messages,
        payload_size_kb=args.size,
        batch_size=args.batch,
        use_compression=not args.no_compression
    )) 