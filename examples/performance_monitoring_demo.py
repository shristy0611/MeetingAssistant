#!/usr/bin/env python
"""
Performance Monitoring Demo

This script demonstrates how to use the AMPTALK performance monitoring system
to track metrics from agents and the system.

Author: AMPTALK Team
Date: 2024
"""

import asyncio
import logging
import random
import time
import os
from datetime import datetime
from typing import Dict, List, Any

from src.core.framework.agent import SimpleAgent
from src.core.framework.message import Message, MessageType, MessagePriority
from src.core.framework.orchestrator import Orchestrator
from src.core.utils.performance_monitor import setup_performance_monitoring


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("performance_demo")


class WorkloadGenerator:
    """Generates realistic workloads for testing performance monitoring."""
    
    def __init__(self, orchestrator: Orchestrator, agent_ids: List[str]):
        """Initialize with the orchestrator and list of agent IDs."""
        self.orchestrator = orchestrator
        self.agent_ids = agent_ids
        self.running = False
        self.task = None
    
    async def start(self, interval_seconds: float = 0.5):
        """Start generating workload at given interval."""
        self.running = True
        self.task = asyncio.create_task(self._generate_workload(interval_seconds))
        logger.info("Workload generator started")
    
    async def stop(self):
        """Stop the workload generator."""
        if self.running:
            self.running = False
            if self.task:
                self.task.cancel()
                try:
                    await self.task
                except asyncio.CancelledError:
                    pass
            logger.info("Workload generator stopped")
    
    async def _generate_workload(self, interval_seconds: float):
        """Generate messages to simulate workload."""
        try:
            while self.running:
                # Choose a random agent
                target_agent_id = random.choice(self.agent_ids)
                
                # Create a message with random payload size
                payload_size = random.randint(1, 10) * 100  # 100-1000 bytes
                payload = {
                    "timestamp": datetime.now().isoformat(),
                    "data": "x" * payload_size,
                    "priority": random.choice([p.value for p in MessagePriority]),
                }
                
                message = Message(
                    message_type=MessageType.STATUS_REQUEST,
                    source_agent_id="workload_generator",
                    target_agent_id=target_agent_id,
                    payload=payload
                )
                
                # Send the message
                await self.orchestrator.send_message_to_agent(message)
                logger.debug(f"Sent message to {target_agent_id} with {payload_size} bytes")
                
                # Wait for next interval with some jitter
                jitter = random.uniform(0.8, 1.2)
                await asyncio.sleep(interval_seconds * jitter)
        except asyncio.CancelledError:
            logger.info("Workload generator task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in workload generator: {e}")


class WorkAgent(SimpleAgent):
    """Agent that simulates various types of work with predictable patterns."""
    
    def __init__(self, work_pattern: str = "constant", *args, **kwargs):
        """
        Initialize with a work pattern.
        
        Patterns:
        - constant: Always same processing time
        - random: Random processing time
        - increasing: Processing time increases with each message
        - wave: Processing time follows a sine wave pattern
        """
        super().__init__(*args, **kwargs)
        self.work_pattern = work_pattern
        self.base_processing_time = 0.05
        self.message_count = 0
        self.memory_growth = kwargs.get("memory_growth", False)
        self.memory_data = []
        
        # Register message handlers
        self.register_message_handler(
            MessageType.STATUS_REQUEST,
            self.handle_status_request
        )
        self.register_message_handler(
            MessageType.COMMAND,
            self.handle_command
        )
    
    async def handle_status_request(self, message: Message) -> List[Message]:
        """Handle a status request with simulated work based on pattern."""
        # Increment counter
        self.message_count += 1
        
        # Determine processing time based on pattern
        processing_time = self._calculate_processing_time()
        
        # Simulate memory growth if enabled
        if self.memory_growth and self.message_count % 10 == 0:
            # Add 1MB of data every 10 messages
            self.memory_data.append("x" * (1024 * 1024))
            logger.debug(f"Agent {self.name} memory growth: {len(self.memory_data)} MB")
        
        # Simulate CPU work
        start_time = time.time()
        end_time = start_time + processing_time
        
        # Busy wait to simulate CPU usage
        busy_wait_portion = 0.3  # 30% of the time is CPU-bound
        busy_wait_time = processing_time * busy_wait_portion
        
        while time.time() < start_time + busy_wait_time:
            # Simulate CPU-intensive calculation
            _ = [i**2 for i in range(1000)]
        
        # Sleep for the rest of the time to simulate I/O or other waiting
        remaining_time = end_time - time.time()
        if remaining_time > 0:
            await asyncio.sleep(remaining_time)
        
        # Return a response
        response = Message(
            message_type=MessageType.STATUS_RESPONSE,
            source_agent_id=self.agent_id,
            target_agent_id=message.source_agent_id,
            payload={
                "count": self.message_count,
                "processing_time": processing_time,
                "pattern": self.work_pattern,
            }
        )
        
        return [response]
    
    async def handle_command(self, message: Message) -> List[Message]:
        """Handle commands to change agent behavior."""
        command = message.payload.get("command", "")
        response_payload = {"status": "unknown_command"}
        
        if command == "change_pattern":
            new_pattern = message.payload.get("pattern")
            if new_pattern in ["constant", "random", "increasing", "wave"]:
                self.work_pattern = new_pattern
                response_payload = {
                    "status": "success",
                    "message": f"Changed work pattern to {new_pattern}"
                }
            else:
                response_payload = {
                    "status": "error",
                    "message": f"Unknown pattern: {new_pattern}"
                }
        elif command == "reset_counter":
            self.message_count = 0
            response_payload = {
                "status": "success",
                "message": "Reset message counter"
            }
        elif command == "clear_memory":
            old_size = len(self.memory_data)
            self.memory_data = []
            response_payload = {
                "status": "success",
                "message": f"Cleared {old_size} MB of memory"
            }
        
        response = Message(
            message_type=MessageType.COMMAND_RESPONSE,
            source_agent_id=self.agent_id,
            target_agent_id=message.source_agent_id,
            payload=response_payload
        )
        
        return [response]
    
    def _calculate_processing_time(self) -> float:
        """Calculate processing time based on the current pattern."""
        if self.work_pattern == "constant":
            return self.base_processing_time
        
        elif self.work_pattern == "random":
            # Random between 0.5x and 2x base time
            return random.uniform(
                self.base_processing_time * 0.5,
                self.base_processing_time * 2
            )
        
        elif self.work_pattern == "increasing":
            # Increase by 5% for each message, up to 3x base
            factor = min(1 + (self.message_count * 0.05), 3)
            return self.base_processing_time * factor
        
        elif self.work_pattern == "wave":
            # Sine wave pattern between 0.5x and 1.5x base
            import math
            factor = 1 + 0.5 * math.sin(self.message_count / 10)
            return self.base_processing_time * factor
        
        # Default
        return self.base_processing_time


async def main():
    """Run the performance monitoring demo."""
    logger.info("Starting performance monitoring demo")
    
    # Set up performance monitoring
    monitor = await setup_performance_monitoring(
        enable_prometheus=True,
        prometheus_port=8000,
        collection_interval_seconds=1.0
    )
    logger.info(f"Performance monitoring started on port 8000")
    
    # Create an orchestrator with monitoring enabled
    orchestrator = Orchestrator(
        name="DemoOrchestrator",
        enable_performance_monitoring=True
    )
    
    # Create some work agents with different patterns
    work_agents = [
        WorkAgent(name="ConstantAgent", work_pattern="constant"),
        WorkAgent(name="RandomAgent", work_pattern="random"),
        WorkAgent(name="IncreasingAgent", work_pattern="increasing"),
        WorkAgent(name="WaveAgent", work_pattern="wave"),
        WorkAgent(name="MemoryAgent", work_pattern="constant", memory_growth=True),
    ]
    
    # Register the agents with the orchestrator
    for agent in work_agents:
        orchestrator.register_agent(agent)
    
    # Start the orchestrator (and agents)
    await orchestrator.start()
    logger.info(f"Started orchestrator with {len(work_agents)} agents")
    
    # Create a workload generator
    workload = WorkloadGenerator(
        orchestrator=orchestrator,
        agent_ids=[agent.agent_id for agent in work_agents]
    )
    
    # Start generating workload
    await workload.start(interval_seconds=0.2)
    
    try:
        # Keep the demo running
        logger.info(
            "Demo is running. Performance metrics available at: "
            "http://localhost:8000/metrics"
        )
        logger.info("Press Ctrl+C to stop...")
        
        # Keep running until interrupted
        while True:
            await asyncio.sleep(10)
            
            # Log some stats every 10 seconds
            status = await orchestrator.get_system_status()
            agent_statuses = status["agents"]
            
            logger.info("=" * 50)
            logger.info(f"System Status: {len(agent_statuses)} agents running")
            logger.info("-" * 50)
            
            for agent_id, agent_status in agent_statuses.items():
                agent_name = agent_status["name"]
                queue_size = agent_status["message_queue_size"]
                processed = agent_status["messages_processed"]
                logger.info(f"Agent: {agent_name} | Queue: {queue_size} | Processed: {processed}")
            
            logger.info("=" * 50)
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    finally:
        # Clean up
        await workload.stop()
        await orchestrator.stop()
        await monitor.stop_collecting()
        logger.info("Demo stopped")


if __name__ == "__main__":
    asyncio.run(main()) 