#!/usr/bin/env python3
"""
Agent Management Demo.

This script demonstrates how to use the agent management interface
to create, configure, monitor, and control agents.

Author: AMPTALK Team
Date: 2024
"""

import os
import sys
import json
import logging
import asyncio
import argparse
from datetime import datetime
from pathlib import Path

from src.core.interface.agent_management.agent_manager import (
    AgentManager, AgentStatus
)
from src.core.agents.base_agent import BaseAgent
from src.core.orchestration.orchestrator import Orchestrator
from src.core.utils.logging_config import get_logger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = get_logger(__name__)

# Sample agent classes for demo
class DemoAgent(BaseAgent):
    """Demo agent for testing."""
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        process_delay: float = 0.1,
        **kwargs
    ):
        """Initialize demo agent."""
        super().__init__(agent_id=agent_id, name=name)
        self.process_delay = process_delay
        self.counter = 0
    
    async def process_message(self, message):
        """Process message."""
        await asyncio.sleep(self.process_delay)
        self.counter += 1
        return {"result": f"Processed by {self.name}", "count": self.counter}
    
    async def start(self):
        """Start agent."""
        logger.info(f"Starting agent: {self.name}")
        await super().start()
    
    async def stop(self):
        """Stop agent."""
        logger.info(f"Stopping agent: {self.name}")
        await super().stop()

class LoggingAgent(BaseAgent):
    """Logging agent for demo."""
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        log_level: str = "INFO",
        **kwargs
    ):
        """Initialize logging agent."""
        super().__init__(agent_id=agent_id, name=name)
        self.log_level = log_level
        self.log_count = 0
    
    async def process_message(self, message):
        """Process message."""
        self.log_count += 1
        logger.info(
            f"[{self.name}] Processing message #{self.log_count}: {message}"
        )
        return {"logged": True, "count": self.log_count}
    
    async def start(self):
        """Start agent."""
        logger.info(f"Starting logging agent: {self.name}")
        await super().start()
    
    async def stop(self):
        """Stop agent."""
        logger.info(f"Stopping logging agent: {self.name}")
        await super().stop()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Agent Management Demo"
    )
    parser.add_argument(
        "--config-dir",
        default="config/agents",
        help="Agent configuration directory"
    )
    parser.add_argument(
        "--demo-duration",
        type=int,
        default=30,
        help="Demo duration in seconds"
    )
    return parser.parse_args()

async def demo_agent_lifecycle(manager: AgentManager):
    """
    Demonstrate agent lifecycle management.
    
    Args:
        manager: Agent manager instance
    """
    try:
        # Register agent types
        manager.register_agent_type("demo", DemoAgent)
        manager.register_agent_type("logging", LoggingAgent)
        
        # Create agents
        demo_agent_id = manager.create_agent(
            name="DemoAgent1",
            agent_type="demo",
            config={"process_delay": 0.2},
            tags=["demo", "test"],
            group="processing"
        )
        
        logging_agent_id = manager.create_agent(
            name="LoggingAgent1",
            agent_type="logging",
            config={"log_level": "DEBUG"},
            tags=["logging", "test"],
            group="monitoring"
        )
        
        # Add dependency
        manager.add_agent_dependency(
            demo_agent_id,
            logging_agent_id
        )
        
        # Start agents
        logger.info("\nStarting agents...")
        
        # Start logging agent first (dependency)
        await manager.start_agent(logging_agent_id)
        
        # Start demo agent
        await manager.start_agent(demo_agent_id)
        
        # Wait a bit
        await asyncio.sleep(2)
        
        # Update configuration
        logger.info("\nUpdating agent configuration...")
        manager.update_agent_config(
            demo_agent_id,
            {"process_delay": 0.1},
            restart=True
        )
        
        # Wait for restart
        await asyncio.sleep(3)
        
        # Stop agents
        logger.info("\nStopping agents...")
        await manager.stop_agent(demo_agent_id)
        await manager.stop_agent(logging_agent_id)
        
        # Wait a bit
        await asyncio.sleep(1)
        
        # Restart agents
        logger.info("\nRestarting agents...")
        await manager.start_agent(logging_agent_id)
        await manager.start_agent(demo_agent_id)
        
        # Wait a bit
        await asyncio.sleep(2)
        
        # Stop all agents
        logger.info("\nStopping all agents...")
        await manager.stop_all_agents()
        
    except Exception as e:
        logger.error(f"Error in agent lifecycle demo: {e}")
        raise

async def demo_agent_grouping(manager: AgentManager):
    """
    Demonstrate agent grouping.
    
    Args:
        manager: Agent manager instance
    """
    try:
        # Create more agents
        for i in range(3):
            manager.create_agent(
                name=f"DemoAgent{i+2}",
                agent_type="demo",
                config={"process_delay": 0.1 * (i+1)},
                tags=["demo", f"group{i%2}"],
                group=f"group{i%2}"
            )
        
        # List groups
        logger.info("\nAgent Groups:")
        for group in manager.list_groups():
            agents = manager.get_group_agents(group)
            logger.info(
                f"Group '{group}': {len(agents)} agents"
            )
            for agent in agents:
                logger.info(
                    f"  - {agent.name} ({agent.agent_type})"
                )
        
        # Start agents by group
        logger.info("\nStarting agents in group 'group0'...")
        results = await manager.start_all_agents(group="group0")
        logger.info(f"Start results: {results}")
        
        # Wait a bit
        await asyncio.sleep(2)
        
        # Stop agents by group
        logger.info("\nStopping agents in group 'group0'...")
        results = await manager.stop_all_agents(group="group0")
        logger.info(f"Stop results: {results}")
        
    except Exception as e:
        logger.error(f"Error in agent grouping demo: {e}")
        raise

async def demo_agent_tagging(manager: AgentManager):
    """
    Demonstrate agent tagging.
    
    Args:
        manager: Agent manager instance
    """
    try:
        # Add tags
        logger.info("\nAdding tags to agents...")
        agents = manager.list_agents()
        for i, agent in enumerate(agents):
            manager.add_agent_tag(
                agent.id,
                f"priority-{i%3+1}"
            )
        
        # List agents by tag
        logger.info("\nAgents by tag:")
        for priority in range(1, 4):
            tag = f"priority-{priority}"
            tagged_agents = manager.list_agents(tag=tag)
            logger.info(
                f"Tag '{tag}': {len(tagged_agents)} agents"
            )
            for agent in tagged_agents:
                logger.info(
                    f"  - {agent.name} ({agent.agent_type})"
                )
        
        # Start agents by tag
        logger.info("\nStarting agents with tag 'priority-1'...")
        results = await manager.start_all_agents(tag="priority-1")
        logger.info(f"Start results: {results}")
        
        # Wait a bit
        await asyncio.sleep(2)
        
        # Stop agents by tag
        logger.info("\nStopping agents with tag 'priority-1'...")
        results = await manager.stop_all_agents(tag="priority-1")
        logger.info(f"Stop results: {results}")
        
    except Exception as e:
        logger.error(f"Error in agent tagging demo: {e}")
        raise

async def demo_agent_metrics(manager: AgentManager):
    """
    Demonstrate agent metrics.
    
    Args:
        manager: Agent manager instance
    """
    try:
        # Update metrics
        logger.info("\nUpdating agent metrics...")
        agents = manager.list_agents()
        for agent in agents:
            manager.update_agent_metrics(
                agent.id,
                {
                    "messages_processed": 0,
                    "avg_processing_time_ms": 0,
                    "memory_usage_mb": 0,
                    "last_active": datetime.utcnow().isoformat()
                }
            )
        
        # Simulate metrics updates
        logger.info("\nSimulating metrics updates...")
        for _ in range(5):
            for agent in agents:
                current_metrics = manager.get_agent_metrics(agent.id)
                messages = current_metrics.get("messages_processed", 0)
                
                # Update metrics
                manager.update_agent_metrics(
                    agent.id,
                    {
                        "messages_processed": messages + 10,
                        "avg_processing_time_ms": 50 + (messages % 20),
                        "memory_usage_mb": 50 + (messages % 100),
                        "last_active": datetime.utcnow().isoformat()
                    }
                )
            
            await asyncio.sleep(1)
        
        # Show final metrics
        logger.info("\nFinal agent metrics:")
        for agent in agents:
            metrics = manager.get_agent_metrics(agent.id)
            logger.info(f"Agent: {agent.name}")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value}")
        
    except Exception as e:
        logger.error(f"Error in agent metrics demo: {e}")
        raise

async def demo_agent_events(manager: AgentManager):
    """
    Demonstrate agent events.
    
    Args:
        manager: Agent manager instance
    """
    try:
        # Register event handlers
        logger.info("\nRegistering event handlers...")
        
        def on_agent_started(agent_info, **kwargs):
            logger.info(
                f"EVENT: Agent started: {agent_info.name}"
            )
        
        def on_agent_stopped(agent_info, **kwargs):
            logger.info(
                f"EVENT: Agent stopped: {agent_info.name}"
            )
        
        def on_agent_error(agent_info, error=None, **kwargs):
            logger.info(
                f"EVENT: Agent error: {agent_info.name} - {error}"
            )
        
        def on_agent_config_updated(
            agent_info,
            old_config=None,
            new_config=None,
            **kwargs
        ):
            logger.info(
                f"EVENT: Agent config updated: {agent_info.name}"
            )
        
        # Register handlers
        manager.register_event_handler(
            "agent_started",
            on_agent_started
        )
        manager.register_event_handler(
            "agent_stopped",
            on_agent_stopped
        )
        manager.register_event_handler(
            "agent_error",
            on_agent_error
        )
        manager.register_event_handler(
            "agent_config_updated",
            on_agent_config_updated
        )
        
        # Trigger events
        logger.info("\nTriggering events...")
        
        # Get first agent
        agents = manager.list_agents()
        if not agents:
            return
        
        agent = agents[0]
        
        # Start agent
        await manager.start_agent(agent.id)
        
        # Update config
        manager.update_agent_config(
            agent.id,
            {"process_delay": 0.3}
        )
        
        # Stop agent
        await manager.stop_agent(agent.id)
        
    except Exception as e:
        logger.error(f"Error in agent events demo: {e}")
        raise

async def main():
    """Main function."""
    args = parse_args()
    
    try:
        # Create orchestrator
        orchestrator = Orchestrator()
        
        # Create agent manager
        manager = AgentManager(
            orchestrator=orchestrator,
            config_dir=args.config_dir,
            auto_save=True
        )
        
        # Run demos
        logger.info("\nTesting agent lifecycle management...")
        await demo_agent_lifecycle(manager)
        
        logger.info("\nTesting agent grouping...")
        await demo_agent_grouping(manager)
        
        logger.info("\nTesting agent tagging...")
        await demo_agent_tagging(manager)
        
        logger.info("\nTesting agent metrics...")
        await demo_agent_metrics(manager)
        
        logger.info("\nTesting agent events...")
        await demo_agent_events(manager)
        
        logger.info("\nAll demos completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Stopping demo...")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 