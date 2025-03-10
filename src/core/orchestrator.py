"""
Orchestrator for the AMPTALK multi-agent system.

This module provides the Orchestrator class, which is responsible for
managing the lifecycle of multiple agents, routing messages between them,
and monitoring system resources.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Type

from src.core.agent import Agent, AgentStatus, Message, MessagePriority

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class Orchestrator:
    """
    Orchestrator for managing the multi-agent system.
    
    Responsible for:
    - Creating and initializing agents
    - Routing messages between agents
    - Monitoring agent status and resource usage
    - Managing agent lifecycle
    """

    def __init__(self):
        """Initialize the orchestrator."""
        self.agents: Dict[str, Agent] = {}
        self.agent_tasks: Dict[str, asyncio.Task] = {}
        self.message_router_task: Optional[asyncio.Task] = None
        self.resource_monitor_task: Optional[asyncio.Task] = None
        self.running = False
        self.logger = logging.getLogger("Orchestrator")
        self.logger.info("Orchestrator initialized")

    async def register_agent(self, agent: Agent) -> None:
        """
        Register an agent with the orchestrator.
        
        Args:
            agent: The agent to register
        """
        if agent.agent_id in self.agents:
            raise ValueError(f"Agent with ID {agent.agent_id} already registered")
            
        self.agents[agent.agent_id] = agent
        self.logger.info(f"Registered agent {agent.agent_id} of type {agent.agent_type}")

    async def create_agent(
        self, agent_class: Type[Agent], agent_id: str, agent_type: str, **kwargs
    ) -> Agent:
        """
        Create and register a new agent.
        
        Args:
            agent_class: The Agent class to instantiate
            agent_id: Unique identifier for the agent
            agent_type: Type classification for the agent
            **kwargs: Additional arguments to pass to the agent constructor
            
        Returns:
            The created agent instance
        """
        if agent_id in self.agents:
            raise ValueError(f"Agent with ID {agent_id} already exists")
            
        agent = agent_class(agent_id=agent_id, agent_type=agent_type, **kwargs)
        await self.register_agent(agent)
        return agent

    async def start(self) -> None:
        """Start the orchestrator and all registered agents."""
        if self.running:
            self.logger.warning("Orchestrator is already running")
            return
            
        self.running = True
        self.logger.info("Starting orchestrator")
        
        # Start the message router
        self.message_router_task = asyncio.create_task(self._route_messages())
        
        # Start the resource monitor
        self.resource_monitor_task = asyncio.create_task(self._monitor_resources())
        
        # Start all registered agents
        for agent_id, agent in self.agents.items():
            self.agent_tasks[agent_id] = asyncio.create_task(agent.run())
            
        self.logger.info(f"Started {len(self.agents)} agents")

    async def stop(self) -> None:
        """Stop the orchestrator and all agents."""
        if not self.running:
            self.logger.warning("Orchestrator is not running")
            return
            
        self.running = False
        self.logger.info("Stopping orchestrator")
        
        # Signal all agents to shut down
        for agent_id, agent in self.agents.items():
            agent.status = AgentStatus.SHUTDOWN
            
        # Wait for agent tasks to complete
        if self.agent_tasks:
            await asyncio.gather(*self.agent_tasks.values(), return_exceptions=True)
            
        # Cancel router and monitor tasks
        if self.message_router_task:
            self.message_router_task.cancel()
            
        if self.resource_monitor_task:
            self.resource_monitor_task.cancel()
            
        self.logger.info("Orchestrator stopped")

    async def _route_messages(self) -> None:
        """Route messages between agents."""
        self.logger.info("Message router started")
        
        try:
            while self.running:
                # Collect messages from all agent outboxes
                all_outbox_tasks = []
                for agent_id, agent in self.agents.items():
                    if not agent.outbox.empty():
                        all_outbox_tasks.append(agent.outbox.get())
                
                if not all_outbox_tasks:
                    # No messages to route, wait briefly
                    await asyncio.sleep(0.01)
                    continue
                
                # Get the next available message from any agent
                done, _ = await asyncio.wait(
                    all_outbox_tasks, return_when=asyncio.FIRST_COMPLETED
                )
                
                for task in done:
                    message = task.result()
                    
                    # Route the message to the target agent
                    if message.target_agent in self.agents:
                        target_agent = self.agents[message.target_agent]
                        await target_agent.inbox.put(message)
                        self.logger.debug(
                            f"Routed message {message.message_id} from {message.source_agent} to {message.target_agent}"
                        )
                    else:
                        self.logger.warning(
                            f"Message {message.message_id} directed to unknown agent {message.target_agent}"
                        )
                        
        except asyncio.CancelledError:
            self.logger.info("Message router stopped")
        except Exception as e:
            self.logger.error(f"Message router error: {str(e)}")

    async def _monitor_resources(self) -> None:
        """Monitor system and agent resources."""
        self.logger.info("Resource monitor started")
        
        try:
            while self.running:
                # In a real implementation, we would gather actual resource usage
                # For now, we'll just log agent status
                status_reports = {}
                
                for agent_id, agent in self.agents.items():
                    status_reports[agent_id] = agent.get_status_report()
                
                # Log summary of agent status
                agent_counts = {status.value: 0 for status in AgentStatus}
                for report in status_reports.values():
                    agent_counts[report["status"]] += 1
                    
                self.logger.info(f"Agent status: {agent_counts}")
                
                # Check for error states
                error_agents = [
                    agent_id for agent_id, report in status_reports.items()
                    if report["status"] == AgentStatus.ERROR.value
                ]
                
                if error_agents:
                    self.logger.warning(f"Agents in error state: {error_agents}")
                
                # Wait before next monitoring cycle
                await asyncio.sleep(10)
                
        except asyncio.CancelledError:
            self.logger.info("Resource monitor stopped")
        except Exception as e:
            self.logger.error(f"Resource monitor error: {str(e)}")

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """
        Get an agent by ID.
        
        Args:
            agent_id: The ID of the agent to retrieve
            
        Returns:
            The agent if found, None otherwise
        """
        return self.agents.get(agent_id)

    def get_agents_by_type(self, agent_type: str) -> List[Agent]:
        """
        Get all agents of a specific type.
        
        Args:
            agent_type: The type of agents to retrieve
            
        Returns:
            List of agents matching the specified type
        """
        return [
            agent for agent in self.agents.values() if agent.agent_type == agent_type
        ]

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get the overall system status.
        
        Returns:
            Dict containing system status information
        """
        agent_statuses = {
            agent_id: agent.get_status_report()
            for agent_id, agent in self.agents.items()
        }
        
        return {
            "running": self.running,
            "agent_count": len(self.agents),
            "agents": agent_statuses,
            "timestamp": time.time(),
        } 