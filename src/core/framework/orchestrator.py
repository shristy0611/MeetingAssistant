"""
Orchestrator Module for AMPTALK Multi-Agent Framework.

This module defines the Orchestrator class, which is responsible for managing
the multi-agent system, controlling message flow between agents, and monitoring
the overall system status.

Author: AMPTALK Team
Date: 2024
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Set, Union, Callable, Coroutine
import uuid

from .agent import Agent
from .message import Message, MessageType, MessagePriority
from .communication import CommunicationManager, CommunicationConfig, CommunicationMode

# Optional import for performance monitoring
try:
    from ..utils.performance_monitor import get_monitor, setup_performance_monitoring
    PERFORMANCE_MONITORING_AVAILABLE = True
except ImportError:
    PERFORMANCE_MONITORING_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Orchestrator for the AMPTALK multi-agent system.
    
    The Orchestrator is responsible for managing the lifecycle of agents,
    facilitating communication between agents, and monitoring the system's
    overall health and performance.
    """
    
    def __init__(self, name: str = "AMPTALK Orchestrator", 
                 communication_config: Optional[CommunicationConfig] = None):
        """
        Initialize a new Orchestrator.
        
        Args:
            name: Name for this orchestrator instance
            communication_config: Configuration for agent communication, or None to use defaults
        """
        self.name = name
        self.orchestrator_id = str(uuid.uuid4())
        
        # Agent management
        self.agents: Dict[str, Agent] = {}
        self.agent_groups: Dict[str, Set[str]] = {}
        
        # Communication management
        self.communication_config = communication_config or CommunicationConfig()
        self.communication_manager = CommunicationManager(self.communication_config)
        
        # System state
        self.is_running = False
        self.start_time: Optional[float] = None
        self.stop_time: Optional[float] = None
        
        # Tasks
        self.tasks: List[asyncio.Task] = []
        
        # Configuration
        self.config: Dict[str, Any] = {}
        
        # Hooks for system events
        self.event_hooks: Dict[str, List[Callable[..., Coroutine]]] = {
            "agent_added": [],
            "agent_removed": [],
            "system_started": [],
            "system_stopped": [],
            "message_routed": [],
            "error": []
        }
        
        # Performance monitoring
        self.enable_performance_monitoring = PERFORMANCE_MONITORING_AVAILABLE
        self._performance_monitor = None
        
        logger.info(f"Initialized {self.name} (ID: {self.orchestrator_id})")
    
    def register_agent(self, agent: Agent, groups: Optional[List[str]] = None) -> None:
        """
        Register an agent with the orchestrator.
        
        Args:
            agent: The agent to register
            groups: Optional list of group names to add the agent to
        """
        if agent.agent_id in self.agents:
            logger.warning(f"Agent {agent.name} is already registered")
            return
        
        self.agents[agent.agent_id] = agent
        logger.info(f"Registered agent {agent.name} (ID: {agent.agent_id})")
        
        # Add to groups if specified
        if groups:
            for group in groups:
                self.add_agent_to_group(agent.agent_id, group)
        
        # Trigger event hook
        asyncio.create_task(self._trigger_event("agent_added", agent))
    
    def unregister_agent(self, agent_id: str) -> None:
        """
        Unregister an agent from the orchestrator.
        
        Args:
            agent_id: ID of the agent to unregister
        """
        if agent_id not in self.agents:
            logger.warning(f"Agent with ID {agent_id} is not registered")
            return
        
        agent = self.agents[agent_id]
        
        # Remove from all groups
        for group in list(self.agent_groups.keys()):
            if agent_id in self.agent_groups[group]:
                self.agent_groups[group].remove(agent_id)
        
        # Disconnect from all other agents
        for other_agent in self.agents.values():
            if agent_id in other_agent.connected_agents:
                other_agent.disconnect(agent_id)
        
        # Remove from agents
        del self.agents[agent_id]
        
        logger.info(f"Unregistered agent {agent.name} (ID: {agent_id})")
        
        # Trigger event hook
        asyncio.create_task(self._trigger_event("agent_removed", agent))
    
    def add_agent_to_group(self, agent_id: str, group_name: str) -> None:
        """
        Add an agent to a named group.
        
        Args:
            agent_id: ID of the agent to add
            group_name: Name of the group
        """
        if agent_id not in self.agents:
            logger.warning(f"Cannot add to group: Agent with ID {agent_id} is not registered")
            return
        
        # Create group if it doesn't exist
        if group_name not in self.agent_groups:
            self.agent_groups[group_name] = set()
        
        # Add agent to group
        self.agent_groups[group_name].add(agent_id)
        logger.info(f"Added agent {self.agents[agent_id].name} to group '{group_name}'")
    
    def remove_agent_from_group(self, agent_id: str, group_name: str) -> None:
        """
        Remove an agent from a named group.
        
        Args:
            agent_id: ID of the agent to remove
            group_name: Name of the group
        """
        if group_name not in self.agent_groups:
            logger.warning(f"Group '{group_name}' does not exist")
            return
        
        if agent_id not in self.agent_groups[group_name]:
            logger.warning(f"Agent with ID {agent_id} is not in group '{group_name}'")
            return
        
        # Remove agent from group
        self.agent_groups[group_name].remove(agent_id)
        logger.info(f"Removed agent {self.agents[agent_id].name} from group '{group_name}'")
        
        # Clean up empty groups
        if not self.agent_groups[group_name]:
            del self.agent_groups[group_name]
            logger.info(f"Removed empty group '{group_name}'")
    
    def get_agents_in_group(self, group_name: str) -> List[Agent]:
        """
        Get all agents in a named group.
        
        Args:
            group_name: Name of the group
            
        Returns:
            List of agents in the group
        """
        if group_name not in self.agent_groups:
            logger.warning(f"Group '{group_name}' does not exist")
            return []
        
        return [self.agents[agent_id] for agent_id in self.agent_groups[group_name]]
    
    def connect_agents(self, source_id: str, target_id: str, bidirectional: bool = True) -> None:
        """
        Connect two agents to enable message passing between them.
        
        Args:
            source_id: ID of the source agent
            target_id: ID of the target agent
            bidirectional: If True, also connect target to source
        """
        if source_id not in self.agents:
            logger.warning(f"Source agent with ID {source_id} is not registered")
            return
        
        if target_id not in self.agents:
            logger.warning(f"Target agent with ID {target_id} is not registered")
            return
        
        source_agent = self.agents[source_id]
        target_agent = self.agents[target_id]
        
        # Connect source to target
        source_agent.connect(target_agent)
        
        # Connect target to source if bidirectional
        if bidirectional:
            target_agent.connect(source_agent)
    
    def connect_groups(self, source_group: str, target_group: str, bidirectional: bool = True) -> None:
        """
        Connect all agents in one group to all agents in another group.
        
        Args:
            source_group: Name of the source group
            target_group: Name of the target group
            bidirectional: If True, also connect target agents to source agents
        """
        if source_group not in self.agent_groups:
            logger.warning(f"Source group '{source_group}' does not exist")
            return
        
        if target_group not in self.agent_groups:
            logger.warning(f"Target group '{target_group}' does not exist")
            return
        
        # Connect all agents in source group to all agents in target group
        for source_id in self.agent_groups[source_group]:
            for target_id in self.agent_groups[target_group]:
                if source_id != target_id:  # Don't connect an agent to itself
                    self.connect_agents(source_id, target_id, bidirectional)
    
    def connect_agent_to_group(self, agent_id: str, group_name: str, bidirectional: bool = True) -> None:
        """
        Connect an agent to all agents in a group.
        
        Args:
            agent_id: ID of the agent to connect
            group_name: Name of the group to connect to
            bidirectional: If True, also connect group agents to the specified agent
        """
        if agent_id not in self.agents:
            logger.warning(f"Agent with ID {agent_id} is not registered")
            return
        
        if group_name not in self.agent_groups:
            logger.warning(f"Group '{group_name}' does not exist")
            return
        
        # Connect agent to all agents in group
        for target_id in self.agent_groups[group_name]:
            if agent_id != target_id:  # Don't connect an agent to itself
                self.connect_agents(agent_id, target_id, bidirectional)
    
    def connect_all_agents(self, bidirectional: bool = True) -> None:
        """
        Connect all registered agents to each other.
        
        Args:
            bidirectional: If True, connect in both directions
        """
        agent_ids = list(self.agents.keys())
        
        # Connect all agents to all other agents
        for i, source_id in enumerate(agent_ids):
            for target_id in agent_ids[i+1:]:  # Start from next agent to avoid duplicates
                self.connect_agents(source_id, target_id, bidirectional)
    
    async def start(self, start_agents: bool = True) -> None:
        """
        Start the orchestrator and optionally all registered agents.
        
        Args:
            start_agents: Whether to also start all registered agents
        """
        if self.is_running:
            logger.warning(f"{self.name} is already running")
            return
        
        logger.info(f"Starting {self.name}")
        self.is_running = True
        self.start_time = time.time()
        
        # Initialize performance monitoring if enabled
        if self.enable_performance_monitoring and PERFORMANCE_MONITORING_AVAILABLE:
            await self._init_performance_monitoring()
        
        # Start all agents if requested
        if start_agents:
            for agent in self.agents.values():
                await agent.start()
        
        # Start the monitoring task
        monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.tasks.append(monitoring_task)
        
        # Trigger event hook
        await self._trigger_event("system_started")
        
        logger.info(f"{self.name} started successfully")
    
    async def stop(self, stop_agents: bool = True) -> None:
        """
        Stop the orchestrator and optionally all registered agents.
        
        Args:
            stop_agents: Whether to also stop all registered agents
        """
        if not self.is_running:
            logger.warning(f"{self.name} is not running")
            return
        
        logger.info(f"Stopping {self.name}")
        self.is_running = False
        self.stop_time = time.time()
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        self.tasks = []
        
        # Stop all agents if requested
        if stop_agents:
            for agent in self.agents.values():
                await agent.stop()
        
        # Stop performance monitoring if active
        if self._performance_monitor:
            await self._stop_performance_monitoring()
        
        # Trigger event hook
        await self._trigger_event("system_stopped")
        
        logger.info(f"{self.name} stopped successfully")
    
    async def _monitoring_loop(self) -> None:
        """Monitor the health and performance of the system."""
        logger.info(f"{self.name} started monitoring loop")
        
        # Check every 30 seconds
        check_interval = 30
        
        while self.is_running:
            try:
                # Check agent health
                for agent in self.agents.values():
                    queue_size = agent.input_queue.qsize()
                    
                    # Log warnings for agents with large queues
                    if queue_size > 100:
                        logger.warning(f"Agent {agent.name} has a large queue size: {queue_size}")
                
                # Wait for next check
                await asyncio.sleep(check_interval)
            
            except asyncio.CancelledError:
                # Loop was cancelled, exit gracefully
                logger.info(f"{self.name} monitoring loop cancelled")
                break
            
            except Exception as e:
                # Log unexpected errors but keep monitoring
                logger.error(f"Error in monitoring loop: {str(e)}")
                await self._trigger_event("error", "monitoring_loop", str(e))
                
                # Short delay before retrying
                await asyncio.sleep(5)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get the current status of the system.
        
        Returns:
            Dict: System status information
        """
        agent_stats = {}
        for agent_id, agent in self.agents.items():
            try:
                # Get communication metrics for each agent
                comm_metrics = await agent.get_communication_metrics() if hasattr(agent, 'get_communication_metrics') else {}
                
                agent_stats[agent_id] = {
                    "name": agent.name,
                    "is_running": agent.is_running,
                    "input_queue_size": agent.input_queue.qsize(),
                    "connected_agents": len(agent.connected_agents),
                    "stats": {
                        "messages_received": agent.stats.messages_received,
                        "messages_processed": agent.stats.messages_processed,
                        "messages_sent": agent.stats.messages_sent,
                        "messages_failed": agent.stats.messages_failed,
                        "avg_processing_time": agent.stats.avg_processing_time
                    },
                    "communication": comm_metrics.get("communication", {})
                }
            except Exception as e:
                logger.error(f"Error getting stats for agent {agent_id}: {str(e)}")
                agent_stats[agent_id] = {"name": agent.name, "error": str(e)}
        
        # Get communication manager metrics
        comm_metrics = self.communication_manager.get_metrics() if self.communication_manager else {}
        
        return {
            "id": self.orchestrator_id,
            "name": self.name,
            "is_running": self.is_running,
            "uptime": time.time() - self.start_time if self.start_time else 0,
            "agent_count": len(self.agents),
            "group_count": len(self.agent_groups),
            "agents": agent_stats,
            "communication": comm_metrics
        }
    
    def register_event_hook(self, event_name: str, hook: Callable[..., Coroutine]) -> None:
        """
        Register a hook to be called when a specific event occurs.
        
        Args:
            event_name: Name of the event to hook into
            hook: Async function to call when the event occurs
        """
        if event_name not in self.event_hooks:
            logger.warning(f"Unknown event name: {event_name}")
            self.event_hooks[event_name] = []
        
        self.event_hooks[event_name].append(hook)
        logger.debug(f"Registered hook for event '{event_name}'")
    
    async def _trigger_event(self, event_name: str, *args, **kwargs) -> None:
        """
        Trigger all hooks registered for an event.
        
        Args:
            event_name: Name of the event
            *args, **kwargs: Arguments to pass to the hook functions
        """
        if event_name not in self.event_hooks:
            return
        
        for hook in self.event_hooks[event_name]:
            try:
                await hook(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in event hook for '{event_name}': {str(e)}")
    
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure the orchestrator with the provided settings.
        
        Args:
            config: Dictionary of configuration parameters
        """
        self.config.update(config)
        logger.info(f"{self.name} configured with {len(config)} parameters")
    
    def create_agent_pipeline(self, pipeline_config: List[Dict[str, Any]]) -> None:
        """
        Create a pipeline of connected agents based on a configuration.
        
        Args:
            pipeline_config: List of dictionaries describing the pipeline stages
                Each stage should contain:
                - 'agent_id': ID of the agent
                - 'connects_to': List of agent IDs to connect to (optional)
        """
        # First, validate that all agents exist
        for stage in pipeline_config:
            agent_id = stage.get('agent_id')
            if agent_id not in self.agents:
                logger.error(f"Cannot create pipeline: Agent with ID {agent_id} is not registered")
                return
        
        # Create connections according to the pipeline
        for stage in pipeline_config:
            agent_id = stage.get('agent_id')
            connects_to = stage.get('connects_to', [])
            
            for target_id in connects_to:
                if target_id not in self.agents:
                    logger.warning(f"Pipeline target {target_id} is not a registered agent, skipping")
                    continue
                
                self.connect_agents(agent_id, target_id, bidirectional=False)
        
        logger.info(f"Created agent pipeline with {len(pipeline_config)} stages")
    
    async def broadcast_message(self, message: Message, group_name: Optional[str] = None) -> None:
        """
        Broadcast a message to multiple agents.
        
        Args:
            message: The message to broadcast
            group_name: Optional group name to restrict the broadcast to
        """
        target_agents = []
        
        if group_name:
            # Broadcast to a specific group
            if group_name not in self.agent_groups:
                logger.warning(f"Cannot broadcast: Group '{group_name}' does not exist")
                return
            
            target_agents = [self.agents[agent_id] for agent_id in self.agent_groups[group_name]]
        else:
            # Broadcast to all agents
            target_agents = list(self.agents.values())
        
        # Check if we can use batch broadcasting via communication manager
        # for efficiency when we have many target agents
        if (len(target_agents) > 3 and 
            self.communication_config and 
            self.communication_config.batch_messages):
            
            await self._batch_broadcast(message, target_agents)
        else:
            # Send to each target agent individually
            for agent in target_agents:
                # Create a copy of the message for each agent
                msg_copy = Message.from_dict(message.to_dict())
                msg_copy.target_agent_id = agent.agent_id
                
                # Use source agent if specified, otherwise use orchestrator as source
                if not msg_copy.source_agent_id:
                    msg_copy.source_agent_id = f"orchestrator:{self.orchestrator_id}"
                
                # Use communication manager if available, otherwise fall back to direct delivery
                if self.communication_manager:
                    await self.communication_manager.send_message(msg_copy, agent)
                else:
                    await agent.enqueue_message(msg_copy)
                
                # Trigger event hook
                await self._trigger_event("message_routed", msg_copy, agent)

    async def _batch_broadcast(self, message: Message, target_agents: List[Agent]) -> None:
        """
        Broadcast a message to multiple agents using batching for efficiency.
        
        Args:
            message: The message to broadcast
            target_agents: List of target agents
        """
        for agent in target_agents:
            # Create a copy of the message for each agent
            msg_copy = Message.from_dict(message.to_dict())
            msg_copy.target_agent_id = agent.agent_id
            
            # Use source agent if specified, otherwise use orchestrator as source
            if not msg_copy.source_agent_id:
                msg_copy.source_agent_id = f"orchestrator:{self.orchestrator_id}"
            
            # Since we're batching, we'll just queue the message without waiting for completion
            if self.communication_manager:
                asyncio.create_task(self.communication_manager.send_message(msg_copy, agent))
            else:
                asyncio.create_task(agent.enqueue_message(msg_copy))
            
            # Trigger event hook
            await self._trigger_event("message_routed", msg_copy, agent)
    
    async def _init_performance_monitoring(self):
        """Initialize the performance monitoring system."""
        try:
            self._performance_monitor = await setup_performance_monitoring(
                enable_prometheus=True,
                prometheus_port=8000
            )
            logger.info("Performance monitoring initialized")
        except Exception as e:
            logger.error(f"Failed to initialize performance monitoring: {str(e)}")
            self.enable_performance_monitoring = False
    
    async def _stop_performance_monitoring(self):
        """Stop the performance monitoring system."""
        if not self._performance_monitor:
            return
        
        try:
            await self._performance_monitor.stop_collecting()
            logger.info("Performance monitoring stopped")
        except Exception as e:
            logger.error(f"Error stopping performance monitoring: {str(e)}")
    
    async def start_agent(self, agent_id: str):
        """
        Start a registered agent.
        
        Args:
            agent_id: ID of the agent to start
        """
        if agent_id not in self.agents:
            logger.warning(f"Agent ID {agent_id} is not registered with {self.name}")
            return
        
        agent = self.agents[agent_id]
        await agent.start()
        
        logger.info(f"Started agent {agent.name}")
        
        # Trigger event hook
        await self._trigger_event("agent_started", agent=agent)
    
    async def stop_agent(self, agent_id: str):
        """
        Stop a registered agent.
        
        Args:
            agent_id: ID of the agent to stop
        """
        if agent_id not in self.agents:
            logger.warning(f"Agent ID {agent_id} is not registered with {self.name}")
            return
        
        agent = self.agents[agent_id]
        await agent.stop()
        
        logger.info(f"Stopped agent {agent.name}")
        
        # Trigger event hook
        await self._trigger_event("agent_stopped", agent=agent)
    
    def create_agent_group(self, group_id: str, agent_ids: Optional[List[str]] = None):
        """
        Create a new agent group.
        
        Args:
            group_id: ID for the new group
            agent_ids: Optional list of agent IDs to add to the group
        """
        if group_id in self.agent_groups:
            logger.warning(f"Group ID {group_id} already exists in {self.name}")
            return
        
        # Create an empty group
        self.agent_groups[group_id] = set()
        
        # Add agents if specified
        if agent_ids:
            for agent_id in agent_ids:
                self.add_agent_to_group(agent_id, group_id)
        
        logger.info(f"Created agent group {group_id} with {len(self.agent_groups[group_id])} agents")
    
    def delete_agent_group(self, group_id: str):
        """
        Delete an agent group.
        
        Args:
            group_id: ID of the group to delete
        """
        if group_id not in self.agent_groups:
            logger.warning(f"Group ID {group_id} does not exist in {self.name}")
            return
        
        # Remove the group
        del self.agent_groups[group_id]
        
        logger.info(f"Deleted agent group {group_id}")
    
    async def send_message_to_agent(self, message: Message) -> bool:
        """
        Send a message to a specific agent.
        
        Args:
            message: The message to send
            
        Returns:
            bool: True if the message was sent successfully
        """
        if not message.target_agent_id:
            logger.error("Cannot route message: No target agent specified")
            return False
        
        if message.target_agent_id not in self.agents:
            logger.error(f"Cannot route message: Target agent {message.target_agent_id} not found")
            return False
        
        # Get the target agent
        target_agent = self.agents[message.target_agent_id]
        
        # Set the source if not specified
        if not message.source_agent_id:
            message.source_agent_id = f"orchestrator:{self.orchestrator_id}"
        
        # Use the communication manager if available
        if self.communication_manager:
            success = await self.communication_manager.send_message(message, target_agent)
        else:
            # Enqueue the message directly
            await target_agent.enqueue_message(message)
            success = True
        
        # Trigger event hook
        await self._trigger_event("message_routed", message=message, target_agent=target_agent)
        
        return success
    
    async def send_message_to_group(self, message: Message, group_id: str) -> bool:
        """
        Send a message to all agents in a group.
        
        Args:
            message: The message to send
            group_id: ID of the target group
            
        Returns:
            bool: True if the message was sent to at least one agent
        """
        if group_id not in self.agent_groups:
            logger.error(f"Cannot route message to group: Group {group_id} not found")
            return False
        
        # Get all agents in the group
        agents = self.get_agents_in_group(group_id)
        
        if not agents:
            logger.warning(f"No active agents in group {group_id}")
            return False
        
        # Send the message to each agent
        success = False
        for agent in agents:
            # Clone the message for each agent to avoid shared state
            agent_message = Message.from_dict(message.to_dict())
            agent_message.target_agent_id = agent.agent_id
            
            # Send the message
            await agent.enqueue_message(agent_message)
            success = True
            
            # Trigger event hooks
            await self._trigger_event("message_routed", message=agent_message, target_agent=agent)
        
        return success
    
    async def broadcast_message(self, message: Message) -> bool:
        """
        Broadcast a message to all registered agents.
        
        Args:
            message: The message to broadcast
            
        Returns:
            bool: True if the message was sent to at least one agent
        """
        if not self.agents:
            logger.warning("No agents registered for broadcast")
            return False
        
        # Send the message to each agent
        success = False
        for agent_id, agent in self.agents.items():
            # Clone the message for each agent to avoid shared state
            agent_message = Message.from_dict(message.to_dict())
            agent_message.target_agent_id = agent.agent_id
            
            # Send the message
            await agent.enqueue_message(agent_message)
            success = True
            
            # Trigger event hooks
            await self._trigger_event("message_routed", message=agent_message, target_agent=agent)
        
        return success
    
    async def setup_pipeline(self, agent_ids: List[str]) -> bool:
        """
        Connect agents in a sequential pipeline.
        
        Args:
            agent_ids: List of agent IDs to connect in sequence
            
        Returns:
            bool: True if the pipeline was set up successfully
        """
        if len(agent_ids) < 2:
            logger.error("Cannot set up pipeline: Need at least 2 agents")
            return False
        
        # Check that all agents exist
        for agent_id in agent_ids:
            if agent_id not in self.agents:
                logger.error(f"Cannot set up pipeline: Agent {agent_id} not found")
                return False
        
        # Set up the connections in the pipeline order
        for i in range(len(agent_ids) - 1):
            source_id = agent_ids[i]
            target_id = agent_ids[i + 1]
            
            source_agent = self.agents[source_id]
            target_agent = self.agents[target_id]
            
            # Ensure the agents are connected
            source_agent.connect(target_agent)
            target_agent.connect(source_agent)  # Bidirectional for potential feedback
        
        logger.info(f"Set up pipeline with {len(agent_ids)} agents")
        return True

    def add_agent(self, agent: Agent, groups: Optional[List[str]] = None) -> None:
        """
        Add an agent to the orchestrator.
        
        Args:
            agent: The agent to add
            groups: Optional list of group names to add the agent to
        """
        if agent.agent_id in self.agents:
            logger.warning(f"Agent {agent.agent_id} is already registered")
            return
        
        # Check if agent's communication config matches orchestrator's
        if not hasattr(agent, 'communication_config') or not agent.communication_config:
            # Apply orchestrator's communication config to the agent
            agent.communication_config = self.communication_config
            agent.communication_manager = CommunicationManager(self.communication_config)
            logger.debug(f"Applied orchestrator's communication config to agent {agent.name}")
            
        # Register the agent
        self.agents[agent.agent_id] = agent
        
        # Add to specified groups
        if groups:
            for group_name in groups:
                self.add_agent_to_group(agent.agent_id, group_name)
        
        logger.info(f"Added agent {agent.name} (ID: {agent.agent_id})")
        
        # Trigger event hook
        asyncio.create_task(self._trigger_event("agent_added", agent)) 