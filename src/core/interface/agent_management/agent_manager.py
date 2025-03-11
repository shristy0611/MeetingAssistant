#!/usr/bin/env python3
"""
Agent Management Module.

This module provides a comprehensive agent management interface for the AMPTALK system,
allowing users to create, configure, monitor, and control agents.

Author: AMPTALK Team
Date: 2024
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Set, Callable, Union, Type
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
import uuid

from src.core.utils.logging_config import get_logger
from src.core.agents.base_agent import BaseAgent
from src.core.orchestration.orchestrator import Orchestrator

logger = get_logger(__name__)

class AgentStatus(Enum):
    """Agent status enum."""
    CREATED = "created"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"

@dataclass
class AgentInfo:
    """Agent information dataclass."""
    id: str
    name: str
    agent_type: str
    status: AgentStatus
    config: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    metrics: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    error_count: int = 0
    last_error: Optional[str] = None

class AgentManager:
    """
    Agent management interface.
    
    Features:
    - Agent lifecycle management (create, start, stop, restart)
    - Agent configuration management
    - Agent status monitoring
    - Agent dependency management
    - Agent grouping and tagging
    - Agent metrics collection
    """
    
    def __init__(
        self,
        orchestrator: Optional[Orchestrator] = None,
        config_dir: str = "config/agents",
        auto_save: bool = True
    ):
        """
        Initialize agent manager.
        
        Args:
            orchestrator: Optional orchestrator instance
            config_dir: Agent configuration directory
            auto_save: Whether to auto-save agent configurations
        """
        self.orchestrator = orchestrator
        self.config_dir = Path(config_dir)
        self.auto_save = auto_save
        
        # Initialize storage
        self._setup_storage()
        
        # Agent registry
        self.agents: Dict[str, AgentInfo] = {}
        self.agent_instances: Dict[str, BaseAgent] = {}
        self.agent_types: Dict[str, Type[BaseAgent]] = {}
        
        # Agent groups
        self.groups: Dict[str, Set[str]] = {}
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {
            "agent_created": [],
            "agent_started": [],
            "agent_stopped": [],
            "agent_error": [],
            "agent_config_updated": []
        }
        
        # Load saved configurations
        self._load_agent_configs()
    
    def _setup_storage(self) -> None:
        """Set up agent configuration storage."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def register_agent_type(
        self,
        agent_type: str,
        agent_class: Type[BaseAgent]
    ) -> None:
        """
        Register agent type.
        
        Args:
            agent_type: Agent type identifier
            agent_class: Agent class
        """
        if not issubclass(agent_class, BaseAgent):
            raise ValueError(
                f"Agent class must be a subclass of BaseAgent"
            )
        
        self.agent_types[agent_type] = agent_class
        logger.info(f"Registered agent type: {agent_type}")
    
    def register_event_handler(
        self,
        event: str,
        handler: Callable
    ) -> None:
        """
        Register event handler.
        
        Args:
            event: Event name
            handler: Event handler function
        """
        if event not in self.event_handlers:
            self.event_handlers[event] = []
        
        self.event_handlers[event].append(handler)
        logger.debug(f"Registered handler for event: {event}")
    
    def _trigger_event(
        self,
        event: str,
        agent_id: str,
        **kwargs
    ) -> None:
        """
        Trigger event.
        
        Args:
            event: Event name
            agent_id: Agent ID
            **kwargs: Additional event data
        """
        if event not in self.event_handlers:
            return
        
        agent_info = self.agents.get(agent_id)
        if not agent_info:
            return
        
        for handler in self.event_handlers[event]:
            try:
                handler(agent_info, **kwargs)
            except Exception as e:
                logger.error(
                    f"Error in event handler for {event}: {e}"
                )
    
    def create_agent(
        self,
        name: str,
        agent_type: str,
        config: Dict[str, Any],
        tags: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None,
        group: Optional[str] = None
    ) -> str:
        """
        Create agent.
        
        Args:
            name: Agent name
            agent_type: Agent type
            config: Agent configuration
            tags: Optional tags
            dependencies: Optional dependencies
            group: Optional group
        
        Returns:
            Agent ID
        """
        # Validate agent type
        if agent_type not in self.agent_types:
            raise ValueError(
                f"Unknown agent type: {agent_type}"
            )
        
        # Create agent ID
        agent_id = str(uuid.uuid4())
        
        # Create agent info
        now = datetime.utcnow()
        agent_info = AgentInfo(
            id=agent_id,
            name=name,
            agent_type=agent_type,
            status=AgentStatus.CREATED,
            config=config.copy(),
            created_at=now,
            updated_at=now,
            tags=tags or [],
            dependencies=dependencies or []
        )
        
        # Register agent
        self.agents[agent_id] = agent_info
        
        # Add to group if specified
        if group:
            self.add_agent_to_group(agent_id, group)
        
        # Save configuration
        if self.auto_save:
            self._save_agent_config(agent_id)
        
        # Trigger event
        self._trigger_event("agent_created", agent_id)
        
        logger.info(
            f"Created agent: {name} ({agent_type}) with ID: {agent_id}"
        )
        return agent_id
    
    async def start_agent(self, agent_id: str) -> bool:
        """
        Start agent.
        
        Args:
            agent_id: Agent ID
        
        Returns:
            True if successful
        """
        try:
            agent_info = self.agents.get(agent_id)
            if not agent_info:
                logger.error(f"Agent not found: {agent_id}")
                return False
            
            # Check if already running
            if agent_info.status == AgentStatus.RUNNING:
                logger.warning(f"Agent already running: {agent_id}")
                return True
            
            # Check dependencies
            for dep_id in agent_info.dependencies:
                dep_info = self.agents.get(dep_id)
                if not dep_info or dep_info.status != AgentStatus.RUNNING:
                    logger.error(
                        f"Dependency not running: {dep_id}"
                    )
                    return False
            
            # Update status
            agent_info.status = AgentStatus.INITIALIZING
            agent_info.updated_at = datetime.utcnow()
            
            # Create agent instance if needed
            if agent_id not in self.agent_instances:
                agent_class = self.agent_types[agent_info.agent_type]
                agent_instance = agent_class(
                    agent_id=agent_id,
                    name=agent_info.name,
                    **agent_info.config
                )
                self.agent_instances[agent_id] = agent_instance
            
            # Get agent instance
            agent = self.agent_instances[agent_id]
            
            # Start agent
            if self.orchestrator:
                await self.orchestrator.register_agent(agent)
                await self.orchestrator.start_agent(agent_id)
            else:
                await agent.start()
            
            # Update status
            agent_info.status = AgentStatus.RUNNING
            agent_info.updated_at = datetime.utcnow()
            
            # Trigger event
            self._trigger_event("agent_started", agent_id)
            
            logger.info(f"Started agent: {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting agent {agent_id}: {e}")
            
            # Update status
            if agent_id in self.agents:
                self.agents[agent_id].status = AgentStatus.ERROR
                self.agents[agent_id].updated_at = datetime.utcnow()
                self.agents[agent_id].error_count += 1
                self.agents[agent_id].last_error = str(e)
                
                # Trigger event
                self._trigger_event(
                    "agent_error",
                    agent_id,
                    error=str(e)
                )
            
            return False
    
    async def stop_agent(self, agent_id: str) -> bool:
        """
        Stop agent.
        
        Args:
            agent_id: Agent ID
        
        Returns:
            True if successful
        """
        try:
            agent_info = self.agents.get(agent_id)
            if not agent_info:
                logger.error(f"Agent not found: {agent_id}")
                return False
            
            # Check if already stopped
            if agent_info.status in [
                AgentStatus.STOPPED,
                AgentStatus.CREATED
            ]:
                logger.warning(f"Agent already stopped: {agent_id}")
                return True
            
            # Update status
            agent_info.status = AgentStatus.STOPPING
            agent_info.updated_at = datetime.utcnow()
            
            # Stop agent
            if self.orchestrator and agent_id in self.agent_instances:
                await self.orchestrator.stop_agent(agent_id)
            elif agent_id in self.agent_instances:
                await self.agent_instances[agent_id].stop()
            
            # Update status
            agent_info.status = AgentStatus.STOPPED
            agent_info.updated_at = datetime.utcnow()
            
            # Trigger event
            self._trigger_event("agent_stopped", agent_id)
            
            logger.info(f"Stopped agent: {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping agent {agent_id}: {e}")
            
            # Update status
            if agent_id in self.agents:
                self.agents[agent_id].status = AgentStatus.ERROR
                self.agents[agent_id].updated_at = datetime.utcnow()
                self.agents[agent_id].error_count += 1
                self.agents[agent_id].last_error = str(e)
                
                # Trigger event
                self._trigger_event(
                    "agent_error",
                    agent_id,
                    error=str(e)
                )
            
            return False
    
    async def restart_agent(self, agent_id: str) -> bool:
        """
        Restart agent.
        
        Args:
            agent_id: Agent ID
        
        Returns:
            True if successful
        """
        try:
            # Stop agent
            stop_success = await self.stop_agent(agent_id)
            if not stop_success:
                return False
            
            # Wait for agent to stop
            await asyncio.sleep(1)
            
            # Start agent
            return await self.start_agent(agent_id)
            
        except Exception as e:
            logger.error(f"Error restarting agent {agent_id}: {e}")
            return False
    
    def update_agent_config(
        self,
        agent_id: str,
        config: Dict[str, Any],
        restart: bool = False
    ) -> bool:
        """
        Update agent configuration.
        
        Args:
            agent_id: Agent ID
            config: New configuration
            restart: Whether to restart agent
        
        Returns:
            True if successful
        """
        try:
            agent_info = self.agents.get(agent_id)
            if not agent_info:
                logger.error(f"Agent not found: {agent_id}")
                return False
            
            # Update configuration
            old_config = agent_info.config.copy()
            agent_info.config.update(config)
            agent_info.updated_at = datetime.utcnow()
            
            # Save configuration
            if self.auto_save:
                self._save_agent_config(agent_id)
            
            # Trigger event
            self._trigger_event(
                "agent_config_updated",
                agent_id,
                old_config=old_config,
                new_config=agent_info.config
            )
            
            logger.info(f"Updated configuration for agent: {agent_id}")
            
            # Restart if requested
            if restart and agent_info.status == AgentStatus.RUNNING:
                asyncio.create_task(self.restart_agent(agent_id))
            
            return True
            
        except Exception as e:
            logger.error(
                f"Error updating configuration for agent {agent_id}: {e}"
            )
            return False
    
    def get_agent_info(self, agent_id: str) -> Optional[AgentInfo]:
        """
        Get agent information.
        
        Args:
            agent_id: Agent ID
        
        Returns:
            Agent information
        """
        return self.agents.get(agent_id)
    
    def list_agents(
        self,
        status: Optional[AgentStatus] = None,
        agent_type: Optional[str] = None,
        tag: Optional[str] = None,
        group: Optional[str] = None
    ) -> List[AgentInfo]:
        """
        List agents.
        
        Args:
            status: Optional status filter
            agent_type: Optional type filter
            tag: Optional tag filter
            group: Optional group filter
        
        Returns:
            List of agent information
        """
        agents = list(self.agents.values())
        
        # Apply filters
        if status:
            agents = [a for a in agents if a.status == status]
        
        if agent_type:
            agents = [a for a in agents if a.agent_type == agent_type]
        
        if tag:
            agents = [a for a in agents if tag in a.tags]
        
        if group:
            group_agents = self.groups.get(group, set())
            agents = [a for a in agents if a.id in group_agents]
        
        return agents
    
    def add_agent_to_group(
        self,
        agent_id: str,
        group: str
    ) -> bool:
        """
        Add agent to group.
        
        Args:
            agent_id: Agent ID
            group: Group name
        
        Returns:
            True if successful
        """
        if agent_id not in self.agents:
            logger.error(f"Agent not found: {agent_id}")
            return False
        
        if group not in self.groups:
            self.groups[group] = set()
        
        self.groups[group].add(agent_id)
        logger.info(f"Added agent {agent_id} to group: {group}")
        return True
    
    def remove_agent_from_group(
        self,
        agent_id: str,
        group: str
    ) -> bool:
        """
        Remove agent from group.
        
        Args:
            agent_id: Agent ID
            group: Group name
        
        Returns:
            True if successful
        """
        if group not in self.groups:
            logger.warning(f"Group not found: {group}")
            return False
        
        if agent_id not in self.groups[group]:
            logger.warning(
                f"Agent {agent_id} not in group: {group}"
            )
            return False
        
        self.groups[group].remove(agent_id)
        logger.info(
            f"Removed agent {agent_id} from group: {group}"
        )
        return True
    
    def list_groups(self) -> List[str]:
        """
        List groups.
        
        Returns:
            List of group names
        """
        return list(self.groups.keys())
    
    def get_group_agents(self, group: str) -> List[AgentInfo]:
        """
        Get agents in group.
        
        Args:
            group: Group name
        
        Returns:
            List of agent information
        """
        if group not in self.groups:
            return []
        
        return [
            self.agents[agent_id]
            for agent_id in self.groups[group]
            if agent_id in self.agents
        ]
    
    def add_agent_tag(
        self,
        agent_id: str,
        tag: str
    ) -> bool:
        """
        Add tag to agent.
        
        Args:
            agent_id: Agent ID
            tag: Tag
        
        Returns:
            True if successful
        """
        agent_info = self.agents.get(agent_id)
        if not agent_info:
            logger.error(f"Agent not found: {agent_id}")
            return False
        
        if tag not in agent_info.tags:
            agent_info.tags.append(tag)
            agent_info.updated_at = datetime.utcnow()
            
            # Save configuration
            if self.auto_save:
                self._save_agent_config(agent_id)
            
            logger.info(f"Added tag {tag} to agent: {agent_id}")
        
        return True
    
    def remove_agent_tag(
        self,
        agent_id: str,
        tag: str
    ) -> bool:
        """
        Remove tag from agent.
        
        Args:
            agent_id: Agent ID
            tag: Tag
        
        Returns:
            True if successful
        """
        agent_info = self.agents.get(agent_id)
        if not agent_info:
            logger.error(f"Agent not found: {agent_id}")
            return False
        
        if tag in agent_info.tags:
            agent_info.tags.remove(tag)
            agent_info.updated_at = datetime.utcnow()
            
            # Save configuration
            if self.auto_save:
                self._save_agent_config(agent_id)
            
            logger.info(
                f"Removed tag {tag} from agent: {agent_id}"
            )
        
        return True
    
    def update_agent_metrics(
        self,
        agent_id: str,
        metrics: Dict[str, Any]
    ) -> bool:
        """
        Update agent metrics.
        
        Args:
            agent_id: Agent ID
            metrics: Metrics
        
        Returns:
            True if successful
        """
        agent_info = self.agents.get(agent_id)
        if not agent_info:
            logger.error(f"Agent not found: {agent_id}")
            return False
        
        # Update metrics
        agent_info.metrics.update(metrics)
        agent_info.updated_at = datetime.utcnow()
        
        logger.debug(
            f"Updated metrics for agent: {agent_id}"
        )
        return True
    
    def get_agent_metrics(
        self,
        agent_id: str
    ) -> Dict[str, Any]:
        """
        Get agent metrics.
        
        Args:
            agent_id: Agent ID
        
        Returns:
            Agent metrics
        """
        agent_info = self.agents.get(agent_id)
        if not agent_info:
            logger.error(f"Agent not found: {agent_id}")
            return {}
        
        return agent_info.metrics.copy()
    
    def _save_agent_config(self, agent_id: str) -> bool:
        """
        Save agent configuration.
        
        Args:
            agent_id: Agent ID
        
        Returns:
            True if successful
        """
        try:
            agent_info = self.agents.get(agent_id)
            if not agent_info:
                return False
            
            # Create config file
            config_file = self.config_dir / f"{agent_id}.json"
            
            # Prepare data
            data = {
                "id": agent_info.id,
                "name": agent_info.name,
                "agent_type": agent_info.agent_type,
                "config": agent_info.config,
                "created_at": agent_info.created_at.isoformat(),
                "updated_at": agent_info.updated_at.isoformat(),
                "tags": agent_info.tags,
                "dependencies": agent_info.dependencies
            }
            
            # Save to file
            with open(config_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(
                f"Saved configuration for agent: {agent_id}"
            )
            return True
            
        except Exception as e:
            logger.error(
                f"Error saving configuration for agent {agent_id}: {e}"
            )
            return False
    
    def _load_agent_configs(self) -> None:
        """Load agent configurations."""
        try:
            # Find config files
            config_files = list(self.config_dir.glob("*.json"))
            
            for config_file in config_files:
                try:
                    # Load config
                    with open(config_file) as f:
                        data = json.load(f)
                    
                    # Create agent info
                    agent_id = data["id"]
                    agent_info = AgentInfo(
                        id=agent_id,
                        name=data["name"],
                        agent_type=data["agent_type"],
                        status=AgentStatus.STOPPED,
                        config=data["config"],
                        created_at=datetime.fromisoformat(
                            data["created_at"]
                        ),
                        updated_at=datetime.fromisoformat(
                            data["updated_at"]
                        ),
                        tags=data.get("tags", []),
                        dependencies=data.get("dependencies", [])
                    )
                    
                    # Register agent
                    self.agents[agent_id] = agent_info
                    
                    logger.debug(
                        f"Loaded configuration for agent: {agent_id}"
                    )
                    
                except Exception as e:
                    logger.error(
                        f"Error loading configuration from {config_file}: {e}"
                    )
            
            logger.info(
                f"Loaded {len(self.agents)} agent configurations"
            )
            
        except Exception as e:
            logger.error(f"Error loading agent configurations: {e}")
    
    async def start_all_agents(
        self,
        group: Optional[str] = None,
        agent_type: Optional[str] = None,
        tag: Optional[str] = None
    ) -> Dict[str, bool]:
        """
        Start all agents.
        
        Args:
            group: Optional group filter
            agent_type: Optional type filter
            tag: Optional tag filter
        
        Returns:
            Dict of agent IDs to success status
        """
        # Get agents to start
        agents = self.list_agents(
            agent_type=agent_type,
            tag=tag,
            group=group
        )
        
        # Start agents
        results = {}
        for agent in agents:
            if agent.status != AgentStatus.RUNNING:
                results[agent.id] = await self.start_agent(agent.id)
        
        return results
    
    async def stop_all_agents(
        self,
        group: Optional[str] = None,
        agent_type: Optional[str] = None,
        tag: Optional[str] = None
    ) -> Dict[str, bool]:
        """
        Stop all agents.
        
        Args:
            group: Optional group filter
            agent_type: Optional type filter
            tag: Optional tag filter
        
        Returns:
            Dict of agent IDs to success status
        """
        # Get agents to stop
        agents = self.list_agents(
            agent_type=agent_type,
            tag=tag,
            group=group
        )
        
        # Stop agents
        results = {}
        for agent in agents:
            if agent.status == AgentStatus.RUNNING:
                results[agent.id] = await self.stop_agent(agent.id)
        
        return results
    
    def delete_agent(self, agent_id: str) -> bool:
        """
        Delete agent.
        
        Args:
            agent_id: Agent ID
        
        Returns:
            True if successful
        """
        try:
            # Check if agent exists
            if agent_id not in self.agents:
                logger.error(f"Agent not found: {agent_id}")
                return False
            
            # Remove from groups
            for group in list(self.groups.keys()):
                if agent_id in self.groups[group]:
                    self.groups[group].remove(agent_id)
            
            # Remove agent instance
            if agent_id in self.agent_instances:
                del self.agent_instances[agent_id]
            
            # Remove agent info
            del self.agents[agent_id]
            
            # Remove config file
            config_file = self.config_dir / f"{agent_id}.json"
            if config_file.exists():
                config_file.unlink()
            
            logger.info(f"Deleted agent: {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting agent {agent_id}: {e}")
            return False
    
    def get_agent_dependencies(
        self,
        agent_id: str
    ) -> List[AgentInfo]:
        """
        Get agent dependencies.
        
        Args:
            agent_id: Agent ID
        
        Returns:
            List of dependency agent information
        """
        agent_info = self.agents.get(agent_id)
        if not agent_info:
            logger.error(f"Agent not found: {agent_id}")
            return []
        
        return [
            self.agents[dep_id]
            for dep_id in agent_info.dependencies
            if dep_id in self.agents
        ]
    
    def get_dependent_agents(
        self,
        agent_id: str
    ) -> List[AgentInfo]:
        """
        Get agents that depend on this agent.
        
        Args:
            agent_id: Agent ID
        
        Returns:
            List of dependent agent information
        """
        return [
            agent
            for agent in self.agents.values()
            if agent_id in agent.dependencies
        ]
    
    def add_agent_dependency(
        self,
        agent_id: str,
        dependency_id: str
    ) -> bool:
        """
        Add agent dependency.
        
        Args:
            agent_id: Agent ID
            dependency_id: Dependency agent ID
        
        Returns:
            True if successful
        """
        # Check if agents exist
        if agent_id not in self.agents:
            logger.error(f"Agent not found: {agent_id}")
            return False
        
        if dependency_id not in self.agents:
            logger.error(f"Dependency agent not found: {dependency_id}")
            return False
        
        # Check for circular dependency
        if agent_id in self.agents[dependency_id].dependencies:
            logger.error(
                f"Circular dependency detected: "
                f"{agent_id} <-> {dependency_id}"
            )
            return False
        
        # Add dependency
        agent_info = self.agents[agent_id]
        if dependency_id not in agent_info.dependencies:
            agent_info.dependencies.append(dependency_id)
            agent_info.updated_at = datetime.utcnow()
            
            # Save configuration
            if self.auto_save:
                self._save_agent_config(agent_id)
            
            logger.info(
                f"Added dependency {dependency_id} to agent: {agent_id}"
            )
        
        return True
    
    def remove_agent_dependency(
        self,
        agent_id: str,
        dependency_id: str
    ) -> bool:
        """
        Remove agent dependency.
        
        Args:
            agent_id: Agent ID
            dependency_id: Dependency agent ID
        
        Returns:
            True if successful
        """
        # Check if agent exists
        if agent_id not in self.agents:
            logger.error(f"Agent not found: {agent_id}")
            return False
        
        # Remove dependency
        agent_info = self.agents[agent_id]
        if dependency_id in agent_info.dependencies:
            agent_info.dependencies.remove(dependency_id)
            agent_info.updated_at = datetime.utcnow()
            
            # Save configuration
            if self.auto_save:
                self._save_agent_config(agent_id)
            
            logger.info(
                f"Removed dependency {dependency_id} from agent: {agent_id}"
            )
        
        return True 