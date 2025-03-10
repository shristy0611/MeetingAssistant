"""
Base Agent class for the AMPTALK multi-agent framework.

This module defines the foundational Agent class that all specialized
agents in the AMPTALK system will inherit from. It provides common
functionality for message handling, lifecycle management, and
resource tracking.
"""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class AgentStatus(Enum):
    """Enum representing possible agent statuses."""
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class MessagePriority(Enum):
    """Enum representing message priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class Message:
    """Class representing a message between agents."""

    def __init__(
        self,
        source_agent: str,
        target_agent: str,
        message_type: str,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
    ):
        self.message_id = str(uuid.uuid4())
        self.timestamp = time.time()
        self.source_agent = source_agent
        self.target_agent = target_agent
        self.message_type = message_type
        self.payload = payload
        self.priority = priority

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format."""
        return {
            "message_id": self.message_id,
            "timestamp": self.timestamp,
            "source_agent": self.source_agent,
            "target_agent": self.target_agent,
            "message_type": self.message_type,
            "priority": self.priority.value,
            "payload": self.payload,
        }

    def to_json(self) -> str:
        """Convert message to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create a Message from a dictionary."""
        message = cls(
            source_agent=data["source_agent"],
            target_agent=data["target_agent"],
            message_type=data["message_type"],
            payload=data["payload"],
            priority=MessagePriority(data["priority"]),
        )
        message.message_id = data["message_id"]
        message.timestamp = data["timestamp"]
        return message

    @classmethod
    def from_json(cls, json_str: str) -> "Message":
        """Create a Message from a JSON string."""
        return cls.from_dict(json.loads(json_str))


class Agent(ABC):
    """
    Base class for all agents in the AMPTALK framework.
    
    This abstract class defines the interface and common functionality
    that all specialized agents must implement.
    """

    def __init__(self, agent_id: str, agent_type: str):
        """
        Initialize the base agent.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type classification of the agent
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.status = AgentStatus.INITIALIZING
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.outbox: asyncio.Queue = asyncio.Queue()
        self.logger = logging.getLogger(f"Agent.{agent_type}.{agent_id}")
        self.resource_usage: Dict[str, float] = {
            "cpu": 0.0,
            "memory": 0.0,
            "disk": 0.0,
        }
        self.error_count = 0
        self.start_time = time.time()
        self.logger.info(f"Agent {agent_id} of type {agent_type} initialized")

    async def send_message(
        self,
        target_agent: str,
        message_type: str,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> str:
        """
        Send a message to another agent.
        
        Args:
            target_agent: ID of the recipient agent
            message_type: Type of message being sent
            payload: Message content
            priority: Message priority level
            
        Returns:
            str: Message ID of the sent message
        """
        message = Message(
            source_agent=self.agent_id,
            target_agent=target_agent,
            message_type=message_type,
            payload=payload,
            priority=priority,
        )
        await self.outbox.put(message)
        self.logger.debug(f"Sent message {message.message_id} to {target_agent}")
        return message.message_id

    async def receive_message(self) -> Message:
        """
        Receive a message from the agent's inbox.
        
        Returns:
            Message: The received message
        """
        message = await self.inbox.get()
        self.logger.debug(f"Received message {message.message_id} from {message.source_agent}")
        return message

    async def handle_message(self, message: Message) -> None:
        """
        Process an incoming message.
        
        Args:
            message: The message to process
        """
        try:
            self.status = AgentStatus.BUSY
            await self._process_message(message)
            self.inbox.task_done()
        except Exception as e:
            self.logger.error(f"Error processing message {message.message_id}: {str(e)}")
            self.error_count += 1
            self.status = AgentStatus.ERROR
        finally:
            if self.status != AgentStatus.ERROR:
                self.status = AgentStatus.READY

    @abstractmethod
    async def _process_message(self, message: Message) -> None:
        """
        Process a specific message type. Must be implemented by subclasses.
        
        Args:
            message: The message to process
        """
        pass

    async def run(self) -> None:
        """Main agent execution loop."""
        try:
            await self.initialize()
            self.status = AgentStatus.READY
            
            while self.status != AgentStatus.SHUTDOWN:
                try:
                    message = await asyncio.wait_for(self.receive_message(), timeout=1.0)
                    await self.handle_message(message)
                except asyncio.TimeoutError:
                    # No message received, continue the loop
                    await self.idle_processing()
                    
        except Exception as e:
            self.logger.error(f"Agent {self.agent_id} encountered an error: {str(e)}")
            self.status = AgentStatus.ERROR
        finally:
            await self.shutdown()

    async def initialize(self) -> None:
        """Initialize the agent. Override in subclasses for specific initialization."""
        self.logger.info(f"Agent {self.agent_id} starting initialization")
        # Override in subclasses with specific initialization logic
        self.logger.info(f"Agent {self.agent_id} initialized")

    async def idle_processing(self) -> None:
        """Processing to perform when no messages are available."""
        # Override in subclasses for background processing
        pass

    async def shutdown(self) -> None:
        """Clean up resources before agent termination."""
        self.logger.info(f"Agent {self.agent_id} shutting down")
        self.status = AgentStatus.SHUTDOWN
        # Override in subclasses with specific cleanup logic

    def update_resource_usage(self, cpu: float, memory: float, disk: float) -> None:
        """
        Update the agent's resource usage metrics.
        
        Args:
            cpu: CPU usage percentage (0-100)
            memory: Memory usage in MB
            disk: Disk usage in MB
        """
        self.resource_usage["cpu"] = cpu
        self.resource_usage["memory"] = memory
        self.resource_usage["disk"] = disk

    def get_status_report(self) -> Dict[str, Any]:
        """
        Generate a status report for the agent.
        
        Returns:
            Dict containing the agent's current status information
        """
        uptime = time.time() - self.start_time
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "status": self.status.value,
            "uptime": uptime,
            "error_count": self.error_count,
            "inbox_size": self.inbox.qsize(),
            "outbox_size": self.outbox.qsize(),
            "resource_usage": self.resource_usage,
        } 