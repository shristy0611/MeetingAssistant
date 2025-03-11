"""
Agent Module for AMPTALK Multi-Agent Framework.

This module defines the base Agent class that all specialized agents will inherit from,
establishing the common interface and functionality for all agents in the system.

Author: AMPTALK Team
Date: 2024
"""

import time
import asyncio
import logging
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Set, Callable, Coroutine, Union, Tuple, TypeVar, Type
from dataclasses import dataclass, field
import uuid
import json
from enum import Enum
import traceback
import os

from .message import Message, MessageType, MessagePriority, create_status_request
from ..utils.state_manager import (
    StateManager, create_state_manager, StorageType, CacheStrategy
)
from .communication import CommunicationManager, CommunicationConfig, CommunicationMode

# Optional import for performance monitoring
try:
    from ..utils.performance_monitor import get_monitor
    PERFORMANCE_MONITORING_AVAILABLE = True
except ImportError:
    PERFORMANCE_MONITORING_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Classification of error severity for more nuanced error handling."""
    CRITICAL = "critical"  # System cannot continue, requires immediate attention
    HIGH = "high"          # Significant error affecting functionality, requires attention
    MEDIUM = "medium"      # Error affects some functionality but system can continue
    LOW = "low"            # Minor error, can be logged and monitored
    TRANSIENT = "transient" # Temporary error that may resolve with retry


@dataclass
class ErrorContext:
    """Context information about an error for better recovery handling."""
    
    error_type: str  # The type/class of the error
    error_message: str  # The error message
    severity: ErrorSeverity  # The severity classification
    traceback: str  # Full traceback information
    retry_count: int = 0  # Number of retry attempts
    timestamp: float = field(default_factory=time.time)  # When the error occurred
    recoverable: bool = True  # Whether this error is potentially recoverable
    
    # Additional context that might help with recovery
    context: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_exception(cls, exception: Exception, severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                       recoverable: bool = True, context: Dict[str, Any] = None) -> 'ErrorContext':
        """Create an ErrorContext from an exception."""
        return cls(
            error_type=exception.__class__.__name__,
            error_message=str(exception),
            severity=severity,
            traceback=traceback.format_exc(),
            recoverable=recoverable,
            context=context or {}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary for serialization."""
        return {
            "error_type": self.error_type,
            "error_message": self.error_message,
            "severity": self.severity.value,
            "traceback": self.traceback,
            "retry_count": self.retry_count,
            "timestamp": self.timestamp,
            "recoverable": self.recoverable,
            "context": self.context
        }


@dataclass
class AgentStats:
    """Statistics tracking for an agent's performance and activity."""
    
    messages_received: int = 0
    messages_processed: int = 0
    messages_sent: int = 0
    messages_failed: int = 0
    
    total_processing_time: float = 0.0
    avg_processing_time: float = 0.0
    
    started_at: Optional[float] = None
    stopped_at: Optional[float] = None
    
    # Track message types processed
    message_types_processed: Dict[MessageType, int] = field(default_factory=dict)
    
    # Error tracking
    errors: Dict[str, int] = field(default_factory=dict)  # Error type -> count
    retries: int = 0  # Total retry attempts
    successful_retries: int = 0  # Retries that succeeded
    
    def record_message_received(self, message_type: MessageType) -> None:
        """Record a received message."""
        self.messages_received += 1
        self.message_types_processed.setdefault(message_type, 0)
    
    def record_message_processed(self, processing_time: float, message_type: MessageType) -> None:
        """Record a successfully processed message."""
        self.messages_processed += 1
        self.total_processing_time += processing_time
        self.avg_processing_time = self.total_processing_time / self.messages_processed
        self.message_types_processed[message_type] = self.message_types_processed.get(message_type, 0) + 1
    
    def record_message_sent(self) -> None:
        """Record a sent message."""
        self.messages_sent += 1
    
    def record_message_failed(self, error_type: str = "unknown") -> None:
        """Record a failed message processing attempt."""
        self.messages_failed += 1
        self.errors[error_type] = self.errors.get(error_type, 0) + 1
    
    def record_retry_attempt(self, success: bool = False) -> None:
        """Record a retry attempt."""
        self.retries += 1
        if success:
            self.successful_retries += 1
    
    def mark_started(self) -> None:
        """Mark the agent as started."""
        self.started_at = time.time()
    
    def mark_stopped(self) -> None:
        """Mark the agent as stopped."""
        self.stopped_at = time.time()
    
    def get_uptime(self) -> Optional[float]:
        """Get the agent's uptime in seconds."""
        if self.started_at is None:
            return None
        
        end_time = self.stopped_at or time.time()
        return end_time - self.started_at
    
    def get_success_rate(self) -> float:
        """Calculate the message processing success rate."""
        if self.messages_processed + self.messages_failed == 0:
            return 1.0  # No messages processed yet
        
        return self.messages_processed / (self.messages_processed + self.messages_failed)
    
    def get_retry_success_rate(self) -> float:
        """Calculate the retry success rate."""
        if self.retries == 0:
            return 0.0  # No retries attempted
        
        return self.successful_retries / self.retries
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to a dictionary for reporting."""
        result = {
            "messages": {
                "received": self.messages_received,
                "processed": self.messages_processed,
                "sent": self.messages_sent,
                "failed": self.messages_failed,
                "success_rate": self.get_success_rate()
            },
            "timing": {
                "avg_processing_time": self.avg_processing_time,
                "total_processing_time": self.total_processing_time,
                "uptime": self.get_uptime()
            },
            "message_types": {
                msg_type.value: count 
                for msg_type, count in self.message_types_processed.items()
            },
            "errors": {
                "total": self.messages_failed,
                "types": self.errors,
                "retries": {
                    "total": self.retries,
                    "successful": self.successful_retries,
                    "success_rate": self.get_retry_success_rate()
                }
            }
        }
        
        return result


class Agent(ABC):
    """
    Base class for all agents in the AMPTALK multi-agent system.
    
    This abstract class defines the interface and common functionality that
    all agents must implement. It handles message queuing, processing, and
    communication with other agents.
    """
    
    def __init__(self, agent_id: Optional[str] = None, name: Optional[str] = None, 
                communication_config: Optional[CommunicationConfig] = None):
        """
        Initialize a new agent.
        
        Args:
            agent_id: Unique identifier for this agent, generated if not provided
            name: Human-readable name for this agent
            communication_config: Configuration for agent communication, or None to use defaults
        """
        self.agent_id = agent_id or str(uuid.uuid4())
        self.name = name or f"Agent-{self.agent_id[:8]}"
        
        # Message handling
        self.input_queue = asyncio.Queue()
        self.message_handlers: Dict[MessageType, Callable[[Message], Coroutine]] = {}
        
        # Agent state
        self.is_running = False
        self.connected_agents: Dict[str, 'Agent'] = {}
        self.state: Dict[str, Any] = {}  # Volatile state (in-memory)
        self.tasks: List[asyncio.Task] = []
        
        # Communication manager
        self.communication_config = communication_config
        self.communication_manager = CommunicationManager(communication_config)
        
        # Statistics and monitoring
        self.stats = AgentStats()
        self.enable_performance_monitoring = PERFORMANCE_MONITORING_AVAILABLE
        self._performance_monitor = None
        
        # Error recovery configuration
        self.max_retry_attempts = 3
        self.base_retry_delay = 1.0  # seconds
        self.max_retry_delay = 30.0  # seconds
        self.retry_jitter = 0.1  # 10% jitter
        self.error_persistence_path: Optional[str] = None  # Path to store error logs
        
        # State management
        self.state_manager: Optional[StateManager] = None
        self.state_persistence_enabled = False
        self.state_persistence_interval = 60.0  # seconds
        self.state_persistence_path = None
        
        # Recovery handlers
        self.error_handlers: Dict[str, Callable[[Message, ErrorContext], Coroutine]] = {}
        
        logger.info(f"Agent {self.name} ({self.agent_id}) initialized")
    
    def __str__(self) -> str:
        """Return a string representation of the agent."""
        return f"{self.name} ({self.agent_id})"
    
    @abstractmethod
    async def process_message(self, message: Message) -> Optional[List[Message]]:
        """
        Process an incoming message.
        
        This is the main method that agents must implement to handle
        domain-specific message processing logic.
        
        Args:
            message: The message to process
            
        Returns:
            Optional[List[Message]]: Response messages to be sent, or None
        """
        pass
    
    async def enqueue_message(self, message: Message) -> None:
        """
        Add a message to the agent's input queue for processing.
        
        Args:
            message: The message to queue
        """
        # Record message received in stats
        self.stats.record_message_received(message.message_type)
        
        # Add to processing queue
        await self.input_queue.put(message)
        
        logger.debug(f"Agent {self.name} queued message {message.message_id} "
                    f"of type {message.message_type}")
    
    async def start(self) -> None:
        """Start the agent's message processing loop."""
        if self.is_running:
            logger.warning(f"Agent {self.name} is already running")
            return
        
        logger.info(f"Starting agent {self.name}")
        self.is_running = True
        self.stats.mark_started()
        
        # Initialize state manager if enabled
        if self.state_persistence_enabled:
            await self._initialize_state_manager()
        
        # Register with performance monitor if enabled
        if self.enable_performance_monitoring:
            self._register_with_performance_monitor()
        
        # Start message processing task
        process_task = asyncio.create_task(self._message_processing_loop())
        self.tasks.append(process_task)
        
        # Start state persistence task if enabled
        if self.state_persistence_enabled and self.state_manager:
            persistence_task = asyncio.create_task(self._state_persistence_loop())
            self.tasks.append(persistence_task)
        
        # Load state if available
        await self._load_state()
    
    async def stop(self) -> None:
        """Stop the agent's message processing loop."""
        if not self.is_running:
            logger.warning(f"Agent {self.name} is not running")
            return
        
        logger.info(f"Stopping agent {self.name}")
        self.is_running = False
        self.stats.mark_stopped()
        
        # Unregister from performance monitor if registered
        if self.enable_performance_monitoring and self._performance_monitor:
            self._unregister_from_performance_monitor()
        
        # Save state before stopping if enabled
        if self.state_persistence_enabled and self.state_manager:
            await self._save_state()
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        self.tasks = []
    
    def _register_with_performance_monitor(self) -> None:
        """Register this agent with the performance monitor."""
        if not PERFORMANCE_MONITORING_AVAILABLE:
            return
        
        try:
            # Get the global monitor instance
            monitor = get_monitor(create_if_none=True)
            if monitor:
                monitor.add_agent(self)
                self._performance_monitor = monitor
                logger.debug(f"Agent {self.name} registered with performance monitor")
        except Exception as e:
            logger.warning(f"Failed to register agent {self.name} with performance monitor: {str(e)}")
            self.enable_performance_monitoring = False
    
    def _unregister_from_performance_monitor(self) -> None:
        """Unregister this agent from the performance monitor."""
        if not self._performance_monitor:
            return
        
        try:
            self._performance_monitor.remove_agent(self.agent_id)
            logger.debug(f"Agent {self.name} unregistered from performance monitor")
        except Exception as e:
            logger.warning(f"Failed to unregister agent {self.name} from performance monitor: {str(e)}")
    
    async def _message_processing_loop(self) -> None:
        """Main processing loop that pulls messages from the queue and processes them."""
        logger.info(f"Agent {self.name} started message processing loop")
        
        while self.is_running:
            try:
                # Get message from the queue
                message = await self.input_queue.get()
                
                # Mark the start of processing
                message.metadata.mark_processing_start()
                
                # Use performance monitoring if available
                if self.enable_performance_monitoring and self._performance_monitor:
                    # Use the measure_execution_time context manager
                    async with self._measure_processing_time(message):
                        response_messages = await self._process_with_retry(message)
                else:
                    # Regular processing without performance monitoring
                    response_messages = await self._process_with_retry(message)
                
                # Mark processing as complete
                message.metadata.mark_processing_complete()
                
                # Record stats
                if message.metadata.processing_time is not None:
                    self.stats.record_message_processed(
                        message.metadata.processing_time,
                        message.message_type
                    )
                
                # Send any response messages
                if response_messages:
                    for response in response_messages:
                        await self.send_message(response)
            
            except Exception as e:
                # Handle processing errors that weren't caught by retry mechanism
                error_context = ErrorContext.from_exception(
                    e, 
                    severity=ErrorSeverity.HIGH,
                    context={"message_id": message.message_id, "message_type": message.message_type.value}
                )
                
                logger.error(f"Error processing message {message.message_id}: {str(e)}")
                message.metadata.set_error(str(e))
                self.stats.record_message_failed(error_context.error_type)
                
                # Save error for analysis
                await self._persist_error(error_context)
                
            finally:
                # Mark task as done
                self.input_queue.task_done()
            
            # Check for cancellation
            if not self.is_running:
                break
            
        logger.info(f"Agent {self.name} message processing loop ended")
    
    async def _process_with_retry(self, message: Message) -> Optional[List[Message]]:
        """
        Process a message with exponential backoff retry.
        
        Args:
            message: The message to process
            
        Returns:
            Optional[List[Message]]: Response messages, or None
            
        Raises:
            Exception: If all retry attempts fail
        """
        attempt = 0
        last_error = None
        error_context = None
        
        while attempt <= self.max_retry_attempts:
            try:
                # If this is a retry, update message metadata
                if attempt > 0:
                    message.metadata.retry_count = attempt
                    message.metadata.mark_processing_start()  # Reset processing timer
                
                # Process the message
                if attempt == 0:
                    # First attempt - standard processing
                    return await self.process_message(message)
                else:
                    # Retry attempt - log the retry
                    logger.info(f"Retry attempt {attempt}/{self.max_retry_attempts} for message {message.message_id}")
                    
                    # Check if we have a specific handler for this error type
                    if error_context and error_context.error_type in self.error_handlers:
                        # Use specialized handler for this error type
                        handler = self.error_handlers[error_context.error_type]
                        result = await handler(message, error_context)
                        self.stats.record_retry_attempt(success=True)
                        return result
                    else:
                        # Standard retry
                        result = await self.process_message(message)
                        self.stats.record_retry_attempt(success=True)
                        return result
            
            except Exception as e:
                # Create or update error context
                if error_context is None:
                    # First error - create new context
                    error_context = ErrorContext.from_exception(
                        e,
                        context={"message_id": message.message_id, "message_type": message.message_type.value}
                    )
                else:
                    # Update existing context
                    error_context.error_message = str(e)
                    error_context.retry_count = attempt
                    error_context.timestamp = time.time()
                    error_context.traceback = traceback.format_exc()
                
                last_error = e
                attempt += 1
                
                # Skip retries for non-recoverable errors
                if not self._is_recoverable_error(e):
                    error_context.recoverable = False
                    break
                
                # Calculate retry delay with exponential backoff and jitter
                if attempt <= self.max_retry_attempts:
                    delay = min(self.base_retry_delay * (2 ** (attempt - 1)), self.max_retry_delay)
                    jitter = delay * self.retry_jitter * (2 * random.random() - 1)
                    delay += jitter
                    
                    # Record the retry attempt in stats
                    self.stats.record_retry_attempt(success=False)
                    
                    # Wait before retrying
                    logger.info(f"Waiting {delay:.2f}s before retry attempt {attempt} for message {message.message_id}")
                    await asyncio.sleep(delay)
        
        # If we reach here, all retry attempts failed
        if last_error:
            raise last_error
        return None
    
    def _is_recoverable_error(self, error: Exception) -> bool:
        """
        Determine if an error is potentially recoverable with a retry.
        
        Args:
            error: The exception to evaluate
            
        Returns:
            bool: True if the error might be resolved with a retry
        """
        # Network/IO errors are often transient and can be retried
        if isinstance(error, (ConnectionError, TimeoutError, IOError)):
            return True
        
        # Some errors should not be retried
        if isinstance(error, (ValueError, TypeError, KeyError, AttributeError)):
            return False
        
        # Default to allowing retry for other errors
        return True
    
    async def _persist_error(self, error_context: ErrorContext) -> None:
        """
        Persist error information for later analysis.
        
        Args:
            error_context: The error context to persist
        """
        if not self.error_persistence_path:
            return
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.error_persistence_path, exist_ok=True)
            
            # Create a timestamped filename
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"{self.error_persistence_path}/error-{self.agent_id}-{timestamp}.json"
            
            # Prepare data
            data = {
                "agent_id": self.agent_id,
                "agent_name": self.name,
                "timestamp": time.time(),
                "error": error_context.to_dict()
            }
            
            # Save to file
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.debug(f"Persisted error information to {filename}")
        
        except Exception as e:
            logger.error(f"Failed to persist error information: {str(e)}")
    
    async def _initialize_state_manager(self) -> None:
        """Initialize the state manager for agent state persistence."""
        try:
            storage_type = StorageType.FILE if self.state_persistence_path else StorageType.MEMORY
            
            # Create the state manager
            self.state_manager = await create_state_manager(
                agent_id=self.agent_id,
                storage_type=storage_type,
                base_dir=self.state_persistence_path,
                max_memory_mb=100  # Default to 100MB memory limit
            )
            
            logger.info(f"Initialized state manager for agent {self.name} with {storage_type.value} storage")
        except Exception as e:
            logger.error(f"Failed to initialize state manager: {str(e)}")
            self.state_manager = None
    
    async def _save_state(self) -> None:
        """Save agent state for persistence."""
        if not self.state_manager:
            return
        
        try:
            # Add stats to state data
            state_to_save = {
                "state": self.state,
                "stats": self.stats.to_dict(),
                "last_saved": time.time()
            }
            
            # Save to the state manager
            await self.state_manager.save_state(f"{self.agent_id}_state", state_to_save)
            logger.debug(f"Saved state for agent {self.name}")
        except Exception as e:
            logger.error(f"Failed to save agent state: {str(e)}")
    
    async def _load_state(self) -> None:
        """Load agent state if available."""
        if not self.state_manager:
            return
        
        try:
            # Load state from the state manager
            state_data = await self.state_manager.load_state(f"{self.agent_id}_state")
            
            if state_data and isinstance(state_data, dict):
                # Restore state
                if "state" in state_data:
                    self.state = state_data["state"]
                    logger.info(f"Restored state for agent {self.name}")
            
            # We don't restore stats as they should reflect the current session
        except Exception as e:
            logger.error(f"Failed to load agent state: {str(e)}")
    
    async def _state_persistence_loop(self) -> None:
        """Background loop to periodically save agent state."""
        logger.info(f"Agent {self.name} started state persistence loop")
        
        while self.is_running:
            try:
                await asyncio.sleep(self.state_persistence_interval)
                await self._save_state()
            
            except asyncio.CancelledError:
                # Loop was cancelled, exit gracefully
                logger.info(f"Agent {self.name} state persistence loop cancelled")
                break
            
            except Exception as e:
                logger.error(f"Error in state persistence loop for {self.name}: {str(e)}")
                # Continue loop despite errors
    
    async def get_state_value(self, key: str, default: Any = None) -> Any:
        """
        Get a value from agent state, with optional default.
        
        Args:
            key: The key to look up
            default: Default value if key is not found
            
        Returns:
            The state value or default
        """
        return self.state.get(key, default)
    
    async def set_state_value(self, key: str, value: Any) -> None:
        """
        Set a value in agent state.
        
        Args:
            key: The key to set
            value: The value to store
        """
        self.state[key] = value
        
        # If immediate persistence is enabled, save state right away
        if self.state_persistence_enabled and self.state_manager and key.startswith("critical_"):
            await self._save_state()
    
    async def clear_state(self) -> None:
        """Clear all agent state data."""
        self.state.clear()
        
        # If state manager is available, clear persisted state
        if self.state_manager:
            await self.state_manager.delete_state(f"{self.agent_id}_state")
    
    def register_error_handler(self, error_type: str, 
                           handler: Callable[[Message, ErrorContext], Coroutine]) -> None:
        """
        Register a specialized handler for a specific error type.
        
        Args:
            error_type: The name of the exception class to handle
            handler: Async function to handle this error type
        """
        self.error_handlers[error_type] = handler
        logger.debug(f"Registered error handler for {error_type} in agent {self.name}")
    
    async def _measure_processing_time(self, message: Message):
        """
        Context manager to measure message processing time for performance monitoring.
        
        Args:
            message: The message being processed
        """
        class AsyncMeasureTime:
            def __init__(self, agent):
                self.agent = agent
                self.message = message
                self.start_time = None
            
            async def __aenter__(self):
                self.start_time = time.time()
                return self
            
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                if self.start_time:
                    duration = time.time() - self.start_time
                    
                    # Record the processing time if we have a monitor
                    if self.agent._performance_monitor:
                        attrs = {
                            "agent_id": self.agent.agent_id,
                            "agent_name": self.agent.name,
                            "message_type": self.message.message_type.value
                        }
                        self.agent._performance_monitor.agent_processing_time.record(duration, attrs)
                
                # Don't suppress exceptions
                return False
        
        return AsyncMeasureTime(self)
    
    async def send_message(self, message: Message) -> bool:
        """
        Send a message to another agent.
        
        Args:
            message: The message to send
            
        Returns:
            bool: True if the message was sent successfully, False otherwise
        """
        if not message.target_agent_id:
            logger.error(f"Cannot send message {message.message_id}: No target agent specified")
            return False
        
        # Set the source agent ID if not already set
        if not message.source_agent_id:
            message.source_agent_id = self.agent_id
        
        # Try to find the target agent in connected agents
        target_agent = self.connected_agents.get(message.target_agent_id)
        
        if not target_agent:
            logger.error(f"Cannot send message {message.message_id}: "
                         f"Target agent {message.target_agent_id} not found")
            return False
        
        # Use communication manager if it's configured for non-local transport
        if (self.communication_manager and 
            self.communication_config and 
            self.communication_config.mode != CommunicationMode.LOCAL):
            
            # Use the communication manager to send the message
            return await self.communication_manager.send_message(message, target_agent)
        else:
            # Fall back to direct enqueueing if using local transport or for backward compatibility
            try:
                # Enqueue the message to the target agent
                await target_agent.enqueue_message(message)
                
                # Record in stats
                self.stats.record_message_sent()
                
                return True
            except Exception as e:
                logger.error(f"Error sending message {message.message_id}: {str(e)}")
                return False
    
    def connect(self, other_agent: 'Agent') -> None:
        """
        Connect this agent to another agent for direct messaging.
        
        Args:
            other_agent: The agent to connect to
        """
        if other_agent.agent_id == self.agent_id:
            logger.warning(f"Agent {self.name} cannot connect to itself")
            return
        
        # Add to connected agents
        self.connected_agents[other_agent.agent_id] = other_agent
        
        logger.debug(f"Agent {self.name} connected to {other_agent.name}")
    
    def disconnect(self, agent_id: str) -> None:
        """
        Disconnect this agent from another agent.
        
        Args:
            agent_id: The ID of the agent to disconnect from
        """
        if agent_id in self.connected_agents:
            agent = self.connected_agents[agent_id]
            del self.connected_agents[agent_id]
            
            logger.debug(f"Agent {self.name} disconnected from {agent.name}")
        else:
            logger.warning(f"Agent {self.name} is not connected to agent {agent_id}")
    
    def register_message_handler(self, 
                                message_type: MessageType, 
                                handler: Callable[[Message], Coroutine]) -> None:
        """
        Register a handler for a specific message type.
        
        Args:
            message_type: The type of message to handle
            handler: Async function to handle this message type
        """
        self.message_handlers[message_type] = handler
        logger.debug(f"Registered handler for {message_type} in agent {self.name}")
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the agent.
        
        Returns:
            Dict[str, Any]: Status information
        """
        status = {
            "agent_id": self.agent_id,
            "name": self.name,
            "is_running": self.is_running,
            "queue_size": self.input_queue.qsize(),
            "connected_agents": [agent_id for agent_id in self.connected_agents],
            "stats": self.stats.to_dict(),
            "state": {
                "persistence_enabled": self.state_persistence_enabled,
                "persistence_interval": self.state_persistence_interval,
                "keys": list(self.state.keys())
            },
            "error_recovery": {
                "max_retry_attempts": self.max_retry_attempts,
                "base_retry_delay": self.base_retry_delay,
                "registered_error_handlers": list(self.error_handlers.keys())
            },
            "performance_monitoring": {
                "enabled": self.enable_performance_monitoring,
                "registered": self._performance_monitor is not None
            }
        }
        
        # Add state manager info if available
        if self.state_manager:
            status["state_manager"] = self.state_manager.get_stats()
        
        return status
    
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure the agent with custom settings.
        
        Args:
            config: Configuration dictionary
        """
        # Extract error recovery configuration
        if "error_recovery" in config:
            error_config = config["error_recovery"]
            self.max_retry_attempts = error_config.get("max_retry_attempts", self.max_retry_attempts)
            self.base_retry_delay = error_config.get("base_retry_delay", self.base_retry_delay)
            self.max_retry_delay = error_config.get("max_retry_delay", self.max_retry_delay)
            self.retry_jitter = error_config.get("retry_jitter", self.retry_jitter)
            self.error_persistence_path = error_config.get("persistence_path", self.error_persistence_path)
        
        # Extract state persistence configuration
        if "state_persistence" in config:
            state_config = config["state_persistence"]
            self.state_persistence_enabled = state_config.get("enabled", self.state_persistence_enabled)
            self.state_persistence_path = state_config.get("path", self.state_persistence_path)
            self.state_persistence_interval = state_config.get("interval", self.state_persistence_interval)
        
        # Extract performance monitoring configuration
        if "performance_monitoring" in config:
            pm_config = config["performance_monitoring"]
            self.enable_performance_monitoring = pm_config.get(
                "enabled", 
                self.enable_performance_monitoring and PERFORMANCE_MONITORING_AVAILABLE
            )
        
        logger.info(f"Agent {self.name} configured with custom settings")

    async def get_communication_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about agent communication.
        
        Returns:
            Dictionary containing communication metrics
        """
        metrics = {
            "messages_sent": self.stats.messages_sent,
            "messages_received": self.stats.messages_received,
            "messages_processed": self.stats.messages_processed,
            "messages_failed": self.stats.messages_failed,
            "input_queue_size": self.input_queue.qsize()
        }
        
        # Add communication manager metrics if available
        if self.communication_manager:
            comm_metrics = self.communication_manager.get_metrics()
            metrics.update({
                "communication": comm_metrics
            })
            
        return metrics


class SimpleAgent(Agent):
    """
    A simple agent implementation for testing or basic message handling.
    
    This concrete implementation of Agent can be used directly for basic
    functionality or as a reference for creating more specialized agents.
    """
    
    async def process_message(self, message: Message) -> Optional[List[Message]]:
        """Process a message by using the registered handler for its type."""
        if message.message_type in self.message_handlers:
            handler = self.message_handlers[message.message_type]
            return await handler(message)
        
        logger.debug(f"No handler for message type {message.message_type} in {self.name}")
        return None 