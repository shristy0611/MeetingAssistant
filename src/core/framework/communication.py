"""
Enhanced Communication Module for AMPTALK Multi-Agent Framework.

This module extends the core messaging system with advanced features for 
improved inter-agent communication, including distributed agents, optimized
serialization, prioritized message delivery, and performance enhancements.

Author: AMPTALK Team
Date: 2024
"""

import asyncio
import json
import logging
import time
import zlib
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Callable, Tuple, Type
from dataclasses import dataclass, field, asdict
import uuid

from .message import Message, MessageType, MessagePriority, MessageMetadata

# Configure logging
logger = logging.getLogger(__name__)


class CommunicationMode(Enum):
    """Enum defining communication modes between agents."""
    
    LOCAL = "local"               # Direct in-process communication (default)
    MEMORY_CHANNEL = "memory"     # Shared memory communication (same process, optimized)
    SOCKET = "socket"             # Socket-based communication (different processes)
    HTTP = "http"                 # HTTP-based communication (network)
    WEBSOCKET = "websocket"       # WebSocket communication (bidirectional network)


class CompressionType(Enum):
    """Enum defining compression types for message payload."""
    
    NONE = "none"         # No compression
    ZLIB = "zlib"         # ZLIB compression
    SNAPPY = "snappy"     # Snappy compression (if available)
    LZ4 = "lz4"           # LZ4 compression (if available)


class SerializationType(Enum):
    """Enum defining serialization types for messages."""
    
    JSON = "json"         # JSON serialization (default)
    MSGPACK = "msgpack"   # MessagePack serialization
    PICKLE = "pickle"     # Python Pickle serialization (only for trusted environments)
    PROTOBUF = "protobuf" # Protocol Buffers serialization (requires schema)


@dataclass
class CommunicationConfig:
    """Configuration for inter-agent communication."""
    
    # Communication mode between agents
    mode: CommunicationMode = CommunicationMode.LOCAL
    
    # Serialization and compression
    serialization: SerializationType = SerializationType.JSON
    compression: CompressionType = CompressionType.NONE
    compression_threshold: int = 1024  # Only compress if payload size exceeds this threshold (bytes)
    
    # Performance settings
    use_shared_memory: bool = False    # Use shared memory for local communication when possible
    batch_messages: bool = False       # Batch messages for delivery when possible
    max_batch_size: int = 10           # Maximum number of messages to batch together
    batch_timeout: float = 0.1         # Maximum time to wait for batching (seconds)
    
    # Distributed communication settings
    socket_host: str = "localhost"
    socket_port: int = 8765
    http_host: str = "localhost"
    http_port: int = 8080
    ws_host: str = "localhost"
    ws_port: int = 8766
    connection_timeout: float = 5.0
    
    # Security settings
    use_encryption: bool = False       # Encrypt messages for network transport
    authentication_required: bool = False  # Require authentication for remote agents
    
    # Reliability settings
    enable_message_tracking: bool = True  # Track message delivery status
    retry_failed_deliveries: bool = True  # Automatically retry failed deliveries
    max_delivery_attempts: int = 3     # Maximum number of delivery attempts
    
    # Caching settings
    enable_message_caching: bool = False  # Cache frequently accessed messages
    max_cache_size: int = 1000         # Maximum number of messages to cache


class MessageTracker:
    """
    Tracks message delivery status and performance metrics.
    
    This class provides functionality to monitor message delivery,
    track latency, and generate performance reports.
    """
    
    def __init__(self, config: CommunicationConfig):
        """Initialize the message tracker."""
        self.config = config
        self.delivered_messages: Dict[str, Dict[str, Any]] = {}
        self.pending_messages: Dict[str, Dict[str, Any]] = {}
        self.failed_messages: Dict[str, Dict[str, Any]] = {}
        
        # Performance metrics
        self.total_messages_sent: int = 0
        self.total_messages_delivered: int = 0
        self.total_messages_failed: int = 0
        self.total_bytes_sent: int = 0
        self.serialization_times: List[float] = []
        self.delivery_times: List[float] = []
        
    def track_message(self, message: Message, payload_size: int) -> None:
        """
        Start tracking a message.
        
        Args:
            message: The message to track
            payload_size: Size of the serialized payload in bytes
        """
        if not self.config.enable_message_tracking:
            return
            
        self.pending_messages[message.message_id] = {
            "message_type": message.message_type.value,
            "source_agent_id": message.source_agent_id,
            "target_agent_id": message.target_agent_id,
            "priority": message.priority.value,
            "payload_size": payload_size,
            "sent_at": time.time(),
            "delivery_attempts": 0
        }
        self.total_messages_sent += 1
        self.total_bytes_sent += payload_size
        
    def mark_delivered(self, message_id: str, latency: float) -> None:
        """
        Mark a message as successfully delivered.
        
        Args:
            message_id: ID of the delivered message
            latency: Time taken for delivery in seconds
        """
        if not self.config.enable_message_tracking:
            return
            
        if message_id in self.pending_messages:
            message_data = self.pending_messages.pop(message_id)
            message_data["delivered_at"] = time.time()
            message_data["latency"] = latency
            self.delivered_messages[message_id] = message_data
            self.total_messages_delivered += 1
            self.delivery_times.append(latency)
            
    def mark_failed(self, message_id: str, error: str) -> None:
        """
        Mark a message as failed to deliver.
        
        Args:
            message_id: ID of the failed message
            error: Error message describing the failure
        """
        if not self.config.enable_message_tracking:
            return
            
        if message_id in self.pending_messages:
            message_data = self.pending_messages.pop(message_id)
            message_data["failed_at"] = time.time()
            message_data["error"] = error
            self.failed_messages[message_id] = message_data
            self.total_messages_failed += 1
    
    def record_serialization_time(self, time_taken: float) -> None:
        """
        Record time taken to serialize a message.
        
        Args:
            time_taken: Time in seconds to serialize the message
        """
        if not self.config.enable_message_tracking:
            return
            
        self.serialization_times.append(time_taken)
    
    def get_average_delivery_time(self) -> Optional[float]:
        """
        Get the average message delivery time.
        
        Returns:
            Average delivery time in seconds, or None if no messages delivered
        """
        if not self.delivery_times:
            return None
        return sum(self.delivery_times) / len(self.delivery_times)
    
    def get_average_serialization_time(self) -> Optional[float]:
        """
        Get the average message serialization time.
        
        Returns:
            Average serialization time in seconds, or None if no messages serialized
        """
        if not self.serialization_times:
            return None
        return sum(self.serialization_times) / len(self.serialization_times)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive messaging metrics.
        
        Returns:
            Dictionary of messaging performance metrics
        """
        avg_delivery = self.get_average_delivery_time()
        avg_serialization = self.get_average_serialization_time()
        
        return {
            "total_messages_sent": self.total_messages_sent,
            "total_messages_delivered": self.total_messages_delivered,
            "total_messages_failed": self.total_messages_failed,
            "delivery_success_rate": (self.total_messages_delivered / self.total_messages_sent 
                                     if self.total_messages_sent > 0 else 0),
            "total_bytes_sent": self.total_bytes_sent,
            "average_message_size": (self.total_bytes_sent / self.total_messages_sent 
                                    if self.total_messages_sent > 0 else 0),
            "average_delivery_time": avg_delivery if avg_delivery is not None else 0,
            "average_serialization_time": avg_serialization if avg_serialization is not None else 0,
            "pending_messages": len(self.pending_messages),
            "failed_messages": len(self.failed_messages)
        }
    
    
class MessageSerializer:
    """
    Handles serialization and deserialization of messages.
    
    This class provides optimized serialization based on message size,
    content, and configured serialization type.
    """
    
    def __init__(self, config: CommunicationConfig):
        """Initialize the serializer with the provided configuration."""
        self.config = config
        self._check_dependencies()
        
    def _check_dependencies(self) -> None:
        """Check if required dependencies for the selected serialization are available."""
        if self.config.serialization == SerializationType.MSGPACK:
            try:
                import msgpack
                self._msgpack = msgpack
            except ImportError:
                logger.warning("MessagePack serialization requested but 'msgpack' module not found. "
                             "Falling back to JSON serialization.")
                self.config.serialization = SerializationType.JSON
                
        if self.config.serialization == SerializationType.PROTOBUF:
            try:
                from google.protobuf import json_format
                self._json_format = json_format
            except ImportError:
                logger.warning("Protocol Buffers serialization requested but 'protobuf' module not found. "
                             "Falling back to JSON serialization.")
                self.config.serialization = SerializationType.JSON
                
        if self.config.compression == CompressionType.SNAPPY:
            try:
                import snappy
                self._snappy = snappy
            except ImportError:
                logger.warning("Snappy compression requested but 'snappy' module not found. "
                             "Falling back to ZLIB compression.")
                self.config.compression = CompressionType.ZLIB
                
        if self.config.compression == CompressionType.LZ4:
            try:
                import lz4.frame
                self._lz4 = lz4.frame
            except ImportError:
                logger.warning("LZ4 compression requested but 'lz4' module not found. "
                             "Falling back to ZLIB compression.")
                self.config.compression = CompressionType.ZLIB
    
    def serialize(self, message: Message) -> Tuple[bytes, float]:
        """
        Serialize a message to bytes.
        
        Args:
            message: The message to serialize
            
        Returns:
            Tuple of (serialized_data, serialization_time)
        """
        start_time = time.time()
        
        # Convert message to dict
        message_dict = message.to_dict()
        
        # Serialize based on configured type
        if self.config.serialization == SerializationType.JSON:
            serialized = json.dumps(message_dict).encode('utf-8')
        elif self.config.serialization == SerializationType.MSGPACK:
            serialized = self._msgpack.packb(message_dict)
        elif self.config.serialization == SerializationType.PICKLE:
            import pickle
            serialized = pickle.dumps(message_dict)
        elif self.config.serialization == SerializationType.PROTOBUF:
            # This is a simplified approach - a real implementation would use proper protobuf schemas
            json_str = json.dumps(message_dict)
            serialized = self._json_format.Parse(json_str, self._message_proto()).SerializeToString()
        else:
            # Default to JSON
            serialized = json.dumps(message_dict).encode('utf-8')
        
        # Apply compression if needed
        if (self.config.compression != CompressionType.NONE and 
            len(serialized) >= self.config.compression_threshold):
            
            if self.config.compression == CompressionType.ZLIB:
                serialized = zlib.compress(serialized)
            elif self.config.compression == CompressionType.SNAPPY:
                serialized = self._snappy.compress(serialized)
            elif self.config.compression == CompressionType.LZ4:
                serialized = self._lz4.compress(serialized)
        
        serialization_time = time.time() - start_time
        return serialized, serialization_time
    
    def deserialize(self, data: bytes) -> Message:
        """
        Deserialize bytes to a message.
        
        Args:
            data: Serialized message data
            
        Returns:
            Deserialized Message object
        """
        # Detect and decompress if needed
        try:
            if self.config.compression == CompressionType.ZLIB:
                try:
                    data = zlib.decompress(data)
                except zlib.error:
                    # Not compressed with zlib, assume raw data
                    pass
            elif self.config.compression == CompressionType.SNAPPY:
                try:
                    data = self._snappy.decompress(data)
                except:
                    # Not compressed with snappy, assume raw data
                    pass
            elif self.config.compression == CompressionType.LZ4:
                try:
                    data = self._lz4.decompress(data)
                except:
                    # Not compressed with lz4, assume raw data
                    pass
        except Exception as e:
            logger.warning(f"Error during decompression, treating as uncompressed: {str(e)}")
            
        # Deserialize based on configured type
        try:
            if self.config.serialization == SerializationType.JSON:
                message_dict = json.loads(data.decode('utf-8'))
            elif self.config.serialization == SerializationType.MSGPACK:
                message_dict = self._msgpack.unpackb(data)
            elif self.config.serialization == SerializationType.PICKLE:
                import pickle
                message_dict = pickle.loads(data)
            elif self.config.serialization == SerializationType.PROTOBUF:
                proto_message = self._message_proto()
                proto_message.ParseFromString(data)
                message_dict = self._json_format.MessageToDict(proto_message)
            else:
                # Default to JSON
                message_dict = json.loads(data.decode('utf-8'))
                
            return Message.from_dict(message_dict)
        except Exception as e:
            logger.error(f"Error deserializing message: {str(e)}")
            raise ValueError(f"Failed to deserialize message: {str(e)}")
    
    def _message_proto(self):
        """
        Get a protobuf message instance for serialization/deserialization.
        This is a placeholder that would be implemented with actual protobuf schemas.
        """
        raise NotImplementedError("Protobuf serialization not fully implemented")


class CommunicationManager:
    """
    Manages communication between agents using various transport mechanisms.
    
    This class handles the actual message delivery, including serialization,
    network communication, and delivery tracking.
    """
    
    def __init__(self, config: Optional[CommunicationConfig] = None):
        """
        Initialize the communication manager.
        
        Args:
            config: Communication configuration, or None to use defaults
        """
        self.config = config or CommunicationConfig()
        self.serializer = MessageSerializer(self.config)
        self.tracker = MessageTracker(self.config)
        self.message_cache: Dict[str, bytes] = {}
        
        # State for batched message delivery
        self.batch_queues: Dict[str, List[Message]] = {}  # target_agent_id -> message list
        self.batch_timers: Dict[str, asyncio.Task] = {}  # target_agent_id -> timer task
        
        # Connection pools for remote communication
        self.connection_pools = {}
        
        logger.info(f"Initialized CommunicationManager with mode: {self.config.mode.value}")
        
    async def send_message(self, message: Message, target_agent) -> bool:
        """
        Send a message to a target agent.
        
        This is the main entry point for message delivery. The method chooses
        the appropriate transport mechanism based on the configuration and
        target agent location.
        
        Args:
            message: The message to send
            target_agent: The agent to receive the message
            
        Returns:
            bool: True if the message was sent successfully, False otherwise
        """
        # Set source agent ID if not already set
        if not message.source_agent_id:
            logger.warning(f"Message {message.message_id} has no source agent ID")
            return False
            
        # Ensure the message has the correct target agent ID
        message.target_agent_id = target_agent.agent_id
        
        # Check if batching is enabled
        if self.config.batch_messages:
            return await self._batch_message(message, target_agent)
            
        # Otherwise send immediately
        return await self._deliver_message(message, target_agent)
    
    async def _batch_message(self, message: Message, target_agent) -> bool:
        """
        Add a message to a batch for delivery.
        
        Args:
            message: The message to batch
            target_agent: The target agent
            
        Returns:
            bool: True if the message was successfully added to a batch
        """
        target_id = target_agent.agent_id
        
        # Initialize batch queue if needed
        if target_id not in self.batch_queues:
            self.batch_queues[target_id] = []
            
        # Add message to batch queue
        self.batch_queues[target_id].append(message)
        
        # Start batch timer if not already running
        if target_id not in self.batch_timers or self.batch_timers[target_id].done():
            self.batch_timers[target_id] = asyncio.create_task(
                self._flush_batch_after_timeout(target_id, target_agent)
            )
        
        # Flush immediately if batch is full
        if len(self.batch_queues[target_id]) >= self.config.max_batch_size:
            await self._flush_batch(target_id, target_agent)
            
        return True
    
    async def _flush_batch_after_timeout(self, target_id: str, target_agent) -> None:
        """
        Flush a batch after the configured timeout.
        
        Args:
            target_id: ID of the target agent
            target_agent: The target agent object
        """
        try:
            await asyncio.sleep(self.config.batch_timeout)
            await self._flush_batch(target_id, target_agent)
        except asyncio.CancelledError:
            # Task was cancelled, just exit
            pass
        except Exception as e:
            logger.error(f"Error in batch timeout flush: {str(e)}")
    
    async def _flush_batch(self, target_id: str, target_agent) -> None:
        """
        Flush all batched messages to a target agent.
        
        Args:
            target_id: ID of the target agent
            target_agent: The target agent object
        """
        if target_id not in self.batch_queues or not self.batch_queues[target_id]:
            return
            
        # Get all messages in this batch
        messages = self.batch_queues[target_id]
        self.batch_queues[target_id] = []
        
        # Cancel the timer if it's still running
        if target_id in self.batch_timers and not self.batch_timers[target_id].done():
            self.batch_timers[target_id].cancel()
        
        # Deliver all messages in the batch
        for message in messages:
            await self._deliver_message(message, target_agent)
    
    async def _deliver_message(self, message: Message, target_agent) -> bool:
        """
        Deliver a message to a target agent.
        
        This method handles the actual delivery using the appropriate
        transport mechanism.
        
        Args:
            message: The message to deliver
            target_agent: The target agent object
            
        Returns:
            bool: True if delivered successfully, False otherwise
        """
        delivery_start = time.time()
        
        try:
            # Increment hop count
            message.metadata.hop_count += 1
            
            # Check if the message has expired
            if message.metadata.hop_count > message.metadata.ttl:
                logger.warning(f"Message {message.message_id} exceeded TTL (hop count: {message.metadata.hop_count})")
                self.tracker.mark_failed(message.message_id, "TTL exceeded")
                return False
            
            # Serialize the message
            serialized, serialization_time = self.serializer.serialize(message)
            self.tracker.record_serialization_time(serialization_time)
            
            # Track the message
            self.tracker.track_message(message, len(serialized))
            
            # Cache the serialized message if configured
            if self.config.enable_message_caching:
                self._cache_message(message.message_id, serialized)
            
            # Choose delivery method based on communication mode
            if self.config.mode == CommunicationMode.LOCAL:
                success = await self._local_delivery(message, target_agent)
            elif self.config.mode == CommunicationMode.MEMORY_CHANNEL:
                success = await self._memory_channel_delivery(serialized, target_agent)
            elif self.config.mode == CommunicationMode.SOCKET:
                success = await self._socket_delivery(serialized, target_agent)
            elif self.config.mode == CommunicationMode.HTTP:
                success = await self._http_delivery(serialized, target_agent)
            elif self.config.mode == CommunicationMode.WEBSOCKET:
                success = await self._websocket_delivery(serialized, target_agent)
            else:
                logger.error(f"Unsupported communication mode: {self.config.mode}")
                success = False
            
            # Record delivery metrics
            delivery_time = time.time() - delivery_start
            if success:
                self.tracker.mark_delivered(message.message_id, delivery_time)
            else:
                self.tracker.mark_failed(message.message_id, f"Delivery failed with mode {self.config.mode}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error delivering message {message.message_id}: {str(e)}")
            self.tracker.mark_failed(message.message_id, str(e))
            return False
    
    async def _local_delivery(self, message: Message, target_agent) -> bool:
        """
        Deliver a message locally (in-process).
        
        Args:
            message: The message to deliver
            target_agent: The target agent object
            
        Returns:
            bool: True if delivered successfully
        """
        try:
            # Direct enqueue to target agent's input queue
            await target_agent.enqueue_message(message)
            return True
        except Exception as e:
            logger.error(f"Error in local delivery: {str(e)}")
            return False
    
    async def _memory_channel_delivery(self, serialized: bytes, target_agent) -> bool:
        """
        Deliver a message via shared memory channel.
        
        Args:
            serialized: The serialized message
            target_agent: The target agent object
            
        Returns:
            bool: True if delivered successfully
        """
        try:
            # This is a simplified implementation - actual implementation would use shared memory
            message = self.serializer.deserialize(serialized)
            await target_agent.enqueue_message(message)
            return True
        except Exception as e:
            logger.error(f"Error in memory channel delivery: {str(e)}")
            return False
    
    async def _socket_delivery(self, serialized: bytes, target_agent) -> bool:
        """
        Deliver a message via socket.
        
        Args:
            serialized: The serialized message
            target_agent: The target agent object
            
        Returns:
            bool: True if delivered successfully
        """
        # Placeholder for socket-based delivery
        # This would be implemented with actual socket communication
        logger.warning("Socket delivery not fully implemented")
        return False
    
    async def _http_delivery(self, serialized: bytes, target_agent) -> bool:
        """
        Deliver a message via HTTP.
        
        Args:
            serialized: The serialized message
            target_agent: The target agent object
            
        Returns:
            bool: True if delivered successfully
        """
        # Placeholder for HTTP-based delivery
        # This would be implemented with actual HTTP communication
        logger.warning("HTTP delivery not fully implemented")
        return False
    
    async def _websocket_delivery(self, serialized: bytes, target_agent) -> bool:
        """
        Deliver a message via WebSocket.
        
        Args:
            serialized: The serialized message
            target_agent: The target agent object
            
        Returns:
            bool: True if delivered successfully
        """
        # Placeholder for WebSocket-based delivery
        # This would be implemented with actual WebSocket communication
        logger.warning("WebSocket delivery not fully implemented")
        return False
    
    def _cache_message(self, message_id: str, serialized: bytes) -> None:
        """
        Cache a serialized message.
        
        Args:
            message_id: ID of the message to cache
            serialized: Serialized message data
        """
        if len(self.message_cache) >= self.config.max_cache_size:
            # Simple LRU eviction - remove oldest entry
            oldest_key = next(iter(self.message_cache))
            del self.message_cache[oldest_key]
            
        self.message_cache[message_id] = serialized
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get communication performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        metrics = self.tracker.get_metrics()
        
        # Add additional metrics
        metrics.update({
            "communication_mode": self.config.mode.value,
            "serialization_type": self.config.serialization.value,
            "compression_type": self.config.compression.value,
            "cache_size": len(self.message_cache),
            "cache_hit_rate": 0.0,  # Would be implemented in a full version
            "batch_queues": len(self.batch_queues),
            "total_queued_messages": sum(len(queue) for queue in self.batch_queues.values())
        })
        
        return metrics 