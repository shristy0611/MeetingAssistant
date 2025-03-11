"""
Message Protocol for AMPTALK Multi-Agent Framework.

This module defines the message protocol used for communication between agents
in the AMPTALK system. It includes message types, priorities, and serialization.

Author: AMPTALK Team
Date: 2024
"""

import time
import uuid
import json
from enum import Enum
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict


class MessageType(Enum):
    """Enum defining standard message types for agent communication."""
    
    # Control messages
    INITIALIZE = "initialize"          # System initialization
    SHUTDOWN = "shutdown"              # System shutdown
    STATUS_REQUEST = "status_request"  # Request agent status
    STATUS_RESPONSE = "status_response" # Agent status reply
    
    # Error and recovery
    ERROR = "error"                    # General error message
    RECOVERY_REQUEST = "recovery_request"  # Request recovery action
    RECOVERY_RESPONSE = "recovery_response"  # Recovery action result
    
    # Audio processing
    AUDIO_INPUT = "audio_input"          # Raw audio input
    AUDIO_PROCESSED = "audio_processed"  # Processed audio segments
    AUDIO_ERROR = "audio_error"          # Audio processing error
    
    # Transcription
    TRANSCRIPTION_REQUEST = "transcription_request"    # Request for transcription
    TRANSCRIPTION_RESULT = "transcription_result"      # Transcription result
    TRANSCRIPTION_PARTIAL = "transcription_partial"    # Partial transcription (streaming)
    TRANSCRIPTION_ERROR = "transcription_error"        # Transcription error
    
    # NLP processing
    NLP_REQUEST = "nlp_request"          # Request for NLP processing
    NLP_RESULT = "nlp_result"            # NLP processing result
    ENTITY_DETECTION = "entity_detection" # Entity detection result
    TOPIC_DETECTION = "topic_detection"   # Topic detection result
    INTENT_RECOGNITION = "intent_recognition" # Intent recognition result
    
    # Sentiment analysis
    SENTIMENT_REQUEST = "sentiment_request"  # Request for sentiment analysis
    SENTIMENT_RESULT = "sentiment_result"    # Sentiment analysis result
    PAIN_POINT_DETECTED = "pain_point_detected" # Pain point detection
    
    # Summarization
    SUMMARY_REQUEST = "summary_request"  # Request for summarization
    SUMMARY_RESULT = "summary_result"    # Summarization result
    
    # User interaction
    USER_QUERY = "user_query"            # User query
    SYSTEM_RESPONSE = "system_response"  # System response to user


class MessagePriority(Enum):
    """Enum defining message priorities."""
    
    LOW = 0       # Background tasks, non-time-critical
    NORMAL = 1    # Standard operations
    HIGH = 2      # Time-sensitive operations
    CRITICAL = 3  # System-critical operations


@dataclass
class ErrorDetails:
    """Details about an error that occurred during message processing."""
    
    message: str                      # Human-readable error message
    error_type: str                   # Type/class of the error
    timestamp: float = field(default_factory=time.time)  # When the error occurred
    traceback: Optional[str] = None   # Stack trace if available
    is_recoverable: bool = True       # Whether the error is potentially recoverable
    context: Dict[str, Any] = field(default_factory=dict)  # Additional context
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary for serialization."""
        return {
            "message": self.message,
            "error_type": self.error_type,
            "timestamp": self.timestamp,
            "traceback": self.traceback,
            "is_recoverable": self.is_recoverable,
            "context": self.context
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ErrorDetails':
        """Create an ErrorDetails instance from a dictionary."""
        return cls(**data)


@dataclass
class MessageMetadata:
    """Class for storing metadata about a message."""
    
    # Message creation details
    created_at: float = field(default_factory=time.time)    # Timestamp of creation
    processing_started_at: Optional[float] = None           # When processing began
    processing_completed_at: Optional[float] = None         # When processing completed
    
    # Tracking and routing
    conversation_id: Optional[str] = None         # Group related messages
    parent_message_id: Optional[str] = None       # Reference to parent message
    hop_count: int = 0                            # Number of agent hops so far
    ttl: int = 10                                 # Time-to-live (max hops)
    
    # Processing details
    retry_count: int = 0                          # Number of retries
    max_retries: int = 3                          # Maximum retries
    processing_time: Optional[float] = None       # Time taken to process
    
    # Error handling
    error: Optional[str] = None                   # Simple error message (for backward compatibility)
    error_details: Optional[ErrorDetails] = None  # Detailed error information
    recovery_attempts: List[Dict[str, Any]] = field(default_factory=list)  # Record of recovery attempts
    
    # State tracking for recovery
    checkpoint_data: Dict[str, Any] = field(default_factory=dict)  # State data for recovery
    last_successful_stage: Optional[str] = None   # Last successfully completed processing stage
    
    def mark_processing_start(self) -> None:
        """Mark the start of message processing."""
        self.processing_started_at = time.time()
    
    def mark_processing_complete(self) -> None:
        """Mark the completion of message processing and calculate duration."""
        self.processing_completed_at = time.time()
        if self.processing_started_at:
            self.processing_time = self.processing_completed_at - self.processing_started_at
    
    def increment_hop(self) -> None:
        """Increment the hop count as message moves between agents."""
        self.hop_count += 1
    
    def increment_retry(self) -> None:
        """Increment the retry count when a message is reprocessed."""
        self.retry_count += 1
    
    def set_error(self, error_message: str) -> None:
        """Set simple error information when processing fails (legacy method)."""
        self.error = error_message
    
    def set_detailed_error(self, error_details: ErrorDetails) -> None:
        """Set detailed error information for better recovery handling."""
        self.error = error_details.message  # For backward compatibility
        self.error_details = error_details
    
    def save_checkpoint(self, stage: str, data: Dict[str, Any]) -> None:
        """
        Save a processing checkpoint to aid in recovery.
        
        Args:
            stage: Identifier for the processing stage
            data: State data to preserve
        """
        self.checkpoint_data[stage] = data
        self.last_successful_stage = stage
    
    def record_recovery_attempt(self, strategy: str, timestamp: float = None, 
                               successful: bool = False, notes: str = None) -> None:
        """
        Record an attempt to recover from an error.
        
        Args:
            strategy: The recovery strategy used
            timestamp: When the attempt occurred (default: now)
            successful: Whether the recovery succeeded
            notes: Any additional information
        """
        self.recovery_attempts.append({
            "strategy": strategy,
            "timestamp": timestamp or time.time(),
            "successful": successful,
            "notes": notes
        })
    
    def clear_error(self) -> None:
        """Clear error information after successful recovery."""
        self.error = None
        self.error_details = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to a dictionary for serialization."""
        result = {
            "created_at": self.created_at,
            "processing_started_at": self.processing_started_at,
            "processing_completed_at": self.processing_completed_at,
            "conversation_id": self.conversation_id,
            "parent_message_id": self.parent_message_id,
            "hop_count": self.hop_count,
            "ttl": self.ttl,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "processing_time": self.processing_time,
            "error": self.error,
            "recovery_attempts": self.recovery_attempts,
            "checkpoint_data": self.checkpoint_data,
            "last_successful_stage": self.last_successful_stage
        }
        
        if self.error_details:
            result["error_details"] = self.error_details.to_dict()
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MessageMetadata':
        """Create a MessageMetadata instance from a dictionary."""
        # Make a copy to avoid modifying the input
        data_copy = data.copy()
        
        # Handle error_details separately
        error_details_data = data_copy.pop("error_details", None)
        
        # Create the instance
        instance = cls(**data_copy)
        
        # Set error_details if provided
        if error_details_data:
            instance.error_details = ErrorDetails.from_dict(error_details_data)
        
        return instance


@dataclass
class Message:
    """
    Main message class for agent communication.
    
    This class defines the structure for messages exchanged between agents,
    including routing information, payload, and metadata.
    """
    
    # Core message fields (required)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType = MessageType.STATUS_REQUEST
    source_agent_id: str = "system"
    target_agent_id: str = "system"
    priority: MessagePriority = MessagePriority.NORMAL
    
    # Content field
    payload: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    metadata: MessageMetadata = field(default_factory=MessageMetadata)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to a dictionary for serialization."""
        message_dict = {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "source_agent_id": self.source_agent_id,
            "target_agent_id": self.target_agent_id,
            "priority": self.priority.value,
            "payload": self.payload,
            "metadata": self.metadata.to_dict()
        }
        return message_dict
    
    def to_json(self) -> str:
        """Convert message to a JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create a Message from a dictionary."""
        # Create a copy to avoid modifying the input
        data_copy = data.copy()
        
        # Convert string values back to enums
        if "message_type" in data_copy:
            data_copy["message_type"] = MessageType(data_copy["message_type"])
        if "priority" in data_copy:
            data_copy["priority"] = MessagePriority(data_copy["priority"])
        
        # Handle metadata separately
        metadata_dict = data_copy.pop("metadata", {})
        metadata = MessageMetadata.from_dict(metadata_dict)
        
        # Create the message
        message = cls(
            message_id=data_copy.get("message_id", str(uuid.uuid4())),
            message_type=data_copy.get("message_type", MessageType.STATUS_REQUEST),
            source_agent_id=data_copy.get("source_agent_id", "system"),
            target_agent_id=data_copy.get("target_agent_id", "system"),
            priority=data_copy.get("priority", MessagePriority.NORMAL),
            payload=data_copy.get("payload", {}),
            metadata=metadata
        )
        
        return message
    
    @classmethod
    def from_json(cls, json_str: str) -> "Message":
        """Create a Message from a JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def get_processing_time(self) -> Optional[float]:
        """Get the processing time for this message, if available."""
        return self.metadata.processing_time
    
    def is_expired(self) -> bool:
        """Check if the message has exceeded its TTL (time to live)."""
        return self.metadata.hop_count >= self.metadata.ttl
    
    def can_retry(self) -> bool:
        """Check if the message can be retried if processing fails."""
        return self.metadata.retry_count < self.metadata.max_retries
    
    def get_latency(self) -> Optional[float]:
        """Calculate the total latency from creation to completion."""
        if self.metadata.processing_completed_at:
            return self.metadata.processing_completed_at - self.metadata.created_at
        return None
    
    def create_reply(self, message_type: MessageType, 
                    payload: Dict[str, Any], 
                    priority: Optional[MessagePriority] = None) -> "Message":
        """
        Create a reply message to this message.
        
        Args:
            message_type: Type of the reply message
            payload: Content of the reply
            priority: Priority of the reply (defaults to same as original)
            
        Returns:
            A new message configured as a reply to this one
        """
        # Set up the reply metadata
        metadata = MessageMetadata()
        metadata.conversation_id = self.metadata.conversation_id
        metadata.parent_message_id = self.message_id
        
        # Create the reply message
        reply = Message(
            message_type=message_type,
            source_agent_id=self.target_agent_id,  # Swap source and target
            target_agent_id=self.source_agent_id,
            priority=priority or self.priority,
            payload=payload,
            metadata=metadata
        )
        
        return reply
    
    def create_error_reply(self, error_message: str, error_type: str = "ProcessingError",
                         is_recoverable: bool = True, context: Dict[str, Any] = None) -> "Message":
        """
        Create an error reply message.
        
        Args:
            error_message: Human-readable error description
            error_type: Classification of the error
            is_recoverable: Whether the error can be recovered from
            context: Additional context about the error
            
        Returns:
            A new error message
        """
        # Create error details
        error_details = ErrorDetails(
            message=error_message,
            error_type=error_type,
            is_recoverable=is_recoverable,
            context=context or {}
        )
        
        # Create reply message
        reply = self.create_reply(
            message_type=MessageType.ERROR, 
            payload={"error": error_message, "error_details": error_details.to_dict()},
            priority=MessagePriority.HIGH  # Errors are high priority
        )
        
        # Add error details to metadata
        reply.metadata.set_detailed_error(error_details)
        
        return reply
    
    def checkpoint(self, stage: str, data: Dict[str, Any]) -> None:
        """
        Save checkpoint data for recovery.
        
        Args:
            stage: Name of the processing stage
            data: State data to preserve
        """
        self.metadata.save_checkpoint(stage, data)


# Factory functions for creating common message types

def create_status_request(source_id: str, target_id: str) -> Message:
    """
    Create a status request message.
    
    Args:
        source_id: ID of the requesting agent
        target_id: ID of the agent to request status from
        
    Returns:
        A configured status request message
    """
    return Message(
        message_type=MessageType.STATUS_REQUEST,
        source_agent_id=source_id,
        target_agent_id=target_id,
        priority=MessagePriority.LOW,
        payload={}
    )


def create_recovery_request(source_id: str, target_id: str, 
                          failed_message_id: str,
                          error_details: Dict[str, Any]) -> Message:
    """
    Create a message requesting recovery for a failed operation.
    
    Args:
        source_id: ID of the requesting agent
        target_id: ID of the agent to perform recovery
        failed_message_id: ID of the message that failed
        error_details: Information about the error
        
    Returns:
        A configured recovery request message
    """
    return Message(
        message_type=MessageType.RECOVERY_REQUEST,
        source_agent_id=source_id,
        target_agent_id=target_id,
        priority=MessagePriority.HIGH,
        payload={
            "failed_message_id": failed_message_id,
            "error_details": error_details
        }
    )


def create_audio_input_message(source_id: str, target_id: str, 
                              audio_data: bytes, sample_rate: int,
                              priority: MessagePriority = MessagePriority.HIGH) -> Message:
    """
    Create a message containing audio input data.
    
    Args:
        source_id: ID of the source agent
        target_id: ID of the target agent
        audio_data: Binary audio data
        sample_rate: Audio sample rate in Hz
        priority: Message priority level
        
    Returns:
        A configured audio input message
    """
    return Message(
        message_type=MessageType.AUDIO_INPUT,
        source_agent_id=source_id,
        target_agent_id=target_id,
        priority=priority,
        payload={
            "audio_data": audio_data,
            "sample_rate": sample_rate,
            "timestamp": time.time()
        }
    )


def create_transcription_result_message(source_id: str, target_id: str,
                                       text: str, confidence: float,
                                       start_time: float, end_time: float,
                                       language: str = "en") -> Message:
    """
    Create a message containing transcription results.
    
    Args:
        source_id: ID of the source agent
        target_id: ID of the target agent
        text: Transcribed text
        confidence: Confidence score (0-1)
        start_time: Start time of the audio segment
        end_time: End time of the audio segment
        language: Language code
        
    Returns:
        A configured transcription result message
    """
    return Message(
        message_type=MessageType.TRANSCRIPTION_RESULT,
        source_agent_id=source_id,
        target_agent_id=target_id,
        priority=MessagePriority.NORMAL,
        payload={
            "text": text,
            "confidence": confidence,
            "start_time": start_time,
            "end_time": end_time,
            "language": language,
            "processing_time": time.time()
        }
    ) 