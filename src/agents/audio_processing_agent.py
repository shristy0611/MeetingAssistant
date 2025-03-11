"""
Audio Processing Agent for AMPTALK.

This module defines the AudioProcessingAgent which is responsible for
receiving raw audio input, preprocessing it, and preparing it for transcription.

Author: AMPTALK Team
Date: 2024
"""

import logging
import asyncio
import os
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import io
import wave
import time
from dataclasses import dataclass

from src.core.framework.agent import Agent
from src.core.framework.message import (
    Message, MessageType, MessagePriority, 
    create_audio_input_message
)
from src.core.utils.logging_config import get_logger

# Configure logger
logger = get_logger("amptalk.agents.audio")


@dataclass
class AudioSegment:
    """Class representing a segment of audio."""
    
    audio_data: np.ndarray
    sample_rate: int
    start_time: float
    end_time: float
    channels: int = 1
    segment_id: str = ""
    
    def duration(self) -> float:
        """Get the duration of the audio segment in seconds."""
        return self.end_time - self.start_time
    
    def to_bytes(self, format: str = "wav") -> bytes:
        """
        Convert the audio segment to bytes in the specified format.
        
        Args:
            format: Audio format ('wav' is currently the only supported format)
            
        Returns:
            Bytes representing the audio data
        """
        if format.lower() != "wav":
            raise ValueError(f"Unsupported audio format: {format}")
        
        # Convert to int16 for WAV format
        audio_int16 = (self.audio_data * 32767).astype(np.int16)
        
        # Create an in-memory buffer
        buffer = io.BytesIO()
        
        # Write WAV to the buffer
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)  # 2 bytes for int16
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        # Get the bytes
        buffer.seek(0)
        return buffer.read()


class AudioProcessingConfig:
    """Configuration for audio processing."""
    
    def __init__(self, 
                sample_rate: int = 16000,
                chunk_duration_ms: int = 1000,
                vad_threshold: float = 0.3,
                silence_duration_ms: int = 500,
                max_segment_duration_ms: int = 30000,
                channels: int = 1):
        """
        Initialize audio processing configuration.
        
        Args:
            sample_rate: Target sample rate in Hz
            chunk_duration_ms: Duration of each processing chunk in milliseconds
            vad_threshold: Voice activity detection threshold (0-1)
            silence_duration_ms: Silence duration to mark a segment boundary
            max_segment_duration_ms: Maximum duration for a single segment
            channels: Number of audio channels (1 for mono, 2 for stereo)
        """
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.vad_threshold = vad_threshold
        self.silence_duration_ms = silence_duration_ms
        self.max_segment_duration_ms = max_segment_duration_ms
        self.channels = channels
        
        # Derived values
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        self.silence_samples = int(sample_rate * silence_duration_ms / 1000)
        self.max_segment_samples = int(sample_rate * max_segment_duration_ms / 1000)


class AudioProcessingAgent(Agent):
    """
    Agent responsible for processing audio input.
    
    This agent receives raw audio input, performs preprocessing steps
    such as normalization, noise reduction, and voice activity detection,
    then segments the audio for optimal transcription.
    """
    
    def __init__(self, 
                agent_id: Optional[str] = None, 
                name: str = "AudioProcessingAgent",
                config: Optional[Dict[str, Any]] = None):
        """
        Initialize the AudioProcessingAgent.
        
        Args:
            agent_id: Unique identifier for this agent
            name: Human-readable name for this agent
            config: Configuration dictionary
        """
        super().__init__(agent_id, name)
        
        # Initialize with default configuration
        self.audio_config = AudioProcessingConfig()
        
        # Apply custom configuration if provided
        if config:
            self.configure(config)
        
        # Register supported message types
        self.supported_message_types = {
            MessageType.AUDIO_INPUT,
            MessageType.STATUS_REQUEST,
            MessageType.INITIALIZE,
            MessageType.SHUTDOWN
        }
        
        # Current state
        self.is_processing = False
        self.current_segment: Optional[AudioSegment] = None
        
        logger.info(f"Initialized {self.name} with sample rate {self.audio_config.sample_rate}Hz")
    
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure the agent with custom settings.
        
        Args:
            config: Dictionary of configuration parameters
        """
        super().configure(config)
        
        # Update audio configuration if provided
        if 'audio' in config:
            audio_config = config['audio']
            
            if 'sample_rate' in audio_config:
                self.audio_config.sample_rate = int(audio_config['sample_rate'])
            
            if 'chunk_duration_ms' in audio_config:
                self.audio_config.chunk_duration_ms = int(audio_config['chunk_duration_ms'])
                self.audio_config.chunk_size = int(self.audio_config.sample_rate * 
                                                 self.audio_config.chunk_duration_ms / 1000)
            
            if 'vad_threshold' in audio_config:
                self.audio_config.vad_threshold = float(audio_config['vad_threshold'])
            
            if 'silence_duration_ms' in audio_config:
                self.audio_config.silence_duration_ms = int(audio_config['silence_duration_ms'])
                self.audio_config.silence_samples = int(self.audio_config.sample_rate * 
                                                      self.audio_config.silence_duration_ms / 1000)
            
            if 'max_segment_duration_ms' in audio_config:
                self.audio_config.max_segment_duration_ms = int(audio_config['max_segment_duration_ms'])
                self.audio_config.max_segment_samples = int(self.audio_config.sample_rate * 
                                                          self.audio_config.max_segment_duration_ms / 1000)
            
            if 'channels' in audio_config:
                self.audio_config.channels = int(audio_config['channels'])
        
        logger.info(f"{self.name} configured with sample rate {self.audio_config.sample_rate}Hz")
    
    async def process_message(self, message: Message) -> Optional[List[Message]]:
        """
        Process incoming messages.
        
        Args:
            message: The message to process
            
        Returns:
            Optional list of response messages
        """
        message_type = message.message_type
        
        # Handle status request
        if message_type == MessageType.STATUS_REQUEST:
            return await self._handle_status_request(message)
        
        # Handle initialization
        elif message_type == MessageType.INITIALIZE:
            return await self._handle_initialization(message)
        
        # Handle audio input
        elif message_type == MessageType.AUDIO_INPUT:
            return await self._process_audio_input(message)
        
        # Handle shutdown
        elif message_type == MessageType.SHUTDOWN:
            return await self._handle_shutdown(message)
        
        # Unsupported message type
        else:
            logger.warning(f"Received unsupported message type: {message_type.name}")
            return None
    
    async def _handle_status_request(self, message: Message) -> List[Message]:
        """
        Handle a status request message.
        
        Args:
            message: The status request message
            
        Returns:
            List containing a status response message
        """
        # Get agent status
        status = {
            "is_processing": self.is_processing,
            "sample_rate": self.audio_config.sample_rate,
            "channels": self.audio_config.channels,
            "current_segment": self.current_segment.segment_id if self.current_segment else None
        }
        
        # Create response message
        response = Message(
            message_type=MessageType.STATUS_RESPONSE,
            source_agent_id=self.agent_id,
            target_agent_id=message.source_agent_id,
            priority=message.priority,
            payload={"status": status}
        )
        
        # Link to the request message
        response.metadata.parent_message_id = message.message_id
        
        return [response]
    
    async def _handle_initialization(self, message: Message) -> List[Message]:
        """
        Handle an initialization message.
        
        Args:
            message: The initialization message
            
        Returns:
            List containing a status response message
        """
        try:
            # Apply any configuration from the message
            if 'config' in message.payload:
                self.configure(message.payload['config'])
            
            # Reset state
            self.is_processing = False
            self.current_segment = None
            
            logger.info(f"{self.name} initialized successfully")
            
            # Create response message
            response = Message(
                message_type=MessageType.STATUS_RESPONSE,
                source_agent_id=self.agent_id,
                target_agent_id=message.source_agent_id,
                priority=message.priority,
                payload={"status": "initialized", "success": True}
            )
            
            # Link to the request message
            response.metadata.parent_message_id = message.message_id
            
            return [response]
            
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            
            # Create error response
            response = Message(
                message_type=MessageType.STATUS_RESPONSE,
                source_agent_id=self.agent_id,
                target_agent_id=message.source_agent_id,
                priority=MessagePriority.HIGH,
                payload={
                    "status": "error", 
                    "success": False,
                    "error": str(e)
                }
            )
            
            response.metadata.parent_message_id = message.message_id
            return [response]
    
    async def _handle_shutdown(self, message: Message) -> List[Message]:
        """
        Handle a shutdown message.
        
        Args:
            message: The shutdown message
            
        Returns:
            List containing a status response message
        """
        try:
            # Clean up any resources
            self.is_processing = False
            self.current_segment = None
            
            logger.info(f"{self.name} shutdown successfully")
            
            # Create response message
            response = Message(
                message_type=MessageType.STATUS_RESPONSE,
                source_agent_id=self.agent_id,
                target_agent_id=message.source_agent_id,
                priority=message.priority,
                payload={"status": "shutdown", "success": True}
            )
            
            # Link to the request message
            response.metadata.parent_message_id = message.message_id
            
            return [response]
            
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
            
            # Create error response
            response = Message(
                message_type=MessageType.STATUS_RESPONSE,
                source_agent_id=self.agent_id,
                target_agent_id=message.source_agent_id,
                priority=MessagePriority.HIGH,
                payload={
                    "status": "error", 
                    "success": False,
                    "error": str(e)
                }
            )
            
            response.metadata.parent_message_id = message.message_id
            return [response]
    
    async def _process_audio_input(self, message: Message) -> List[Message]:
        """
        Process audio input message.
        
        Args:
            message: The audio input message
            
        Returns:
            List of processed audio segment messages
        """
        try:
            self.is_processing = True
            
            # Extract audio data from message payload
            payload = message.payload
            
            # In a real implementation, this would decode the audio data
            # For now we'll use a placeholder
            audio_format = payload.get('audio_format', 'wav')
            sample_rate = payload.get('sample_rate', self.audio_config.sample_rate)
            channels = payload.get('channels', self.audio_config.channels)
            
            # Log receipt of audio data
            logger.info(f"Received audio input: format={audio_format}, "
                       f"sample_rate={sample_rate}Hz, channels={channels}")
            
            # In a real implementation, we'd process the actual audio
            # For now, we'll simulate processing with a delay
            await asyncio.sleep(0.5)  # Simulate processing time
            
            # Create a simulated audio segment
            timestamp = time.time()
            segment = AudioSegment(
                # In a real implementation, this would be the actual processed audio
                audio_data=np.zeros(self.audio_config.chunk_size),
                sample_rate=self.audio_config.sample_rate,
                start_time=timestamp,
                end_time=timestamp + 1.0,  # 1 second duration
                channels=self.audio_config.channels,
                segment_id=f"segment_{int(timestamp)}"
            )
            
            self.current_segment = segment
            
            # Create processed audio message
            processed_message = Message(
                message_type=MessageType.AUDIO_PROCESSED,
                source_agent_id=self.agent_id,
                # Transcription agent would typically be the target
                # We'll send back to the source for now
                target_agent_id=message.source_agent_id,
                priority=message.priority,
                payload={
                    "segment_id": segment.segment_id,
                    "sample_rate": segment.sample_rate,
                    "channels": segment.channels,
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                    "duration": segment.duration(),
                    # In a real implementation, this would be the actual processed audio
                    "audio_data": "[PROCESSED_AUDIO_DATA]"
                }
            )
            
            # Link to the input message
            processed_message.metadata.parent_message_id = message.message_id
            
            logger.info(f"Processed audio segment: {segment.segment_id}, "
                       f"duration: {segment.duration():.2f}s")
            
            self.is_processing = False
            return [processed_message]
            
        except Exception as e:
            logger.error(f"Error processing audio input: {str(e)}")
            self.is_processing = False
            
            # Create error message
            error_message = Message(
                message_type=MessageType.AUDIO_ERROR,
                source_agent_id=self.agent_id,
                target_agent_id=message.source_agent_id,
                priority=MessagePriority.HIGH,
                payload={"error": str(e)}
            )
            
            error_message.metadata.parent_message_id = message.message_id
            return [error_message]
    
    def _detect_voice_activity(self, audio_data: np.ndarray) -> bool:
        """
        Detect if voice activity is present in the audio.
        
        Args:
            audio_data: Numpy array of audio samples
            
        Returns:
            True if voice activity is detected, False otherwise
        """
        # In a real implementation, this would use a VAD algorithm
        # For now, we'll use a simple energy threshold
        energy = np.mean(np.abs(audio_data))
        return energy > self.audio_config.vad_threshold
    
    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Normalize audio to have a peak amplitude of 1.0.
        
        Args:
            audio_data: Numpy array of audio samples
            
        Returns:
            Normalized audio data
        """
        if np.max(np.abs(audio_data)) > 0:
            return audio_data / np.max(np.abs(audio_data))
        return audio_data
    
    def _resample_audio(self, audio_data: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
        """
        Resample audio to the target sample rate.
        
        Args:
            audio_data: Numpy array of audio samples
            original_rate: Original sample rate in Hz
            target_rate: Target sample rate in Hz
            
        Returns:
            Resampled audio data
        """
        # In a real implementation, this would use a resampling library
        # For now, we'll return the original data with a message
        logger.info(f"Resampling audio from {original_rate}Hz to {target_rate}Hz")
        return audio_data 