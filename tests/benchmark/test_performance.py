"""
Performance benchmarks for the AMPTALK system.

This module contains benchmarks to measure the performance of
key components of the system, providing insights for optimization.

Author: AMPTALK Team
Date: 2024
"""

import pytest
import asyncio
import time
import numpy as np
from typing import List, Dict, Any

from src.core.framework.agent import SimpleAgent
from src.agents.audio_processing_agent import AudioProcessingAgent
from src.agents.transcription_agent import TranscriptionAgent
from src.core.framework.message import (
    Message, MessageType, MessagePriority, create_audio_input_message
)


@pytest.fixture
async def benchmark_agents():
    """Create a set of agents for benchmarking."""
    # Create agents
    simple_agent = SimpleAgent(name="BenchmarkSimpleAgent")
    audio_agent = AudioProcessingAgent(name="BenchmarkAudioAgent")
    transcription_agent = TranscriptionAgent(
        name="BenchmarkTranscriptionAgent",
        config={
            'whisper': {
                'model_size': 'tiny',  # Use smallest model for benchmarks
                'device': 'cpu'
            }
        }
    )
    
    # Start all agents
    await simple_agent.start()
    await audio_agent.start()
    await transcription_agent.start()
    
    # Return agents for benchmark use
    yield {
        "simple_agent": simple_agent,
        "audio_agent": audio_agent,
        "transcription_agent": transcription_agent
    }
    
    # Cleanup
    await simple_agent.stop()
    await audio_agent.stop()
    await transcription_agent.stop()


@pytest.mark.benchmark
@pytest.mark.slow
def test_message_serialization_performance(benchmark):
    """Benchmark message serialization performance."""
    def create_and_serialize_message():
        # Create a message with realistic payload
        message = Message(
            message_type=MessageType.AUDIO_PROCESSED,
            source_agent_id="benchmark_source",
            target_agent_id="benchmark_target",
            priority=MessagePriority.NORMAL,
            payload={
                "segment_id": "benchmark_segment",
                "sample_rate": 16000,
                "channels": 1,
                "start_time": 0.0,
                "end_time": 1.0,
                "duration": 1.0,
                "text": "This is a benchmark message with some text content to simulate a realistic payload",
                "confidence": 0.95,
                "metadata": {
                    "field1": "value1",
                    "field2": 123,
                    "field3": [1, 2, 3, 4, 5],
                    "field4": {
                        "nested": "value",
                        "another": 456
                    }
                }
            }
        )
        
        # Serialize to JSON
        json_data = message.to_json()
        
        # Deserialize back
        restored_message = Message.from_json(json_data)
        
        # Verify round-trip integrity
        assert restored_message.message_type == message.message_type
        assert restored_message.payload["segment_id"] == "benchmark_segment"
    
    # Run the benchmark
    benchmark(create_and_serialize_message)


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.asyncio
async def test_agent_messaging_throughput(benchmark_agents):
    """Benchmark message processing throughput between agents."""
    simple_agent = benchmark_agents["simple_agent"]
    
    # Number of messages to benchmark
    message_count = 100
    processed_messages = 0
    start_time = None
    
    # Set up a collector to track when all messages are processed
    collector = []
    
    # Custom handler to collect results
    async def collect_response(message):
        nonlocal processed_messages
        processed_messages += 1
        collector.append(message)
        return []
    
    # Register the collector
    simple_agent.register_message_handler(
        MessageType.STATUS_RESPONSE, 
        collect_response
    )
    
    # Function to measure
    async def message_throughput_test():
        nonlocal start_time
        
        # Reset counters
        processed_messages = 0
        collector.clear()
        
        # Start timing
        start_time = time.time()
        
        # Send multiple messages
        for i in range(message_count):
            message = Message(
                message_type=MessageType.STATUS_REQUEST,
                source_agent_id="benchmark",
                target_agent_id=simple_agent.agent_id,
                priority=MessagePriority.NORMAL
            )
            await simple_agent.enqueue_message(message)
        
        # Wait for all messages to be processed
        while processed_messages < message_count:
            await asyncio.sleep(0.01)
        
        # Calculate throughput
        elapsed = time.time() - start_time
        throughput = message_count / elapsed
        
        # Return metrics for reporting
        return {
            "messages_per_second": throughput,
            "total_time": elapsed,
            "message_count": message_count
        }
    
    # Run the benchmark and get results
    result = await message_throughput_test()
    print(f"\nMessage Throughput: {result['messages_per_second']:.2f} msgs/sec")
    print(f"Total Time: {result['total_time']:.4f} seconds")
    print(f"Messages Processed: {result['message_count']}")
    
    # Verify all messages were processed
    assert processed_messages == message_count


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.asyncio
async def test_audio_processing_performance(benchmark_agents):
    """Benchmark audio processing performance."""
    audio_agent = benchmark_agents["audio_agent"]
    
    # Create test audio data (1 second of silence at 16kHz)
    dummy_audio = b'\0' * 32000
    
    # Function to measure
    async def process_audio():
        # Create message
        message = create_audio_input_message(
            source_id="benchmark",
            target_id=audio_agent.agent_id,
            audio_data=dummy_audio,
            sample_rate=16000
        )
        
        # Measure processing time
        start_time = time.time()
        result = await audio_agent.process_message(message)
        elapsed = time.time() - start_time
        
        # Verify result
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].message_type == MessageType.AUDIO_PROCESSED
        
        return {
            "processing_time": elapsed,
            "real_time_factor": elapsed / 1.0  # 1 second of audio
        }
    
    # Run multiple iterations to get an average
    iterations = 5
    total_time = 0
    total_rtf = 0
    
    for i in range(iterations):
        result = await process_audio()
        total_time += result["processing_time"]
        total_rtf += result["real_time_factor"]
    
    avg_time = total_time / iterations
    avg_rtf = total_rtf / iterations
    
    print(f"\nAudio Processing Performance:")
    print(f"Average Processing Time: {avg_time:.4f} seconds")
    print(f"Average Real-Time Factor: {avg_rtf:.4f}x")
    
    # Assert reasonable performance (adjust thresholds as needed)
    assert avg_rtf < 1.0, "Audio processing should be faster than real-time"


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.asyncio
async def test_transcription_performance(benchmark_agents):
    """Benchmark transcription performance."""
    transcription_agent = benchmark_agents["transcription_agent"]
    
    # Make sure model is loaded
    await transcription_agent._load_model()
    
    # Create a mock audio processed message
    def create_test_message():
        return Message(
            message_type=MessageType.AUDIO_PROCESSED,
            source_agent_id="benchmark",
            target_agent_id=transcription_agent.agent_id,
            priority=MessagePriority.NORMAL,
            payload={
                "segment_id": f"benchmark_segment_{int(time.time())}",
                "sample_rate": 16000,
                "channels": 1,
                "start_time": 0.0,
                "end_time": 1.0,
                "duration": 1.0,
                "audio_data": "[PROCESSED_AUDIO_DATA]"
            }
        )
    
    # Function to measure
    async def process_transcription():
        message = create_test_message()
        
        # Measure processing time
        start_time = time.time()
        result = await transcription_agent.process_message(message)
        elapsed = time.time() - start_time
        
        # Verify result
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].message_type == MessageType.TRANSCRIPTION_RESULT
        
        # Get the RTF from the payload
        rtf = result[0].payload.get("real_time_factor", 0)
        
        return {
            "processing_time": elapsed,
            "real_time_factor": rtf
        }
    
    # Run multiple iterations to get an average
    iterations = 3
    total_time = 0
    total_rtf = 0
    
    for i in range(iterations):
        result = await process_transcription()
        total_time += result["processing_time"]
        total_rtf += result["real_time_factor"]
    
    avg_time = total_time / iterations
    avg_rtf = total_rtf / iterations
    
    print(f"\nTranscription Performance:")
    print(f"Average Processing Time: {avg_time:.4f} seconds")
    print(f"Average Real-Time Factor: {avg_rtf:.4f}x")
    
    # Assert reasonable performance (adjust thresholds based on hardware)
    assert avg_time < 5.0, "Transcription should complete in reasonable time" 