"""
Pytest Configuration for AMPTALK Tests

This module contains shared fixtures and configuration for the AMPTALK test suite.
It sets up the async test environment and provides common utilities for testing
the multi-agent system.

Author: AMPTALK Team
Date: 2024
"""

import os
import pytest
import asyncio
from typing import AsyncGenerator, Generator

from src.core.framework.agent import Agent
from src.core.framework.message import Message, MessageType, MessagePriority
from src.core.framework.orchestrator import Orchestrator


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def orchestrator() -> AsyncGenerator[Orchestrator, None]:
    """Create a test orchestrator instance."""
    orchestrator = Orchestrator(name="Test Orchestrator")
    yield orchestrator
    await orchestrator.stop()


@pytest.fixture
def test_config() -> dict:
    """Provide test configuration settings."""
    return {
        "audio_agent": {
            "audio": {
                "sample_rate": 16000,
                "chunk_duration_ms": 100,
                "vad_threshold": 0.3
            }
        },
        "transcription_agent": {
            "whisper": {
                "model_size": "tiny",  # Use smallest model for tests
                "device": "cpu",
                "language": "en"
            }
        }
    }


@pytest.fixture
def mock_audio_data() -> bytes:
    """Generate mock audio data for testing."""
    # Create 1 second of silence at 16kHz
    return b'\0' * 32000


class MockAgent(Agent):
    """A simple mock agent for testing."""
    
    def __init__(self, agent_id: str = None, name: str = "MockAgent"):
        super().__init__(agent_id, name)
        self.received_messages = []
        self.should_fail = False
    
    async def process_message(self, message: Message) -> list[Message]:
        """Process a message and optionally return responses."""
        self.received_messages.append(message)
        
        if self.should_fail:
            raise Exception("Mock processing error")
        
        # Echo the message back as a response
        response = Message(
            message_type=MessageType.STATUS_RESPONSE,
            source_agent_id=self.agent_id,
            target_agent_id=message.source_agent_id,
            priority=message.priority,
            payload={"echo": message.payload}
        )
        return [response]


@pytest.fixture
async def mock_agent() -> AsyncGenerator[MockAgent, None]:
    """Create a mock agent for testing."""
    agent = MockAgent()
    await agent.start()
    yield agent
    await agent.stop()


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers",
        "e2e: marks tests as end-to-end tests"
    ) 