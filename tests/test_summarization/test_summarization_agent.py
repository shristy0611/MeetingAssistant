"""Tests for the Summarization Agent."""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from typing import List, Dict, Any

from src.agents.summarization.summarization_agent import SummarizationAgent, SummarizationConfig
from src.core.message import Message


@pytest.fixture
def mock_pipeline():
    """Mock the transformers pipeline."""
    def mock_pipeline_factory(*args, **kwargs):
        mock = Mock()
        if args[0] == "summarization":
            mock.return_value = [{"summary_text": "Mocked summary"}]
        elif args[0] == "zero-shot-classification":
            mock.return_value = {
                "labels": ["action item"],
                "scores": [0.8]
            }
        elif args[0] == "feature-extraction":
            mock.return_value = [[[0.1, 0.2, 0.3]]]  # Mock embeddings
        return mock
    return mock_pipeline_factory


@pytest.fixture
def mock_tokenizer():
    """Mock the AutoTokenizer."""
    mock = Mock()
    mock.from_pretrained.return_value = Mock()
    return mock


@pytest.fixture
def summarization_agent(mock_pipeline, mock_tokenizer):
    """Create a SummarizationAgent instance with mocked dependencies."""
    with patch("src.agents.summarization.summarization_agent.pipeline", mock_pipeline), \
         patch("src.agents.summarization.summarization_agent.AutoTokenizer", mock_tokenizer):
        agent = SummarizationAgent("test_agent")
        return agent


def test_initialization(summarization_agent):
    """Test agent initialization."""
    assert summarization_agent.agent_id == "test_agent"
    assert isinstance(summarization_agent.config, SummarizationConfig)


def test_summarize_meeting_empty_text(summarization_agent):
    """Test summarization with empty text."""
    result = summarization_agent.summarize_meeting("")
    assert result["summary"] == ""
    assert result["segment_summaries"] == []
    assert result["action_items"] == []
    assert result["key_points"] == []
    assert result["timeline"] == []
    assert result["priorities"] == []


def test_summarize_meeting(summarization_agent):
    """Test meeting summarization with sample text."""
    text = """
    Team meeting - Project X
    Alice will implement the new feature by next week.
    Bob is responsible for testing.
    Key points discussed:
    1. Timeline needs to be adjusted
    2. Budget is on track
    Meeting started at 2pm and ended at 3pm.
    """
    
    result = summarization_agent.summarize_meeting(text)
    
    assert isinstance(result, dict)
    assert "summary" in result
    assert "segment_summaries" in result
    assert "action_items" in result
    assert "key_points" in result
    assert "timeline" in result
    assert "priorities" in result


def test_extract_action_items(summarization_agent):
    """Test action item extraction."""
    text = "Alice will implement the feature by next week. Bob needs to review the code."
    
    action_items = summarization_agent.extract_action_items(text)
    
    assert isinstance(action_items, list)
    assert len(action_items) > 0
    assert all(isinstance(item, dict) for item in action_items)
    assert all("text" in item for item in action_items)
    assert all("confidence" in item for item in action_items)


def test_identify_key_points(summarization_agent):
    """Test key point identification."""
    text = """
    Important point: The project timeline needs adjustment.
    Another key point: The budget is on track.
    Random note: Coffee machine needs cleaning.
    """
    
    key_points = summarization_agent.identify_key_points(text)
    
    assert isinstance(key_points, list)
    assert len(key_points) > 0
    assert all(isinstance(point, dict) for point in key_points)
    assert all("text" in point for point in key_points)
    assert all("confidence" in point for point in key_points)


def test_generate_timeline(summarization_agent):
    """Test timeline generation."""
    text = """
    Meeting started at 2pm.
    First, we discussed the project status.
    At 2:30pm, we moved on to technical challenges.
    The meeting concluded at 3pm with action items.
    """
    
    timeline = summarization_agent.generate_timeline(text)
    
    assert isinstance(timeline, list)
    assert len(timeline) > 0
    assert all(isinstance(entry, dict) for entry in timeline)
    assert all("position" in entry for entry in timeline)
    assert all("summary" in entry for entry in timeline)
    assert all("time_references" in entry for entry in timeline)


def test_tag_priorities(summarization_agent):
    """Test priority tagging for action items."""
    action_items = [
        {"text": "Critical: Fix security vulnerability immediately"},
        {"text": "Update documentation when possible"}
    ]
    
    prioritized_items = summarization_agent.tag_priorities(action_items)
    
    assert isinstance(prioritized_items, list)
    assert len(prioritized_items) == len(action_items)
    assert all(isinstance(item, dict) for item in prioritized_items)
    assert all("priority" in item for item in prioritized_items)
    assert all("priority_confidence" in item for item in prioritized_items)


def test_segment_text_by_topic(summarization_agent):
    """Test text segmentation by topic."""
    text = """
    Topic 1: Project Status
    Everything is on track.
    
    Topic 2: Technical Issues
    Several bugs were identified.
    
    Topic 3: Next Steps
    We need to plan the next sprint.
    """
    
    segments = summarization_agent._segment_text_by_topic(text)
    
    assert isinstance(segments, list)
    assert len(segments) > 0
    assert all(isinstance(segment, str) for segment in segments)


@pytest.mark.asyncio
async def test_handle_summarize(summarization_agent):
    """Test summarize message handler."""
    message = Message(
        sender="test_sender",
        receiver="test_agent",
        type="summarize",
        data={"text": "Test meeting text"}
    )
    
    response = await summarization_agent.handle_summarize(message)
    
    assert isinstance(response, Message)
    assert response.type == "summarization_results"
    assert "result" in response.data
    assert "original_text" in response.data


@pytest.mark.asyncio
async def test_handle_extract_action_items(summarization_agent):
    """Test action item extraction message handler."""
    message = Message(
        sender="test_sender",
        receiver="test_agent",
        type="extract_action_items",
        data={"text": "Alice will implement the feature"}
    )
    
    response = await summarization_agent.handle_extract_action_items(message)
    
    assert isinstance(response, Message)
    assert response.type == "action_item_results"
    assert "result" in response.data
    assert "original_text" in response.data


@pytest.mark.asyncio
async def test_handle_identify_key_points(summarization_agent):
    """Test key point identification message handler."""
    message = Message(
        sender="test_sender",
        receiver="test_agent",
        type="identify_key_points",
        data={"text": "Important point: Project is on track"}
    )
    
    response = await summarization_agent.handle_identify_key_points(message)
    
    assert isinstance(response, Message)
    assert response.type == "key_point_results"
    assert "result" in response.data
    assert "original_text" in response.data


@pytest.mark.asyncio
async def test_handle_generate_timeline(summarization_agent):
    """Test timeline generation message handler."""
    message = Message(
        sender="test_sender",
        receiver="test_agent",
        type="generate_timeline",
        data={"text": "Meeting started at 2pm"}
    )
    
    response = await summarization_agent.handle_generate_timeline(message)
    
    assert isinstance(response, Message)
    assert response.type == "timeline_results"
    assert "result" in response.data
    assert "original_text" in response.data


@pytest.mark.asyncio
async def test_handle_tag_priorities(summarization_agent):
    """Test priority tagging message handler."""
    message = Message(
        sender="test_sender",
        receiver="test_agent",
        type="tag_priorities",
        data={"action_items": [{"text": "Critical task"}]}
    )
    
    response = await summarization_agent.handle_tag_priorities(message)
    
    assert isinstance(response, Message)
    assert response.type == "priority_results"
    assert "result" in response.data


@pytest.mark.asyncio
async def test_shutdown(summarization_agent):
    """Test agent shutdown."""
    await summarization_agent.shutdown()
    # Add assertions for cleanup if needed


def test_cosine_similarity(summarization_agent):
    """Test cosine similarity calculation."""
    vec1 = [1, 0, 1]
    vec2 = [0, 1, 0]
    
    similarity = summarization_agent._cosine_similarity(vec1, vec2)
    
    assert isinstance(similarity, float)
    assert -1 <= similarity <= 1


def test_split_into_sentences(summarization_agent):
    """Test sentence splitting."""
    text = "First sentence. Second sentence! Third sentence?"
    
    sentences = summarization_agent._split_into_sentences(text)
    
    assert isinstance(sentences, list)
    assert len(sentences) == 3
    assert all(isinstance(s, str) for s in sentences)


def test_extract_assignee(summarization_agent):
    """Test assignee extraction from action items."""
    text = "John will implement the feature"
    
    assignee = summarization_agent._extract_assignee(text)
    
    assert assignee == "John"


def test_extract_deadline(summarization_agent):
    """Test deadline extraction from action items."""
    text = "Complete the task by next week"
    
    deadline = summarization_agent._extract_deadline(text)
    
    assert isinstance(deadline, str)
    assert "by next week" in deadline


def test_extract_time_references(summarization_agent):
    """Test time reference extraction."""
    text = "Meeting at 2pm, ended 3 hours later"
    
    time_refs = summarization_agent._extract_time_references(text)
    
    assert isinstance(time_refs, list)
    assert len(time_refs) > 0
    assert all(isinstance(ref, str) for ref in time_refs) 