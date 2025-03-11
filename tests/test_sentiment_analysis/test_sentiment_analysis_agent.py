"""Tests for the Sentiment Analysis Agent."""

import pytest
import torch
from unittest.mock import MagicMock, patch
import numpy as np

from src.agents.sentiment_analysis.sentiment_analysis_agent import SentimentAnalysisAgent, SentimentAnalysisConfig
from src.core.message import Message


@pytest.fixture
def config():
    """Create a test configuration."""
    return SentimentAnalysisConfig(
        sentiment_model_name="distilbert-base-uncased-finetuned-sst-2-english",
        emotion_model_name="j-hartmann/emotion-english-distilroberta-base",
        pain_point_model_name="distilbert-base-uncased",
        context_window_size=3
    )


@pytest.fixture
def mock_sentiment_model():
    """Create a mock sentiment model."""
    model = MagicMock()
    # Mock the forward pass to return logits for positive sentiment
    model.return_value = MagicMock(logits=torch.tensor([[0.1, 0.9]]))
    return model


@pytest.fixture
def mock_emotion_model():
    """Create a mock emotion model."""
    model = MagicMock()
    # Mock the forward pass to return logits for emotions (joy, sadness, anger, fear, surprise, disgust)
    model.return_value = MagicMock(logits=torch.tensor([[1.5, -0.5, -1.0, -2.0, 0.8, -1.5]]))
    model.config = MagicMock()
    model.config.id2label = {0: "joy", 1: "sadness", 2: "anger", 3: "fear", 4: "surprise", 5: "disgust"}
    return model


@pytest.fixture
def mock_pain_point_pipeline():
    """Create a mock pain point pipeline."""
    pipeline = MagicMock()
    pipeline.return_value = {
        "labels": ["product issue", "service complaint", "pricing concern"],
        "scores": [0.8, 0.3, 0.6]
    }
    return pipeline


@pytest.fixture
def agent(config, mock_sentiment_model, mock_emotion_model, mock_pain_point_pipeline):
    """Create a test agent with mocked models."""
    with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer, \
         patch("transformers.AutoModelForSequenceClassification.from_pretrained") as mock_model_fn, \
         patch("transformers.pipeline") as mock_pipeline:
        
        # Configure mocks
        mock_tokenizer.return_value = MagicMock()
        mock_model_fn.side_effect = [mock_sentiment_model, mock_emotion_model]
        mock_pipeline.return_value = mock_pain_point_pipeline
        
        # Create agent
        agent = SentimentAnalysisAgent("test_agent", config)
        
        # Replace the models with mocks for testing
        agent.sentiment_model = mock_sentiment_model
        agent.emotion_model = mock_emotion_model
        agent.pain_point_pipeline = mock_pain_point_pipeline
        
        return agent


@pytest.mark.asyncio
async def test_initialization(agent, config):
    """Test agent initialization."""
    assert agent.agent_id == "test_agent"
    assert agent.config.sentiment_model_name == config.sentiment_model_name
    assert agent.config.emotion_model_name == config.emotion_model_name
    assert agent.config.pain_point_model_name == config.pain_point_model_name
    assert agent.config.context_window_size == config.context_window_size


@pytest.mark.asyncio
async def test_analyze_sentiment(agent, mock_sentiment_model):
    """Test sentiment analysis."""
    result = agent.analyze_sentiment("This is a great product!")
    
    assert result["sentiment"] == "positive"
    assert result["confidence"] > 0.5
    assert isinstance(result["scores"], list)


@pytest.mark.asyncio
async def test_track_emotions(agent, mock_emotion_model):
    """Test emotion tracking."""
    result = agent.track_emotions("I am so happy with this product!")
    
    assert "emotions" in result
    assert "dominant_emotion" in result
    assert "all_scores" in result
    assert result["dominant_emotion"] == "joy"


@pytest.mark.asyncio
async def test_identify_pain_points(agent, mock_pain_point_pipeline):
    """Test pain point identification."""
    result = agent.identify_pain_points("The product is broken and doesn't work properly.")
    
    assert "pain_points" in result
    assert "keyword_matches" in result
    assert "has_pain_points" in result
    assert result["has_pain_points"] is True
    assert len(result["pain_points"]) > 0
    assert "product issue" in [p["category"] for p in result["pain_points"]]


@pytest.mark.asyncio
async def test_analyze_with_context(agent):
    """Test context-aware analysis."""
    # Add some context
    agent._update_context("test_convo", "I like this product.")
    agent._update_context("test_convo", "But it has some issues.")
    
    # Analyze with context
    result = agent.analyze_with_context("Now it's completely broken!", "test_convo")
    
    assert "sentiment" in result
    assert "emotions" in result
    assert "pain_points" in result
    assert "context_size" in result
    assert "sentiment_trend" in result
    assert result["context_size"] == 3


@pytest.mark.asyncio
async def test_context_window_size(agent):
    """Test that context window size is respected."""
    # Add more messages than the context window size
    for i in range(5):
        agent._update_context("test_convo", f"Message {i}")
    
    context = agent._get_context("test_convo")
    assert len(context) == agent.config.context_window_size
    assert context[0] == "Message 2"  # Should have the 3 most recent messages


@pytest.mark.asyncio
async def test_handle_sentiment(agent):
    """Test handling sentiment analysis message."""
    message = Message(
        sender="test_sender",
        receiver="test_agent",
        type="analyze_sentiment",
        data={"text": "This is a great product!"}
    )
    
    response = await agent.handle_sentiment(message)
    
    assert response.sender == "test_agent"
    assert response.receiver == "test_sender"
    assert response.type == "sentiment_results"
    assert "result" in response.data
    assert "original_text" in response.data
    assert response.data["result"]["sentiment"] == "positive"


@pytest.mark.asyncio
async def test_handle_emotion(agent):
    """Test handling emotion tracking message."""
    message = Message(
        sender="test_sender",
        receiver="test_agent",
        type="track_emotions",
        data={"text": "I am so happy with this product!"}
    )
    
    response = await agent.handle_emotion(message)
    
    assert response.sender == "test_agent"
    assert response.receiver == "test_sender"
    assert response.type == "emotion_results"
    assert "result" in response.data
    assert "original_text" in response.data
    assert response.data["result"]["dominant_emotion"] == "joy"


@pytest.mark.asyncio
async def test_handle_pain_points(agent):
    """Test handling pain point identification message."""
    message = Message(
        sender="test_sender",
        receiver="test_agent",
        type="identify_pain_points",
        data={"text": "The product is broken and doesn't work properly."}
    )
    
    response = await agent.handle_pain_points(message)
    
    assert response.sender == "test_agent"
    assert response.receiver == "test_sender"
    assert response.type == "pain_point_results"
    assert "result" in response.data
    assert "original_text" in response.data
    assert response.data["result"]["has_pain_points"] is True


@pytest.mark.asyncio
async def test_handle_context_analysis(agent):
    """Test handling context analysis message."""
    message = Message(
        sender="test_sender",
        receiver="test_agent",
        type="analyze_with_context",
        data={
            "text": "Now it's completely broken!",
            "conversation_id": "test_convo"
        }
    )
    
    response = await agent.handle_context_analysis(message)
    
    assert response.sender == "test_agent"
    assert response.receiver == "test_sender"
    assert response.type == "context_analysis_results"
    assert "result" in response.data
    assert "original_text" in response.data


@pytest.mark.asyncio
async def test_handle_empty_text(agent):
    """Test handling empty text."""
    message = Message(
        sender="test_sender",
        receiver="test_agent",
        type="analyze_sentiment",
        data={"text": ""}
    )
    
    response = await agent.handle_sentiment(message)
    
    assert response.sender == "test_agent"
    assert response.receiver == "test_sender"
    assert response.type == "error"
    assert "error" in response.data


@pytest.mark.asyncio
async def test_shutdown(agent):
    """Test agent shutdown."""
    # Add some context
    agent._update_context("test_convo", "Test message")
    
    # Shutdown
    await agent.shutdown()
    
    # Verify conversation history is cleared
    assert len(agent.conversation_history) == 0


@pytest.mark.asyncio
async def test_analyze_sentiment_trend_insufficient_data(agent):
    """Test sentiment trend analysis with insufficient data."""
    result = agent.analyze_sentiment_trend("test_convo")
    
    assert result["trend"] == "insufficient_data"
    assert len(result["change_points"]) == 0
    assert len(result["sentiment_scores"]) == 0
    assert len(result["smoothed_scores"]) == 0


@pytest.mark.asyncio
async def test_analyze_sentiment_trend(agent):
    """Test sentiment trend analysis."""
    # Add some sentiment history
    agent.sentiment_history["test_convo"] = [-0.8, -0.6, -0.3, 0.1, 0.4, 0.7]
    
    result = agent.analyze_sentiment_trend("test_convo")
    
    assert result["trend"] in ["improving", "deteriorating", "stable"]
    assert "change_points" in result
    assert "sentiment_scores" in result
    assert "smoothed_scores" in result
    assert "statistics" in result
    assert "mean" in result["statistics"]
    assert "median" in result["statistics"]
    assert "std_dev" in result["statistics"]


@pytest.mark.asyncio
async def test_analyze_emotion_trend_insufficient_data(agent):
    """Test emotion trend analysis with insufficient data."""
    result = agent.analyze_emotion_trend("test_convo")
    
    assert result["trend"] == "insufficient_data"
    assert len(result["emotions"]) == 0


@pytest.mark.asyncio
async def test_analyze_emotion_trend(agent):
    """Test emotion trend analysis."""
    # Add some emotion history
    agent.emotion_history["test_convo"] = [
        {"emotions": [{"emotion": "joy", "score": 0.8}, {"emotion": "sadness", "score": 0.1}]},
        {"emotions": [{"emotion": "joy", "score": 0.7}, {"emotion": "sadness", "score": 0.2}]},
        {"emotions": [{"emotion": "joy", "score": 0.5}, {"emotion": "sadness", "score": 0.4}]},
        {"emotions": [{"emotion": "joy", "score": 0.3}, {"emotion": "sadness", "score": 0.6}]}
    ]
    
    result = agent.analyze_emotion_trend("test_convo")
    
    assert result["trend"] == "analyzed"
    assert "emotions" in result
    assert "joy" in result["emotions"]
    assert "sadness" in result["emotions"]
    assert "trend" in result["emotions"]["joy"]
    assert "scores" in result["emotions"]["joy"]
    assert "smoothed_scores" in result["emotions"]["joy"]


@pytest.mark.asyncio
async def test_analyze_pain_point_trend_insufficient_data(agent):
    """Test pain point trend analysis with insufficient data."""
    result = agent.analyze_pain_point_trend("test_convo")
    
    assert result["trend"] == "insufficient_data"
    assert len(result["categories"]) == 0


@pytest.mark.asyncio
async def test_analyze_pain_point_trend(agent):
    """Test pain point trend analysis."""
    # Add some pain point history
    agent.pain_point_history["test_convo"] = [
        {
            "pain_points": [{"category": "product issue", "confidence": 0.8}],
            "keyword_matches": ["broken", "issue"]
        },
        {
            "pain_points": [{"category": "product issue", "confidence": 0.7}],
            "keyword_matches": ["issue"]
        },
        {
            "pain_points": [{"category": "service complaint", "confidence": 0.6}],
            "keyword_matches": ["problem"]
        }
    ]
    
    result = agent.analyze_pain_point_trend("test_convo")
    
    assert result["trend"] == "analyzed"
    assert "categories" in result
    assert "product issue" in result["categories"]
    assert "keywords" in result
    assert "issue" in result["keywords"]


@pytest.mark.asyncio
async def test_exponential_smoothing(agent):
    """Test exponential smoothing function."""
    scores = [0.1, 0.3, 0.2, 0.5, 0.4]
    smoothed = agent._apply_exponential_smoothing(scores)
    
    assert len(smoothed) == len(scores)
    assert smoothed[0] == scores[0]  # First value should be the same
    
    # Check that smoothing reduces variance
    assert np.std(smoothed) <= np.std(scores)


@pytest.mark.asyncio
async def test_detect_trend_direction(agent):
    """Test trend direction detection."""
    # Upward trend
    scores = [0.1, 0.2, 0.3, 0.4, 0.5]
    direction = agent._detect_trend_direction(scores)
    assert direction == "improving"
    
    # Downward trend
    scores = [0.5, 0.4, 0.3, 0.2, 0.1]
    direction = agent._detect_trend_direction(scores)
    assert direction == "deteriorating"
    
    # Stable trend
    scores = [0.3, 0.31, 0.29, 0.3, 0.31]
    direction = agent._detect_trend_direction(scores)
    assert direction == "stable"


@pytest.mark.asyncio
async def test_detect_change_points(agent):
    """Test change point detection."""
    # Create a series with a change point
    scores = [0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1]
    change_points = agent._detect_change_points(scores)
    
    assert len(change_points) > 0
    assert 3 in change_points  # Should detect change at index 3


@pytest.mark.asyncio
async def test_update_sentiment_history(agent):
    """Test updating sentiment history."""
    sentiment_result = {"sentiment": "positive", "confidence": 0.8}
    agent._update_sentiment_history("test_convo", sentiment_result)
    
    assert "test_convo" in agent.sentiment_history
    assert len(agent.sentiment_history["test_convo"]) == 1
    assert agent.sentiment_history["test_convo"][0] == 0.8
    
    # Test negative sentiment
    sentiment_result = {"sentiment": "negative", "confidence": 0.7}
    agent._update_sentiment_history("test_convo", sentiment_result)
    
    assert len(agent.sentiment_history["test_convo"]) == 2
    assert agent.sentiment_history["test_convo"][1] == -0.7


@pytest.mark.asyncio
async def test_handle_trend_analysis(agent):
    """Test handling trend analysis message."""
    # Add some sentiment history
    agent.sentiment_history["test_convo"] = [-0.8, -0.6, -0.3, 0.1, 0.4, 0.7]
    
    message = Message(
        sender="test_sender",
        receiver="test_agent",
        type="analyze_trend",
        data={
            "conversation_id": "test_convo",
            "trend_type": "sentiment"
        }
    )
    
    response = await agent.handle_trend_analysis(message)
    
    assert response.sender == "test_agent"
    assert response.receiver == "test_sender"
    assert response.type == "trend_analysis_results"
    assert "result" in response.data
    assert "trend_type" in response.data
    assert response.data["trend_type"] == "sentiment"
    assert "conversation_id" in response.data
    assert response.data["conversation_id"] == "test_convo"


@pytest.mark.asyncio
async def test_handle_trend_analysis_invalid_type(agent):
    """Test handling trend analysis message with invalid trend type."""
    message = Message(
        sender="test_sender",
        receiver="test_agent",
        type="analyze_trend",
        data={
            "conversation_id": "test_convo",
            "trend_type": "invalid_type"
        }
    )
    
    response = await agent.handle_trend_analysis(message)
    
    assert response.sender == "test_agent"
    assert response.receiver == "test_sender"
    assert response.type == "error"
    assert "error" in response.data 