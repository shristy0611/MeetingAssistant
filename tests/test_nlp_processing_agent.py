"""Tests for the NLP Processing Agent."""

import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
import torch
import numpy as np

from src.core.message import Message
from src.agents.nlp_processing_agent import NLPProcessingAgent, NLPProcessingConfig

@pytest.fixture
def config():
    """Create a test configuration."""
    return NLPProcessingConfig(
        entity_model_name="test-entity-model",
        topic_model_name="test-topic-model",
        intent_model_name="test-intent-model",
        language_model_name="test-language-model",
        device="cpu"  # Use CPU for testing
    )

@pytest.fixture
def model_manager():
    """Create a mock model manager."""
    return MagicMock()

@pytest.fixture
def agent(config, model_manager):
    """Create an NLP Processing Agent for testing."""
    agent = NLPProcessingAgent(
        agent_id="test-nlp-agent",
        config=config,
        model_manager=model_manager
    )
    return agent

@pytest.mark.asyncio
async def test_initialization(agent, config, model_manager):
    """Test agent initialization."""
    assert agent.agent_id == "test-nlp-agent"
    assert agent.config == config
    assert agent.model_manager == model_manager
    assert agent._entity_model is None
    assert agent._topic_model is None
    assert agent._intent_model is None
    assert agent._language_model is None
    assert isinstance(agent._context_history, dict)

@pytest.mark.asyncio
async def test_handle_process_text_empty_text(agent):
    """Test handling process_text with empty text."""
    message = Message(
        sender="test-sender",
        receiver=agent.agent_id,
        type="process_text",
        data={}
    )
    
    response = await agent.handle_process_text(message)
    
    assert response.type == "error"
    assert "No text provided" in response.data["error"]

@pytest.mark.asyncio
async def test_update_context(agent):
    """Test context updating functionality."""
    conversation_id = "test-conversation"
    texts = ["Hello", "How are you?", "I'm fine", "What about you?"]
    
    for text in texts:
        agent._update_context(conversation_id, text)
        
    context = agent._get_context(conversation_id)
    
    assert len(context) == 4
    assert context == texts
    
    # Test context window limiting
    agent.config.context_window_size = 2
    agent._update_context(conversation_id, "Another message")
    
    context = agent._get_context(conversation_id)
    assert len(context) == 2
    assert context == ["What about you?", "Another message"]

@pytest.mark.asyncio
async def test_detect_entities(agent):
    """Test entity detection with mocked model."""
    # Mock the entity model loading
    with patch.object(agent, '_load_entity_model', new_callable=AsyncMock) as mock_load:
        # Mock tokenizer and model
        agent._entity_tokenizer = MagicMock()
        agent._entity_tokenizer.convert_ids_to_tokens.return_value = ["[CLS]", "John", "works", "at", "Google", "in", "New", "York", "[SEP]"]
        agent._entity_tokenizer.all_special_tokens = ["[CLS]", "[SEP]"]
        
        agent._entity_model = MagicMock()
        agent._entity_model.config.id2label = {
            0: "O",
            1: "B-PER",
            2: "I-PER",
            3: "B-ORG",
            4: "I-ORG",
            5: "B-LOC",
            6: "I-LOC"
        }
        
        # Mock tokenizer output
        mock_inputs = {
            "input_ids": torch.tensor([[101, 2, 3, 4, 5, 6, 7, 8, 102]])
        }
        agent._entity_tokenizer.return_value = mock_inputs
        
        # Mock model output
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.zeros((1, 9, 7))
        # Set high values for specific entity labels
        # John: B-PER
        mock_outputs.logits[0, 1, 1] = 10.0
        # Google: B-ORG
        mock_outputs.logits[0, 4, 3] = 10.0
        # New: B-LOC
        mock_outputs.logits[0, 6, 5] = 10.0
        # York: I-LOC
        mock_outputs.logits[0, 7, 6] = 10.0
        
        agent._entity_model.return_value = mock_outputs
        
        # Test entity detection
        entities = await agent.detect_entities("John works at Google in New York")
        
        # Verify entity model was loaded
        mock_load.assert_called_once()
        
        # Check entities were correctly extracted
        assert len(entities) == 3
        assert entities[0]["type"] == "PER"
        assert entities[0]["text"] == "John"
        assert entities[1]["type"] == "ORG"
        assert entities[1]["text"] == "Google"
        assert entities[2]["type"] == "LOC"
        assert entities[2]["text"] == "New York"

@pytest.mark.asyncio
async def test_model_topics(agent):
    """Test topic modeling with mocked model."""
    # Mock the topic model loading
    with patch.object(agent, '_load_topic_model', new_callable=AsyncMock) as mock_load:
        # Mock topic model
        agent._topic_model = MagicMock()
        agent._topic_model.fit_transform.return_value = ([0, 1, 0], [0.8, 0.9, 0.7])
        
        # Mock topic info
        mock_topic_info = MagicMock()
        mock_topic_info.to_dict.return_value = [
            {"Topic": 0, "Count": 2, "Name": "Topic 0"},
            {"Topic": 1, "Count": 1, "Name": "Topic 1"}
        ]
        agent._topic_model.get_topic_info.return_value = mock_topic_info
        
        # Mock topic keywords
        agent._topic_model.get_topic.side_effect = lambda topic_id: [
            ("technology", 0.9), ("computer", 0.8), ("software", 0.7)
        ] if topic_id == 0 else [
            ("health", 0.9), ("medicine", 0.8), ("doctor", 0.7)
        ]
        
        # Test topic modeling
        texts = [
            "I need to upgrade my computer software",
            "The doctor prescribed new medicine for my health",
            "Technology is advancing rapidly in the software industry"
        ]
        
        topics = await agent.model_topics(texts)
        
        # Verify topic model was loaded
        mock_load.assert_called_once()
        
        # Check topics were correctly extracted
        assert "topic_representations" in topics
        assert "document_topics" in topics
        assert "topic_info" in topics
        
        # Check topic representations
        assert "0" in topics["topic_representations"]
        assert "1" in topics["topic_representations"]
        assert topics["topic_representations"]["0"]["keywords"] == ["technology", "computer", "software"]
        assert topics["topic_representations"]["1"]["keywords"] == ["health", "medicine", "doctor"]
        
        # Check document topics
        assert len(topics["document_topics"]) == 3
        assert topics["document_topics"][0]["topic_id"] == 0
        assert topics["document_topics"][1]["topic_id"] == 1
        assert topics["document_topics"][2]["topic_id"] == 0

@pytest.mark.asyncio
async def test_recognize_intent(agent):
    """Test intent recognition with mocked model."""
    # Mock the intent model loading
    with patch.object(agent, '_load_intent_model', new_callable=AsyncMock) as mock_load:
        # Mock tokenizer and model
        agent._intent_tokenizer = MagicMock()
        agent._intent_model = MagicMock()
        
        # Mock model output for different intents
        mock_outputs = MagicMock()
        
        # For question intent
        mock_question_output = MagicMock()
        mock_question_output.logits = torch.tensor([[0.1, 0.2, 0.9]])  # High score for entailment (index 2)
        
        # For greeting intent
        mock_greeting_output = MagicMock()
        mock_greeting_output.logits = torch.tensor([[0.1, 0.2, 0.7]])  # Medium score for entailment
        
        # For other intents
        mock_other_output = MagicMock()
        mock_other_output.logits = torch.tensor([[0.1, 0.2, 0.3]])  # Low score for entailment
        
        # Set up side effect to return different outputs for different inputs
        def side_effect(text, hyp, **kwargs):
            if "question" in hyp:
                agent._intent_model.return_value = mock_question_output
            elif "greeting" in hyp:
                agent._intent_model.return_value = mock_greeting_output
            else:
                agent._intent_model.return_value = mock_other_output
            return MagicMock()
            
        agent._intent_tokenizer.side_effect = side_effect
        
        # Test intent recognition for a question
        intent = await agent.recognize_intent("What time is it?")
        
        # Verify intent model was loaded
        mock_load.assert_called_once()
        
        # Check intent was correctly recognized
        assert intent["intent"] == "question"
        assert intent["confidence"] > 0.7
        
        # Test with custom intents
        custom_intents = ["ask_time", "ask_weather", "ask_name"]
        intent = await agent.recognize_intent("What time is it?", custom_intents)
        
        # Check custom intents were used
        assert "all_intents" in intent
        assert len(intent["all_intents"]) == 3
        assert intent["all_intents"][0]["intent"] == "ask_time"

@pytest.mark.asyncio
async def test_detect_language(agent):
    """Test language detection with mocked model."""
    # Mock the language model loading
    with patch.object(agent, '_load_language_model', new_callable=AsyncMock) as mock_load:
        # Mock tokenizer and model
        agent._language_tokenizer = MagicMock()
        agent._language_model = MagicMock()
        
        # Mock model output
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.zeros((1, 3))
        mock_outputs.logits[0, 1] = 10.0  # High score for English
        agent._language_model.return_value = mock_outputs
        
        # Mock language labels
        agent._language_model.config.id2label = {
            0: "fr_French",
            1: "en_English",
            2: "es_Spanish"
        }
        
        # Test language detection
        language = await agent.detect_language("Hello, how are you?")
        
        # Verify language model was loaded
        mock_load.assert_called_once()
        
        # Check language was correctly detected
        assert language["language"] == "en_English"
        assert language["language_code"] == "en"
        assert language["confidence"] > 0.9

@pytest.mark.asyncio
async def test_analyze_with_context(agent):
    """Test contextual analysis."""
    # Set up context
    conversation_id = "test-conversation"
    context = [
        "My name is John Smith.",
        "I work at Google in New York.",
        "What do you think about the weather?"
    ]
    
    for text in context:
        agent._update_context(conversation_id, text)
    
    # Mock entity detection
    with patch.object(agent, 'detect_entities', new_callable=AsyncMock) as mock_detect:
        # Mock entity detection for context
        mock_detect.side_effect = [
            # Context entities
            [
                {"type": "PER", "text": "John Smith", "confidence": 0.9},
                {"type": "ORG", "text": "Google", "confidence": 0.9},
                {"type": "LOC", "text": "New York", "confidence": 0.9}
            ],
            # Current message entities
            [
                {"type": "PER", "text": "John", "confidence": 0.9}
            ]
        ]
        
        # Test contextual analysis
        analysis = await agent.analyze_with_context("John, do you like it there?", conversation_id)
        
        # Check analysis results
        assert analysis["has_context"] is True
        assert analysis["context_size"] == 3
        assert analysis["is_followup"] is True
        assert len(analysis["referenced_entities"]) == 1
        assert analysis["referenced_entities"][0]["current"]["text"] == "John"
        assert analysis["referenced_entities"][0]["referenced"]["text"] == "John Smith"

@pytest.mark.asyncio
async def test_shutdown(agent):
    """Test agent shutdown."""
    # Set up models and context
    agent._entity_model = MagicMock()
    agent._topic_model = MagicMock()
    agent._intent_model = MagicMock()
    agent._language_model = MagicMock()
    agent._context_history = {"test": ["message1", "message2"]}
    
    # Mock super().shutdown
    with patch('src.core.agent.Agent.shutdown', new_callable=AsyncMock) as mock_super:
        await agent.shutdown()
        
        # Verify models were cleared
        assert agent._entity_model is None
        assert agent._topic_model is None
        assert agent._intent_model is None
        assert agent._language_model is None
        
        # Verify context was cleared
        assert agent._context_history == {}
        
        # Verify super().shutdown was called
        mock_super.assert_called_once() 