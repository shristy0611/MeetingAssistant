"""Tests for the SpeculativeDecoder class."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from src.models.speculative_decoding import (
    SpeculativeDecoder,
    SpeculativeDecodingConfig,
    VerificationStrategy
)

class SimpleDummyModel(nn.Module):
    """A very simple dummy model for testing."""
    def __init__(self, vocab_size=100):
        super().__init__()
        self.vocab_size = vocab_size
        
    def forward(self, input_ids):
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        # Create mock logits that favor token IDs that are multiples of 5
        logits = torch.ones((batch_size, seq_len, self.vocab_size))
        logits[:, :, ::5] = 10.0  # Make multiples of 5 more likely
        
        # Create a dummy output object with a logits attribute
        output = MagicMock()
        output.logits = logits
        return output

@pytest.fixture
def target_model():
    """Create a dummy target model."""
    return SimpleDummyModel(vocab_size=100)

@pytest.fixture
def draft_model():
    """Create a dummy draft model."""
    return SimpleDummyModel(vocab_size=100)

@pytest.fixture
def tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.decode = lambda x: f"Token-{x.item()}"
    return tokenizer

@pytest.fixture
def decoder(target_model, draft_model, tokenizer):
    """Create a SpeculativeDecoder instance."""
    config = SpeculativeDecodingConfig(
        gamma=3,
        top_k=50,
        top_p=0.9,
        temperature=0.7,
        verification_strategy=VerificationStrategy.BASIC
    )
    return SpeculativeDecoder(target_model, draft_model, config, tokenizer)

def test_initialization(decoder, target_model, draft_model, tokenizer):
    """Test decoder initialization."""
    assert decoder.target_model == target_model
    assert decoder.draft_model == draft_model
    assert decoder.tokenizer == tokenizer
    assert decoder.config.gamma == 3
    assert decoder.config.top_k == 50
    assert decoder.config.top_p == 0.9
    assert decoder.config.temperature == 0.7
    assert decoder.config.verification_strategy == VerificationStrategy.BASIC

def test_get_logits(decoder, target_model):
    """Test getting logits from a model."""
    input_ids = torch.tensor([[1, 2, 3]])
    logits = decoder._get_logits(target_model, input_ids)
    assert logits.shape == (1, 100)  # (batch_size, vocab_size)

def test_sample_token(decoder):
    """Test token sampling."""
    logits = torch.ones((1, 100))
    logits[0, 5] = 100.0  # Make token 5 very likely
    
    token_id, probs = decoder._sample_token(
        logits,
        temperature=1.0,
        top_k=0,
        top_p=1.0
    )
    
    assert token_id.shape == (1, 1)
    assert probs.shape == (1, 100)
    assert torch.isclose(torch.sum(probs), torch.tensor(1.0))

def test_compute_acceptance_probability_basic(decoder):
    """Test basic acceptance probability computation."""
    target_probs = torch.zeros((1, 100))
    target_probs[0, 5] = 0.8
    
    draft_probs = torch.zeros((1, 100))
    draft_probs[0, 5] = 0.4
    
    token_id = torch.tensor([[5]])
    
    prob = decoder._compute_acceptance_probability(
        target_probs,
        draft_probs,
        token_id,
        VerificationStrategy.BASIC
    )
    
    # Should be min(1, 0.8/0.4) = 1.0
    assert prob == 1.0
    
    # Test with draft_prob > target_prob
    draft_probs[0, 5] = 0.9
    prob = decoder._compute_acceptance_probability(
        target_probs,
        draft_probs,
        token_id,
        VerificationStrategy.BASIC
    )
    
    # Should be min(1, 0.8/0.9) ≈ 0.889
    assert 0.88 <= prob <= 0.89

def test_compute_acceptance_probability_optimal_transport(decoder):
    """Test optimal transport acceptance probability computation."""
    target_probs = torch.zeros((1, 100))
    target_probs[0, 5] = 0.8
    target_probs[0, 10] = 0.2
    
    draft_probs = torch.zeros((1, 100))
    draft_probs[0, 5] = 0.7
    draft_probs[0, 10] = 0.3
    
    token_id = torch.tensor([[5]])
    
    prob = decoder._compute_acceptance_probability(
        target_probs,
        draft_probs,
        token_id,
        VerificationStrategy.OPTIMAL_TRANSPORT
    )
    
    # Should compute based on distribution similarity and direct ratio
    assert 0 <= prob <= 1

def test_compute_acceptance_probability_adaptive(decoder):
    """Test adaptive acceptance probability computation."""
    target_probs = torch.zeros((1, 100))
    target_probs[0, 5] = 0.8
    target_probs[0, 10] = 0.2
    
    draft_probs = torch.zeros((1, 100))
    draft_probs[0, 5] = 0.7
    draft_probs[0, 10] = 0.3
    
    token_id = torch.tensor([[5]])
    
    prob = decoder._compute_acceptance_probability(
        target_probs,
        draft_probs,
        token_id,
        VerificationStrategy.ADAPTIVE
    )
    
    # Should compute based on model confidence and direct ratio
    assert 0 <= prob <= 1

def test_adjust_gamma(decoder):
    """Test gamma adjustment."""
    # Test with dynamic_gamma=False
    decoder.config.dynamic_gamma = False
    new_gamma = decoder._adjust_gamma(0.9)
    assert new_gamma == decoder.config.gamma  # Should remain unchanged
    
    # Test with dynamic_gamma=True, high acceptance rate
    decoder.config.dynamic_gamma = True
    decoder.current_gamma = 3
    decoder.config.max_gamma = 5
    decoder.config.dynamic_threshold = 0.7
    
    new_gamma = decoder._adjust_gamma(0.8)  # Above threshold
    assert new_gamma == 4  # Should increase
    
    # Test with dynamic_gamma=True, low acceptance rate
    decoder.current_gamma = 3
    decoder.config.min_gamma = 1
    
    new_gamma = decoder._adjust_gamma(0.6)  # Below threshold
    assert new_gamma == 2  # Should decrease

@patch('torch.multinomial')
def test_generate(mock_multinomial, decoder):
    """Test text generation with speculative decoding."""
    # Mock multinomial to deterministically return token 5
    mock_multinomial.return_value = torch.tensor([[5]])
    
    # Initial input
    input_ids = torch.tensor([[1, 2, 3]])
    
    # Test with a small number of tokens for faster testing
    result = decoder.generate(input_ids, max_new_tokens=5)
    
    # Should generate 5 new tokens
    assert result.shape[1] == input_ids.shape[1] + 5
    assert decoder.stats["total_tokens"] == 5
    assert decoder.stats["speedup_factor"] > 0

def test_generate_with_callback(decoder):
    """Test generation with callback function."""
    callback_called = 0
    
    def callback(text):
        nonlocal callback_called
        callback_called += 1
        assert isinstance(text, str)
    
    # Initial input
    input_ids = torch.tensor([[1, 2, 3]])
    
    # Generate just a few tokens
    decoder.generate(input_ids, max_new_tokens=3, callback=callback)
    
    # Callback should be called for each token
    assert callback_called == 3

def test_get_stats(decoder):
    """Test getting decoder statistics."""
    # Initially, stats should be zeros
    stats = decoder.get_stats()
    assert stats["total_tokens"] == 0
    assert stats["accepted_drafts"] == 0
    
    # After generation, stats should be populated
    input_ids = torch.tensor([[1, 2, 3]])
    decoder.generate(input_ids, max_new_tokens=2)
    
    stats = decoder.get_stats()
    assert stats["total_tokens"] == 2
    assert "tokens_per_second" in stats
    assert "inference_time" in stats
    assert "speedup_factor" in stats 