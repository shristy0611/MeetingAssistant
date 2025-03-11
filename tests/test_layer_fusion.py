"""Tests for the LayerFusion module."""

import pytest
import torch
import torch.nn as nn
import tempfile
import os
from pathlib import Path
import json
from unittest.mock import patch, MagicMock

from src.models.layer_fusion import (
    LayerFusion,
    FusionPattern,
    FusionBackend,
    fuse_onnx_model
)

class SimpleTransformerModel(nn.Module):
    """Simple transformer model for testing fusion."""
    def __init__(self, hidden_size=64, num_heads=4, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.embedding = nn.Embedding(1000, hidden_size)
        
        # Multi-head attention
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        
        # FFN
        self.ffn1 = nn.Linear(hidden_size, hidden_size * 4)
        self.ffn2 = nn.Linear(hidden_size * 4, hidden_size)
        self.gelu = nn.GELU()
        
        # Layer norm
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        
        # Self-attention
        residual = x
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        batch_size, seq_len, _ = q.size()
        q = q.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        
        # Simplified attention (for testing only)
        scores = torch.matmul(q, k.transpose(-1, -2))
        scores = scores / (q.size(-1) ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # Output projection
        context = self.o_proj(context)
        x = self.ln1(residual + context)
        
        # FFN
        residual = x
        x = self.ffn1(x)
        x = self.gelu(x)
        x = self.ffn2(x)
        x = self.ln2(residual + x)
        
        return x

@pytest.fixture
def model():
    """Create a simple model for testing."""
    return SimpleTransformerModel()

@pytest.fixture
def sample_input():
    """Create sample input for testing."""
    return torch.randint(0, 1000, (2, 10))  # batch_size=2, seq_len=10

@pytest.mark.parametrize("backend", [FusionBackend.TORCH_JIT])
def test_initialize_layer_fusion(model, backend):
    """Test initialization of LayerFusion."""
    fusion = LayerFusion(model, backend=backend)
    
    assert fusion.model is model
    assert fusion.backend == backend
    assert fusion.model_type == "pytorch"
    assert len(fusion.enabled_patterns) > 0

def test_analyze_pytorch_fusion_opportunities(model):
    """Test analysis of fusion opportunities in PyTorch model."""
    fusion = LayerFusion(model, backend=FusionBackend.TORCH_JIT)
    opportunities = fusion.analyze_fusion_opportunities()
    
    assert isinstance(opportunities, dict)
    assert len(opportunities) > 0

@pytest.mark.skip(reason="Requires ONNX model")
def test_onnx_fusion():
    """Test fusion with ONNX backend."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_path = os.path.join(tmp_dir, "model.onnx")
        output_path = os.path.join(tmp_dir, "model_fused.onnx")
        
        # Mock ONNX model
        with patch("onnx.load") as mock_load:
            mock_model = MagicMock()
            mock_model.graph = MagicMock()
            mock_load.return_value = mock_model
            
            # Mock ONNX Runtime session
            with patch("onnxruntime.InferenceSession") as mock_session:
                mock_session.return_value = MagicMock()
                
                # Write dummy ONNX file
                with open(input_path, "wb") as f:
                    f.write(b"dummy")
                
                # Test fusion
                fusion = LayerFusion(input_path, backend=FusionBackend.ONNX)
                result = fusion.apply_fusion(output_path)
                
                assert result == output_path
                mock_session.assert_called_once()

@patch("torch.jit.script")
@patch("torch.jit.freeze")
def test_torch_jit_fusion(mock_freeze, mock_script, model, sample_input):
    """Test fusion with PyTorch JIT backend."""
    # Mock scripted model
    scripted_model = MagicMock()
    mock_script.return_value = scripted_model
    
    # Mock frozen model
    frozen_model = MagicMock()
    mock_freeze.return_value = frozen_model
    
    # Test fusion
    fusion = LayerFusion(model, backend=FusionBackend.TORCH_JIT)
    result = fusion.apply_fusion()
    
    assert result is frozen_model
    mock_script.assert_called_once_with(model)
    mock_freeze.assert_called_once_with(scripted_model)

def test_save_fusion_metadata(model):
    """Test saving fusion metadata."""
    fusion = LayerFusion(model, backend=FusionBackend.TORCH_JIT)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        metadata_path = os.path.join(tmp_dir, "metadata.json")
        fusion.save_fusion_metadata(metadata_path)
        
        assert os.path.exists(metadata_path)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        assert metadata["backend"] == FusionBackend.TORCH_JIT.value
        assert "enabled_patterns" in metadata
        assert isinstance(metadata["enabled_patterns"], list)

@patch("src.models.layer_fusion.LayerFusion")
def test_fuse_onnx_model_helper(mock_layer_fusion):
    """Test the fuse_onnx_model helper function."""
    # Mock LayerFusion instance
    mock_instance = MagicMock()
    mock_layer_fusion.return_value = mock_instance
    mock_instance.apply_fusion.return_value = "output_path"
    
    # Test helper function
    result = fuse_onnx_model(
        model_path="model.onnx",
        output_path="model_fused.onnx",
        patterns=[FusionPattern.ATTENTION_QKV.value]
    )
    
    assert result == "output_path"
    mock_layer_fusion.assert_called_once()
    mock_instance.apply_fusion.assert_called_once_with("model_fused.onnx") 