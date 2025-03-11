"""Tests for the ModelCompressor class."""

import pytest
import torch
import torch.nn as nn
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.models.model_compression import (
    CompressionType,
    SharingStrategy,
    CompressionConfig,
    ModelCompressor,
    compress_model
)

class SimpleModel(nn.Module):
    """Simple model for testing compression."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

@pytest.fixture
def model():
    """Create a simple model for testing."""
    return SimpleModel()

@pytest.fixture
def config():
    """Create a compression configuration for testing."""
    return CompressionConfig(
        compression_types=[CompressionType.WEIGHT_SHARING],
        target_ratio=0.5,
        sharing_strategy=SharingStrategy.CLUSTER,
        num_clusters=16
    )

def test_compression_config():
    """Test compression configuration initialization."""
    config = CompressionConfig(
        compression_types=[CompressionType.WEIGHT_SHARING],
        target_ratio=0.5,
        sharing_strategy=SharingStrategy.CLUSTER,
        num_clusters=16,
        extra_param="test"
    )
    
    assert config.compression_types == [CompressionType.WEIGHT_SHARING]
    assert config.target_ratio == 0.5
    assert config.sharing_strategy == SharingStrategy.CLUSTER
    assert config.num_clusters == 16
    assert config.extra_config["extra_param"] == "test"

def test_model_compressor_initialization(model, config):
    """Test ModelCompressor initialization."""
    compressor = ModelCompressor(model, config)
    
    assert compressor.model == model
    assert compressor.config == config
    assert compressor.original_size > 0
    assert compressor.compressed_size is None

def test_cluster_sharing(model, config):
    """Test cluster-based weight sharing."""
    compressor = ModelCompressor(model, config)
    compressed_model = compressor._apply_cluster_sharing(model)
    
    # Check that weights were modified
    assert torch.equal(compressed_model.fc1.weight, model.fc1.weight) == False
    assert compressed_model.fc1.weight.shape == model.fc1.weight.shape

def test_low_rank_compression(model):
    """Test low-rank factorization compression."""
    config = CompressionConfig(
        compression_types=[CompressionType.LOW_RANK],
        rank_ratio=0.5
    )
    compressor = ModelCompressor(model, config)
    compressed_model = compressor._apply_low_rank(model)
    
    # Check that weights were modified
    assert torch.equal(compressed_model.fc1.weight, model.fc1.weight) == False
    assert compressed_model.fc1.weight.shape == model.fc1.weight.shape

def test_model_size_calculation(model, config):
    """Test model size calculation."""
    compressor = ModelCompressor(model, config)
    size = compressor._get_model_size()
    
    # Calculate expected size
    expected_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    assert size == expected_size

def test_compression_stats(model, config):
    """Test compression statistics generation."""
    compressor = ModelCompressor(model, config)
    compressed_model = compressor.compress()
    stats = compressor.get_compression_stats()
    
    assert "original_size" in stats
    assert "compressed_size" in stats
    assert "compression_ratio" in stats
    assert "elapsed_time" in stats
    assert "techniques_applied" in stats
    assert stats["compression_ratio"] > 0

def test_save_compressed_model(model, config):
    """Test saving compressed model and metadata."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "compressed_model.pt")
        
        compressor = ModelCompressor(model, config)
        compressed_model = compressor.compress()
        compressor.save_compressed_model(model_path)
        
        # Check that files were created
        assert os.path.exists(model_path)
        assert os.path.exists(os.path.splitext(model_path)[0] + "_metadata.json")
        
        # Load and check metadata
        with open(os.path.splitext(model_path)[0] + "_metadata.json", 'r') as f:
            metadata = json.load(f)
            assert "compression_ratio" in metadata

def test_compress_model_helper(model):
    """Test the compress_model helper function."""
    compressed_model, stats = compress_model(
        model=model,
        compression_types=["weight_sharing"],
        target_ratio=0.5,
        sharing_strategy="cluster",
        num_clusters=16
    )
    
    assert isinstance(compressed_model, nn.Module)
    assert isinstance(stats, dict)
    assert "compression_ratio" in stats

def test_multiple_compression_techniques(model):
    """Test applying multiple compression techniques."""
    config = CompressionConfig(
        compression_types=[
            CompressionType.WEIGHT_SHARING,
            CompressionType.LOW_RANK
        ],
        target_ratio=0.5,
        sharing_strategy=SharingStrategy.CLUSTER,
        num_clusters=16,
        rank_ratio=0.5
    )
    
    compressor = ModelCompressor(model, config)
    compressed_model = compressor.compress()
    stats = compressor.get_compression_stats()
    
    assert len(stats["techniques_applied"]) == 2
    assert stats["compression_ratio"] > 0

def test_invalid_sharing_strategy(model):
    """Test error handling for invalid sharing strategy."""
    with pytest.raises(ValueError):
        config = CompressionConfig(
            compression_types=[CompressionType.WEIGHT_SHARING],
            sharing_strategy="invalid"
        )

def test_model_path_loading():
    """Test loading model from path."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model = SimpleModel()
        model_path = os.path.join(tmp_dir, "model.pt")
        torch.save(model, model_path)
        
        config = CompressionConfig(
            compression_types=[CompressionType.WEIGHT_SHARING]
        )
        
        compressor = ModelCompressor(model_path, config)
        assert isinstance(compressor.model, nn.Module) 