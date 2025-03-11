"""Tests for the ModelManager class."""

import pytest
import torch
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch
from src.models.model_manager import ModelManager

@pytest.fixture
def model_manager():
    """Create a ModelManager instance for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield ModelManager(cache_dir=tmp_dir)

def test_init(model_manager):
    """Test ModelManager initialization."""
    assert isinstance(model_manager.cache_dir, Path)
    assert model_manager.max_cached_models == 3
    assert model_manager.memory_threshold == 0.85
    assert isinstance(model_manager._model_cache, dict)
    assert isinstance(model_manager._model_metadata, dict)

def test_device_setup():
    """Test device setup logic."""
    # Test CPU fallback
    manager = ModelManager(enable_gpu=False)
    assert manager.device == "cpu"
    
    # Test GPU detection
    with patch("torch.cuda.is_available", return_value=True):
        manager = ModelManager(enable_gpu=True)
        assert manager.device == "cuda"
    
    # Test explicit device setting
    manager = ModelManager(device="cuda:0")
    assert manager.device == "cuda:0"

@pytest.mark.asyncio
async def test_model_loading(model_manager):
    """Test model loading functionality."""
    mock_model = Mock()
    mock_model.to = Mock(return_value=mock_model)
    
    with patch("whisper.load_model", return_value=mock_model):
        # Test loading a new model
        model = model_manager.load_model(
            "test_model",
            "path/to/model",
            "whisper"
        )
        assert model == mock_model
        assert "test_model" in model_manager._model_cache
        assert "test_model" in model_manager._model_metadata
        
        # Test loading same model again (should use cache)
        cached_model = model_manager.load_model(
            "test_model",
            "path/to/model",
            "whisper"
        )
        assert cached_model == model
        
def test_memory_management(model_manager):
    """Test memory management and LRU cache behavior."""
    mock_models = [Mock() for _ in range(4)]
    for model in mock_models:
        model.to = Mock(return_value=model)
    
    with patch("whisper.load_model", side_effect=mock_models):
        # Load max_cached_models + 1 models
        for i in range(4):
            model_manager.load_model(
                f"model_{i}",
                f"path/to/model_{i}",
                "whisper"
            )
        
        # Check that oldest model was unloaded
        assert "model_0" not in model_manager._model_cache
        assert len(model_manager._model_cache) == model_manager.max_cached_models

def test_model_unloading(model_manager):
    """Test explicit model unloading."""
    mock_model = Mock()
    mock_model.to = Mock(return_value=mock_model)
    
    with patch("whisper.load_model", return_value=mock_model):
        model_manager.load_model("test_model", "path/to/model", "whisper")
        assert "test_model" in model_manager._model_cache
        
        model_manager.unload_model("test_model")
        assert "test_model" not in model_manager._model_cache

def test_clear_cache(model_manager):
    """Test cache clearing functionality."""
    mock_model = Mock()
    mock_model.to = Mock(return_value=mock_model)
    
    with patch("whisper.load_model", return_value=mock_model):
        model_manager.load_model("test_model", "path/to/model", "whisper")
        assert len(model_manager._model_cache) == 1
        
        model_manager.clear_cache()
        assert len(model_manager._model_cache) == 0

def test_context_manager(model_manager):
    """Test context manager functionality."""
    mock_model = Mock()
    mock_model.to = Mock(return_value=mock_model)
    
    with patch("whisper.load_model", return_value=mock_model):
        with model_manager as mm:
            mm.load_model("test_model", "path/to/model", "whisper")
            assert len(mm._model_cache) == 1
        
        # Cache should be cleared after context exit
        assert len(model_manager._model_cache) == 0

def test_metadata_handling(model_manager):
    """Test model metadata handling."""
    mock_model = Mock()
    mock_model.to = Mock(return_value=mock_model)
    
    with patch("whisper.load_model", return_value=mock_model):
        metadata = {"version": "1.0", "language": "en"}
        model_manager.load_model(
            "test_model",
            "path/to/model",
            "whisper",
            metadata=metadata
        )
        
        stored_metadata = model_manager.get_model_metadata("test_model")
        assert stored_metadata["version"] == "1.0"
        assert stored_metadata["language"] == "en"
        assert "last_loaded" in stored_metadata

def test_error_handling(model_manager):
    """Test error handling during model loading."""
    with pytest.raises(FileNotFoundError):
        model_manager.load_model(
            "invalid_model",
            "nonexistent/path",
            "whisper"
        )
        
    with pytest.raises(ValueError):
        model_manager.load_model(
            "invalid_type",
            "path/to/model",
            "unsupported_type"
        ) 