"""Tests for the ModelPruning module."""

import pytest
import torch
import torch.nn as nn
import tempfile
from pathlib import Path
import json
from src.models.model_pruning import ModelPruner, ImportanceMetric

class SimpleConvNet(nn.Module):
    """Simple CNN for testing pruning."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 32 * 8 * 8)
        return self.fc(x)

@pytest.fixture
def model():
    """Create a simple model for testing."""
    return SimpleConvNet()

@pytest.fixture
def sample_input():
    """Create sample input tensor."""
    return torch.randn(1, 3, 32, 32)

@pytest.fixture
def pruner(model):
    """Create a ModelPruner instance."""
    return ModelPruner(model)

def test_importance_metric_sensitivity(model, sample_input):
    """Test sensitivity score calculation."""
    metric = ImportanceMetric()
    conv_layer = model.conv1
    
    sensitivity = metric.sensitivity_score(conv_layer, sample_input)
    assert sensitivity.shape == conv_layer.weight.shape[:1]
    assert torch.all(sensitivity >= 0)

def test_importance_metric_uniqueness(model):
    """Test uniqueness score calculation."""
    metric = ImportanceMetric()
    conv_layer = model.conv1
    
    uniqueness = metric.uniqueness_score(conv_layer)
    assert uniqueness.shape == conv_layer.weight.shape[:1]
    assert torch.all(uniqueness >= 0)

def test_layer_sensitivity_analysis(pruner, sample_input):
    """Test layer sensitivity analysis."""
    sensitivity_scores = pruner.analyze_layer_sensitivity(sample_input)
    
    assert isinstance(sensitivity_scores, dict)
    assert len(sensitivity_scores) > 0
    assert all(isinstance(score, float) for score in sensitivity_scores.values())

def test_structured_pruning(pruner):
    """Test structured pruning."""
    layer = pruner.model.conv1
    original_params = torch.sum(layer.weight.data != 0).item()
    
    weights, mask = pruner.structured_pruning(layer, amount=0.3)
    pruned_params = torch.sum(weights != 0).item()
    
    assert pruned_params < original_params
    assert weights.shape == layer.weight.shape
    assert mask.shape == layer.weight.shape

def test_unstructured_pruning(pruner):
    """Test unstructured pruning."""
    layer = pruner.model.fc
    original_params = torch.sum(layer.weight.data != 0).item()
    
    weights, mask = pruner.unstructured_pruning(layer, amount=0.3)
    pruned_params = torch.sum(weights != 0).item()
    
    assert pruned_params < original_params
    assert weights.shape == layer.weight.shape
    assert mask.shape == layer.weight.shape

def test_zero_shot_pruning(pruner, sample_input):
    """Test zero-shot pruning."""
    layer = pruner.model.conv1
    original_params = torch.sum(layer.weight.data != 0).item()
    
    weights, mask = pruner.zero_shot_pruning(layer, sample_input, amount=0.3)
    pruned_params = torch.sum(weights != 0).item()
    
    assert pruned_params < original_params
    assert weights.shape == layer.weight.shape
    assert mask.shape == layer.weight.shape

def test_adaptive_pruning(pruner, sample_input):
    """Test adaptive pruning."""
    initial_params = sum(p.numel() for p in pruner.model.parameters() if p.requires_grad)
    
    pruner.adaptive_pruning(
        sample_input,
        initial_amount=0.1,
        max_amount=0.3,
        step_size=0.1
    )
    
    # Check pruning history
    assert len(pruner.pruning_history) > 0
    assert all(isinstance(entry, dict) for entry in pruner.pruning_history)
    
    # Check sparsity increased
    final_params = sum(
        torch.sum(p != 0).item()
        for p in pruner.model.parameters()
        if p.requires_grad
    )
    assert final_params < initial_params

def test_save_load_pruning_state(pruner, sample_input, tmp_path):
    """Test saving and loading pruning state."""
    # Apply some pruning
    pruner.adaptive_pruning(sample_input, max_amount=0.2)
    original_history = pruner.pruning_history.copy()
    
    # Save state
    save_path = tmp_path / "pruning_state.json"
    pruner.save_pruning_state(save_path)
    
    # Clear history
    pruner.pruning_history = []
    
    # Load state
    pruner.load_pruning_state(save_path)
    
    assert pruner.pruning_history == original_history

def test_get_model_sparsity(pruner, sample_input):
    """Test sparsity calculation."""
    # Apply pruning
    pruner.adaptive_pruning(sample_input, max_amount=0.2)
    
    sparsity = pruner.get_model_sparsity()
    assert isinstance(sparsity, dict)
    assert len(sparsity) > 0
    assert all(0 <= s <= 1 for s in sparsity.values())

def test_get_compression_stats(pruner, sample_input):
    """Test compression statistics calculation."""
    # Apply pruning
    pruner.adaptive_pruning(sample_input, max_amount=0.2)
    
    stats = pruner.get_compression_stats()
    
    assert isinstance(stats, dict)
    assert "overall_sparsity" in stats
    assert "params_before" in stats
    assert "params_after" in stats
    assert "compression_ratio" in stats
    assert "layer_stats" in stats
    
    assert 0 <= stats["overall_sparsity"] <= 1
    assert stats["params_before"] > stats["params_after"]
    assert stats["compression_ratio"] >= 1.0 