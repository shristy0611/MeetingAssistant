"""Tests for the OperatorOptimizer class."""

import pytest
import torch
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.models.operator_optimization import (
    OptimizationTarget,
    OperatorType,
    OperatorOptimizer,
    detect_optimal_target,
    optimize_operators
)

@pytest.fixture
def mock_onnx_model():
    """Create a temporary ONNX model file for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "model.onnx")
        
        # Create an empty file
        with open(model_path, 'wb') as f:
            f.write(b'dummy')
        
        yield model_path

@pytest.mark.skipif(not hasattr(pytest, 'importorskip'), reason="importorskip not available")
def test_detect_optimal_target():
    """Test detection of optimal optimization target."""
    pytest.importorskip("torch")
    
    target = detect_optimal_target()
    assert isinstance(target, OptimizationTarget)
    assert target.value in ["cpu", "cpu_avx", "cpu_avx2", "cpu_avx512", "gpu_cuda", "gpu_mps"]

@patch("src.models.operator_optimization.onnx.load")
@patch("src.models.operator_optimization.onnx.save")
@patch("src.models.operator_optimization.ort.InferenceSession")
def test_operator_optimizer_initialization(mock_session, mock_save, mock_load, mock_onnx_model):
    """Test initialization of the OperatorOptimizer."""
    # Mock the ONNX model
    mock_model = MagicMock()
    mock_model.graph.node = []
    mock_load.return_value = mock_model
    
    # Create optimizer
    optimizer = OperatorOptimizer(
        model=mock_onnx_model,
        target=OptimizationTarget.CPU,
        config={"cpu_threads": 4}
    )
    
    assert optimizer.target == OptimizationTarget.CPU
    assert optimizer.model_path == mock_onnx_model
    assert optimizer.config == {"cpu_threads": 4}
    assert len(optimizer.enabled_optimizations) > 0
    assert len(optimizer.operator_optimization_map) > 0

@patch("src.models.operator_optimization.onnx.load")
@patch("src.models.operator_optimization.onnx.save")
@patch("src.models.operator_optimization.ort.InferenceSession")
def test_analyze_model(mock_session, mock_save, mock_load, mock_onnx_model):
    """Test model analysis functionality."""
    # Create a mock model with some nodes
    mock_model = MagicMock()
    mock_node1 = MagicMock()
    mock_node1.op_type = "MatMul"
    mock_node2 = MagicMock()
    mock_node2.op_type = "Add"
    mock_node3 = MagicMock()
    mock_node3.op_type = "Relu"
    mock_node4 = MagicMock()
    mock_node4.op_type = "UnknownOp"
    
    mock_model.graph.node = [mock_node1, mock_node2, mock_node3, mock_node4]
    mock_load.return_value = mock_model
    
    # Create optimizer
    optimizer = OperatorOptimizer(
        model=mock_onnx_model,
        target=OptimizationTarget.CPU
    )
    
    # Analyze model
    result = optimizer.analyze_model()
    
    # Check that the known operators were counted correctly
    assert result.get("matmul", 0) == 1
    assert result.get("element_wise", 0) == 1
    assert result.get("activation", 0) == 1

@patch("src.models.operator_optimization.onnx.load")
@patch("src.models.operator_optimization.ort.InferenceSession")
def test_apply_optimizations(mock_session, mock_load, mock_onnx_model):
    """Test applying operator optimizations."""
    # Mock the ONNX model
    mock_model = MagicMock()
    mock_model.graph.node = []
    mock_load.return_value = mock_model
    
    # Mock the inference session
    mock_session_instance = MagicMock()
    mock_session.return_value = mock_session_instance
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = os.path.join(tmp_dir, "optimized_model.onnx")
        
        # Create optimizer
        optimizer = OperatorOptimizer(
            model=mock_onnx_model,
            target=OptimizationTarget.CPU
        )
        
        # Apply optimizations
        result = optimizer.apply_optimizations(output_path)
        
        # Check that the optimization session was created
        mock_session.assert_called_once()
        
        # Check that the result is the output path
        assert result == output_path
        
        # Check that optimization stats were updated
        assert "elapsed_time" in optimizer.optimization_stats
        assert "target" in optimizer.optimization_stats
        assert "optimizations_applied" in optimizer.optimization_stats
        assert optimizer.optimization_stats["target"] == OptimizationTarget.CPU.value

@patch("src.models.operator_optimization.onnx.load")
@patch("src.models.operator_optimization.ort.InferenceSession")
def test_save_optimization_metadata(mock_session, mock_load, mock_onnx_model):
    """Test saving optimization metadata."""
    # Mock the ONNX model
    mock_model = MagicMock()
    mock_model.graph.node = []
    mock_load.return_value = mock_model
    
    # Mock the inference session
    mock_session_instance = MagicMock()
    mock_session.return_value = mock_session_instance
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = os.path.join(tmp_dir, "optimized_model.onnx")
        metadata_path = os.path.join(tmp_dir, "metadata.json")
        
        # Create optimizer
        optimizer = OperatorOptimizer(
            model=mock_onnx_model,
            target=OptimizationTarget.CPU
        )
        
        # Apply optimizations and update stats
        optimizer.apply_optimizations(output_path)
        
        # Save metadata
        optimizer.save_optimization_metadata(metadata_path)
        
        # Check that metadata file was created
        assert os.path.exists(metadata_path)
        
        # Check metadata content
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        assert metadata["target"] == OptimizationTarget.CPU.value
        assert "enabled_optimizations" in metadata
        assert "optimization_stats" in metadata

@patch("src.models.operator_optimization.OperatorOptimizer")
def test_optimize_operators_helper(mock_optimizer_class, mock_onnx_model):
    """Test the optimize_operators helper function."""
    # Mock optimizer instance
    mock_instance = MagicMock()
    mock_optimizer_class.return_value = mock_instance
    mock_instance.analyze_model.return_value = {"matmul": 1}
    mock_instance.apply_optimizations.return_value = "output_path"
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = os.path.join(tmp_dir, "optimized_model.onnx")
        
        # Call helper function
        result = optimize_operators(
            model_path=mock_onnx_model,
            output_path=output_path,
            target=OptimizationTarget.CPU,
            operator_types=[OperatorType.MATMUL.value],
            config={"cpu_threads": 4}
        )
        
        # Check that optimizer was created with correct parameters
        mock_optimizer_class.assert_called_with(
            model=mock_onnx_model,
            target=OptimizationTarget.CPU,
            config={"cpu_threads": 4, "enabled_optimizations": [OperatorType.MATMUL.value]}
        )
        
        # Check that analyze_model was called
        mock_instance.analyze_model.assert_called_once()
        
        # Check that apply_optimizations was called with correct path
        mock_instance.apply_optimizations.assert_called_with(output_path)
        
        # Check that save_optimization_metadata was called
        mock_instance.save_optimization_metadata.assert_called_once()
        
        # Check that the result is the output path
        assert result == "output_path" 