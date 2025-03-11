"""Tests for the ResourceManager class."""

import pytest
import time
import json
import tempfile
import os
from unittest.mock import patch, MagicMock
from typing import List

import torch
import psutil

from src.core.resource_management import (
    ResourceType,
    OptimizationStrategy,
    ResourceLimits,
    ResourceMetrics,
    ResourceManager,
    create_resource_manager
)

@pytest.fixture
def resource_manager():
    """Create a resource manager for testing."""
    return ResourceManager(
        limits=ResourceLimits(
            max_memory_mb=1000,
            max_cpu_percent=80,
            max_gpu_memory_mb=1000,
            max_power_watts=100,
            max_temp_celsius=80
        ),
        strategy=OptimizationStrategy.ADAPTIVE,
        monitoring_interval=0.1
    )

def test_resource_manager_initialization():
    """Test ResourceManager initialization."""
    manager = ResourceManager()
    assert manager.limits is not None
    assert manager.strategy == OptimizationStrategy.ADAPTIVE
    assert manager._monitoring_thread is None
    assert isinstance(manager.current_metrics, ResourceMetrics)
    assert isinstance(manager.historical_metrics, list)

def test_resource_limits():
    """Test resource limits configuration."""
    limits = ResourceLimits(
        max_memory_mb=1000,
        max_cpu_percent=80,
        max_gpu_memory_mb=1000,
        max_power_watts=100,
        max_temp_celsius=80
    )
    
    assert limits.max_memory_mb == 1000
    assert limits.max_cpu_percent == 80
    assert limits.max_gpu_memory_mb == 1000
    assert limits.max_power_watts == 100
    assert limits.max_temp_celsius == 80

def test_resource_metrics():
    """Test resource metrics data class."""
    metrics = ResourceMetrics(
        memory_used_mb=500,
        memory_percent=50,
        cpu_percent=60,
        gpu_memory_mb=300,
        gpu_utilization=40,
        power_watts=50,
        temperature_celsius=70
    )
    
    assert metrics.memory_used_mb == 500
    assert metrics.memory_percent == 50
    assert metrics.cpu_percent == 60
    assert metrics.gpu_memory_mb == 300
    assert metrics.gpu_utilization == 40
    assert metrics.power_watts == 50
    assert metrics.temperature_celsius == 70

def test_monitoring_start_stop(resource_manager):
    """Test starting and stopping resource monitoring."""
    # Start monitoring
    resource_manager.start_monitoring()
    assert resource_manager._monitoring_thread is not None
    assert resource_manager._monitoring_thread.is_alive()
    
    # Stop monitoring
    resource_manager.stop_monitoring()
    assert not resource_manager._monitoring_thread.is_alive()
    assert resource_manager._monitoring_thread is None

def test_metrics_collection(resource_manager):
    """Test resource metrics collection."""
    metrics = resource_manager._collect_metrics()
    
    assert isinstance(metrics, ResourceMetrics)
    assert metrics.memory_used_mb > 0
    assert 0 <= metrics.memory_percent <= 100
    assert 0 <= metrics.cpu_percent <= 100

@patch('psutil.Process')
@patch('psutil.cpu_percent')
def test_memory_limit_handling(mock_cpu_percent, mock_process, resource_manager):
    """Test memory limit handling."""
    # Mock process memory info
    mock_memory_info = MagicMock()
    mock_memory_info.rss = 2000 * 1024 * 1024  # 2000 MB
    mock_process.return_value.memory_info.return_value = mock_memory_info
    mock_process.return_value.memory_percent.return_value = 90
    mock_cpu_percent.return_value = 50
    
    # Collect metrics (should trigger limit handling)
    metrics = resource_manager._collect_metrics()
    
    # Check that metrics were collected
    assert metrics.memory_used_mb > resource_manager.limits.max_memory_mb
    assert metrics.memory_percent > 0

@patch('torch.cuda.is_available')
@patch('torch.cuda.memory_allocated')
@patch('torch.cuda.utilization')
def test_gpu_metrics(mock_utilization, mock_memory, mock_available, resource_manager):
    """Test GPU metrics collection."""
    # Mock GPU availability and metrics
    mock_available.return_value = True
    mock_memory.return_value = 500 * 1024 * 1024  # 500 MB
    mock_utilization.return_value = 60
    
    metrics = resource_manager._collect_metrics()
    
    assert metrics.gpu_memory_mb == 500
    assert metrics.gpu_utilization == 60

def test_metrics_saving(resource_manager):
    """Test saving metrics to file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        metrics_path = os.path.join(tmp_dir, "metrics.json")
        
        # Collect some metrics
        resource_manager.start_monitoring()
        time.sleep(0.2)  # Allow some metrics to be collected
        resource_manager.stop_monitoring()
        
        # Save metrics
        resource_manager.save_metrics(metrics_path)
        
        # Check that file was created
        assert os.path.exists(metrics_path)
        
        # Load and check metrics
        with open(metrics_path, 'r') as f:
            data = json.load(f)
            assert "current" in data
            assert "historical" in data
            assert isinstance(data["historical"], list)

def test_resource_manager_context():
    """Test ResourceManager context manager."""
    with ResourceManager(monitoring_interval=0.1) as manager:
        assert manager._monitoring_thread is not None
        assert manager._monitoring_thread.is_alive()
        time.sleep(0.2)
    
    assert not manager._monitoring_thread.is_alive()
    assert manager._monitoring_thread is None

def test_create_resource_manager():
    """Test resource manager creation helper."""
    manager = create_resource_manager(
        max_memory_mb=1000,
        max_cpu_percent=80,
        max_gpu_memory_mb=1000,
        max_power_watts=100,
        max_temp_celsius=80,
        strategy="dynamic",
        monitoring_interval=0.1
    )
    
    assert isinstance(manager, ResourceManager)
    assert manager.limits.max_memory_mb == 1000
    assert manager.limits.max_cpu_percent == 80
    assert manager.strategy == OptimizationStrategy.DYNAMIC

def test_callback_execution():
    """Test resource metrics callback."""
    metrics_list: List[ResourceMetrics] = []
    
    def callback(metrics: ResourceMetrics):
        metrics_list.append(metrics)
    
    manager = ResourceManager(
        monitoring_interval=0.1,
        callback=callback
    )
    
    manager.start_monitoring()
    time.sleep(0.2)  # Allow some metrics to be collected
    manager.stop_monitoring()
    
    assert len(metrics_list) > 0
    assert all(isinstance(m, ResourceMetrics) for m in metrics_list)

def test_optimization_strategies():
    """Test different optimization strategies."""
    for strategy in OptimizationStrategy:
        manager = ResourceManager(
            strategy=strategy,
            monitoring_interval=0
        )
        
        # Test that strategy-specific optimization doesn't raise errors
        manager._optimize_resources()

def test_historical_metrics():
    """Test historical metrics management."""
    manager = ResourceManager(monitoring_interval=0.1)
    
    # Collect some metrics
    manager.start_monitoring()
    time.sleep(0.2)
    manager.stop_monitoring()
    
    # Check historical metrics
    assert len(manager.get_historical_metrics()) > 0
    
    # Clear metrics
    manager.clear_historical_metrics()
    assert len(manager.get_historical_metrics()) == 0

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_limit_handling():
    """Test GPU memory limit handling."""
    manager = ResourceManager(
        limits=ResourceLimits(max_gpu_memory_mb=100),
        monitoring_interval=0.1
    )
    
    # Allocate some GPU memory
    if torch.cuda.is_available():
        x = torch.zeros(1000, 1000).cuda()  # Allocate memory
        
        # Check metrics
        metrics = manager._collect_metrics()
        assert metrics.gpu_memory_mb > 0
        
        # Clean up
        del x
        torch.cuda.empty_cache() 