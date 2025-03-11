"""
Tests for the state management module.

These tests verify the functionality of the state management utilities, including
memory management and state persistence features.

Author: AMPTALK Team
Date: 2024
"""

import os
import tempfile
import time
import pytest
import asyncio
import json
from typing import Dict, Any, List

from src.core.utils.state_manager import (
    StorageType, CacheStrategy, StateMetadata, MemoryOptimizer,
    StateManager, create_state_manager
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for state files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.mark.asyncio
async def test_memory_optimizer_initialization():
    """Test the initialization of the MemoryOptimizer."""
    optimizer = MemoryOptimizer(
        strategy=CacheStrategy.LRU,
        max_items=500,
        max_size_mb=50,
        default_ttl=60
    )
    
    # Check initialization
    assert optimizer.strategy == CacheStrategy.LRU
    assert optimizer.max_items == 500
    assert optimizer.max_size_bytes == 50 * 1024 * 1024  # 50 MB in bytes
    assert optimizer.default_ttl == 60
    assert optimizer.item_count == 0
    assert optimizer.current_size_bytes == 0


@pytest.mark.asyncio
async def test_memory_optimizer_cache_eviction():
    """Test the cache eviction strategies in MemoryOptimizer."""
    # Create an optimizer with a small capacity
    optimizer = MemoryOptimizer(
        strategy=CacheStrategy.LRU,
        max_items=3,  # Only 3 items allowed
        max_size_mb=1,
        default_ttl=None
    )
    
    # Add 3 items
    await optimizer.cache_item("key1", {"data": "item1"})
    await optimizer.cache_item("key2", {"data": "item2"})
    await optimizer.cache_item("key3", {"data": "item3"})
    
    # Verify all 3 items are in cache
    assert "key1" in optimizer.metadata
    assert "key2" in optimizer.metadata
    assert "key3" in optimizer.metadata
    assert optimizer.item_count == 3
    
    # Add a 4th item, should evict the least recently used (key1)
    await optimizer.cache_item("key4", {"data": "item4"})
    
    # Verify eviction
    assert "key1" not in optimizer.metadata
    assert "key2" in optimizer.metadata
    assert "key3" in optimizer.metadata
    assert "key4" in optimizer.metadata
    assert optimizer.item_count == 3


@pytest.mark.asyncio
async def test_memory_optimizer_ttl_expiration():
    """Test that items are evicted when their TTL expires."""
    # Create an optimizer with a small TTL
    optimizer = MemoryOptimizer(
        strategy=CacheStrategy.TTL,
        max_items=10,
        max_size_mb=1,
        default_ttl=0.1  # 100ms TTL
    )
    
    # Add an item
    await optimizer.cache_item("key1", {"data": "item1"})
    
    # Verify it's in cache
    assert "key1" in optimizer.metadata
    
    # Wait for TTL to expire
    await asyncio.sleep(0.2)
    
    # Access the item, which should trigger eviction
    result = await optimizer.get_item("key1")
    
    # Verify eviction
    assert result is None
    assert "key1" not in optimizer.metadata


@pytest.mark.asyncio
async def test_state_manager_memory_storage():
    """Test the StateManager with memory storage."""
    # Create a state manager with memory storage
    manager = StateManager(storage_type=StorageType.MEMORY)
    
    # Save some state
    test_data = {"key": "value", "nested": {"data": [1, 2, 3]}}
    await manager.save_state("test_state", test_data)
    
    # Load the state
    loaded_data = await manager.load_state("test_state")
    
    # Verify the data was loaded correctly
    assert loaded_data == test_data
    
    # Delete the state
    result = await manager.delete_state("test_state")
    assert result is True
    
    # Verify it's gone
    loaded_data = await manager.load_state("test_state")
    assert loaded_data is None


@pytest.mark.asyncio
async def test_state_manager_file_storage(temp_dir):
    """Test the StateManager with file storage."""
    # Create a state manager with file storage
    manager = StateManager(
        storage_type=StorageType.FILE,
        base_dir=temp_dir,
        prefix="test",
        compression=True
    )
    
    # Save some state
    test_data = {"key": "value", "nested": {"data": [1, 2, 3]}}
    await manager.save_state("test_state", test_data)
    
    # Verify the file was created
    file_path = manager._get_file_path("test_state")
    assert os.path.exists(file_path)
    
    # Load the state
    loaded_data = await manager.load_state("test_state")
    
    # Verify the data was loaded correctly
    assert loaded_data == test_data
    
    # Delete the state
    result = await manager.delete_state("test_state")
    assert result is True
    
    # Verify the file is gone
    assert not os.path.exists(file_path)


@pytest.mark.asyncio
async def test_state_manager_snapshot():
    """Test the snapshot context manager."""
    # Create a state manager
    manager = StateManager(storage_type=StorageType.MEMORY)
    
    # Use the snapshot context manager
    async with manager.snapshot("test_agent") as state:
        # Modify the state
        state["key"] = "value"
        state["count"] = 42
    
    # Verify the state was saved
    loaded_data = await manager.load_state("test_agent_state")
    assert loaded_data == {"key": "value", "count": 42}
    
    # Use the snapshot again to modify the state
    async with manager.snapshot("test_agent") as state:
        # Verify the existing state
        assert state["key"] == "value"
        assert state["count"] == 42
        
        # Modify it
        state["key"] = "new_value"
        state["nested"] = {"data": True}
    
    # Verify the state was updated
    loaded_data = await manager.load_state("test_agent_state")
    assert loaded_data == {"key": "new_value", "count": 42, "nested": {"data": True}}


@pytest.mark.asyncio
async def test_create_state_manager():
    """Test the create_state_manager utility function."""
    # Create a state manager
    manager = await create_state_manager(
        agent_id="test_agent",
        storage_type=StorageType.MEMORY,
        max_memory_mb=50
    )
    
    # Verify the manager is configured correctly
    assert manager.storage_type == StorageType.MEMORY
    assert manager.prefix == "agent_test_agent"
    assert manager.memory_optimizer.max_size_bytes == 50 * 1024 * 1024
    
    # Test basic functionality
    await manager.save_state("test", {"data": "value"})
    loaded = await manager.load_state("test")
    assert loaded == {"data": "value"}


@pytest.mark.asyncio
async def test_complex_object_serialization(temp_dir):
    """Test serialization of complex objects."""
    # Create a state manager with file storage
    manager = StateManager(
        storage_type=StorageType.FILE,
        base_dir=temp_dir
    )
    
    # Create a complex object (not JSON serializable)
    class TestObject:
        def __init__(self, name):
            self.name = name
            self.created_at = time.time()
        
        def get_name(self):
            return self.name
    
    test_obj = TestObject("test_instance")
    
    # Save the object
    await manager.save_state("complex_object", test_obj)
    
    # Load the object
    loaded_obj = await manager.load_state("complex_object")
    
    # Verify the object was loaded correctly
    assert isinstance(loaded_obj, TestObject)
    assert loaded_obj.name == "test_instance"
    assert loaded_obj.get_name() == "test_instance"


@pytest.mark.asyncio
async def test_memory_management_under_load():
    """Test memory management under load with many items."""
    # Create an optimizer with moderate capacity
    optimizer = MemoryOptimizer(
        strategy=CacheStrategy.LRU,
        max_items=100,
        max_size_mb=1,
        default_ttl=None
    )
    
    # Add 200 items (should keep only the last 100)
    for i in range(200):
        data = {"index": i, "data": f"item_{i}", "payload": "x" * 100}
        await optimizer.cache_item(f"key_{i}", data)
    
    # Verify we have exactly 100 items
    assert optimizer.item_count == 100
    
    # Verify we have the most recent 100 items
    for i in range(100):
        assert f"key_{i}" not in optimizer.metadata
    
    for i in range(100, 200):
        assert f"key_{i}" in optimizer.metadata
    
    # Get statistics
    stats = optimizer.get_stats()
    assert stats["item_count"] == 100
    assert stats["max_items"] == 100


@pytest.mark.asyncio
async def test_state_manager_list_states(temp_dir):
    """Test listing all stored states."""
    # Create a state manager with file storage
    manager = StateManager(
        storage_type=StorageType.FILE,
        base_dir=temp_dir
    )
    
    # Save multiple states
    await manager.save_state("state1", {"data": 1})
    await manager.save_state("state2", {"data": 2})
    await manager.save_state("state3", {"data": 3})
    
    # List all states
    states = await manager.list_states()
    
    # Verify we have all the states
    assert len(states) == 3
    assert "state1" in states
    assert "state2" in states
    assert "state3" in states
    
    # Delete one state
    await manager.delete_state("state2")
    
    # List again
    states = await manager.list_states()
    
    # Verify the state was removed
    assert len(states) == 2
    assert "state1" in states
    assert "state2" not in states
    assert "state3" in states
    
    # Clear all states
    await manager.clear_all()
    
    # List again
    states = await manager.list_states()
    
    # Verify all states were removed
    assert len(states) == 0 