"""
Tests for the DataPersistence class.
"""

import pytest
import time
import os
import shutil
import json
from pathlib import Path

from src.core.data_persistence import DataPersistence

class TestDataPersistence:
    """Tests for the DataPersistence class."""
    
    @pytest.fixture
    def test_dir(self):
        """Create a temporary test directory."""
        # Create a unique test directory
        test_dir = Path("./test_data_persistence")
        
        # Ensure it's empty
        if test_dir.exists():
            shutil.rmtree(test_dir)
        
        test_dir.mkdir(exist_ok=True)
        
        yield test_dir
        
        # Clean up after tests
        shutil.rmtree(test_dir)
    
    @pytest.fixture
    def persistence(self, test_dir):
        """Create a DataPersistence instance for testing."""
        return DataPersistence(
            base_dir=str(test_dir),
            memory_cache_size=5,
            disk_cache_size=10
        )
    
    def test_init(self, persistence, test_dir):
        """Test initialization."""
        assert persistence.base_dir == test_dir
        assert persistence.memory_cache_size == 5
        assert persistence.disk_cache_size == 10
        
        # Check that base directories are created
        assert test_dir.exists()
        assert (test_dir / "versions").exists()
        
        # Check that caches are initially empty
        assert len(persistence.memory_cache) == 0
        assert len(persistence.disk_cache_index) == 0
    
    def test_put_get_basic(self, persistence):
        """Test basic put and get operations."""
        # Put an item
        result = persistence.put("key1", "value1")
        assert result is True
        
        # Get the item
        value = persistence.get("key1")
        assert value == "value1"
        
        # Should be in memory cache
        assert "key1" in persistence.memory_cache
        assert persistence.memory_cache["key1"][0] == "value1"  # value
        
        # Should also be in disk cache index
        assert "key1" in persistence.disk_cache_index
        assert persistence.disk_cache_index["key1"]["version"] == 1
    
    def test_versioning(self, persistence):
        """Test versioning functionality."""
        # Put an item multiple times
        persistence.put("key1", "value1")
        persistence.put("key1", "value2")
        persistence.put("key1", "value3")
        
        # Get latest version
        value = persistence.get("key1")
        assert value == "value3"
        
        # Get specific versions
        value1 = persistence.get("key1", version=1)
        assert value1 == "value1"
        
        value2 = persistence.get("key1", version=2)
        assert value2 == "value2"
        
        value3 = persistence.get("key1", version=3)
        assert value3 == "value3"
        
        # Check disk cache index
        assert persistence.disk_cache_index["key1"]["version"] == 3
        
        # Get list of versions
        versions = persistence.get_versions("key1")
        assert versions == [1, 2, 3]
    
    def test_memory_cache_eviction(self, persistence):
        """Test memory cache eviction."""
        # Fill memory cache
        for i in range(5):
            persistence.put(f"key{i}", f"value{i}")
        
        # All should be in memory cache
        assert len(persistence.memory_cache) == 5
        
        # Add one more to trigger eviction
        persistence.put("key5", "value5")
        
        # Memory cache should still have 5 items, but key0 should be evicted
        assert len(persistence.memory_cache) == 5
        assert "key0" not in persistence.memory_cache
        assert "key5" in persistence.memory_cache
        
        # But key0 should still be retrievable from disk
        value = persistence.get("key0")
        assert value == "value0"
    
    def test_integrity_verification(self, persistence):
        """Test data integrity verification."""
        # Put an item
        persistence.put("key1", "value1")
        
        # Verify checksum is stored
        memory_item = persistence.memory_cache["key1"]
        assert memory_item[3]  # checksum exists
        
        # Tamper with the value directly (simulating corruption)
        # Note: This is implementation-dependent, but shows the concept
        correct_checksum = memory_item[3]
        tampered_item = ("tampered_value", memory_item[1], memory_item[2], correct_checksum)
        persistence.memory_cache["key1"] = tampered_item
        
        # Get should detect the tampering and fail integrity check
        value = persistence.get("key1")
        
        # The item should be retrieved from disk since memory cache failed integrity check
        assert value == "value1"
        
        # Memory cache should no longer have the tampered item
        assert "key1" not in persistence.memory_cache
    
    def test_remove(self, persistence):
        """Test remove operation."""
        # Put an item with multiple versions
        persistence.put("key1", "value1")
        persistence.put("key1", "value2")
        
        # Remove the item
        result = persistence.remove("key1")
        assert result is True
        
        # Should be gone from memory cache
        assert "key1" not in persistence.memory_cache
        
        # Should be gone from disk cache index
        assert "key1" not in persistence.disk_cache_index
        
        # Should not be retrievable
        value = persistence.get("key1")
        assert value is None
    
    def test_persistence_across_instances(self, test_dir):
        """Test that data persists across different instances."""
        # Create first instance and store data
        persistence1 = DataPersistence(
            base_dir=str(test_dir),
            memory_cache_size=5,
            disk_cache_size=10
        )
        
        persistence1.put("key1", "value1")
        persistence1.put("key2", "value2")
        
        # Create second instance and verify data
        persistence2 = DataPersistence(
            base_dir=str(test_dir),
            memory_cache_size=5,
            disk_cache_size=10
        )
        
        value1 = persistence2.get("key1")
        value2 = persistence2.get("key2")
        
        assert value1 == "value1"
        assert value2 == "value2"
    
    def test_get_stats(self, persistence):
        """Test get_stats method."""
        # Add some items
        persistence.put("key1", "value1")
        persistence.put("key2", "value2")
        
        # Get stats
        stats = persistence.get_stats()
        
        # Verify stats
        assert stats["memory_cache_size"] == 2
        assert stats["disk_cache_size"] == 2
        assert stats["memory_cache_max"] == 5
        assert stats["disk_cache_max"] == 10
    
    def test_thread_safety(self, persistence):
        """Basic test that operations don't raise exceptions when locked."""
        # Acquire lock manually to simulate concurrent access
        with persistence._lock:
            # These operations should wait for lock to be released
            persistence.put("key1", "value1")
            value = persistence.get("key1")
            
            # Operations completed after waiting for lock
            assert value == "value1" 