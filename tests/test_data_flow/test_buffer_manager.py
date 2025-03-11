"""
Tests for the BufferManager class.
"""

import pytest
import time
from src.core.buffer_manager import BufferManager

class TestBufferManager:
    """Tests for the BufferManager class."""
    
    @pytest.fixture
    def buffer_manager(self):
        """Create a BufferManager instance for testing."""
        return BufferManager(
            max_size=5,
            ttl=0.5,  # Short TTL for testing
            cleanup_interval=0.2,  # Short cleanup interval for testing
            a1_ratio=0.4  # 2 items in A1, 3 in AM
        )
    
    def test_init(self, buffer_manager):
        """Test initialization."""
        assert buffer_manager.max_size == 5
        assert buffer_manager.default_ttl == 0.5
        assert buffer_manager.cleanup_interval == 0.2
        assert buffer_manager.a1_size == 2
        assert buffer_manager.am_size == 3
        
        # Check that caches are initially empty
        assert len(buffer_manager.a1_cache) == 0
        assert len(buffer_manager.am_cache) == 0
        assert len(buffer_manager.lru_cache) == 0
        assert len(buffer_manager.ttl_cache) == 0
    
    def test_put_get_basic(self, buffer_manager):
        """Test basic put and get operations."""
        # Put an item without TTL
        buffer_manager.put("key1", "value1", ttl=None)
        
        # Should be in A1 cache
        assert len(buffer_manager.a1_cache) == 1
        assert "key1" in buffer_manager.a1_cache
        
        # Get the item
        value = buffer_manager.get("key1")
        assert value == "value1"
        
        # Should now be in AM cache (promoted)
        assert len(buffer_manager.a1_cache) == 0
        assert len(buffer_manager.am_cache) == 1
        assert "key1" in buffer_manager.am_cache
    
    def test_put_get_ttl(self, buffer_manager):
        """Test put and get with TTL."""
        # Put an item with TTL
        buffer_manager.put("key1", "value1", ttl=0.1)
        
        # Should be in TTL cache
        assert len(buffer_manager.ttl_cache) == 1
        assert "key1" in buffer_manager.ttl_cache
        
        # Get the item right away
        value = buffer_manager.get("key1")
        assert value == "value1"
        
        # Wait for TTL to expire
        time.sleep(0.2)
        
        # Get after TTL expired
        value = buffer_manager.get("key1")
        assert value is None
        assert "key1" not in buffer_manager.ttl_cache
    
    def test_2q_cache_eviction(self, buffer_manager):
        """Test 2Q cache eviction strategy."""
        # Fill A1 cache
        buffer_manager.put("key1", "value1")
        buffer_manager.put("key2", "value2")
        
        # Both should be in A1
        assert len(buffer_manager.a1_cache) == 2
        assert "key1" in buffer_manager.a1_cache
        assert "key2" in buffer_manager.a1_cache
        
        # Access key1 to promote to AM
        value = buffer_manager.get("key1")
        assert value == "value1"
        
        # key1 should be in AM, key2 in A1
        assert len(buffer_manager.a1_cache) == 1
        assert len(buffer_manager.am_cache) == 1
        assert "key1" in buffer_manager.am_cache
        assert "key2" in buffer_manager.a1_cache
        
        # Add more items to fill A1 again
        buffer_manager.put("key3", "value3")
        
        # A1 should be full, pushing key2 to AM
        assert len(buffer_manager.a1_cache) == 1
        assert len(buffer_manager.am_cache) == 2
        assert "key3" in buffer_manager.a1_cache
        assert "key1" in buffer_manager.am_cache
        assert "key2" in buffer_manager.am_cache
        
        # Add more items to fill AM
        buffer_manager.put("key4", "value4")
        buffer_manager.put("key5", "value5")
        buffer_manager.put("key6", "value6")
        
        # A1 should have newer items, AM should have evicted the oldest
        assert len(buffer_manager.a1_cache) == 2
        assert len(buffer_manager.am_cache) == 3
        
        # key1 or key2 should have been evicted from AM (key1 is older)
        assert "key1" not in buffer_manager.am_cache
    
    def test_cleanup_expired(self, buffer_manager):
        """Test cleanup of expired items."""
        # Add some items with TTL
        buffer_manager.put("key1", "value1", ttl=0.1)
        buffer_manager.put("key2", "value2", ttl=10)  # Long TTL
        
        # Both should be in TTL cache
        assert len(buffer_manager.ttl_cache) == 2
        
        # Wait for key1 to expire
        time.sleep(0.2)
        
        # Manually trigger cleanup
        buffer_manager.cleanup_expired()
        
        # key1 should be gone, key2 should remain
        assert len(buffer_manager.ttl_cache) == 1
        assert "key1" not in buffer_manager.ttl_cache
        assert "key2" in buffer_manager.ttl_cache
    
    def test_clear(self, buffer_manager):
        """Test clear operation."""
        # Add some items
        buffer_manager.put("key1", "value1")
        buffer_manager.put("key2", "value2", ttl=1)
        
        # Verify items are stored
        assert len(buffer_manager.a1_cache) == 1
        assert len(buffer_manager.ttl_cache) == 1
        
        # Clear all caches
        buffer_manager.clear()
        
        # All caches should be empty
        assert len(buffer_manager.a1_cache) == 0
        assert len(buffer_manager.am_cache) == 0
        assert len(buffer_manager.lru_cache) == 0
        assert len(buffer_manager.ttl_cache) == 0
    
    def test_get_stats(self, buffer_manager):
        """Test get_stats method."""
        # Add some items
        buffer_manager.put("key1", "value1")
        buffer_manager.put("key2", "value2", ttl=1)
        
        # Get stats
        stats = buffer_manager.get_stats()
        
        # Verify stats
        assert stats["a1_cache_size"] == 1
        assert stats["ttl_cache_size"] == 1
        assert stats["am_cache_size"] == 0
        assert stats["lru_cache_size"] == 0
        assert stats["total_items"] == 2
    
    def test_thread_safety(self, buffer_manager):
        """Basic test that operations don't raise exceptions when locked."""
        # Acquire lock manually to simulate concurrent access
        with buffer_manager._lock:
            # These operations should wait for lock to be released
            buffer_manager.put("key1", "value1")
            value = buffer_manager.get("key1")
            
            # Operations completed after waiting for lock
            assert value == "value1" 