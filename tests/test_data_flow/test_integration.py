"""
Integration tests for data flow components.
"""

import pytest
import time
import shutil
import json
import os
from pathlib import Path
import random
from datetime import datetime

from src.core.buffer_manager import BufferManager
from src.core.data_persistence import DataPersistence

class TestDataFlowIntegration:
    """Integration tests for data flow components."""
    
    @pytest.fixture
    def test_dir(self):
        """Create a temporary test directory."""
        test_dir = Path("./test_data_flow_integration")
        
        if test_dir.exists():
            shutil.rmtree(test_dir)
        
        test_dir.mkdir(exist_ok=True)
        
        yield test_dir
        
        shutil.rmtree(test_dir)
    
    @pytest.fixture
    def buffer_manager(self):
        """Create a BufferManager instance for testing."""
        return BufferManager(
            max_size=10,
            ttl=5.0,
            cleanup_interval=1.0
        )
    
    @pytest.fixture
    def data_persistence(self, test_dir):
        """Create a DataPersistence instance for testing."""
        return DataPersistence(
            base_dir=str(test_dir),
            memory_cache_size=5,
            disk_cache_size=20
        )
    
    def test_data_flow_pipeline(self, buffer_manager, data_persistence):
        """Test the data flow through buffer and persistence systems."""
        # Create test data
        def create_test_item(item_id):
            return {
                "id": item_id,
                "timestamp": datetime.now().isoformat(),
                "value": random.random() * 100,
                "type": random.choice(["temperature", "humidity", "pressure"]),
                "source": random.choice(["sensor_1", "sensor_2", "sensor_3"])
            }
        
        # Process data through the pipeline
        for i in range(20):
            item_id = f"item_{i}"
            raw_data = create_test_item(item_id)
            
            # 1. Store raw data in buffer
            buffer_manager.put(item_id, raw_data)
            
            # 2. Process data
            processed_data = raw_data.copy()
            processed_data["processed"] = True
            processed_data["processing_time"] = datetime.now().isoformat()
            
            if processed_data["type"] == "temperature":
                # Example processing logic
                processed_data["value_f"] = processed_data["value"] * 9/5 + 32
            
            # 3. Update buffer with processed data
            buffer_manager.put(item_id, processed_data)
            
            # 4. Store processed data in persistence system
            data_persistence.put(item_id, processed_data)
        
        # Verify data in buffer
        buffer_stats = buffer_manager.get_stats()
        assert buffer_stats["total_items"] == 20
        
        # Verify data in persistence
        persistence_stats = data_persistence.get_stats()
        assert persistence_stats["memory_cache_size"] == 5  # Limited by size
        assert persistence_stats["disk_cache_size"] == 20
        
        # Test retrieval path: buffer -> persistence
        for i in range(10):  # First 10 items should be in buffer
            item_id = f"item_{i}"
            
            # Should be in buffer
            buffer_data = buffer_manager.get(item_id)
            assert buffer_data is not None
            assert buffer_data["id"] == item_id
            assert buffer_data["processed"] is True
            
            # And also in persistence
            persistence_data = data_persistence.get(item_id)
            assert persistence_data is not None
            assert persistence_data["id"] == item_id
            assert persistence_data["processed"] is True
            
            # Data should be identical
            assert buffer_data == persistence_data
        
        # Test data flow with TTL expiration
        special_item = "special_item"
        special_data = create_test_item(special_item)
        
        # Store with short TTL
        buffer_manager.put(special_item, special_data, ttl=0.5)
        data_persistence.put(special_item, special_data)
        
        # Verify in both systems
        assert buffer_manager.get(special_item) is not None
        assert data_persistence.get(special_item) is not None
        
        # Wait for TTL to expire in buffer
        time.sleep(0.6)
        
        # Should be gone from buffer but still in persistence
        assert buffer_manager.get(special_item) is None
        assert data_persistence.get(special_item) is not None
        
        # Test retrieval from persistence when not in buffer
        not_in_buffer = "not_in_buffer"
        not_in_buffer_data = create_test_item(not_in_buffer)
        
        # Store only in persistence
        data_persistence.put(not_in_buffer, not_in_buffer_data)
        
        # Simulate application logic retrieving data
        def get_data_from_system(item_id):
            # Try buffer first (faster)
            data = buffer_manager.get(item_id)
            if data is not None:
                return data, "buffer"
            
            # Fall back to persistence
            data = data_persistence.get(item_id)
            if data is not None:
                return data, "persistence"
            
            return None, None
        
        # Test retrieval flow
        data, source = get_data_from_system(not_in_buffer)
        assert data is not None
        assert source == "persistence"
        
        # Test versioning with both systems
        versioned_item = "versioned_item"
        
        # Create multiple versions
        for i in range(3):
            version_data = create_test_item(versioned_item)
            version_data["version"] = i + 1
            
            # Update both systems
            buffer_manager.put(versioned_item, version_data)
            data_persistence.put(versioned_item, version_data)
        
        # Latest version should be available in both
        buffer_latest = buffer_manager.get(versioned_item)
        persistence_latest = data_persistence.get(versioned_item)
        
        assert buffer_latest["version"] == 3
        assert persistence_latest["version"] == 3
        
        # But only persistence can retrieve older versions
        persistence_v1 = data_persistence.get(versioned_item, version=1)
        persistence_v2 = data_persistence.get(versioned_item, version=2)
        
        assert persistence_v1["version"] == 1
        assert persistence_v2["version"] == 2 