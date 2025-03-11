"""
Data Flow Management Demo for AMPTALK

This script demonstrates the integrated use of StreamProcessor, BufferManager, and DataPersistence
to handle real-time data processing, caching, and storage in the AMPTALK system.
"""

import time
import logging
import json
import random
from typing import Dict, Any
from datetime import datetime

from core.stream_processor import StreamProcessor
from core.buffer_manager import BufferManager
from core.data_persistence import DataPersistence

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DataFlowDemo")

class DataFlowManager:
    """
    Demonstrates the integrated use of data flow components.
    """
    
    def __init__(self):
        """Initialize the data flow manager with all components."""
        logger.info("Initializing Data Flow Manager")
        
        # Initialize components
        self.buffer = BufferManager(
            max_size=100,
            ttl=60.0,  # 1 minute TTL
            cleanup_interval=10.0
        )
        
        self.persistence = DataPersistence(
            base_dir="./data/demo",
            memory_cache_size=50,
            disk_cache_size=500
        )
        
        # Note: StreamProcessor requires PyFlink, which would be initialized here
        # self.stream_processor = StreamProcessor(window_size=500)
        
        logger.info("Data Flow Manager initialized")
    
    def generate_sample_data(self, num_items: int = 10) -> None:
        """
        Generate and process sample data through the system.
        
        Args:
            num_items: Number of items to generate
        """
        logger.info(f"Generating {num_items} sample data items")
        
        for i in range(num_items):
            # Generate sample data
            item_id = f"item_{i}"
            data = {
                "id": item_id,
                "timestamp": datetime.now().isoformat(),
                "value": random.random() * 100,
                "type": random.choice(["temperature", "humidity", "pressure"]),
                "source": random.choice(["sensor_1", "sensor_2", "sensor_3"]),
                "processed": False
            }
            
            # Store in buffer first (short-term storage)
            self.buffer.put(item_id, data)
            logger.info(f"Added item {item_id} to buffer")
            
            # Process the data (simulating stream processing)
            processed_data = self._process_data(data)
            
            # Update buffer with processed data
            self.buffer.put(item_id, processed_data)
            logger.info(f"Updated item {item_id} in buffer with processed data")
            
            # Store in persistence system (long-term storage)
            self.persistence.put(item_id, processed_data)
            logger.info(f"Stored item {item_id} in persistence system")
            
            # Small delay to simulate real-time processing
            time.sleep(0.2)
    
    def _process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data (simulating stream processing).
        
        Args:
            data: Raw data to process
            
        Returns:
            Processed data
        """
        # Create a copy to avoid modifying the original
        processed = data.copy()
        
        # Add some processing logic
        if processed["type"] == "temperature":
            # Convert to Fahrenheit if it's temperature
            processed["value_f"] = processed["value"] * 9/5 + 32
        
        # Add processing metadata
        processed["processed"] = True
        processed["processing_time"] = datetime.now().isoformat()
        
        return processed
    
    def retrieve_data(self, item_ids: list) -> None:
        """
        Retrieve and display data from the system.
        
        Args:
            item_ids: List of item IDs to retrieve
        """
        logger.info(f"Retrieving {len(item_ids)} items")
        
        for item_id in item_ids:
            # Try to get from buffer first (faster)
            data = self.buffer.get(item_id)
            source = "buffer"
            
            if data is None:
                # If not in buffer, try persistence system
                data = self.persistence.get(item_id)
                source = "persistence"
            
            if data:
                logger.info(f"Retrieved item {item_id} from {source}: {json.dumps(data, indent=2)}")
            else:
                logger.info(f"Item {item_id} not found in any storage layer")
    
    def show_stats(self) -> None:
        """Display statistics from all components."""
        logger.info("System Statistics:")
        
        # Get buffer stats
        buffer_stats = self.buffer.get_stats()
        logger.info(f"Buffer stats: {json.dumps(buffer_stats, indent=2)}")
        
        # Get persistence stats
        persistence_stats = self.persistence.get_stats()
        logger.info(f"Persistence stats: {json.dumps(persistence_stats, indent=2)}")

def main():
    """Run the data flow demo."""
    logger.info("Starting Data Flow Demo")
    
    # Initialize the manager
    manager = DataFlowManager()
    
    # Generate sample data
    manager.generate_sample_data(15)
    
    # Show system stats
    manager.show_stats()
    
    # Retrieve some items
    manager.retrieve_data(["item_2", "item_5", "item_10", "nonexistent_item"])
    
    # Wait a bit to simulate passage of time
    logger.info("Waiting to demonstrate TTL expiration...")
    time.sleep(5)
    
    # Generate more data
    manager.generate_sample_data(5)
    
    # Show updated stats
    manager.show_stats()
    
    logger.info("Data Flow Demo completed")

if __name__ == "__main__":
    main() 