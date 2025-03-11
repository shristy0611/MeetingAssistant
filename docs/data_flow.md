# Data Flow Management

This document describes the Data Flow Management components of the AMPTALK system, which provide efficient real-time data processing, caching, and storage capabilities.

## Overview

The Data Flow Management system consists of three main components:

1. **StreamProcessor**: Handles real-time data streaming and processing using PyFlink
2. **BufferManager**: Provides sophisticated in-memory caching with multiple strategies
3. **DataPersistence**: Implements multi-layer storage with versioning and data integrity

Together, these components enable efficient data flow through the AMPTALK system, from raw audio input to processed transcriptions and analysis results.

## Components

### StreamProcessor

The StreamProcessor is built on Apache Flink's Python API (PyFlink) and provides high-performance, low-latency stream processing capabilities.

Key features:
- Real-time data streaming with millisecond latency
- Windowed calculations support
- Session management
- State persistence
- Backpressure handling

Example usage:
```python
from core.stream_processor import StreamProcessor

# Initialize the stream processor
processor = StreamProcessor(window_size=1000)  # 1 second window

# Create a data source
schema = {"timestamp": Types.LONG(), "value": Types.FLOAT()}
source = processor.create_source("sensor_data", schema)

# Add watermark strategy for handling out-of-order events
source_with_watermark = processor.add_watermark(source, "timestamp")

# Add window
windowed_stream = processor.add_window(source_with_watermark)

# Apply processing function
def process_func(value):
    # Process the data
    return processed_value

processed_stream = processor.process(windowed_stream, process_func)

# Execute the job
processor.execute("Sensor Data Processing")
```

### BufferManager

The BufferManager implements a sophisticated buffer management system with hybrid caching strategies, including LRU, TTL, and 2Q.

Key features:
- Multiple caching strategies (LRU, TTL, 2Q)
- Memory limits with automatic eviction
- Backpressure handling
- Adaptive batching
- Thread-safe operations

Example usage:
```python
from core.buffer_manager import BufferManager

# Initialize the buffer manager
buffer = BufferManager(
    max_size=1000,       # Maximum number of items
    ttl=60.0,            # Default TTL in seconds
    cleanup_interval=10.0,  # Clean up every 10 seconds
    a1_ratio=0.25        # 25% of cache for short-term items in 2Q
)

# Store an item
buffer.put("key1", "value1")

# Store an item with TTL
buffer.put("key2", "value2", ttl=30.0)  # 30 seconds TTL

# Retrieve an item
value = buffer.get("key1")

# Get cache statistics
stats = buffer.get_stats()
print(f"Cache stats: {stats}")

# Clear the cache
buffer.clear()
```

### DataPersistence

The DataPersistence component implements a multi-layer storage strategy with in-memory cache, local disk cache, and versioning support.

Key features:
- Multi-layer storage (memory, disk)
- Versioning with incremental updates
- Data integrity verification
- Automatic cache invalidation

Example usage:
```python
from core.data_persistence import DataPersistence

# Initialize the persistence system
persistence = DataPersistence(
    base_dir="./data",           # Base directory for disk storage
    memory_cache_size=1000,      # Maximum items in memory cache
    disk_cache_size=10000        # Maximum items in disk cache
)

# Store an item
persistence.put("key1", "value1")

# Retrieve an item
value = persistence.get("key1")

# Store multiple versions
persistence.put("versioned_key", "version1")
persistence.put("versioned_key", "version2")
persistence.put("versioned_key", "version3")

# Get latest version
latest = persistence.get("versioned_key")

# Get specific version
v1 = persistence.get("versioned_key", version=1)
v2 = persistence.get("versioned_key", version=2)

# Get list of available versions
versions = persistence.get_versions("versioned_key")
print(f"Available versions: {versions}")

# Remove an item
persistence.remove("key1")
```

## Integration

These components work together to provide a complete data flow solution:

1. Raw data is processed through the StreamProcessor in real-time
2. Processed data is stored in the BufferManager for quick access
3. Data is also persisted in the DataPersistence system for long-term storage

See the `src/demo_data_flow.py` file for a complete example of how these components work together.

## Requirements

- Python 3.8+
- PyFlink (for StreamProcessor only)
- Standard Python libraries (collections, threading, time, etc.)

## Testing

Comprehensive test coverage is provided:
- Unit tests for each component
- Integration tests for the complete data flow
- Performance benchmarks

Run the tests with:
```
pytest tests/test_data_flow/
``` 