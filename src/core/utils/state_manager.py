"""
State Manager for AMPTALK Multi-Agent Framework.

This module provides utilities for managing agent state, including memory management
and state persistence mechanisms. It helps agents store, retrieve, and optimize
their state data throughout their lifecycle.

Author: AMPTALK Team
Date: 2024
"""

import os
import json
import logging
import time
import asyncio
import aiofiles
from typing import Dict, Any, Optional, List, Set, Tuple, Callable, Union
import tempfile
import pickle
import gzip
import hashlib
from enum import Enum
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class StorageType(Enum):
    """Types of storage supported for state persistence."""
    
    MEMORY = "memory"        # In-memory only (volatile)
    FILE = "file"            # File-based persistence
    SHARED_MEMORY = "shm"    # Shared memory (not implemented yet)
    DATABASE = "db"          # Database backend (not implemented yet)


class CacheStrategy(Enum):
    """Cache eviction strategies for memory management."""
    
    LRU = "lru"              # Least Recently Used
    LFU = "lfu"              # Least Frequently Used
    FIFO = "fifo"            # First In First Out
    TTL = "ttl"              # Time To Live
    SIZE_BASED = "size"      # Size-based eviction


@dataclass
class StateMetadata:
    """Metadata about a state entry for caching and persistence."""
    
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    last_modified_at: float = field(default_factory=time.time)
    access_count: int = 0
    size_bytes: int = 0
    version: int = 1
    checksum: Optional[str] = None
    ttl: Optional[float] = None  # Time to live in seconds
    
    def update_access(self) -> None:
        """Update access metadata."""
        self.last_accessed_at = time.time()
        self.access_count += 1
    
    def update_modification(self, size_bytes: int, data: Any = None) -> None:
        """Update modification metadata."""
        self.last_modified_at = time.time()
        self.size_bytes = size_bytes
        self.version += 1
        
        if data is not None:
            # Calculate checksum
            try:
                if isinstance(data, (dict, list, str, int, float, bool, type(None))):
                    # Use json for simple data types
                    json_str = json.dumps(data, sort_keys=True)
                    self.checksum = hashlib.md5(json_str.encode('utf-8')).hexdigest()
                else:
                    # Use pickle for complex objects
                    pickle_data = pickle.dumps(data)
                    self.checksum = hashlib.md5(pickle_data).hexdigest()
            except Exception as e:
                logger.warning(f"Failed to calculate checksum: {str(e)}")
    
    def is_expired(self) -> bool:
        """Check if the entry has expired based on TTL."""
        if self.ttl is None:
            return False
        
        return (time.time() - self.last_modified_at) > self.ttl


class MemoryOptimizer:
    """
    Manages memory usage optimization for agent state.
    
    This class implements various caching strategies and eviction policies
    to help keep memory usage under control while ensuring frequently used
    state is readily available.
    """
    
    def __init__(self, strategy: CacheStrategy = CacheStrategy.LRU, 
                max_items: int = 1000, max_size_mb: float = 100,
                default_ttl: Optional[float] = None):
        """
        Initialize the memory optimizer.
        
        Args:
            strategy: Cache eviction strategy to use
            max_items: Maximum number of items to keep in memory
            max_size_mb: Maximum memory usage in MB
            default_ttl: Default time-to-live for cached items in seconds
        """
        self.strategy = strategy
        self.max_items = max_items
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.default_ttl = default_ttl
        
        # Current memory usage
        self.current_size_bytes = 0
        self.item_count = 0
        
        # Cache metadata
        self.metadata: Dict[str, StateMetadata] = {}
        
        # Access order for LRU/FIFO
        self.access_order: List[str] = []
        
        # Lock for thread safety
        self.lock = asyncio.Lock()
        
        logger.info(f"Initialized MemoryOptimizer with {strategy.value} strategy, "
                   f"max {max_items} items and {max_size_mb} MB")
    
    def _estimate_size(self, data: Any) -> int:
        """
        Estimate the memory size of an object in bytes.
        
        Args:
            data: The object to measure
            
        Returns:
            Estimated size in bytes
        """
        try:
            if isinstance(data, (dict, list, str, int, float, bool, type(None))):
                # Use json for simple data types
                json_str = json.dumps(data)
                return len(json_str.encode('utf-8'))
            else:
                # Use pickle for complex objects
                return len(pickle.dumps(data))
        except Exception as e:
            logger.warning(f"Failed to estimate size: {str(e)}")
            # Return a reasonable default
            return 1024  # 1 KB default
    
    async def cache_item(self, key: str, data: Any, ttl: Optional[float] = None) -> None:
        """
        Add or update an item in the cache.
        
        Args:
            key: Unique identifier for the item
            data: The data to cache
            ttl: Time-to-live in seconds (overrides default)
        """
        async with self.lock:
            # Calculate size
            size_bytes = self._estimate_size(data)
            
            # Create or update metadata
            if key in self.metadata:
                # Update existing entry
                old_size = self.metadata[key].size_bytes
                self.current_size_bytes = self.current_size_bytes - old_size + size_bytes
                
                # Update metadata
                self.metadata[key].update_modification(size_bytes, data)
                self.metadata[key].update_access()
                
                # If TTL is provided, update it
                if ttl is not None:
                    self.metadata[key].ttl = ttl
                
                # Update access order (LRU)
                if self.strategy == CacheStrategy.LRU and key in self.access_order:
                    self.access_order.remove(key)
                    self.access_order.append(key)
            else:
                # Add new entry
                self.metadata[key] = StateMetadata(
                    size_bytes=size_bytes,
                    ttl=ttl if ttl is not None else self.default_ttl
                )
                self.current_size_bytes += size_bytes
                self.item_count += 1
                
                # Add to access order
                if self.strategy in [CacheStrategy.LRU, CacheStrategy.FIFO]:
                    self.access_order.append(key)
            
            # Check if we need to evict items
            await self._check_eviction()
    
    async def get_item(self, key: str) -> Optional[Tuple[Any, StateMetadata]]:
        """
        Get an item from the cache.
        
        Args:
            key: The key to retrieve
            
        Returns:
            Tuple of (data, metadata) if found, None otherwise
        """
        async with self.lock:
            # Check if the key exists and hasn't expired
            if key in self.metadata:
                metadata = self.metadata[key]
                
                # Check if expired
                if metadata.is_expired():
                    await self._evict_item(key)
                    return None
                
                # Update access metadata
                metadata.update_access()
                
                # Update access order (LRU)
                if self.strategy == CacheStrategy.LRU and key in self.access_order:
                    self.access_order.remove(key)
                    self.access_order.append(key)
                
                # Return a tuple of (data, metadata) - actual implementation
                # would retrieve the data from the storage
                return (None, metadata)
            
            return None
    
    async def remove_item(self, key: str) -> bool:
        """
        Remove an item from the cache.
        
        Args:
            key: The key to remove
            
        Returns:
            True if the item was removed, False otherwise
        """
        async with self.lock:
            return await self._evict_item(key)
    
    async def _evict_item(self, key: str) -> bool:
        """
        Evict a specific item from the cache.
        
        Args:
            key: The key to evict
            
        Returns:
            True if the item was evicted, False otherwise
        """
        if key in self.metadata:
            # Update size tracking
            self.current_size_bytes -= self.metadata[key].size_bytes
            self.item_count -= 1
            
            # Remove metadata
            del self.metadata[key]
            
            # Remove from access order
            if key in self.access_order:
                self.access_order.remove(key)
            
            return True
        
        return False
    
    async def _check_eviction(self) -> None:
        """Check if items need to be evicted based on constraints."""
        # First check TTL for all items
        expired_keys = [k for k, v in self.metadata.items() if v.is_expired()]
        for key in expired_keys:
            await self._evict_item(key)
        
        # Continue eviction if we're still over limits
        while (self.item_count > self.max_items or 
               self.current_size_bytes > self.max_size_bytes):
            
            if len(self.metadata) == 0:
                break
            
            # Choose the item to evict based on the strategy
            if self.strategy == CacheStrategy.LRU:
                # Least Recently Used
                if self.access_order:
                    key_to_evict = self.access_order[0]  # Oldest accessed item
                else:
                    key_to_evict = next(iter(self.metadata))
            
            elif self.strategy == CacheStrategy.LFU:
                # Least Frequently Used
                key_to_evict = min(self.metadata.items(), 
                                 key=lambda x: x[1].access_count)[0]
            
            elif self.strategy == CacheStrategy.FIFO:
                # First In First Out
                if self.access_order:
                    key_to_evict = self.access_order[0]  # Oldest item
                else:
                    key_to_evict = next(iter(self.metadata))
            
            elif self.strategy == CacheStrategy.SIZE_BASED:
                # Largest items first
                key_to_evict = max(self.metadata.items(),
                                  key=lambda x: x[1].size_bytes)[0]
            
            else:
                # Default to first item
                key_to_evict = next(iter(self.metadata))
            
            # Evict the chosen item
            await self._evict_item(key_to_evict)
    
    async def clear(self) -> None:
        """Clear all items from the cache."""
        async with self.lock:
            self.metadata.clear()
            self.access_order.clear()
            self.current_size_bytes = 0
            self.item_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache."""
        return {
            "strategy": self.strategy.value,
            "item_count": self.item_count,
            "max_items": self.max_items,
            "current_size_mb": self.current_size_bytes / (1024 * 1024),
            "max_size_mb": self.max_size_bytes / (1024 * 1024),
            "memory_usage_percent": (self.current_size_bytes / self.max_size_bytes) * 100 if self.max_size_bytes > 0 else 0
        }


class StateManager:
    """
    Manages state storage and retrieval for agents.
    
    This class provides a unified interface for storing and retrieving state data,
    handling serialization, compression, and persistence across different storage backends.
    """
    
    def __init__(self, 
                storage_type: StorageType = StorageType.MEMORY,
                base_dir: Optional[str] = None,
                prefix: str = "agent_state",
                compression: bool = True,
                memory_optimizer: Optional[MemoryOptimizer] = None):
        """
        Initialize the state manager.
        
        Args:
            storage_type: Type of storage to use
            base_dir: Base directory for file storage (if applicable)
            prefix: Prefix for state files
            compression: Whether to compress stored data
            memory_optimizer: Custom memory optimizer (or None to use default)
        """
        self.storage_type = storage_type
        self.base_dir = base_dir
        self.prefix = prefix
        self.compression = compression
        
        # Set up the appropriate storage directory
        if storage_type == StorageType.FILE:
            if base_dir is None:
                self.base_dir = os.path.join(tempfile.gettempdir(), "amptalk_state")
            
            os.makedirs(self.base_dir, exist_ok=True)
            logger.info(f"Using file storage at {self.base_dir}")
        
        # Set up memory cache
        self.memory_data: Dict[str, Any] = {}
        self.memory_optimizer = memory_optimizer or MemoryOptimizer()
        
        # Lock for thread safety
        self.lock = asyncio.Lock()
        
        logger.info(f"Initialized StateManager with {storage_type.value} storage")
    
    def _get_file_path(self, state_id: str) -> str:
        """
        Get the file path for a state ID.
        
        Args:
            state_id: The state identifier
            
        Returns:
            Full path to the state file
        """
        safe_id = state_id.replace("/", "_").replace(":", "_")
        return os.path.join(self.base_dir, f"{self.prefix}_{safe_id}.state")
    
    async def save_state(self, state_id: str, data: Any) -> None:
        """
        Save state data.
        
        Args:
            state_id: Unique identifier for the state
            data: The data to store
        """
        async with self.lock:
            # Always keep a copy in memory cache
            self.memory_data[state_id] = data
            
            # Update memory cache metadata
            await self.memory_optimizer.cache_item(state_id, data)
            
            # Handle persistence based on storage type
            if self.storage_type == StorageType.FILE:
                await self._save_to_file(state_id, data)
            
            # Other storage types would be handled here
    
    async def _save_to_file(self, state_id: str, data: Any) -> None:
        """
        Save state data to a file.
        
        Args:
            state_id: Unique identifier for the state
            data: The data to store
        """
        file_path = self._get_file_path(state_id)
        
        try:
            # Serialize the data
            if isinstance(data, (dict, list, str, int, float, bool, type(None))):
                # Use JSON for simple types
                serialized = json.dumps(data).encode('utf-8')
            else:
                # Use pickle for complex objects
                serialized = pickle.dumps(data)
            
            # Compress if needed
            if self.compression:
                serialized = gzip.compress(serialized)
            
            # Write to file
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(serialized)
            
            logger.debug(f"Saved state {state_id} to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save state {state_id} to file: {str(e)}")
            raise
    
    async def load_state(self, state_id: str) -> Optional[Any]:
        """
        Load state data.
        
        Args:
            state_id: Unique identifier for the state
            
        Returns:
            The loaded state data, or None if not found
        """
        async with self.lock:
            # First check memory cache
            if state_id in self.memory_data:
                # Update access metadata
                await self.memory_optimizer.get_item(state_id)
                return self.memory_data[state_id]
            
            # If not in memory, check persistent storage
            if self.storage_type == StorageType.FILE:
                data = await self._load_from_file(state_id)
                if data is not None:
                    # Update memory cache
                    self.memory_data[state_id] = data
                    await self.memory_optimizer.cache_item(state_id, data)
                return data
            
            return None
    
    async def _load_from_file(self, state_id: str) -> Optional[Any]:
        """
        Load state data from a file.
        
        Args:
            state_id: Unique identifier for the state
            
        Returns:
            The loaded state data, or None if not found
        """
        file_path = self._get_file_path(state_id)
        
        if not os.path.exists(file_path):
            return None
        
        try:
            # Read from file
            async with aiofiles.open(file_path, 'rb') as f:
                serialized = await f.read()
            
            # Decompress if needed
            if self.compression:
                try:
                    serialized = gzip.decompress(serialized)
                except Exception as e:
                    logger.warning(f"Failed to decompress state {state_id}: {str(e)}")
                    # Continue with the raw data
            
            # Deserialize
            try:
                # First try JSON
                data = json.loads(serialized.decode('utf-8'))
            except:
                # Fall back to pickle
                data = pickle.loads(serialized)
            
            logger.debug(f"Loaded state {state_id} from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load state {state_id} from file: {str(e)}")
            return None
    
    async def delete_state(self, state_id: str) -> bool:
        """
        Delete state data.
        
        Args:
            state_id: Unique identifier for the state
            
        Returns:
            True if the state was deleted, False otherwise
        """
        async with self.lock:
            # Remove from memory cache
            if state_id in self.memory_data:
                del self.memory_data[state_id]
                await self.memory_optimizer.remove_item(state_id)
            
            # Remove from persistent storage
            if self.storage_type == StorageType.FILE:
                return await self._delete_from_file(state_id)
            
            return True
    
    async def _delete_from_file(self, state_id: str) -> bool:
        """
        Delete state data from a file.
        
        Args:
            state_id: Unique identifier for the state
            
        Returns:
            True if the state was deleted, False otherwise
        """
        file_path = self._get_file_path(state_id)
        
        if not os.path.exists(file_path):
            return False
        
        try:
            os.remove(file_path)
            logger.debug(f"Deleted state {state_id} from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete state {state_id} from file: {str(e)}")
            return False
    
    async def clear_all(self) -> bool:
        """
        Clear all state data.
        
        Returns:
            True if all state was cleared, False otherwise
        """
        async with self.lock:
            # Clear memory cache
            self.memory_data.clear()
            await self.memory_optimizer.clear()
            
            # Clear persistent storage
            if self.storage_type == StorageType.FILE:
                return await self._clear_all_files()
            
            return True
    
    async def _clear_all_files(self) -> bool:
        """
        Clear all state files.
        
        Returns:
            True if all files were cleared, False otherwise
        """
        try:
            for filename in os.listdir(self.base_dir):
                if filename.startswith(f"{self.prefix}_") and filename.endswith(".state"):
                    file_path = os.path.join(self.base_dir, filename)
                    os.remove(file_path)
            
            logger.debug(f"Cleared all state files from {self.base_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear all state files: {str(e)}")
            return False
    
    async def list_states(self) -> List[str]:
        """
        List all available state IDs.
        
        Returns:
            List of state IDs
        """
        async with self.lock:
            states = set(self.memory_data.keys())
            
            # Add states from persistent storage
            if self.storage_type == StorageType.FILE:
                file_states = await self._list_file_states()
                states.update(file_states)
            
            return list(states)
    
    async def _list_file_states(self) -> Set[str]:
        """
        List all state IDs from files.
        
        Returns:
            Set of state IDs
        """
        states = set()
        
        try:
            prefix_len = len(f"{self.prefix}_")
            suffix_len = len(".state")
            
            for filename in os.listdir(self.base_dir):
                if filename.startswith(f"{self.prefix}_") and filename.endswith(".state"):
                    # Extract state ID from filename
                    state_id = filename[prefix_len:-suffix_len]
                    states.add(state_id)
        except Exception as e:
            logger.error(f"Failed to list state files: {str(e)}")
        
        return states
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the state manager.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            "storage_type": self.storage_type.value,
            "memory_cache_items": len(self.memory_data),
            "memory_optimizer": self.memory_optimizer.get_stats()
        }
        
        if self.storage_type == StorageType.FILE:
            try:
                state_files = [f for f in os.listdir(self.base_dir) 
                             if f.startswith(f"{self.prefix}_") and f.endswith(".state")]
                stats["file_count"] = len(state_files)
                
                # Calculate total file size
                total_size = sum(os.path.getsize(os.path.join(self.base_dir, f)) for f in state_files)
                stats["total_file_size_mb"] = total_size / (1024 * 1024)
            except Exception as e:
                logger.error(f"Failed to get file stats: {str(e)}")
        
        return stats
    
    @asynccontextmanager
    async def snapshot(self, agent_id: str) -> Dict[str, Any]:
        """
        Create a snapshot of an agent's state that can be modified and saved back.
        
        Args:
            agent_id: The ID of the agent
            
        Yields:
            A dictionary of state data that can be modified
        """
        state_id = f"{agent_id}_state"
        
        # Load existing state or start with empty dict
        state_data = await self.load_state(state_id) or {}
        
        try:
            # Yield the state for modification
            yield state_data
        finally:
            # Save the modified state back
            await self.save_state(state_id, state_data)


# Global utility functions

async def create_state_manager(
    agent_id: str,
    storage_type: StorageType = StorageType.FILE,
    base_dir: Optional[str] = None,
    max_memory_mb: float = 100
) -> StateManager:
    """
    Create a configured state manager for an agent.
    
    Args:
        agent_id: ID of the agent
        storage_type: Type of storage to use
        base_dir: Base directory for file storage
        max_memory_mb: Maximum memory usage in MB
        
    Returns:
        Configured StateManager instance
    """
    # Create memory optimizer with reasonable defaults
    memory_optimizer = MemoryOptimizer(
        strategy=CacheStrategy.LRU,
        max_items=1000,
        max_size_mb=max_memory_mb,
        default_ttl=3600  # 1 hour default TTL
    )
    
    # Create state manager
    manager = StateManager(
        storage_type=storage_type,
        base_dir=base_dir,
        prefix=f"agent_{agent_id}",
        compression=True,
        memory_optimizer=memory_optimizer
    )
    
    return manager 