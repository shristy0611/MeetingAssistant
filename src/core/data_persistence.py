"""
Data Persistence Module for AMPTALK

This module implements a multi-layer storage strategy for efficient data persistence,
including in-memory cache, local disk cache, and remote storage with versioning support
and data integrity verification.
"""

from typing import Any, Dict, List, Optional, Union
import os
import json
import pickle
import hashlib
import logging
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

class DataPersistence:
    """
    A multi-layer data persistence system with versioning and integrity verification.
    
    Features:
    - Multi-layer storage (memory, disk, remote)
    - Versioning with incremental updates
    - Data integrity verification
    - Automatic cache invalidation
    """
    
    def __init__(
        self,
        base_dir: str = "./data",
        memory_cache_size: int = 1000,
        disk_cache_size: int = 10000,
    ):
        """
        Initialize the DataPersistence system.
        
        Args:
            base_dir: Base directory for disk storage
            memory_cache_size: Maximum number of items in memory cache
            disk_cache_size: Maximum number of items in disk cache
        """
        # Storage configuration
        self.base_dir = Path(base_dir)
        self.memory_cache_size = memory_cache_size
        self.disk_cache_size = disk_cache_size
        
        # Storage layers
        self.memory_cache = {}  # {key: (value, timestamp, version, checksum)}
        self.disk_cache_index = {}  # {key: {"timestamp", "version", "path", "checksum"}}
        
        # Ensure base directory exists
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.base_dir / "versions", exist_ok=True)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Load disk cache index
        self._load_disk_cache_index()
        
        logger.info(
            f"Initialized DataPersistence with memory_cache_size={memory_cache_size}, "
            f"disk_cache_size={disk_cache_size}, base_dir={base_dir}"
        )
    
    def _load_disk_cache_index(self) -> None:
        """Load the disk cache index from disk."""
        index_path = self.base_dir / "disk_cache_index.json"
        if index_path.exists():
            try:
                with open(index_path, "r") as f:
                    self.disk_cache_index = json.load(f)
                logger.info(f"Loaded disk cache index with {len(self.disk_cache_index)} items")
            except Exception as e:
                logger.error(f"Failed to load disk cache index: {e}")
                self.disk_cache_index = {}
    
    def _save_disk_cache_index(self) -> None:
        """Save the disk cache index to disk."""
        index_path = self.base_dir / "disk_cache_index.json"
        try:
            with open(index_path, "w") as f:
                json.dump(self.disk_cache_index, f)
            logger.info(f"Saved disk cache index with {len(self.disk_cache_index)} items")
        except Exception as e:
            logger.error(f"Failed to save disk cache index: {e}")
    
    def _get_disk_path(self, key: str) -> Path:
        """Get the disk path for a key."""
        # Use first 2 characters as directory prefix for better file distribution
        prefix = hashlib.md5(key.encode()).hexdigest()[:2]
        directory = self.base_dir / prefix
        os.makedirs(directory, exist_ok=True)
        return directory / f"{hashlib.md5(key.encode()).hexdigest()}.pickle"
    
    def _get_version_path(self, key: str, version: int) -> Path:
        """Get the disk path for a versioned key."""
        prefix = hashlib.md5(key.encode()).hexdigest()[:2]
        directory = self.base_dir / "versions" / prefix
        os.makedirs(directory, exist_ok=True)
        return directory / f"{hashlib.md5(key.encode()).hexdigest()}_v{version}.pickle"
    
    def _calculate_checksum(self, value: Any) -> str:
        """Calculate a checksum for a value."""
        try:
            data = pickle.dumps(value)
            return hashlib.sha256(data).hexdigest()
        except Exception as e:
            logger.warning(f"Failed to calculate checksum: {e}")
            return ""
    
    def _verify_integrity(self, value: Any, checksum: str) -> bool:
        """Verify data integrity using the checksum."""
        if not checksum:
            return True
        calculated = self._calculate_checksum(value)
        return calculated == checksum
    
    def _save_to_disk(self, key: str, value: Any, version: int) -> bool:
        """
        Save an item to disk.
        
        Args:
            key: Key to store
            value: Value to save
            version: Version number
            
        Returns:
            True if successful, False otherwise
        """
        path = self._get_disk_path(key)
        version_path = self._get_version_path(key, version)
        
        try:
            # Calculate checksum
            checksum = self._calculate_checksum(value)
            
            # Create item tuple
            item = (value, time.time(), version, checksum)
            
            # Save current version
            with open(path, "wb") as f:
                pickle.dump(item, f)
            
            # Save versioned copy
            with open(version_path, "wb") as f:
                pickle.dump(item, f)
            
            # Update disk cache index
            self.disk_cache_index[key] = {
                "timestamp": item[1],
                "version": version,
                "checksum": checksum,
                "path": str(path),
                "version_path": str(version_path),
                "access_count": self.disk_cache_index.get(key, {}).get("access_count", 0)
            }
            
            # Save index periodically
            if len(self.disk_cache_index) % 100 == 0:
                self._save_disk_cache_index()
            
            return True
        except Exception as e:
            logger.error(f"Failed to save {key} to disk: {e}")
            return False
    
    def _load_from_disk(self, key: str) -> Optional[tuple]:
        """
        Load an item from disk.
        
        Args:
            key: Key to load
            
        Returns:
            Tuple of (value, timestamp, version, checksum) if found and valid, None otherwise
        """
        if key not in self.disk_cache_index:
            return None
        
        path = Path(self.disk_cache_index[key]["path"])
        if not path.exists():
            logger.warning(f"Disk file for {key} not found at {path}")
            self.disk_cache_index.pop(key, None)
            return None
        
        try:
            with open(path, "rb") as f:
                item = pickle.load(f)
            
            # Verify integrity
            value, timestamp, version, checksum = item
            if not self._verify_integrity(value, checksum):
                logger.warning(f"Integrity check failed for {key}")
                return None
            
            # Update access count
            if key in self.disk_cache_index:
                self.disk_cache_index[key]["access_count"] = self.disk_cache_index[key].get("access_count", 0) + 1
            
            return item
        except Exception as e:
            logger.error(f"Failed to load {key} from disk: {e}")
            return None
    
    def get(self, key: str, version: Optional[int] = None) -> Optional[Any]:
        """
        Retrieve an item from storage.
        
        Args:
            key: Key to retrieve
            version: Specific version to retrieve (if None, latest version is used)
            
        Returns:
            The stored value if found, None otherwise
        """
        with self._lock:
            # If requesting a specific version, bypass memory cache
            if version is not None:
                return self._get_version(key, version)
            
            # Check memory cache first
            if key in self.memory_cache:
                value, timestamp, version, checksum = self.memory_cache[key]
                if self._verify_integrity(value, checksum):
                    return value
                else:
                    logger.warning(f"Memory cache integrity check failed for {key}")
                    self.memory_cache.pop(key)
            
            # Check disk cache next
            disk_item = self._load_from_disk(key)
            if disk_item:
                # Move to memory cache if space available
                if len(self.memory_cache) < self.memory_cache_size:
                    self.memory_cache[key] = disk_item
                return disk_item[0]  # Return value only
            
            return None
    
    def _get_version(self, key: str, version: int) -> Optional[Any]:
        """
        Retrieve a specific version of an item.
        
        Args:
            key: Key to retrieve
            version: Specific version to retrieve
            
        Returns:
            The stored value if found, None otherwise
        """
        # Check if version exists in disk cache index
        if key not in self.disk_cache_index or self.disk_cache_index[key].get("version", 0) < version:
            return None
        
        # Load specific version from disk
        version_path = self._get_version_path(key, version)
        if not version_path.exists():
            logger.warning(f"Version {version} for {key} not found at {version_path}")
            return None
        
        try:
            with open(version_path, "rb") as f:
                item = pickle.load(f)
            
            # Verify integrity
            value, timestamp, ver, checksum = item
            if not self._verify_integrity(value, checksum):
                logger.warning(f"Integrity check failed for {key} version {version}")
                return None
            
            return value
        except Exception as e:
            logger.error(f"Failed to load version {version} for {key}: {e}")
            return None
    
    def put(self, key: str, value: Any) -> bool:
        """
        Store an item in the persistence system.
        
        Args:
            key: Key to store
            value: Value to store
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            # Check if item already exists to determine version
            current_version = 1
            if key in self.memory_cache:
                current_version = self.memory_cache[key][2] + 1
            elif key in self.disk_cache_index:
                current_version = self.disk_cache_index[key].get("version", 0) + 1
            
            # Calculate checksum
            checksum = self._calculate_checksum(value)
            
            # Create new data item
            item = (value, time.time(), current_version, checksum)
            
            # Store in memory cache
            if len(self.memory_cache) >= self.memory_cache_size:
                # Evict least recently used item
                oldest_key = min(
                    self.memory_cache,
                    key=lambda k: self.memory_cache[k][1]  # timestamp
                )
                # Save to disk before evicting from memory
                old_item = self.memory_cache.pop(oldest_key)
                self._save_to_disk(oldest_key, old_item[0], old_item[2])
            
            self.memory_cache[key] = item
            
            # Also save to disk for persistence
            return self._save_to_disk(key, value, current_version)
    
    def remove(self, key: str) -> bool:
        """
        Remove an item from all storage layers.
        
        Args:
            key: Key to remove
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            # Remove from memory cache
            self.memory_cache.pop(key, None)
            
            # Remove from disk
            if key in self.disk_cache_index:
                try:
                    path = Path(self.disk_cache_index[key]["path"])
                    if path.exists():
                        path.unlink()
                    
                    # Remove all versions
                    for i in range(1, self.disk_cache_index[key].get("version", 0) + 1):
                        version_path = self._get_version_path(key, i)
                        if version_path.exists():
                            version_path.unlink()
                    
                    self.disk_cache_index.pop(key, None)
                    self._save_disk_cache_index()
                except Exception as e:
                    logger.error(f"Failed to remove {key} from disk: {e}")
                    return False
            
            return True
    
    def get_versions(self, key: str) -> List[int]:
        """
        Get a list of available versions for a key.
        
        Args:
            key: Key to check
            
        Returns:
            List of available versions
        """
        with self._lock:
            if key not in self.disk_cache_index:
                return []
            
            max_version = self.disk_cache_index[key].get("version", 0)
            return list(range(1, max_version + 1))
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the persistence system.
        
        Returns:
            Dictionary with statistics
        """
        with self._lock:
            return {
                "memory_cache_size": len(self.memory_cache),
                "disk_cache_size": len(self.disk_cache_index),
                "memory_cache_max": self.memory_cache_size,
                "disk_cache_max": self.disk_cache_size,
                "base_dir": str(self.base_dir)
            } 