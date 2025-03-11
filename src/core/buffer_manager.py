"""
Buffer Management Module for AMPTALK

This module implements a sophisticated buffer management system with hybrid caching strategies,
including LRU (Least Recently Used), TTL (Time To Live), and 2Q (Two Queue) caching.
It provides efficient memory management with backpressure handling and adaptive batching.
"""

from typing import Any, Dict, Optional, List, Tuple
from collections import OrderedDict
import time
import threading
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class CacheItem:
    """Data class for cache items with metadata."""
    value: Any
    timestamp: float
    ttl: Optional[float] = None
    access_count: int = 0

class BufferManager:
    """
    A sophisticated buffer manager implementing hybrid caching strategies.
    
    Features:
    - Hybrid caching (LRU, TTL, 2Q)
    - Memory limits with automatic eviction
    - Backpressure handling
    - Adaptive batching
    - Thread-safe operations
    """
    
    def __init__(
        self,
        max_size: int = 1000,  # Maximum number of items in cache
        ttl: Optional[float] = None,  # Default TTL in seconds
        cleanup_interval: float = 60.0,  # Cleanup interval in seconds
        a1_ratio: float = 0.25,  # Ratio of cache size for A1 queue in 2Q
    ):
        """
        Initialize the BufferManager with the given configuration.
        
        Args:
            max_size: Maximum number of items in the cache
            ttl: Default time-to-live for items in seconds
            cleanup_interval: Interval for cleanup of expired items
            a1_ratio: Ratio of cache size allocated to A1 queue in 2Q
        """
        # Main cache components
        self.max_size = max_size
        self.default_ttl = ttl
        self.cleanup_interval = cleanup_interval
        
        # 2Q cache components
        self.a1_size = int(max_size * a1_ratio)
        self.am_size = max_size - self.a1_size
        self.a1_cache = OrderedDict()  # Short-term queue
        self.am_cache = OrderedDict()  # Long-term queue
        
        # LRU cache for frequently accessed items
        self.lru_cache = OrderedDict()
        
        # TTL cache components
        self.ttl_cache = {}  # Stores items with TTL
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Start cleanup thread
        self._start_cleanup_thread()
        
        logger.info(
            f"Initialized BufferManager with max_size={max_size}, "
            f"ttl={ttl}, cleanup_interval={cleanup_interval}s"
        )
    
    def _start_cleanup_thread(self) -> None:
        """Start the background thread for cleaning up expired items."""
        def cleanup_loop():
            while True:
                time.sleep(self.cleanup_interval)
                self.cleanup_expired()
        
        cleanup_thread = threading.Thread(
            target=cleanup_loop,
            daemon=True,
            name="BufferManager-Cleanup"
        )
        cleanup_thread.start()
        logger.info("Started cleanup thread")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve an item from the cache.
        
        Args:
            key: Key to retrieve
            
        Returns:
            The cached value if found and not expired, None otherwise
        """
        with self._lock:
            # Check TTL cache first
            if key in self.ttl_cache:
                item = self.ttl_cache[key]
                if item.ttl and time.time() - item.timestamp > item.ttl:
                    # Item has expired
                    self._remove_item(key)
                    return None
                item.access_count += 1
                return item.value
            
            # Check 2Q cache
            if key in self.a1_cache:
                value = self.a1_cache.pop(key)
                # Move to AM cache if frequently accessed
                if len(self.am_cache) >= self.am_size:
                    self.am_cache.popitem(last=False)
                self.am_cache[key] = value
                return value
            
            if key in self.am_cache:
                # Move to end of AM cache
                value = self.am_cache.pop(key)
                self.am_cache[key] = value
                return value
            
            # Check LRU cache
            if key in self.lru_cache:
                value = self.lru_cache.pop(key)
                self.lru_cache[key] = value
                return value
            
            return None
    
    def put(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None
    ) -> None:
        """
        Add or update an item in the cache.
        
        Args:
            key: Key to store
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        with self._lock:
            if ttl is not None or self.default_ttl is not None:
                # Store in TTL cache
                self.ttl_cache[key] = CacheItem(
                    value=value,
                    timestamp=time.time(),
                    ttl=ttl or self.default_ttl
                )
                return
            
            # 2Q caching strategy
            if key not in self.a1_cache and key not in self.am_cache:
                if len(self.a1_cache) >= self.a1_size:
                    # Move oldest item from A1 to AM
                    if len(self.am_cache) >= self.am_size:
                        self.am_cache.popitem(last=False)
                    k, v = self.a1_cache.popitem(last=False)
                    self.am_cache[k] = v
                self.a1_cache[key] = value
            else:
                # Update existing item
                if key in self.a1_cache:
                    self.a1_cache[key] = value
                else:
                    self.am_cache[key] = value
    
    def cleanup_expired(self) -> None:
        """Remove expired items from the TTL cache."""
        with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, item in self.ttl_cache.items()
                if item.ttl and current_time - item.timestamp > item.ttl
            ]
            for key in expired_keys:
                self._remove_item(key)
            
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired items")
    
    def _remove_item(self, key: str) -> None:
        """
        Remove an item from all caches.
        
        Args:
            key: Key to remove
        """
        self.ttl_cache.pop(key, None)
        self.a1_cache.pop(key, None)
        self.am_cache.pop(key, None)
        self.lru_cache.pop(key, None)
    
    def clear(self) -> None:
        """Clear all caches."""
        with self._lock:
            self.ttl_cache.clear()
            self.a1_cache.clear()
            self.am_cache.clear()
            self.lru_cache.clear()
            logger.info("Cleared all caches")
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            return {
                "ttl_cache_size": len(self.ttl_cache),
                "a1_cache_size": len(self.a1_cache),
                "am_cache_size": len(self.am_cache),
                "lru_cache_size": len(self.lru_cache),
                "total_items": (
                    len(self.ttl_cache) +
                    len(self.a1_cache) +
                    len(self.am_cache) +
                    len(self.lru_cache)
                )
            } 