"""
Model Caching Utility for AMPTALK.

This module provides a caching mechanism for AI models to optimize loading times
and memory usage. It maintains a pool of loaded models for reuse across agents.

Author: AMPTALK Team
Date: 2024
"""

import logging
import threading
import time
from typing import Dict, Any, Optional, Tuple, Callable, List
import os
import gc
import weakref

from src.core.utils.logging_config import get_logger

# Configure logger
logger = get_logger("amptalk.utils.model_cache")


class ModelCache:
    """
    Cache for AI models to optimize loading and memory usage.
    
    This class implements a singleton pattern to provide a centralized
    cache for models across the application. It supports:
    - Model reuse across multiple agents
    - Automatic unloading of unused models
    - Memory usage tracking
    - Cache size limits
    """
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        """Implement singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelCache, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """Initialize the model cache."""
        # Skip re-initialization if already initialized
        if self._initialized:
            return
            
        # Cache of models with their metadata
        self._cache: Dict[str, Dict[str, Any]] = {}
        
        # Cache usage statistics
        self._hits = 0
        self._misses = 0
        
        # Configuration
        self._max_models = int(os.environ.get("AMPTALK_MODEL_CACHE_SIZE", "5"))
        self._ttl_seconds = int(os.environ.get("AMPTALK_MODEL_CACHE_TTL", "3600"))  # 1 hour
        self._memory_limit_mb = int(os.environ.get("AMPTALK_MODEL_CACHE_MEMORY_LIMIT", "0"))  # 0 = unlimited
        
        # Internal state
        self._current_memory_usage = 0
        self._last_cleanup_time = time.time()
        
        # References to model owners for automatic cleanup
        self._model_owners: Dict[str, List[weakref.ref]] = {}
        
        self._initialized = True
        logger.info(f"Model cache initialized with max_models={self._max_models}, "
                    f"ttl={self._ttl_seconds}s, memory_limit={self._memory_limit_mb}MB")
    
    def configure(self, max_models: Optional[int] = None, 
                 ttl_seconds: Optional[int] = None,
                 memory_limit_mb: Optional[int] = None) -> None:
        """
        Configure the cache parameters.
        
        Args:
            max_models: Maximum number of models to keep in cache
            ttl_seconds: Time-to-live for cached models in seconds
            memory_limit_mb: Memory limit for the cache in MB
        """
        with self._lock:
            if max_models is not None:
                self._max_models = max_models
            
            if ttl_seconds is not None:
                self._ttl_seconds = ttl_seconds
            
            if memory_limit_mb is not None:
                self._memory_limit_mb = memory_limit_mb
            
            logger.info(f"Model cache reconfigured: max_models={self._max_models}, "
                        f"ttl={self._ttl_seconds}s, memory_limit={self._memory_limit_mb}MB")
    
    def get_model(self, model_key: str, owner: Optional[Any] = None) -> Optional[Any]:
        """
        Get a model from the cache.
        
        Args:
            model_key: Unique key identifying the model
            owner: Optional reference to the owner of the model
            
        Returns:
            The cached model instance or None if not found
        """
        with self._lock:
            # Check if model is in cache
            if model_key in self._cache:
                entry = self._cache[model_key]
                
                # Update last access time
                entry["last_access"] = time.time()
                entry["access_count"] += 1
                
                # Register owner if provided
                if owner is not None:
                    self._register_owner(model_key, owner)
                
                self._hits += 1
                logger.debug(f"Cache hit for model: {model_key} (hits: {self._hits}, misses: {self._misses})")
                return entry["model"]
            
            self._misses += 1
            logger.debug(f"Cache miss for model: {model_key} (hits: {self._hits}, misses: {self._misses})")
            return None
    
    def _register_owner(self, model_key: str, owner: Any) -> None:
        """Register an owner for a model for automatic cleanup."""
        if model_key not in self._model_owners:
            self._model_owners[model_key] = []
        
        # Check if owner is already registered
        for ref in self._model_owners[model_key]:
            if ref() is owner:
                return
        
        # Add weak reference to owner
        self._model_owners[model_key].append(weakref.ref(
            owner, 
            lambda _: self._owner_finalized(model_key, owner)
        ))
    
    def _owner_finalized(self, model_key: str, owner: Any) -> None:
        """Handle owner object being garbage collected."""
        with self._lock:
            if model_key in self._model_owners:
                # Remove references to this owner
                self._model_owners[model_key] = [
                    ref for ref in self._model_owners[model_key] 
                    if ref() is not None and ref() is not owner
                ]
                
                # If no more owners, schedule model for unloading
                if not self._model_owners[model_key]:
                    logger.debug(f"No more owners for model: {model_key}, scheduling for unloading")
                    # Mark model for potential unloading
                    if model_key in self._cache:
                        self._cache[model_key]["orphaned"] = True
    
    def put_model(self, model_key: str, model: Any, 
                 size_mb: Optional[float] = None,
                 owner: Optional[Any] = None,
                 unload_fn: Optional[Callable[[Any], None]] = None) -> None:
        """
        Add a model to the cache.
        
        Args:
            model_key: Unique key identifying the model
            model: The model instance to cache
            size_mb: Estimated memory size of the model in MB
            owner: Optional reference to the owner of the model
            unload_fn: Optional function to call when unloading the model
        """
        with self._lock:
            # Make room in the cache if needed
            self._ensure_capacity(size_mb)
            
            # Add or update the model in cache
            now = time.time()
            
            if model_key in self._cache:
                # Update existing entry
                self._cache[model_key].update({
                    "model": model,
                    "last_access": now,
                    "access_count": self._cache[model_key].get("access_count", 0) + 1,
                    "size_mb": size_mb,
                    "unload_fn": unload_fn,
                    "orphaned": False
                })
                logger.debug(f"Updated existing model in cache: {model_key}")
            else:
                # Create new entry
                self._cache[model_key] = {
                    "model": model,
                    "created": now,
                    "last_access": now,
                    "access_count": 1,
                    "size_mb": size_mb,
                    "unload_fn": unload_fn,
                    "orphaned": False
                }
                logger.debug(f"Added new model to cache: {model_key}")
            
            # Register owner if provided
            if owner is not None:
                self._register_owner(model_key, owner)
            
            # Update memory usage
            if size_mb is not None:
                self._current_memory_usage += size_mb
                logger.debug(f"Current cache memory usage: {self._current_memory_usage:.2f}MB")
            
            # Periodically clean up
            if now - self._last_cleanup_time > 300:  # Every 5 minutes
                self._cleanup()
    
    def _ensure_capacity(self, new_model_size_mb: Optional[float] = None) -> None:
        """
        Ensure there's enough capacity for a new model.
        
        Args:
            new_model_size_mb: Size of the new model in MB
        """
        # If no size limit or model has no size, just check count
        if (self._memory_limit_mb <= 0 or new_model_size_mb is None) and len(self._cache) < self._max_models:
            return
            
        # Check memory limit if specified
        if self._memory_limit_mb > 0 and new_model_size_mb is not None:
            # If adding this model would exceed memory limit, evict models
            while self._current_memory_usage + new_model_size_mb > self._memory_limit_mb and self._cache:
                self._evict_model()
        
        # Check model count limit
        while len(self._cache) >= self._max_models and self._cache:
            self._evict_model()
    
    def _evict_model(self) -> None:
        """Evict the least recently used model from the cache."""
        if not self._cache:
            return
            
        # First try to evict orphaned models
        orphaned_models = [(k, v) for k, v in self._cache.items() if v.get("orphaned", False)]
        if orphaned_models:
            # Evict the least recently accessed orphaned model
            model_key = min(orphaned_models, key=lambda x: x[1]["last_access"])[0]
        else:
            # Evict the least recently accessed model
            model_key = min(self._cache.items(), key=lambda x: x[1]["last_access"])[0]
        
        self._unload_model(model_key)
    
    def _unload_model(self, model_key: str) -> None:
        """
        Unload a model from the cache.
        
        Args:
            model_key: Key of the model to unload
        """
        if model_key not in self._cache:
            return
            
        entry = self._cache[model_key]
        
        # Call custom unload function if provided
        if entry.get("unload_fn") is not None:
            try:
                entry["unload_fn"](entry["model"])
            except Exception as e:
                logger.warning(f"Error unloading model {model_key}: {e}")
        
        # Update memory usage
        if entry.get("size_mb") is not None:
            self._current_memory_usage -= entry["size_mb"]
        
        # Remove from cache
        del self._cache[model_key]
        
        # Remove from owners
        if model_key in self._model_owners:
            del self._model_owners[model_key]
        
        logger.info(f"Unloaded model from cache: {model_key}")
        
        # Force garbage collection to reclaim memory
        gc.collect()
    
    def unload_model(self, model_key: str) -> bool:
        """
        Explicitly unload a model from the cache.
        
        Args:
            model_key: Key of the model to unload
            
        Returns:
            True if the model was unloaded, False otherwise
        """
        with self._lock:
            if model_key in self._cache:
                self._unload_model(model_key)
                return True
            return False
    
    def _cleanup(self) -> None:
        """
        Clean up expired models from the cache.
        """
        now = time.time()
        
        # Identify expired models
        expired_models = []
        for model_key, entry in self._cache.items():
            # Check TTL expiration
            if now - entry["last_access"] > self._ttl_seconds:
                expired_models.append(model_key)
            # Check if orphaned and older than 5 minutes
            elif entry.get("orphaned", False) and now - entry["last_access"] > 300:
                expired_models.append(model_key)
        
        # Unload expired models
        for model_key in expired_models:
            logger.debug(f"Unloading expired model: {model_key}")
            self._unload_model(model_key)
        
        self._last_cleanup_time = now
    
    def clear(self) -> None:
        """
        Clear all models from the cache.
        """
        with self._lock:
            # Make a copy of keys to avoid modification during iteration
            model_keys = list(self._cache.keys())
            
            for model_key in model_keys:
                self._unload_model(model_key)
            
            # Reset counters
            self._hits = 0
            self._misses = 0
            self._current_memory_usage = 0
            
            logger.info("Model cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            return {
                "model_count": len(self._cache),
                "memory_usage_mb": self._current_memory_usage,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0,
                "max_models": self._max_models,
                "memory_limit_mb": self._memory_limit_mb,
                "ttl_seconds": self._ttl_seconds,
                "models": {
                    key: {
                        "created": entry["created"],
                        "last_access": entry["last_access"],
                        "access_count": entry["access_count"],
                        "size_mb": entry["size_mb"],
                        "orphaned": entry.get("orphaned", False),
                        "age_seconds": time.time() - entry["created"],
                        "idle_seconds": time.time() - entry["last_access"]
                    }
                    for key, entry in self._cache.items()
                }
            }


# Create the singleton instance
model_cache = ModelCache()


def get_model_cache() -> ModelCache:
    """
    Get the singleton model cache instance.
    
    Returns:
        The global ModelCache instance
    """
    return model_cache 