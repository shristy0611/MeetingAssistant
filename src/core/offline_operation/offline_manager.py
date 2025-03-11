"""
Offline Operation Management System.

This module handles offline operation capabilities for the AMPTALK system,
ensuring functionality without constant internet connectivity.

Author: AMPTALK Team
Date: 2024
"""

import os
import json
import time
import sqlite3
import logging
import asyncio
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from src.core.utils.logging_config import get_logger

logger = get_logger(__name__)

class SyncStatus(Enum):
    """Synchronization status."""
    SYNCED = "synced"
    PENDING = "pending"
    CONFLICT = "conflict"
    ERROR = "error"

class OperationMode(Enum):
    """System operation modes."""
    ONLINE = "online"
    OFFLINE = "offline"
    HYBRID = "hybrid"  # Partial online functionality

@dataclass
class SyncInfo:
    """Synchronization information."""
    component: str
    last_sync: datetime
    status: SyncStatus
    pending_changes: int
    data_size_bytes: int
    priority: int  # Lower is higher priority

class OfflineManager:
    """
    Manages offline operation capabilities.
    
    Features:
    - Local data storage and caching
    - Automatic mode switching
    - Sync queue management
    - Conflict resolution
    - Bandwidth optimization
    """
    
    def __init__(
        self,
        offline_dir: str = "offline_data",
        max_cache_size_mb: int = 1000,
        sync_interval: int = 3600,  # 1 hour
        min_bandwidth_kbps: int = 50  # Minimum bandwidth for online mode
    ):
        """
        Initialize the offline manager.
        
        Args:
            offline_dir: Directory for offline data
            max_cache_size_mb: Maximum cache size in MB
            sync_interval: Interval between sync attempts in seconds
            min_bandwidth_kbps: Minimum bandwidth requirement for online mode
        """
        self.offline_dir = offline_dir
        self.max_cache_size_mb = max_cache_size_mb
        self.sync_interval = sync_interval
        self.min_bandwidth_kbps = min_bandwidth_kbps
        
        self.current_mode = OperationMode.ONLINE
        self.sync_queue: Dict[str, SyncInfo] = {}
        
        # Initialize storage
        self._setup_storage()
    
    def _setup_storage(self) -> None:
        """Set up offline storage."""
        os.makedirs(self.offline_dir, exist_ok=True)
        
        # Initialize SQLite database
        db_path = os.path.join(self.offline_dir, "offline.db")
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS offline_data (
                    id TEXT PRIMARY KEY,
                    component TEXT NOT NULL,
                    data BLOB NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    modified_at TIMESTAMP NOT NULL,
                    sync_status TEXT NOT NULL,
                    priority INTEGER NOT NULL
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sync_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    component TEXT NOT NULL,
                    sync_time TIMESTAMP NOT NULL,
                    status TEXT NOT NULL,
                    details TEXT
                )
            """)
            
            conn.commit()
        
        logger.info(f"Initialized offline storage in {self.offline_dir}")
    
    async def check_connectivity(self) -> Tuple[bool, float]:
        """
        Check internet connectivity and bandwidth.
        
        Returns:
            Tuple of (is_connected, bandwidth_kbps)
        """
        try:
            # Perform bandwidth test
            start_time = time.time()
            
            # Try to download a small test file
            async with aiohttp.ClientSession() as session:
                async with session.get("https://www.google.com/favicon.ico") as response:
                    if response.status != 200:
                        return False, 0.0
                    
                    # Download the file
                    data = await response.read()
                    
                    # Calculate bandwidth
                    duration = time.time() - start_time
                    size_kb = len(data) / 1024
                    bandwidth_kbps = size_kb / duration
                    
                    return True, bandwidth_kbps
        
        except Exception:
            return False, 0.0
    
    async def update_operation_mode(self) -> None:
        """Update operation mode based on connectivity."""
        is_connected, bandwidth = await self.check_connectivity()
        
        if not is_connected:
            new_mode = OperationMode.OFFLINE
        elif bandwidth < self.min_bandwidth_kbps:
            new_mode = OperationMode.HYBRID
        else:
            new_mode = OperationMode.ONLINE
        
        if new_mode != self.current_mode:
            logger.info(f"Switching to {new_mode.value} mode")
            self.current_mode = new_mode
    
    def store_offline_data(
        self,
        component: str,
        data_id: str,
        data: bytes,
        priority: int = 1
    ) -> bool:
        """
        Store data for offline use.
        
        Args:
            component: Component identifier
            data_id: Unique data identifier
            data: Binary data to store
            priority: Sync priority (1-5, lower is higher)
        
        Returns:
            True if successful
        """
        try:
            # Check cache size
            if not self._check_cache_size(len(data)):
                logger.warning("Cache size limit reached")
                self._cleanup_cache()
            
            # Store data
            now = datetime.utcnow()
            
            with sqlite3.connect(os.path.join(self.offline_dir, "offline.db")) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO offline_data
                    (id, component, data, created_at, modified_at, sync_status, priority)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    data_id,
                    component,
                    data,
                    now,
                    now,
                    SyncStatus.PENDING.value,
                    priority
                ))
                conn.commit()
            
            # Update sync queue
            if component not in self.sync_queue:
                self.sync_queue[component] = SyncInfo(
                    component=component,
                    last_sync=now,
                    status=SyncStatus.PENDING,
                    pending_changes=1,
                    data_size_bytes=len(data),
                    priority=priority
                )
            else:
                self.sync_queue[component].pending_changes += 1
                self.sync_queue[component].data_size_bytes += len(data)
                self.sync_queue[component].priority = min(
                    self.sync_queue[component].priority,
                    priority
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing offline data: {e}")
            return False
    
    def get_offline_data(
        self,
        component: str,
        data_id: str
    ) -> Optional[bytes]:
        """
        Retrieve offline data.
        
        Args:
            component: Component identifier
            data_id: Data identifier
        
        Returns:
            Stored data if available
        """
        try:
            with sqlite3.connect(os.path.join(self.offline_dir, "offline.db")) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT data FROM offline_data
                    WHERE component = ? AND id = ?
                """, (component, data_id))
                
                result = cursor.fetchone()
                return result[0] if result else None
                
        except Exception as e:
            logger.error(f"Error retrieving offline data: {e}")
            return None
    
    def _check_cache_size(self, additional_bytes: int) -> bool:
        """Check if adding data would exceed cache size limit."""
        try:
            with sqlite3.connect(os.path.join(self.offline_dir, "offline.db")) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT SUM(LENGTH(data)) FROM offline_data")
                current_size = cursor.fetchone()[0] or 0
                
                max_size_bytes = self.max_cache_size_mb * 1024 * 1024
                return (current_size + additional_bytes) <= max_size_bytes
                
        except Exception as e:
            logger.error(f"Error checking cache size: {e}")
            return False
    
    def _cleanup_cache(self) -> None:
        """Clean up cache by removing old, synced data."""
        try:
            with sqlite3.connect(os.path.join(self.offline_dir, "offline.db")) as conn:
                cursor = conn.cursor()
                
                # Remove synced data older than 7 days
                cursor.execute("""
                    DELETE FROM offline_data
                    WHERE sync_status = ? AND modified_at < ?
                """, (
                    SyncStatus.SYNCED.value,
                    datetime.utcnow() - timedelta(days=7)
                ))
                
                # If still need space, remove oldest pending data
                if not self._check_cache_size(0):
                    cursor.execute("""
                        DELETE FROM offline_data
                        WHERE id IN (
                            SELECT id FROM offline_data
                            ORDER BY priority DESC, modified_at ASC
                            LIMIT 100
                        )
                    """)
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error cleaning up cache: {e}")
    
    async def sync_data(self, component: str) -> bool:
        """
        Synchronize offline data for a component.
        
        Args:
            component: Component to sync
        
        Returns:
            True if successful
        """
        if component not in self.sync_queue:
            return True
        
        sync_info = self.sync_queue[component]
        
        try:
            # Check connectivity
            is_connected, bandwidth = await self.check_connectivity()
            if not is_connected:
                logger.warning("No connectivity, sync deferred")
                return False
            
            # Get pending data
            with sqlite3.connect(os.path.join(self.offline_dir, "offline.db")) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, data, modified_at
                    FROM offline_data
                    WHERE component = ? AND sync_status = ?
                    ORDER BY priority ASC, modified_at ASC
                """, (component, SyncStatus.PENDING.value))
                
                pending_data = cursor.fetchall()
                
                if not pending_data:
                    return True
                
                # Sync each item
                success_count = 0
                for data_id, data, modified_at in pending_data:
                    try:
                        # TODO: Implement actual sync with server
                        # For now, just mark as synced
                        cursor.execute("""
                            UPDATE offline_data
                            SET sync_status = ?, modified_at = ?
                            WHERE id = ?
                        """, (
                            SyncStatus.SYNCED.value,
                            datetime.utcnow(),
                            data_id
                        ))
                        
                        success_count += 1
                        
                    except Exception as e:
                        logger.error(f"Error syncing data {data_id}: {e}")
                
                # Update sync history
                cursor.execute("""
                    INSERT INTO sync_history
                    (component, sync_time, status, details)
                    VALUES (?, ?, ?, ?)
                """, (
                    component,
                    datetime.utcnow(),
                    SyncStatus.SYNCED.value if success_count == len(pending_data)
                    else SyncStatus.ERROR.value,
                    f"Synced {success_count}/{len(pending_data)} items"
                ))
                
                conn.commit()
            
            # Update sync queue
            if success_count == len(pending_data):
                del self.sync_queue[component]
            else:
                sync_info.pending_changes -= success_count
            
            return success_count == len(pending_data)
            
        except Exception as e:
            logger.error(f"Error syncing component {component}: {e}")
            return False
    
    async def start_sync_monitor(self) -> None:
        """Start periodic sync monitoring."""
        while True:
            try:
                # Update operation mode
                await self.update_operation_mode()
                
                if self.current_mode != OperationMode.OFFLINE:
                    # Sort components by priority
                    components = sorted(
                        self.sync_queue.keys(),
                        key=lambda x: (
                            self.sync_queue[x].priority,
                            -self.sync_queue[x].pending_changes
                        )
                    )
                    
                    # Attempt sync for each component
                    for component in components:
                        await self.sync_data(component)
                
                # Sleep until next check
                await asyncio.sleep(self.sync_interval)
                
            except Exception as e:
                logger.error(f"Error in sync monitor: {e}")
                await asyncio.sleep(60)  # Wait before retrying 