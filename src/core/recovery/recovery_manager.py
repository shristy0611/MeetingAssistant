"""
Recovery Management System.

This module provides comprehensive recovery procedures for the AMPTALK system,
ensuring resilience against various types of failures.

Author: AMPTALK Team
Date: 2024
"""

import os
import json
import time
import logging
import asyncio
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from src.core.utils.logging_config import get_logger

logger = get_logger(__name__)

class RecoveryType(Enum):
    """Types of recovery procedures."""
    SYSTEM_STATE = "system_state"
    DATA = "data"
    AGENT = "agent"
    CRASH = "crash"
    NETWORK = "network"

class RecoveryStatus(Enum):
    """Status of recovery operations."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class RecoveryPoint:
    """Snapshot of system state for recovery."""
    timestamp: datetime
    type: RecoveryType
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    checksum: str

@dataclass
class RecoveryOperation:
    """Details of a recovery operation."""
    id: str
    type: RecoveryType
    status: RecoveryStatus
    start_time: datetime
    end_time: Optional[datetime]
    success: bool
    details: Dict[str, Any]

class RecoveryManager:
    """
    Manages system recovery procedures.
    
    Features:
    - System state recovery
    - Data recovery
    - Agent recovery
    - Crash recovery
    - Network failure recovery
    """
    
    def __init__(
        self,
        recovery_dir: str = "recovery_data",
        max_recovery_points: int = 10,
        auto_recovery: bool = True,
        recovery_interval: int = 300  # 5 minutes
    ):
        """
        Initialize the recovery manager.
        
        Args:
            recovery_dir: Directory for recovery data
            max_recovery_points: Maximum number of recovery points to maintain
            auto_recovery: Whether to enable automatic recovery
            recovery_interval: Interval between recovery point creation
        """
        self.recovery_dir = recovery_dir
        self.max_recovery_points = max_recovery_points
        self.auto_recovery = auto_recovery
        self.recovery_interval = recovery_interval
        
        self.recovery_points: Dict[str, RecoveryPoint] = {}
        self.active_recoveries: Dict[str, RecoveryOperation] = {}
        self.recovery_handlers: Dict[RecoveryType, List[Callable]] = {
            rt: [] for rt in RecoveryType
        }
        
        # Initialize storage
        self._setup_storage()
    
    def _setup_storage(self) -> None:
        """Set up recovery storage."""
        os.makedirs(self.recovery_dir, exist_ok=True)
        
        # Create subdirectories for different recovery types
        for recovery_type in RecoveryType:
            os.makedirs(
                os.path.join(self.recovery_dir, recovery_type.value),
                exist_ok=True
            )
        
        logger.info(f"Initialized recovery storage in {self.recovery_dir}")
    
    def register_recovery_handler(
        self,
        recovery_type: RecoveryType,
        handler: Callable
    ) -> None:
        """
        Register a handler for a specific recovery type.
        
        Args:
            recovery_type: Type of recovery
            handler: Handler function
        """
        self.recovery_handlers[recovery_type].append(handler)
        logger.info(f"Registered handler for {recovery_type.value}")
    
    async def create_recovery_point(
        self,
        recovery_type: RecoveryType,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Create a recovery point.
        
        Args:
            recovery_type: Type of recovery point
            data: Data to store
            metadata: Additional metadata
        
        Returns:
            Recovery point ID if successful
        """
        try:
            # Generate recovery point
            timestamp = datetime.utcnow()
            recovery_id = f"{recovery_type.value}_{timestamp.timestamp()}"
            
            recovery_point = RecoveryPoint(
                timestamp=timestamp,
                type=recovery_type,
                data=data,
                metadata=metadata or {},
                checksum=self._calculate_checksum(data)
            )
            
            # Save recovery point
            path = os.path.join(
                self.recovery_dir,
                recovery_type.value,
                f"{recovery_id}.json"
            )
            
            with open(path, 'w') as f:
                json.dump({
                    'timestamp': recovery_point.timestamp.isoformat(),
                    'type': recovery_point.type.value,
                    'data': recovery_point.data,
                    'metadata': recovery_point.metadata,
                    'checksum': recovery_point.checksum
                }, f)
            
            # Update recovery points
            self.recovery_points[recovery_id] = recovery_point
            
            # Clean up old recovery points
            await self._cleanup_recovery_points(recovery_type)
            
            logger.info(f"Created recovery point: {recovery_id}")
            return recovery_id
            
        except Exception as e:
            logger.error(f"Error creating recovery point: {e}")
            return None
    
    def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """Calculate checksum for data integrity."""
        import hashlib
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()
    
    async def _cleanup_recovery_points(
        self,
        recovery_type: RecoveryType
    ) -> None:
        """Clean up old recovery points."""
        try:
            # Get recovery points for type
            type_points = {
                k: v for k, v in self.recovery_points.items()
                if v.type == recovery_type
            }
            
            # Sort by timestamp
            sorted_points = sorted(
                type_points.items(),
                key=lambda x: x[1].timestamp,
                reverse=True
            )
            
            # Remove excess points
            for recovery_id, _ in sorted_points[self.max_recovery_points:]:
                path = os.path.join(
                    self.recovery_dir,
                    recovery_type.value,
                    f"{recovery_id}.json"
                )
                
                try:
                    os.remove(path)
                    del self.recovery_points[recovery_id]
                except OSError:
                    pass
            
        except Exception as e:
            logger.error(f"Error cleaning up recovery points: {e}")
    
    async def start_recovery(
        self,
        recovery_type: RecoveryType,
        recovery_id: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Start a recovery operation.
        
        Args:
            recovery_type: Type of recovery
            recovery_id: Specific recovery point ID (optional)
            **kwargs: Additional parameters for recovery
        
        Returns:
            Operation ID if successful
        """
        try:
            # Find latest recovery point if none specified
            if not recovery_id:
                type_points = {
                    k: v for k, v in self.recovery_points.items()
                    if v.type == recovery_type
                }
                if not type_points:
                    logger.error(f"No recovery points for {recovery_type.value}")
                    return None
                
                recovery_id = max(
                    type_points.items(),
                    key=lambda x: x[1].timestamp
                )[0]
            
            # Get recovery point
            recovery_point = self.recovery_points.get(recovery_id)
            if not recovery_point:
                logger.error(f"Recovery point not found: {recovery_id}")
                return None
            
            # Create operation
            operation_id = f"recovery_{datetime.utcnow().timestamp()}"
            operation = RecoveryOperation(
                id=operation_id,
                type=recovery_type,
                status=RecoveryStatus.IN_PROGRESS,
                start_time=datetime.utcnow(),
                end_time=None,
                success=False,
                details={
                    'recovery_point_id': recovery_id,
                    'params': kwargs
                }
            )
            
            self.active_recoveries[operation_id] = operation
            
            # Execute handlers
            success = True
            for handler in self.recovery_handlers[recovery_type]:
                try:
                    await handler(recovery_point, **kwargs)
                except Exception as e:
                    logger.error(f"Handler failed: {e}")
                    success = False
            
            # Update operation status
            operation.status = (
                RecoveryStatus.COMPLETED if success
                else RecoveryStatus.FAILED
            )
            operation.end_time = datetime.utcnow()
            operation.success = success
            
            logger.info(
                f"Recovery operation {operation_id} "
                f"{'completed' if success else 'failed'}"
            )
            return operation_id
            
        except Exception as e:
            logger.error(f"Error starting recovery: {e}")
            return None
    
    def get_recovery_status(self, operation_id: str) -> Optional[RecoveryOperation]:
        """Get status of a recovery operation."""
        return self.active_recoveries.get(operation_id)
    
    async def monitor_system_health(self) -> None:
        """Monitor system health and trigger auto-recovery if needed."""
        while True:
            try:
                # Check system health
                # TODO: Implement system health checks
                
                # Create recovery points periodically
                for recovery_type in RecoveryType:
                    await self.create_recovery_point(
                        recovery_type=recovery_type,
                        data={},  # TODO: Collect actual system state
                        metadata={'auto_generated': True}
                    )
                
                await asyncio.sleep(self.recovery_interval)
                
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(60)  # Wait before retrying 