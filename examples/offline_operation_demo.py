#!/usr/bin/env python3
"""
Offline Operation Demo.

This script demonstrates how to use the offline operation system
to handle scenarios with limited or no connectivity.

Author: AMPTALK Team
Date: 2024
"""

import os
import sys
import uuid
import asyncio
import argparse
import logging
from datetime import datetime
from typing import Optional

from src.core.offline_operation.offline_manager import (
    OfflineManager,
    OperationMode,
    SyncStatus
)
from src.core.utils.logging_config import get_logger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = get_logger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Offline Operation Demo")
    parser.add_argument(
        "--mode",
        choices=[m.value for m in OperationMode],
        help="Force operation mode"
    )
    parser.add_argument(
        "--data-dir",
        default="offline_data",
        help="Directory for offline data"
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=1000,
        help="Maximum cache size in MB"
    )
    parser.add_argument(
        "--sync-interval",
        type=int,
        default=60,
        help="Sync interval in seconds"
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Run sync monitor"
    )
    return parser.parse_args()

async def simulate_data_generation(
    manager: OfflineManager,
    component: str,
    interval: float = 5.0
) -> None:
    """
    Simulate data generation for testing.
    
    Args:
        manager: Offline manager instance
        component: Component identifier
        interval: Interval between data generation in seconds
    """
    while True:
        try:
            # Generate test data
            data_id = str(uuid.uuid4())
            data = f"Test data generated at {datetime.utcnow()}".encode()
            
            # Store data
            success = manager.store_offline_data(
                component=component,
                data_id=data_id,
                data=data,
                priority=1
            )
            
            if success:
                logger.info(f"Stored data {data_id} for {component}")
                
                # Try to retrieve data
                retrieved = manager.get_offline_data(component, data_id)
                if retrieved:
                    logger.info(f"Retrieved data: {retrieved.decode()}")
                else:
                    logger.error("Failed to retrieve data")
            else:
                logger.error("Failed to store data")
            
            await asyncio.sleep(interval)
            
        except Exception as e:
            logger.error(f"Error generating data: {e}")
            await asyncio.sleep(1)

async def monitor_mode_changes(manager: OfflineManager) -> None:
    """Monitor operation mode changes."""
    previous_mode = manager.current_mode
    
    while True:
        try:
            await manager.update_operation_mode()
            
            if manager.current_mode != previous_mode:
                logger.info(
                    f"Operation mode changed: "
                    f"{previous_mode.value} -> {manager.current_mode.value}"
                )
                previous_mode = manager.current_mode
            
            await asyncio.sleep(5)
            
        except Exception as e:
            logger.error(f"Error monitoring mode: {e}")
            await asyncio.sleep(1)

async def main():
    """Main function."""
    args = parse_args()
    
    try:
        # Initialize offline manager
        manager = OfflineManager(
            offline_dir=args.data_dir,
            max_cache_size_mb=args.cache_size,
            sync_interval=args.sync_interval
        )
        
        # Force operation mode if specified
        if args.mode:
            manager.current_mode = OperationMode[args.mode.upper()]
            logger.info(f"Forced operation mode: {args.mode}")
        
        # Create tasks
        tasks = [
            # Generate data for different components
            simulate_data_generation(manager, "audio", 7.0),
            simulate_data_generation(manager, "transcription", 10.0),
            simulate_data_generation(manager, "summary", 15.0),
            # Monitor mode changes
            monitor_mode_changes(manager)
        ]
        
        # Add sync monitor if requested
        if args.monitor:
            tasks.append(manager.start_sync_monitor())
        
        # Run tasks
        await asyncio.gather(*tasks)
        
    except KeyboardInterrupt:
        logger.info("Stopping demo...")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 