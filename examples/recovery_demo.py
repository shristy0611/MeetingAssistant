#!/usr/bin/env python3
"""
Recovery Procedures Demo.

This script demonstrates how to use the recovery management system
to handle various types of system failures and recovery scenarios.

Author: AMPTALK Team
Date: 2024
"""

import os
import sys
import asyncio
import argparse
import logging
import random
from datetime import datetime
from typing import Dict, Any

from src.core.recovery.recovery_manager import (
    RecoveryManager,
    RecoveryType,
    RecoveryStatus
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
    parser = argparse.ArgumentParser(description="Recovery Procedures Demo")
    parser.add_argument(
        "--recovery-dir",
        default="recovery_data",
        help="Directory for recovery data"
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=10,
        help="Maximum recovery points to maintain"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Recovery point creation interval in seconds"
    )
    parser.add_argument(
        "--simulate-failures",
        action="store_true",
        help="Simulate random system failures"
    )
    return parser.parse_args()

async def simulate_system_state() -> Dict[str, Any]:
    """Simulate system state data."""
    return {
        'timestamp': datetime.utcnow().isoformat(),
        'memory_usage': random.uniform(50, 90),
        'cpu_usage': random.uniform(20, 80),
        'active_agents': random.randint(3, 8),
        'pending_tasks': random.randint(0, 10),
        'network_latency': random.uniform(10, 100)
    }

async def system_state_handler(recovery_point, **kwargs):
    """Handle system state recovery."""
    logger.info("Restoring system state...")
    state = recovery_point.data
    
    # Simulate state restoration
    await asyncio.sleep(2)
    
    logger.info(f"Restored system state from {state['timestamp']}")
    logger.info(f"Memory: {state['memory_usage']}%")
    logger.info(f"CPU: {state['cpu_usage']}%")
    logger.info(f"Active agents: {state['active_agents']}")

async def simulate_failure():
    """Simulate a random system failure."""
    failure_types = [
        "Memory overflow",
        "CPU spike",
        "Network disconnect",
        "Agent crash",
        "Data corruption"
    ]
    
    failure = random.choice(failure_types)
    logger.error(f"SIMULATED FAILURE: {failure}")
    
    if failure == "Memory overflow":
        return RecoveryType.SYSTEM_STATE
    elif failure == "Data corruption":
        return RecoveryType.DATA
    elif failure == "Agent crash":
        return RecoveryType.AGENT
    elif failure == "Network disconnect":
        return RecoveryType.NETWORK
    else:
        return RecoveryType.CRASH

async def main():
    """Main function."""
    args = parse_args()
    
    try:
        # Initialize recovery manager
        manager = RecoveryManager(
            recovery_dir=args.recovery_dir,
            max_recovery_points=args.max_points,
            recovery_interval=args.interval
        )
        
        # Register recovery handlers
        manager.register_recovery_handler(
            RecoveryType.SYSTEM_STATE,
            system_state_handler
        )
        
        # Start tasks
        tasks = [
            # Create recovery points periodically
            asyncio.create_task(
                manager.monitor_system_health()
            )
        ]
        
        # Main demo loop
        while True:
            try:
                # Generate system state
                state = await simulate_system_state()
                
                # Create recovery point
                recovery_id = await manager.create_recovery_point(
                    recovery_type=RecoveryType.SYSTEM_STATE,
                    data=state,
                    metadata={'demo': True}
                )
                
                if recovery_id:
                    logger.info(
                        f"Created recovery point {recovery_id} "
                        f"with {len(state)} state items"
                    )
                
                # Simulate failures if enabled
                if args.simulate_failures:
                    if random.random() < 0.2:  # 20% chance of failure
                        failure_type = await simulate_failure()
                        
                        # Attempt recovery
                        operation_id = await manager.start_recovery(
                            recovery_type=failure_type
                        )
                        
                        if operation_id:
                            # Wait for recovery to complete
                            while True:
                                operation = manager.get_recovery_status(operation_id)
                                if operation.status in (
                                    RecoveryStatus.COMPLETED,
                                    RecoveryStatus.FAILED
                                ):
                                    logger.info(
                                        f"Recovery operation {operation_id} "
                                        f"{'succeeded' if operation.success else 'failed'}"
                                    )
                                    break
                                await asyncio.sleep(1)
                
                await asyncio.sleep(args.interval)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(5)
        
        # Cancel tasks
        for task in tasks:
            task.cancel()
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
    except KeyboardInterrupt:
        logger.info("Stopping demo...")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 