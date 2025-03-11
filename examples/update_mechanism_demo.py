#!/usr/bin/env python3
"""
Update Mechanism Demo.

This script demonstrates how to use the update mechanism to manage
component versions and perform updates.

Author: AMPTALK Team
Date: 2024
"""

import os
import sys
import asyncio
import argparse
import logging
from typing import Optional

from src.core.update_mechanism.update_manager import (
    UpdateManager,
    UpdateStatus,
    UpdatePriority
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
    parser = argparse.ArgumentParser(description="Update Mechanism Demo")
    parser.add_argument(
        "--component",
        help="Component to update"
    )
    parser.add_argument(
        "--version",
        help="Target version"
    )
    parser.add_argument(
        "--priority",
        choices=[p.value for p in UpdatePriority],
        help="Minimum update priority"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check for updates"
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Monitor for updates continuously"
    )
    return parser.parse_args()

def update_progress_callback(
    component: str,
    status: UpdateStatus,
    progress: float
) -> None:
    """Handle update progress notifications."""
    if status == UpdateStatus.DOWNLOADING:
        logger.info(f"Downloading {component}: {progress:.1%}")
    else:
        logger.info(f"Update status for {component}: {status.value}")

async def check_updates(manager: UpdateManager) -> None:
    """Check for available updates."""
    updates = await manager.check_for_updates()
    
    if not updates:
        logger.info("No updates available")
        return
    
    logger.info("Available updates:")
    for component, versions in updates.items():
        for version in versions:
            logger.info(
                f"  {component} {version.version} "
                f"({version.priority.value}, {version.release_date})"
            )

async def update_component(
    manager: UpdateManager,
    component: str,
    version: str
) -> None:
    """Update a specific component."""
    logger.info(f"Updating {component} to version {version}")
    
    success = await manager.update_component(component, version)
    
    if success:
        logger.info(f"Successfully updated {component} to {version}")
    else:
        logger.error(f"Failed to update {component}")

async def update_all(
    manager: UpdateManager,
    priority: Optional[str] = None
) -> None:
    """Update all components."""
    logger.info("Updating all components")
    
    priority_enum = UpdatePriority[priority.upper()] if priority else None
    results = await manager.update_all(priority_enum)
    
    successes = sum(1 for success in results.values() if success)
    failures = len(results) - successes
    
    logger.info(f"Update complete: {successes} succeeded, {failures} failed")
    
    if failures > 0:
        logger.info("Failed components:")
        for component, success in results.items():
            if not success:
                logger.info(f"  {component}")

async def monitor_updates(manager: UpdateManager) -> None:
    """Monitor for updates continuously."""
    logger.info("Starting update monitor")
    
    try:
        await manager.start_update_checker()
    except asyncio.CancelledError:
        logger.info("Update monitor stopped")

async def main():
    """Main function."""
    args = parse_args()
    
    try:
        # Initialize update manager
        manager = UpdateManager()
        
        # Register progress callback
        manager.register_progress_callback(update_progress_callback)
        
        if args.monitor:
            # Monitor for updates
            await monitor_updates(manager)
            
        elif args.check_only:
            # Check for updates
            await check_updates(manager)
            
        elif args.component and args.version:
            # Update specific component
            await update_component(manager, args.component, args.version)
            
        else:
            # Update all components
            await update_all(manager, args.priority)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 