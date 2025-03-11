#!/usr/bin/env python3
"""
System Configuration Demo.

This script demonstrates how to use the system configuration management interface
to view and modify system settings.

Author: AMPTALK Team
Date: 2024
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

from src.core.interface.config.system_config import SystemConfig
from src.core.utils.logging_config import get_logger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = get_logger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="System Configuration Demo"
    )
    parser.add_argument(
        "--config-dir",
        default="config",
        help="Configuration directory"
    )
    return parser.parse_args()

def demo_section_management(config: SystemConfig):
    """
    Demonstrate section management.
    
    Args:
        config: System configuration instance
    """
    try:
        # Add custom section
        config.add_section(
            name="custom",
            description="Custom settings",
            settings={
                "feature_flag": True,
                "api_url": "https://api.example.com",
                "retry_count": 3
            },
            validators={
                "retry_count": lambda x: isinstance(x, int) and x > 0
            },
            required=["api_url"]
        )
        logger.info("Added custom section")
        
        # List all sections
        logger.info("\nConfiguration Sections:")
        for name, section in config.sections.items():
            logger.info(
                f"\n{name}: {section.description}"
            )
            for key, value in section.settings.items():
                logger.info(f"  {key}: {value}")
        
        # Validate sections
        logger.info("\nValidating Sections:")
        for name in config.sections:
            errors = config.validate_section(name)
            if errors:
                logger.warning(
                    f"{name}: {len(errors)} validation errors"
                )
                for error in errors:
                    logger.warning(f"  - {error}")
            else:
                logger.info(f"{name}: Valid")
        
    except Exception as e:
        logger.error(f"Error in section management demo: {e}")
        raise

def demo_setting_updates(config: SystemConfig):
    """
    Demonstrate setting updates.
    
    Args:
        config: System configuration instance
    """
    try:
        # Update system settings
        logger.info("\nUpdating Settings:")
        
        success = config.update_setting(
            "system",
            "log_level",
            "DEBUG"
        )
        logger.info(
            f"Updated system.log_level: "
            f"{'succeeded' if success else 'failed'}"
        )
        
        success = config.update_setting(
            "performance",
            "max_workers",
            8
        )
        logger.info(
            f"Updated performance.max_workers: "
            f"{'succeeded' if success else 'failed'}"
        )
        
        # Try invalid update
        success = config.update_setting(
            "performance",
            "max_workers",
            -1  # Invalid value
        )
        logger.info(
            f"Invalid update attempt: "
            f"{'succeeded' if success else 'failed'}"
        )
        
        # Get updated values
        logger.info("\nUpdated Values:")
        logger.info(
            f"system.log_level: "
            f"{config.get_setting('system', 'log_level')}"
        )
        logger.info(
            f"performance.max_workers: "
            f"{config.get_setting('performance', 'max_workers')}"
        )
        
    except Exception as e:
        logger.error(f"Error in setting updates demo: {e}")
        raise

def demo_backup_restore(config: SystemConfig):
    """
    Demonstrate backup and restore.
    
    Args:
        config: System configuration instance
    """
    try:
        # Save current config
        logger.info("\nSaving Configuration:")
        success = config.save()
        logger.info(
            f"Save operation: "
            f"{'succeeded' if success else 'failed'}"
        )
        
        # Make some changes
        config.update_setting(
            "system",
            "debug",
            True
        )
        config.update_setting(
            "security",
            "max_login_attempts",
            5
        )
        
        # Create backup
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        config.save()
        logger.info(f"Created backup: {timestamp}")
        
        # Make more changes
        config.update_setting(
            "system",
            "name",
            "AMPTALK_TEST"
        )
        
        # Restore backup
        logger.info("\nRestoring Backup:")
        success = config.restore_backup(timestamp)
        logger.info(
            f"Restore operation: "
            f"{'succeeded' if success else 'failed'}"
        )
        
        # Verify restored values
        logger.info("\nRestored Values:")
        logger.info(
            f"system.name: "
            f"{config.get_setting('system', 'name')}"
        )
        logger.info(
            f"system.debug: "
            f"{config.get_setting('system', 'debug')}"
        )
        
    except Exception as e:
        logger.error(f"Error in backup/restore demo: {e}")
        raise

def demo_change_tracking(config: SystemConfig):
    """
    Demonstrate change tracking.
    
    Args:
        config: System configuration instance
    """
    try:
        # Get all changes
        all_changes = config.get_changes()
        logger.info(f"\nTotal Changes: {len(all_changes)}")
        
        # Get changes by section
        system_changes = config.get_changes(section="system")
        logger.info(f"System Changes: {len(system_changes)}")
        
        # Get changes by key
        debug_changes = config.get_changes(
            section="system",
            key="debug"
        )
        logger.info(f"Debug Setting Changes: {len(debug_changes)}")
        
        # Print change history
        logger.info("\nChange History:")
        for change in all_changes:
            logger.info(
                f"{change['timestamp']}: "
                f"{change['section']}.{change['key']} = "
                f"{change['old_value']} -> {change['new_value']}"
            )
        
    except Exception as e:
        logger.error(f"Error in change tracking demo: {e}")
        raise

def demo_section_reset(config: SystemConfig):
    """
    Demonstrate section reset.
    
    Args:
        config: System configuration instance
    """
    try:
        # Show current performance settings
        logger.info("\nCurrent Performance Settings:")
        perf_section = config.get_section("performance")
        for key, value in perf_section.settings.items():
            logger.info(f"{key}: {value}")
        
        # Reset performance section
        success = config.reset_section("performance")
        logger.info(
            f"\nReset performance section: "
            f"{'succeeded' if success else 'failed'}"
        )
        
        # Show reset settings
        logger.info("\nReset Performance Settings:")
        perf_section = config.get_section("performance")
        for key, value in perf_section.settings.items():
            logger.info(f"{key}: {value}")
        
    except Exception as e:
        logger.error(f"Error in section reset demo: {e}")
        raise

def main():
    """Main function."""
    args = parse_args()
    
    try:
        # Initialize configuration
        config = SystemConfig(
            config_dir=args.config_dir,
            backup_count=5,
            auto_save=True
        )
        
        # Run demos
        logger.info("\nTesting section management...")
        demo_section_management(config)
        
        logger.info("\nTesting setting updates...")
        demo_setting_updates(config)
        
        logger.info("\nTesting backup and restore...")
        demo_backup_restore(config)
        
        logger.info("\nTesting change tracking...")
        demo_change_tracking(config)
        
        logger.info("\nTesting section reset...")
        demo_section_reset(config)
        
        logger.info("\nAll demos completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Stopping demo...")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 