#!/usr/bin/env python3
"""
Key Management Demo.

This script demonstrates how to use the key management system
to handle cryptographic keys securely.

Author: AMPTALK Team
Date: 2024
"""

import os
import sys
import asyncio
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

from src.core.security.key_manager import (
    KeyManager,
    KeyType,
    KeyAlgorithm,
    KeyStatus
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
    parser = argparse.ArgumentParser(description="Key Management Demo")
    parser.add_argument(
        "--keys-dir",
        default="security/keys",
        help="Directory for key storage"
    )
    parser.add_argument(
        "--rotation-days",
        type=int,
        default=90,
        help="Days between key rotations"
    )
    return parser.parse_args()

async def demo_key_generation(manager: KeyManager):
    """
    Demonstrate key generation.
    
    Args:
        manager: Key manager instance
    """
    try:
        # Generate RSA key pair
        rsa_key_id = manager.generate_key(
            key_type=KeyType.ENCRYPTION,
            algorithm=KeyAlgorithm.RSA,
            tags={'purpose': 'encryption', 'environment': 'demo'}
        )
        logger.info(f"Generated RSA key: {rsa_key_id}")
        
        # Generate AES key
        aes_key_id = manager.generate_key(
            key_type=KeyType.ENCRYPTION,
            algorithm=KeyAlgorithm.AES,
            expiry_days=30,
            tags={'purpose': 'data_encryption', 'environment': 'demo'}
        )
        logger.info(f"Generated AES key: {aes_key_id}")
        
        # Generate signing key
        signing_key_id = manager.generate_key(
            key_type=KeyType.SIGNING,
            algorithm=KeyAlgorithm.RSA,
            tags={'purpose': 'signing', 'environment': 'demo'}
        )
        logger.info(f"Generated signing key: {signing_key_id}")
        
        # Generate authentication key
        auth_key_id = manager.generate_key(
            key_type=KeyType.AUTHENTICATION,
            algorithm=KeyAlgorithm.AES,
            tags={'purpose': 'authentication', 'environment': 'demo'}
        )
        logger.info(f"Generated authentication key: {auth_key_id}")
        
    except Exception as e:
        logger.error(f"Error in key generation demo: {e}")
        raise

async def demo_key_retrieval(manager: KeyManager):
    """
    Demonstrate key retrieval.
    
    Args:
        manager: Key manager instance
    """
    try:
        # List all active keys
        active_keys = manager.list_keys(
            status=KeyStatus.ACTIVE
        )
        logger.info(f"Found {len(active_keys)} active keys")
        
        # List encryption keys
        encryption_keys = manager.list_keys(
            key_type=KeyType.ENCRYPTION
        )
        logger.info(f"Found {len(encryption_keys)} encryption keys")
        
        # List RSA keys
        rsa_keys = manager.list_keys(
            algorithm=KeyAlgorithm.RSA
        )
        logger.info(f"Found {len(rsa_keys)} RSA keys")
        
        # List keys by tag
        demo_keys = manager.list_keys(
            tags={'environment': 'demo'}
        )
        logger.info(f"Found {len(demo_keys)} demo keys")
        
        # Get key with private material
        if encryption_keys:
            key_id = encryption_keys[0].id
            metadata, material = manager.get_key(
                key_id=key_id,
                include_private=True
            )
            logger.info(
                f"Retrieved key {key_id} "
                f"({metadata.algorithm.value}, version {metadata.version})"
            )
        
    except Exception as e:
        logger.error(f"Error in key retrieval demo: {e}")
        raise

async def demo_key_rotation(manager: KeyManager):
    """
    Demonstrate key rotation.
    
    Args:
        manager: Key manager instance
    """
    try:
        # Get an active key
        active_keys = manager.list_keys(status=KeyStatus.ACTIVE)
        if not active_keys:
            logger.warning("No active keys found for rotation")
            return
        
        key_id = active_keys[0].id
        
        # Rotate key
        new_key_id = manager.rotate_key(
            key_id=key_id,
            expiry_days=60
        )
        logger.info(f"Rotated key {key_id} to {new_key_id}")
        
        # Verify old key status
        old_key = manager.list_keys(
            status=KeyStatus.ROTATED,
            tags={'environment': 'demo'}
        )
        logger.info(f"Found {len(old_key)} rotated keys")
        
    except Exception as e:
        logger.error(f"Error in key rotation demo: {e}")
        raise

async def demo_key_backup(manager: KeyManager):
    """
    Demonstrate key backup and restore.
    
    Args:
        manager: Key manager instance
    """
    try:
        # Get an active key
        active_keys = manager.list_keys(status=KeyStatus.ACTIVE)
        if not active_keys:
            logger.warning("No active keys found for backup")
            return
        
        key_id = active_keys[0].id
        
        # Delete key
        success = manager.delete_key(key_id)
        logger.info(f"Deleted key {key_id}: {success}")
        
        # Restore from backup
        success = manager.restore_key(key_id)
        logger.info(f"Restored key {key_id}: {success}")
        
        # Verify key status
        restored_key = manager.list_keys(
            status=KeyStatus.ACTIVE,
            tags={'environment': 'demo'}
        )
        logger.info(f"Found {len(restored_key)} restored keys")
        
    except Exception as e:
        logger.error(f"Error in key backup demo: {e}")
        raise

async def main():
    """Main function."""
    args = parse_args()
    
    try:
        # Initialize key manager
        manager = KeyManager(
            keys_dir=args.keys_dir,
            rotation_days=args.rotation_days
        )
        
        # Run demos
        logger.info("\nTesting key generation...")
        await demo_key_generation(manager)
        
        logger.info("\nTesting key retrieval...")
        await demo_key_retrieval(manager)
        
        logger.info("\nTesting key rotation...")
        await demo_key_rotation(manager)
        
        logger.info("\nTesting key backup...")
        await demo_key_backup(manager)
        
        logger.info("\nAll demos completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Stopping demo...")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 