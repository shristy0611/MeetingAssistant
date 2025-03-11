#!/usr/bin/env python3
"""
End-to-End Encryption Demo.

This script demonstrates how to use the encryption management system
to secure data transmission and storage.

Author: AMPTALK Team
Date: 2024
"""

import os
import sys
import json
import asyncio
import argparse
import logging
from datetime import datetime
from typing import Dict, Any

from src.core.security.encryption_manager import (
    EncryptionManager,
    EncryptionType,
    KeyType
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
    parser = argparse.ArgumentParser(description="Encryption Demo")
    parser.add_argument(
        "--keys-dir",
        default="security/keys",
        help="Directory for key storage"
    )
    parser.add_argument(
        "--key-size",
        type=int,
        default=2048,
        help="RSA key size in bits"
    )
    parser.add_argument(
        "--rotation-days",
        type=int,
        default=30,
        help="Days between key rotations"
    )
    parser.add_argument(
        "--encryption-type",
        choices=[t.value for t in EncryptionType],
        default=EncryptionType.HYBRID.value,
        help="Type of encryption to use"
    )
    return parser.parse_args()

async def demo_encryption(manager: EncryptionManager, encryption_type: EncryptionType):
    """
    Demonstrate encryption functionality.
    
    Args:
        manager: Encryption manager instance
        encryption_type: Type of encryption to use
    """
    try:
        # Test data
        test_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'message': "This is a secret message",
            'metadata': {
                'priority': "high",
                'category': "test"
            }
        }
        
        logger.info(f"Original data: {test_data}")
        
        # Convert to bytes
        data_bytes = json.dumps(test_data).encode()
        
        # Encrypt data
        encrypted_data, nonce = manager.encrypt_data(
            data=data_bytes,
            encryption_type=encryption_type
        )
        
        logger.info(
            f"Encrypted data size: {len(encrypted_data)} bytes, "
            f"Nonce size: {len(nonce)} bytes"
        )
        
        # Decrypt data
        decrypted_bytes = manager.decrypt_data(
            encrypted_data=encrypted_data,
            nonce=nonce,
            encryption_type=encryption_type
        )
        
        # Convert back to dict
        decrypted_data = json.loads(decrypted_bytes.decode())
        
        logger.info(f"Decrypted data: {decrypted_data}")
        
        # Verify data integrity
        assert test_data == decrypted_data, "Data integrity check failed"
        logger.info("Data integrity verified")
        
    except Exception as e:
        logger.error(f"Error in encryption demo: {e}")
        raise

async def demo_key_rotation(manager: EncryptionManager):
    """
    Demonstrate key rotation functionality.
    
    Args:
        manager: Encryption manager instance
    """
    try:
        # Export original public key
        original_key = manager.export_public_key()
        logger.info("Exported original public key")
        
        # Rotate keys
        manager.rotate_keys()
        logger.info("Rotated all keys")
        
        # Export new public key
        new_key = manager.export_public_key()
        logger.info("Exported new public key")
        
        # Verify keys are different
        assert original_key != new_key, "Key rotation failed"
        logger.info("Key rotation verified")
        
    except Exception as e:
        logger.error(f"Error in key rotation demo: {e}")
        raise

async def demo_key_import_export(manager: EncryptionManager):
    """
    Demonstrate key import/export functionality.
    
    Args:
        manager: Encryption manager instance
    """
    try:
        # Export public key
        public_key = manager.export_public_key()
        logger.info("Exported public key")
        
        # Import public key
        manager.import_public_key(public_key)
        logger.info("Imported public key")
        
        # Verify key paths
        key_path = os.path.join(
            manager.keys_dir,
            KeyType.DATA.value,
            "imported_key.json"
        )
        assert os.path.exists(key_path), "Key import failed"
        logger.info("Key import verified")
        
    except Exception as e:
        logger.error(f"Error in key import/export demo: {e}")
        raise

async def main():
    """Main function."""
    args = parse_args()
    
    try:
        # Initialize encryption manager
        manager = EncryptionManager(
            keys_dir=args.keys_dir,
            key_size=args.key_size,
            rotation_days=args.rotation_days
        )
        
        # Run demos
        encryption_type = EncryptionType(args.encryption_type)
        
        logger.info(f"\nTesting {encryption_type.value} encryption...")
        await demo_encryption(manager, encryption_type)
        
        logger.info("\nTesting key rotation...")
        await demo_key_rotation(manager)
        
        logger.info("\nTesting key import/export...")
        await demo_key_import_export(manager)
        
        logger.info("\nAll demos completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Stopping demo...")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 