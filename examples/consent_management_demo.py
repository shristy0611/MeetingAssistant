#!/usr/bin/env python3
"""
Consent Management Demo.

This script demonstrates how to use the consent management system
to handle user consent for data processing.

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

from src.core.privacy.consent_manager import (
    ConsentManager,
    ConsentStatus,
    ConsentPurpose,
    ConsentRecord
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
    parser = argparse.ArgumentParser(description="Consent Management Demo")
    parser.add_argument(
        "--storage-dir",
        default="privacy/consent",
        help="Directory for consent storage"
    )
    parser.add_argument(
        "--expiry-days",
        type=int,
        default=365,
        help="Default consent expiry period in days"
    )
    return parser.parse_args()

async def demo_consent_recording(manager: ConsentManager):
    """
    Demonstrate consent recording.
    
    Args:
        manager: Consent manager instance
    """
    try:
        # Record essential consent
        essential_id = manager.record_consent(
            user_id="user123",
            purpose=ConsentPurpose.ESSENTIAL,
            status=ConsentStatus.GRANTED,
            version="1.0",
            ip_address="192.168.1.100",
            user_agent="Demo/1.0",
            metadata={'device': 'desktop', 'location': 'US'}
        )
        logger.info(f"Recorded essential consent: {essential_id}")
        
        # Record analytics consent (denied)
        analytics_id = manager.record_consent(
            user_id="user123",
            purpose=ConsentPurpose.ANALYTICS,
            status=ConsentStatus.DENIED,
            version="1.0",
            ip_address="192.168.1.100",
            user_agent="Demo/1.0"
        )
        logger.info(f"Recorded analytics consent: {analytics_id}")
        
        # Record marketing consent with expiry
        marketing_id = manager.record_consent(
            user_id="user123",
            purpose=ConsentPurpose.MARKETING,
            status=ConsentStatus.GRANTED,
            version="1.0",
            expiry_days=30,
            ip_address="192.168.1.100",
            user_agent="Demo/1.0",
            metadata={'preferences': ['email', 'sms']}
        )
        logger.info(f"Recorded marketing consent: {marketing_id}")
        
        # Record third-party consent
        third_party_id = manager.record_consent(
            user_id="user123",
            purpose=ConsentPurpose.THIRD_PARTY,
            status=ConsentStatus.GRANTED,
            version="1.0",
            ip_address="192.168.1.100",
            user_agent="Demo/1.0",
            metadata={'partners': ['analytics_co', 'ads_co']}
        )
        logger.info(f"Recorded third-party consent: {third_party_id}")
        
        return marketing_id  # Return for withdrawal demo
        
    except Exception as e:
        logger.error(f"Error in consent recording demo: {e}")
        raise

async def demo_consent_checking(manager: ConsentManager):
    """
    Demonstrate consent checking.
    
    Args:
        manager: Consent manager instance
    """
    try:
        # Check various consent types
        for purpose in ConsentPurpose:
            record = manager.check_consent(
                user_id="user123",
                purpose=purpose
            )
            
            if record:
                logger.info(
                    f"Consent for {purpose.value}: "
                    f"{record.status.value}"
                )
            else:
                logger.info(
                    f"No consent record found for {purpose.value}"
                )
        
    except Exception as e:
        logger.error(f"Error in consent checking demo: {e}")
        raise

async def demo_consent_withdrawal(
    manager: ConsentManager,
    consent_id: str
):
    """
    Demonstrate consent withdrawal.
    
    Args:
        manager: Consent manager instance
        consent_id: Consent record ID to withdraw
    """
    try:
        # Withdraw consent
        success = manager.withdraw_consent(
            record_id=consent_id,
            reason="No longer interested in marketing communications"
        )
        
        logger.info(
            f"Consent withdrawal "
            f"{'succeeded' if success else 'failed'}"
        )
        
        # Verify withdrawal
        record = manager.check_consent(
            user_id="user123",
            purpose=ConsentPurpose.MARKETING
        )
        
        if record:
            logger.info(
                f"Updated marketing consent status: "
                f"{record.status.value}"
            )
        
    except Exception as e:
        logger.error(f"Error in consent withdrawal demo: {e}")
        raise

async def demo_consent_history(manager: ConsentManager):
    """
    Demonstrate consent history retrieval.
    
    Args:
        manager: Consent manager instance
    """
    try:
        # Get all consent history
        all_history = manager.get_consent_history(
            user_id="user123"
        )
        logger.info(f"Found {len(all_history)} consent records")
        
        # Get marketing consent history
        marketing_history = manager.get_consent_history(
            user_id="user123",
            purpose=ConsentPurpose.MARKETING
        )
        logger.info(
            f"Found {len(marketing_history)} marketing consent records"
        )
        
        # Get recent consent history
        start_date = datetime.utcnow() - timedelta(hours=1)
        recent_history = manager.get_consent_history(
            user_id="user123",
            start_date=start_date
        )
        logger.info(
            f"Found {len(recent_history)} recent consent records"
        )
        
        # Print detailed history
        logger.info("\nDetailed consent history:")
        for record in all_history:
            logger.info(
                f"- {record.purpose.value}: {record.status.value} "
                f"(Version: {record.version})"
            )
        
    except Exception as e:
        logger.error(f"Error in consent history demo: {e}")
        raise

async def demo_consent_verification(manager: ConsentManager):
    """
    Demonstrate consent verification.
    
    Args:
        manager: Consent manager instance
    """
    try:
        # Get a consent record
        record = manager.check_consent(
            user_id="user123",
            purpose=ConsentPurpose.ESSENTIAL
        )
        
        if record:
            # Verify integrity
            is_valid = manager.verify_consent(record.id)
            logger.info(
                f"Consent record integrity: "
                f"{'valid' if is_valid else 'invalid'}"
            )
            
            # Try verification with original data
            original_data = {
                'id': record.id,
                'user_id': record.user_id,
                'purpose': record.purpose.value,
                'status': record.status.value,
                'version': record.version,
                'granted_at': (
                    record.granted_at.isoformat()
                    if record.granted_at else None
                ),
                'expires_at': (
                    record.expires_at.isoformat()
                    if record.expires_at else None
                ),
                'ip_address': record.ip_address,
                'user_agent': record.user_agent,
                'metadata': record.metadata
            }
            
            is_valid = manager.verify_consent(
                record.id,
                original_data
            )
            logger.info(
                f"Consent record verification with original data: "
                f"{'valid' if is_valid else 'invalid'}"
            )
        
    except Exception as e:
        logger.error(f"Error in consent verification demo: {e}")
        raise

async def demo_consent_cleanup(manager: ConsentManager):
    """
    Demonstrate consent cleanup.
    
    Args:
        manager: Consent manager instance
    """
    try:
        # Record consent with short expiry
        expired_id = manager.record_consent(
            user_id="user456",
            purpose=ConsentPurpose.MARKETING,
            status=ConsentStatus.GRANTED,
            version="1.0",
            expiry_days=0,  # Expire immediately
            ip_address="192.168.1.101",
            user_agent="Demo/1.0"
        )
        logger.info(f"Recorded expired consent: {expired_id}")
        
        # Clean up expired records
        count = manager.cleanup_expired_consent()
        logger.info(f"Cleaned up {count} expired consent records")
        
        # Verify cleanup
        record = manager.check_consent(
            user_id="user456",
            purpose=ConsentPurpose.MARKETING
        )
        
        if record:
            logger.info(f"Expired consent status: {record.status.value}")
        
    except Exception as e:
        logger.error(f"Error in consent cleanup demo: {e}")
        raise

async def main():
    """Main function."""
    args = parse_args()
    
    try:
        # Initialize consent manager
        manager = ConsentManager(
            storage_dir=args.storage_dir,
            default_expiry_days=args.expiry_days
        )
        
        # Run demos
        logger.info("\nTesting consent recording...")
        consent_id = await demo_consent_recording(manager)
        
        logger.info("\nTesting consent checking...")
        await demo_consent_checking(manager)
        
        logger.info("\nTesting consent withdrawal...")
        await demo_consent_withdrawal(manager, consent_id)
        
        logger.info("\nTesting consent history...")
        await demo_consent_history(manager)
        
        logger.info("\nTesting consent verification...")
        await demo_consent_verification(manager)
        
        logger.info("\nTesting consent cleanup...")
        await demo_consent_cleanup(manager)
        
        logger.info("\nAll demos completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Stopping demo...")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 