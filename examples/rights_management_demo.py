#!/usr/bin/env python3
"""
Data Subject Rights Management Demo.

This script demonstrates how to use the data subject rights management system
to handle privacy rights requests.

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

from src.core.privacy.rights_manager import (
    RightsManager,
    RightType,
    RequestStatus,
    RequestPriority,
    RightsRequest
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
    parser = argparse.ArgumentParser(
        description="Data Subject Rights Management Demo"
    )
    parser.add_argument(
        "--storage-dir",
        default="privacy/rights",
        help="Directory for request storage"
    )
    parser.add_argument(
        "--deadline-days",
        type=int,
        default=30,
        help="Default deadline for requests in days"
    )
    return parser.parse_args()

async def demo_request_submission(manager: RightsManager):
    """
    Demonstrate request submission.
    
    Args:
        manager: Rights manager instance
    """
    try:
        # Submit access request
        access_id = manager.submit_request(
            user_id="user123",
            right_type=RightType.ACCESS,
            description="Request access to all personal data",
            data_scope=["profile", "preferences", "activity"],
            priority=RequestPriority.MEDIUM,
            metadata={'reason': 'personal audit'}
        )
        logger.info(f"Submitted access request: {access_id}")
        
        # Submit rectification request
        rectification_id = manager.submit_request(
            user_id="user123",
            right_type=RightType.RECTIFICATION,
            description="Update email address",
            data_scope=["profile.email"],
            proof_of_identity="id_proof_123",
            priority=RequestPriority.HIGH,
            metadata={
                'old_value': 'old@example.com',
                'new_value': 'new@example.com'
            }
        )
        logger.info(f"Submitted rectification request: {rectification_id}")
        
        # Submit erasure request
        erasure_id = manager.submit_request(
            user_id="user123",
            right_type=RightType.ERASURE,
            description="Delete all account data",
            data_scope=["profile", "preferences", "activity"],
            proof_of_identity="id_proof_123",
            priority=RequestPriority.URGENT,
            metadata={'reason': 'leaving platform'}
        )
        logger.info(f"Submitted erasure request: {erasure_id}")
        
        # Submit portability request
        portability_id = manager.submit_request(
            user_id="user123",
            right_type=RightType.PORTABILITY,
            description="Export data in machine-readable format",
            data_scope=["profile", "activity"],
            proof_of_identity="id_proof_123",
            priority=RequestPriority.LOW,
            metadata={'format': 'json'}
        )
        logger.info(f"Submitted portability request: {portability_id}")
        
        return access_id  # Return for other demos
        
    except Exception as e:
        logger.error(f"Error in request submission demo: {e}")
        raise

async def demo_request_processing(
    manager: RightsManager,
    request_id: str
):
    """
    Demonstrate request processing.
    
    Args:
        manager: Rights manager instance
        request_id: Request ID to process
    """
    try:
        # Update status to processing
        success = manager.update_request_status(
            request_id=request_id,
            status=RequestStatus.PROCESSING
        )
        logger.info(
            f"Updated request status to processing: "
            f"{'succeeded' if success else 'failed'}"
        )
        
        # Simulate processing
        await asyncio.sleep(2)
        
        # Complete request with response
        success = manager.update_request_status(
            request_id=request_id,
            status=RequestStatus.COMPLETED,
            response={
                'completed_at': datetime.utcnow().isoformat(),
                'data': {
                    'profile': {
                        'name': 'John Smith',
                        'email': 'john@example.com'
                    },
                    'preferences': {
                        'theme': 'dark',
                        'notifications': True
                    },
                    'activity': {
                        'last_login': '2024-03-01T12:00:00Z',
                        'sessions': 42
                    }
                }
            }
        )
        logger.info(
            f"Completed request with response: "
            f"{'succeeded' if success else 'failed'}"
        )
        
    except Exception as e:
        logger.error(f"Error in request processing demo: {e}")
        raise

async def demo_request_listing(manager: RightsManager):
    """
    Demonstrate request listing.
    
    Args:
        manager: Rights manager instance
    """
    try:
        # List all requests
        all_requests = manager.list_requests()
        logger.info(f"Found {len(all_requests)} total requests")
        
        # List by right type
        access_requests = manager.list_requests(
            right_type=RightType.ACCESS
        )
        logger.info(f"Found {len(access_requests)} access requests")
        
        # List by status
        completed_requests = manager.list_requests(
            status=RequestStatus.COMPLETED
        )
        logger.info(f"Found {len(completed_requests)} completed requests")
        
        # List by priority
        urgent_requests = manager.list_requests(
            priority=RequestPriority.URGENT
        )
        logger.info(f"Found {len(urgent_requests)} urgent requests")
        
        # List recent requests
        start_date = datetime.utcnow() - timedelta(hours=1)
        recent_requests = manager.list_requests(
            start_date=start_date
        )
        logger.info(f"Found {len(recent_requests)} recent requests")
        
        # Print detailed list
        logger.info("\nDetailed request list:")
        for request in all_requests:
            logger.info(
                f"- {request.right_type.value}: {request.status.value} "
                f"(Priority: {request.priority.value})"
            )
        
    except Exception as e:
        logger.error(f"Error in request listing demo: {e}")
        raise

async def demo_compliance_reporting(manager: RightsManager):
    """
    Demonstrate compliance reporting.
    
    Args:
        manager: Rights manager instance
    """
    try:
        # Generate report for last 30 days
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=30)
        
        report = manager.generate_compliance_report(
            start_date=start_date,
            end_date=end_date
        )
        
        # Print report summary
        logger.info("\nCompliance Report Summary:")
        logger.info(f"Period: {report['period']['start']} to {report['period']['end']}")
        
        stats = report['statistics']
        logger.info("\nStatistics:")
        logger.info(f"Total Requests: {stats['total_requests']}")
        logger.info(f"Completed: {stats['completed_requests']}")
        logger.info(f"Rejected: {stats['rejected_requests']}")
        logger.info(f"Overdue: {stats['overdue_requests']}")
        logger.info(f"Completion Rate: {stats['completion_rate']:.2%}")
        logger.info(f"Average Completion Time: {stats['avg_completion_time']}")
        
        logger.info("\nRequests by Type:")
        for type_name, count in report['requests_by_type'].items():
            logger.info(f"{type_name}: {count}")
        
    except Exception as e:
        logger.error(f"Error in compliance reporting demo: {e}")
        raise

async def demo_overdue_checking(manager: RightsManager):
    """
    Demonstrate overdue request checking.
    
    Args:
        manager: Rights manager instance
    """
    try:
        # Submit request with short deadline
        overdue_id = manager.submit_request(
            user_id="user456",
            right_type=RightType.ACCESS,
            description="Request with short deadline",
            data_scope=["profile"],
            deadline_days=0  # Immediate deadline
        )
        logger.info(f"Submitted request with short deadline: {overdue_id}")
        
        # Check overdue requests
        overdue = manager.check_overdue_requests()
        logger.info(f"Found {len(overdue)} overdue requests")
        
        # Print overdue details
        for request in overdue:
            logger.info(
                f"Overdue request {request.id}: "
                f"{request.right_type.value} "
                f"(Due: {request.deadline})"
            )
        
    except Exception as e:
        logger.error(f"Error in overdue checking demo: {e}")
        raise

async def main():
    """Main function."""
    args = parse_args()
    
    try:
        # Initialize rights manager
        manager = RightsManager(
            storage_dir=args.storage_dir,
            default_deadline_days=args.deadline_days
        )
        
        # Run demos
        logger.info("\nTesting request submission...")
        request_id = await demo_request_submission(manager)
        
        logger.info("\nTesting request processing...")
        await demo_request_processing(manager, request_id)
        
        logger.info("\nTesting request listing...")
        await demo_request_listing(manager)
        
        logger.info("\nTesting compliance reporting...")
        await demo_compliance_reporting(manager)
        
        logger.info("\nTesting overdue checking...")
        await demo_overdue_checking(manager)
        
        logger.info("\nAll demos completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Stopping demo...")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 