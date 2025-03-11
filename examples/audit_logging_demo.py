#!/usr/bin/env python3
"""
Audit Logging Demo.

This script demonstrates how to use the audit logging system
to track and analyze security events.

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

from src.core.security.audit_logger import (
    AuditLogger,
    AuditEventType,
    AuditEventSeverity
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
    parser = argparse.ArgumentParser(description="Audit Logging Demo")
    parser.add_argument(
        "--logs-dir",
        default="security/audit_logs",
        help="Directory for log storage"
    )
    parser.add_argument(
        "--retention-days",
        type=int,
        default=365,
        help="Days to retain logs"
    )
    return parser.parse_args()

async def demo_event_logging(logger: AuditLogger):
    """
    Demonstrate event logging.
    
    Args:
        logger: Audit logger instance
    """
    try:
        # Log authentication event
        auth_event_id = logger.log_event(
            event_type=AuditEventType.AUTH,
            severity=AuditEventSeverity.INFO,
            resource="login_api",
            action="user_login",
            status="success",
            details={
                'username': "admin",
                'method': "password",
                'attempts': 1
            },
            user_id="user_123",
            ip_address="192.168.1.100",
            user_agent="Demo/1.0"
        )
        logger.info(f"Logged authentication event: {auth_event_id}")
        
        # Log access control event
        access_event_id = logger.log_event(
            event_type=AuditEventType.ACCESS,
            severity=AuditEventSeverity.WARNING,
            resource="admin_panel",
            action="access_denied",
            status="failure",
            details={
                'required_role': "admin",
                'user_role': "user",
                'resource': "/admin/users"
            },
            user_id="user_456",
            ip_address="192.168.1.101",
            user_agent="Demo/1.0"
        )
        logger.info(f"Logged access control event: {access_event_id}")
        
        # Log data operation event
        data_event_id = logger.log_event(
            event_type=AuditEventType.DATA,
            severity=AuditEventSeverity.INFO,
            resource="user_data",
            action="data_export",
            status="success",
            details={
                'records': 100,
                'format': "csv",
                'size_bytes': 15000
            },
            user_id="user_789",
            ip_address="192.168.1.102",
            user_agent="Demo/1.0"
        )
        logger.info(f"Logged data operation event: {data_event_id}")
        
        # Log security event
        security_event_id = logger.log_event(
            event_type=AuditEventType.SECURITY,
            severity=AuditEventSeverity.CRITICAL,
            resource="firewall",
            action="intrusion_detected",
            status="blocked",
            details={
                'source_ip': "10.0.0.100",
                'port': 22,
                'attempts': 50,
                'rule': "brute_force_ssh"
            },
            metadata={
                'alert_id': "IDS_001",
                'mitigation': "ip_blocked"
            }
        )
        logger.info(f"Logged security event: {security_event_id}")
        
    except Exception as e:
        logger.error(f"Error in event logging demo: {e}")
        raise

async def demo_event_retrieval(logger: AuditLogger):
    """
    Demonstrate event retrieval.
    
    Args:
        logger: Audit logger instance
    """
    try:
        # Get recent events
        start_date = datetime.utcnow() - timedelta(hours=1)
        events = logger.get_events(
            start_date=start_date,
            limit=10
        )
        logger.info(f"Retrieved {len(events)} recent events")
        
        # Get events by type
        auth_events = logger.get_events(
            event_type=AuditEventType.AUTH,
            limit=5
        )
        logger.info(f"Retrieved {len(auth_events)} authentication events")
        
        # Get events by severity
        critical_events = logger.get_events(
            severity=AuditEventSeverity.CRITICAL,
            limit=5
        )
        logger.info(f"Retrieved {len(critical_events)} critical events")
        
        # Get events by user
        user_events = logger.get_events(
            user_id="user_123",
            limit=5
        )
        logger.info(f"Retrieved {len(user_events)} events for user_123")
        
    except Exception as e:
        logger.error(f"Error in event retrieval demo: {e}")
        raise

async def demo_report_generation(logger: AuditLogger):
    """
    Demonstrate report generation.
    
    Args:
        logger: Audit logger instance
    """
    try:
        # Generate summary report
        start_date = datetime.utcnow() - timedelta(days=7)
        end_date = datetime.utcnow()
        
        summary_report = logger.generate_report(
            start_date=start_date,
            end_date=end_date,
            report_type="summary"
        )
        
        logger.info("Generated summary report:")
        logger.info(f"Total events: {summary_report['total_events']}")
        logger.info(f"Event types: {summary_report['event_types']}")
        logger.info(f"Severities: {summary_report['severities']}")
        logger.info(f"Unique users: {summary_report['unique_users']}")
        logger.info(f"Unique resources: {summary_report['unique_resources']}")
        
        # Generate detailed report
        detailed_report = logger.generate_report(
            start_date=start_date,
            end_date=end_date,
            report_type="detailed"
        )
        
        logger.info("\nGenerated detailed report:")
        logger.info(
            f"Generated at: {detailed_report['metadata']['generated_at']}"
        )
        logger.info(
            f"Total events: {detailed_report['metadata']['total_events']}"
        )
        
    except Exception as e:
        logger.error(f"Error in report generation demo: {e}")
        raise

async def demo_log_rotation(logger: AuditLogger):
    """
    Demonstrate log rotation.
    
    Args:
        logger: Audit logger instance
    """
    try:
        # Generate many events to trigger rotation
        for i in range(100):
            logger.log_event(
                event_type=AuditEventType.SYSTEM,
                severity=AuditEventSeverity.INFO,
                resource="demo",
                action="test_rotation",
                status="success",
                details={'iteration': i},
                metadata={'test': True}
            )
        
        logger.info("Generated events to test log rotation")
        
        # List archive directory
        archive_dir = os.path.join(logger.logs_dir, "archive")
        archives = os.listdir(archive_dir)
        logger.info(f"Archive contains {len(archives)} files")
        
    except Exception as e:
        logger.error(f"Error in log rotation demo: {e}")
        raise

async def main():
    """Main function."""
    args = parse_args()
    
    try:
        # Initialize audit logger
        logger = AuditLogger(
            logs_dir=args.logs_dir,
            retention_days=args.retention_days
        )
        
        # Run demos
        logger.info("\nTesting event logging...")
        await demo_event_logging(logger)
        
        logger.info("\nTesting event retrieval...")
        await demo_event_retrieval(logger)
        
        logger.info("\nTesting report generation...")
        await demo_report_generation(logger)
        
        logger.info("\nTesting log rotation...")
        await demo_log_rotation(logger)
        
        logger.info("\nAll demos completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Stopping demo...")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 