#!/usr/bin/env python3
"""
Access Control Demo.

This script demonstrates how to use the access control system
to manage authentication and authorization.

Author: AMPTALK Team
Date: 2024
"""

import os
import sys
import asyncio
import argparse
import logging
from datetime import datetime
from typing import Dict, Set

from src.core.security.access_control import (
    AccessControlManager,
    Role,
    Resource,
    Permission,
    User,
    Session
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
    parser = argparse.ArgumentParser(description="Access Control Demo")
    parser.add_argument(
        "--config-dir",
        default="security/access_control",
        help="Directory for configuration storage"
    )
    return parser.parse_args()

async def demo_user_management(manager: AccessControlManager):
    """
    Demonstrate user management.
    
    Args:
        manager: Access control manager instance
    """
    try:
        # Create admin user
        admin = manager.create_user(
            username="admin",
            password="admin123",
            roles={Role.ADMIN},
            metadata={'department': 'IT', 'level': 'senior'}
        )
        logger.info(f"Created admin user: {admin.username}")
        
        # Create manager user
        manager_user = manager.create_user(
            username="manager",
            password="manager123",
            roles={Role.MANAGER},
            metadata={'department': 'Operations', 'level': 'mid'}
        )
        logger.info(f"Created manager user: {manager_user.username}")
        
        # Create regular user
        user = manager.create_user(
            username="user",
            password="user123",
            roles={Role.USER},
            metadata={'department': 'Support', 'level': 'junior'}
        )
        logger.info(f"Created regular user: {user.username}")
        
        # Create guest user
        guest = manager.create_user(
            username="guest",
            password="guest123",
            roles={Role.GUEST}
        )
        logger.info(f"Created guest user: {guest.username}")
        
    except Exception as e:
        logger.error(f"Error in user management demo: {e}")
        raise

async def demo_authentication(manager: AccessControlManager):
    """
    Demonstrate authentication.
    
    Args:
        manager: Access control manager instance
    """
    try:
        # Test valid authentication
        session = manager.authenticate(
            username="admin",
            password="admin123",
            ip_address="127.0.0.1",
            user_agent="Demo/1.0"
        )
        
        if session:
            logger.info(
                f"Authentication successful for admin. "
                f"Token: {session.token[:20]}..."
            )
            
            # Validate token
            user = manager.validate_token(session.token)
            if user:
                logger.info(f"Token validation successful for {user.username}")
            
            # Test permission
            has_permission = manager.check_permission(
                user=user,
                resource=Resource.SYSTEM,
                permission=Permission.ADMIN
            )
            logger.info(
                f"Admin has system admin permission: {has_permission}"
            )
        
        # Test invalid authentication
        invalid_session = manager.authenticate(
            username="admin",
            password="wrong_password",
            ip_address="127.0.0.1",
            user_agent="Demo/1.0"
        )
        
        logger.info(
            f"Authentication with wrong password: "
            f"{'failed' if not invalid_session else 'succeeded'}"
        )
        
    except Exception as e:
        logger.error(f"Error in authentication demo: {e}")
        raise

async def demo_authorization(manager: AccessControlManager):
    """
    Demonstrate authorization.
    
    Args:
        manager: Access control manager instance
    """
    try:
        # Test different role permissions
        test_cases = [
            ("admin", Resource.SYSTEM, Permission.ADMIN),
            ("manager", Resource.AGENT, Permission.EXECUTE),
            ("user", Resource.DATA, Permission.WRITE),
            ("guest", Resource.CONFIG, Permission.READ)
        ]
        
        for username, resource, permission in test_cases:
            # Authenticate user
            session = manager.authenticate(
                username=username,
                password=f"{username}123",
                ip_address="127.0.0.1",
                user_agent="Demo/1.0"
            )
            
            if session:
                user = manager.validate_token(session.token)
                if user:
                    # Check permission
                    has_permission = manager.check_permission(
                        user=user,
                        resource=resource,
                        permission=permission
                    )
                    logger.info(
                        f"User {username} "
                        f"{'has' if has_permission else 'does not have'} "
                        f"{permission.value} permission on {resource.value}"
                    )
                    
                    # Get all permissions
                    permissions = manager.get_user_permissions(user)
                    logger.info(
                        f"User {username} permissions: "
                        f"{len(permissions)} resources"
                    )
        
    except Exception as e:
        logger.error(f"Error in authorization demo: {e}")
        raise

async def demo_session_management(manager: AccessControlManager):
    """
    Demonstrate session management.
    
    Args:
        manager: Access control manager instance
    """
    try:
        # Create multiple sessions for same user
        sessions = []
        for i in range(6):  # Try to create 6 sessions (max is 5)
            session = manager.authenticate(
                username="admin",
                password="admin123",
                ip_address=f"127.0.0.{i+1}",
                user_agent=f"Demo/1.0 Client {i+1}"
            )
            
            if session:
                sessions.append(session)
                logger.info(
                    f"Created session {i+1} for admin "
                    f"from {session.ip_address}"
                )
        
        logger.info(f"Created {len(sessions)} sessions for admin")
        
        # Invalidate a session
        if sessions:
            session_id = sessions[0].id
            success = manager.invalidate_session(session_id)
            logger.info(
                f"Session invalidation "
                f"{'succeeded' if success else 'failed'}"
            )
        
    except Exception as e:
        logger.error(f"Error in session management demo: {e}")
        raise

async def main():
    """Main function."""
    args = parse_args()
    
    try:
        # Initialize access control manager
        manager = AccessControlManager(
            config_dir=args.config_dir
        )
        
        # Run demos
        logger.info("\nTesting user management...")
        await demo_user_management(manager)
        
        logger.info("\nTesting authentication...")
        await demo_authentication(manager)
        
        logger.info("\nTesting authorization...")
        await demo_authorization(manager)
        
        logger.info("\nTesting session management...")
        await demo_session_management(manager)
        
        logger.info("\nAll demos completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Stopping demo...")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 