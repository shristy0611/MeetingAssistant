"""
Access Control Manager.

This module provides comprehensive access control capabilities for the AMPTALK system,
ensuring secure authentication and authorization.

Author: AMPTALK Team
Date: 2024
"""

import os
import json
import jwt
import bcrypt
import logging
from enum import Enum
from typing import Dict, List, Optional, Set, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from src.core.utils.logging_config import get_logger

logger = get_logger(__name__)

class Role(Enum):
    """User roles."""
    ADMIN = "admin"
    MANAGER = "manager"
    USER = "user"
    GUEST = "guest"

class Permission(Enum):
    """System permissions."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    MANAGE = "manage"
    ADMIN = "admin"

class Resource(Enum):
    """System resources."""
    SYSTEM = "system"
    AGENT = "agent"
    DATA = "data"
    CONFIG = "config"
    USER = "user"
    LOG = "log"

@dataclass
class User:
    """User information."""
    id: str
    username: str
    password_hash: bytes
    roles: Set[Role]
    permissions: Dict[Resource, Set[Permission]]
    metadata: Dict[str, Any]
    created_at: datetime
    last_login: Optional[datetime]
    is_active: bool

@dataclass
class Session:
    """User session information."""
    id: str
    user_id: str
    token: str
    created_at: datetime
    expires_at: datetime
    last_activity: datetime
    ip_address: str
    user_agent: str
    is_active: bool

class AccessControlManager:
    """
    Manages access control for the system.
    
    Features:
    - User authentication
    - Role-based access control (RBAC)
    - Permission management
    - Access policies
    - Session management
    """
    
    def __init__(
        self,
        config_dir: str = "security/access_control",
        token_secret: Optional[str] = None,
        token_expiry: int = 3600,  # 1 hour
        max_sessions: int = 5
    ):
        """
        Initialize the access control manager.
        
        Args:
            config_dir: Directory for configuration storage
            token_secret: Secret for JWT token generation
            token_expiry: Token expiry time in seconds
            max_sessions: Maximum concurrent sessions per user
        """
        self.config_dir = config_dir
        self.token_secret = token_secret or os.urandom(32).hex()
        self.token_expiry = token_expiry
        self.max_sessions = max_sessions
        
        # Initialize storage
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Session] = {}
        self.role_permissions: Dict[Role, Dict[Resource, Set[Permission]]] = {}
        
        # Initialize configuration
        self._setup_storage()
        self._load_config()
        self._setup_default_roles()
    
    def _setup_storage(self) -> None:
        """Set up configuration storage."""
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Create subdirectories
        for subdir in ['users', 'sessions', 'roles']:
            os.makedirs(
                os.path.join(self.config_dir, subdir),
                exist_ok=True
            )
        
        logger.info(f"Initialized access control storage in {self.config_dir}")
    
    def _load_config(self) -> None:
        """Load configuration from storage."""
        try:
            # Load users
            users_dir = os.path.join(self.config_dir, 'users')
            for filename in os.listdir(users_dir):
                if filename.endswith('.json'):
                    with open(os.path.join(users_dir, filename), 'r') as f:
                        data = json.load(f)
                        user = User(
                            id=data['id'],
                            username=data['username'],
                            password_hash=bytes.fromhex(data['password_hash']),
                            roles={Role(r) for r in data['roles']},
                            permissions={
                                Resource(r): {Permission(p) for p in perms}
                                for r, perms in data['permissions'].items()
                            },
                            metadata=data['metadata'],
                            created_at=datetime.fromisoformat(data['created_at']),
                            last_login=(
                                datetime.fromisoformat(data['last_login'])
                                if data.get('last_login') else None
                            ),
                            is_active=data['is_active']
                        )
                        self.users[user.id] = user
            
            # Load sessions
            sessions_dir = os.path.join(self.config_dir, 'sessions')
            for filename in os.listdir(sessions_dir):
                if filename.endswith('.json'):
                    with open(os.path.join(sessions_dir, filename), 'r') as f:
                        data = json.load(f)
                        session = Session(
                            id=data['id'],
                            user_id=data['user_id'],
                            token=data['token'],
                            created_at=datetime.fromisoformat(data['created_at']),
                            expires_at=datetime.fromisoformat(data['expires_at']),
                            last_activity=datetime.fromisoformat(
                                data['last_activity']
                            ),
                            ip_address=data['ip_address'],
                            user_agent=data['user_agent'],
                            is_active=data['is_active']
                        )
                        if session.is_active and session.expires_at > datetime.utcnow():
                            self.sessions[session.id] = session
            
            logger.info("Loaded access control configuration")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def _setup_default_roles(self) -> None:
        """Set up default role permissions."""
        # Admin role
        self.role_permissions[Role.ADMIN] = {
            resource: set(Permission)
            for resource in Resource
        }
        
        # Manager role
        self.role_permissions[Role.MANAGER] = {
            Resource.SYSTEM: {Permission.READ},
            Resource.AGENT: {Permission.READ, Permission.WRITE, Permission.EXECUTE},
            Resource.DATA: {Permission.READ, Permission.WRITE},
            Resource.CONFIG: {Permission.READ},
            Resource.USER: {Permission.READ},
            Resource.LOG: {Permission.READ}
        }
        
        # User role
        self.role_permissions[Role.USER] = {
            Resource.SYSTEM: {Permission.READ},
            Resource.AGENT: {Permission.READ, Permission.EXECUTE},
            Resource.DATA: {Permission.READ},
            Resource.CONFIG: set(),
            Resource.USER: {Permission.READ},
            Resource.LOG: set()
        }
        
        # Guest role
        self.role_permissions[Role.GUEST] = {
            Resource.SYSTEM: {Permission.READ},
            Resource.AGENT: {Permission.READ},
            Resource.DATA: set(),
            Resource.CONFIG: set(),
            Resource.USER: set(),
            Resource.LOG: set()
        }
    
    def create_user(
        self,
        username: str,
        password: str,
        roles: Set[Role],
        metadata: Optional[Dict[str, Any]] = None
    ) -> User:
        """
        Create a new user.
        
        Args:
            username: Username
            password: Plain text password
            roles: User roles
            metadata: Additional user metadata
        
        Returns:
            Created user
        """
        try:
            # Check if username exists
            if any(u.username == username for u in self.users.values()):
                raise ValueError(f"Username {username} already exists")
            
            # Hash password
            password_hash = bcrypt.hashpw(
                password.encode(),
                bcrypt.gensalt()
            )
            
            # Calculate permissions
            permissions: Dict[Resource, Set[Permission]] = {}
            for role in roles:
                for resource, perms in self.role_permissions[role].items():
                    if resource not in permissions:
                        permissions[resource] = set()
                    permissions[resource].update(perms)
            
            # Create user
            user = User(
                id=f"user_{datetime.utcnow().timestamp()}",
                username=username,
                password_hash=password_hash,
                roles=roles,
                permissions=permissions,
                metadata=metadata or {},
                created_at=datetime.utcnow(),
                last_login=None,
                is_active=True
            )
            
            # Save user
            self.users[user.id] = user
            self._save_user(user)
            
            logger.info(f"Created user: {username}")
            return user
            
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            raise
    
    def _save_user(self, user: User) -> None:
        """Save user to storage."""
        try:
            path = os.path.join(
                self.config_dir,
                'users',
                f"{user.id}.json"
            )
            
            with open(path, 'w') as f:
                json.dump({
                    'id': user.id,
                    'username': user.username,
                    'password_hash': user.password_hash.hex(),
                    'roles': [r.value for r in user.roles],
                    'permissions': {
                        r.value: [p.value for p in perms]
                        for r, perms in user.permissions.items()
                    },
                    'metadata': user.metadata,
                    'created_at': user.created_at.isoformat(),
                    'last_login': (
                        user.last_login.isoformat()
                        if user.last_login else None
                    ),
                    'is_active': user.is_active
                }, f)
            
        except Exception as e:
            logger.error(f"Error saving user: {e}")
            raise
    
    def authenticate(
        self,
        username: str,
        password: str,
        ip_address: str,
        user_agent: str
    ) -> Optional[Session]:
        """
        Authenticate a user.
        
        Args:
            username: Username
            password: Plain text password
            ip_address: Client IP address
            user_agent: Client user agent
        
        Returns:
            Session if authentication successful
        """
        try:
            # Find user
            user = next(
                (u for u in self.users.values() if u.username == username),
                None
            )
            
            if not user or not user.is_active:
                return None
            
            # Verify password
            if not bcrypt.checkpw(password.encode(), user.password_hash):
                return None
            
            # Check session limit
            active_sessions = [
                s for s in self.sessions.values()
                if s.user_id == user.id and s.is_active
            ]
            if len(active_sessions) >= self.max_sessions:
                # Invalidate oldest session
                oldest_session = min(
                    active_sessions,
                    key=lambda s: s.last_activity
                )
                self.invalidate_session(oldest_session.id)
            
            # Generate token
            token_data = {
                'user_id': user.id,
                'username': user.username,
                'roles': [r.value for r in user.roles],
                'exp': datetime.utcnow() + timedelta(seconds=self.token_expiry)
            }
            token = jwt.encode(token_data, self.token_secret, algorithm='HS256')
            
            # Create session
            session = Session(
                id=f"session_{datetime.utcnow().timestamp()}",
                user_id=user.id,
                token=token,
                created_at=datetime.utcnow(),
                expires_at=token_data['exp'],
                last_activity=datetime.utcnow(),
                ip_address=ip_address,
                user_agent=user_agent,
                is_active=True
            )
            
            # Save session
            self.sessions[session.id] = session
            self._save_session(session)
            
            # Update user
            user.last_login = datetime.utcnow()
            self._save_user(user)
            
            logger.info(f"User {username} authenticated successfully")
            return session
            
        except Exception as e:
            logger.error(f"Error authenticating user: {e}")
            return None
    
    def _save_session(self, session: Session) -> None:
        """Save session to storage."""
        try:
            path = os.path.join(
                self.config_dir,
                'sessions',
                f"{session.id}.json"
            )
            
            with open(path, 'w') as f:
                json.dump({
                    'id': session.id,
                    'user_id': session.user_id,
                    'token': session.token,
                    'created_at': session.created_at.isoformat(),
                    'expires_at': session.expires_at.isoformat(),
                    'last_activity': session.last_activity.isoformat(),
                    'ip_address': session.ip_address,
                    'user_agent': session.user_agent,
                    'is_active': session.is_active
                }, f)
            
        except Exception as e:
            logger.error(f"Error saving session: {e}")
            raise
    
    def validate_token(self, token: str) -> Optional[User]:
        """
        Validate a JWT token.
        
        Args:
            token: JWT token
        
        Returns:
            User if token is valid
        """
        try:
            # Verify token
            data = jwt.decode(
                token,
                self.token_secret,
                algorithms=['HS256']
            )
            
            # Get user
            user = self.users.get(data['user_id'])
            if not user or not user.is_active:
                return None
            
            # Find session
            session = next(
                (s for s in self.sessions.values()
                if s.token == token and s.is_active),
                None
            )
            
            if not session:
                return None
            
            # Update session activity
            session.last_activity = datetime.utcnow()
            self._save_session(session)
            
            return user
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None
        except Exception as e:
            logger.error(f"Error validating token: {e}")
            return None
    
    def invalidate_session(self, session_id: str) -> bool:
        """
        Invalidate a session.
        
        Args:
            session_id: Session ID
        
        Returns:
            True if successful
        """
        try:
            session = self.sessions.get(session_id)
            if not session:
                return False
            
            session.is_active = False
            self._save_session(session)
            
            logger.info(f"Invalidated session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error invalidating session: {e}")
            return False
    
    def check_permission(
        self,
        user: User,
        resource: Resource,
        permission: Permission
    ) -> bool:
        """
        Check if user has permission.
        
        Args:
            user: User to check
            resource: Resource to access
            permission: Required permission
        
        Returns:
            True if user has permission
        """
        try:
            return (
                resource in user.permissions and
                permission in user.permissions[resource]
            )
        except Exception as e:
            logger.error(f"Error checking permission: {e}")
            return False
    
    def get_user_permissions(
        self,
        user: User,
        resource: Optional[Resource] = None
    ) -> Dict[Resource, Set[Permission]]:
        """
        Get user permissions.
        
        Args:
            user: User to get permissions for
            resource: Optional specific resource
        
        Returns:
            Dictionary of resource permissions
        """
        try:
            if resource:
                return {
                    resource: user.permissions.get(resource, set())
                }
            return user.permissions
            
        except Exception as e:
            logger.error(f"Error getting permissions: {e}")
            return {} 