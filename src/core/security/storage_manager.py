"""
Secure Storage Manager.

This module provides comprehensive secure storage capabilities for the AMPTALK system,
ensuring data protection at rest and during operations.

Author: AMPTALK Team
Date: 2024
"""

import os
import json
import shutil
import sqlite3
import logging
import tempfile
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import serialization

from src.core.security.encryption_manager import EncryptionManager, EncryptionType
from src.core.utils.logging_config import get_logger

logger = get_logger(__name__)

class StorageType(Enum):
    """Types of secure storage."""
    FILE = "file"
    DATABASE = "database"
    MEMORY = "memory"
    TEMPORARY = "temporary"
    BACKUP = "backup"

class StorageOperation(Enum):
    """Types of storage operations."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    BACKUP = "backup"
    RESTORE = "restore"

class SecureStorageManager:
    """
    Manages secure storage operations.
    
    Features:
    - Encrypted file storage
    - Secure database operations
    - Memory protection
    - Secure temporary storage
    - Secure backup and recovery
    """
    
    def __init__(
        self,
        storage_dir: str = "secure_storage",
        encryption_manager: Optional[EncryptionManager] = None,
        max_file_size_mb: int = 100,
        enable_backup: bool = True
    ):
        """
        Initialize the secure storage manager.
        
        Args:
            storage_dir: Base directory for secure storage
            encryption_manager: Optional encryption manager instance
            max_file_size_mb: Maximum file size in MB
            enable_backup: Whether to enable automatic backups
        """
        self.storage_dir = storage_dir
        self.max_file_size = max_file_size_mb * 1024 * 1024  # Convert to bytes
        self.enable_backup = enable_backup
        
        # Initialize or use provided encryption manager
        self.encryption_manager = encryption_manager or EncryptionManager()
        
        # Initialize storage
        self._setup_storage()
        
        # Initialize secure database
        self.db_path = os.path.join(storage_dir, "secure.db")
        self._setup_database()
    
    def _setup_storage(self) -> None:
        """Set up secure storage directories."""
        # Create main storage directory
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Create subdirectories for different storage types
        for storage_type in StorageType:
            os.makedirs(
                os.path.join(self.storage_dir, storage_type.value),
                exist_ok=True
            )
        
        logger.info(f"Initialized secure storage in {self.storage_dir}")
    
    def _setup_database(self) -> None:
        """Set up secure database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create tables
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS secure_data (
                        id TEXT PRIMARY KEY,
                        storage_type TEXT NOT NULL,
                        data BLOB NOT NULL,
                        metadata TEXT,
                        created_at TIMESTAMP NOT NULL,
                        modified_at TIMESTAMP NOT NULL
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS storage_audit (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        operation TEXT NOT NULL,
                        storage_type TEXT NOT NULL,
                        data_id TEXT NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        details TEXT
                    )
                """)
                
                conn.commit()
            
            logger.info("Initialized secure database")
            
        except Exception as e:
            logger.error(f"Error setting up database: {e}")
            raise
    
    def _audit_operation(
        self,
        operation: StorageOperation,
        storage_type: StorageType,
        data_id: str,
        details: Optional[Dict] = None
    ) -> None:
        """Record a storage operation in the audit log."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO storage_audit
                    (operation, storage_type, data_id, timestamp, details)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    operation.value,
                    storage_type.value,
                    data_id,
                    datetime.utcnow().isoformat(),
                    json.dumps(details) if details else None
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error recording audit: {e}")
    
    @contextmanager
    def secure_temporary_file(self):
        """
        Context manager for secure temporary file handling.
        
        Yields:
            Path to secure temporary file
        """
        temp_dir = os.path.join(
            self.storage_dir,
            StorageType.TEMPORARY.value
        )
        
        try:
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(
                dir=temp_dir,
                delete=False
            )
            
            try:
                yield temp_file.name
            finally:
                # Secure delete
                if os.path.exists(temp_file.name):
                    self._secure_delete_file(temp_file.name)
                
        except Exception as e:
            logger.error(f"Error handling temporary file: {e}")
            raise
    
    def _secure_delete_file(self, file_path: str) -> None:
        """Securely delete a file by overwriting with random data."""
        try:
            if os.path.exists(file_path):
                # Get file size
                file_size = os.path.getsize(file_path)
                
                # Overwrite with random data
                with open(file_path, 'wb') as f:
                    f.write(os.urandom(file_size))
                
                # Delete file
                os.unlink(file_path)
                
        except Exception as e:
            logger.error(f"Error securely deleting file: {e}")
            raise
    
    def store_file(
        self,
        file_path: str,
        data: Union[str, bytes],
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Securely store a file.
        
        Args:
            file_path: Path to store the file
            data: Data to store
            metadata: Optional metadata
        
        Returns:
            File ID
        """
        try:
            if isinstance(data, str):
                data = data.encode()
            
            # Check file size
            if len(data) > self.max_file_size:
                raise ValueError("File size exceeds maximum limit")
            
            # Generate file ID
            file_id = f"file_{datetime.utcnow().timestamp()}"
            
            # Encrypt data
            encrypted_data, nonce = self.encryption_manager.encrypt_data(
                data=data,
                encryption_type=EncryptionType.HYBRID
            )
            
            # Store encrypted data
            full_path = os.path.join(self.storage_dir, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            with open(full_path, 'wb') as f:
                f.write(encrypted_data)
            
            # Store metadata in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO secure_data
                    (id, storage_type, data, metadata, created_at, modified_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    file_id,
                    StorageType.FILE.value,
                    nonce,
                    json.dumps(metadata) if metadata else None,
                    datetime.utcnow().isoformat(),
                    datetime.utcnow().isoformat()
                ))
                conn.commit()
            
            # Record operation
            self._audit_operation(
                operation=StorageOperation.WRITE,
                storage_type=StorageType.FILE,
                data_id=file_id,
                details={'path': file_path}
            )
            
            # Create backup if enabled
            if self.enable_backup:
                self._backup_file(file_id, full_path, encrypted_data, nonce)
            
            return file_id
            
        except Exception as e:
            logger.error(f"Error storing file: {e}")
            raise
    
    def retrieve_file(self, file_id: str) -> Tuple[bytes, Optional[Dict]]:
        """
        Retrieve a securely stored file.
        
        Args:
            file_id: File ID
        
        Returns:
            Tuple of (file_data, metadata)
        """
        try:
            # Get file metadata
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT data, metadata FROM secure_data
                    WHERE id = ? AND storage_type = ?
                """, (file_id, StorageType.FILE.value))
                
                result = cursor.fetchone()
                if not result:
                    raise ValueError(f"File {file_id} not found")
                
                nonce, metadata_json = result
            
            # Get file path
            file_path = self._get_file_path(file_id)
            
            # Read encrypted data
            with open(file_path, 'rb') as f:
                encrypted_data = f.read()
            
            # Decrypt data
            decrypted_data = self.encryption_manager.decrypt_data(
                encrypted_data=encrypted_data,
                nonce=nonce,
                encryption_type=EncryptionType.HYBRID
            )
            
            # Parse metadata
            metadata = json.loads(metadata_json) if metadata_json else None
            
            # Record operation
            self._audit_operation(
                operation=StorageOperation.READ,
                storage_type=StorageType.FILE,
                data_id=file_id
            )
            
            return decrypted_data, metadata
            
        except Exception as e:
            logger.error(f"Error retrieving file: {e}")
            raise
    
    def _get_file_path(self, file_id: str) -> str:
        """Get full file path from ID."""
        return os.path.join(
            self.storage_dir,
            StorageType.FILE.value,
            file_id
        )
    
    def _backup_file(
        self,
        file_id: str,
        file_path: str,
        encrypted_data: bytes,
        nonce: bytes
    ) -> None:
        """Create a secure backup of a file."""
        try:
            backup_dir = os.path.join(
                self.storage_dir,
                StorageType.BACKUP.value
            )
            
            backup_path = os.path.join(backup_dir, f"{file_id}.backup")
            
            # Store encrypted data and nonce
            with open(backup_path, 'wb') as f:
                f.write(len(nonce).to_bytes(4, 'big'))
                f.write(nonce)
                f.write(encrypted_data)
            
            # Record operation
            self._audit_operation(
                operation=StorageOperation.BACKUP,
                storage_type=StorageType.BACKUP,
                data_id=file_id,
                details={'backup_path': backup_path}
            )
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            raise
    
    def restore_file(self, file_id: str) -> bool:
        """
        Restore a file from backup.
        
        Args:
            file_id: File ID to restore
        
        Returns:
            True if successful
        """
        try:
            backup_path = os.path.join(
                self.storage_dir,
                StorageType.BACKUP.value,
                f"{file_id}.backup"
            )
            
            if not os.path.exists(backup_path):
                raise ValueError(f"Backup not found for {file_id}")
            
            # Read backup data
            with open(backup_path, 'rb') as f:
                nonce_size = int.from_bytes(f.read(4), 'big')
                nonce = f.read(nonce_size)
                encrypted_data = f.read()
            
            # Restore file
            file_path = self._get_file_path(file_id)
            with open(file_path, 'wb') as f:
                f.write(encrypted_data)
            
            # Update database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE secure_data
                    SET data = ?, modified_at = ?
                    WHERE id = ? AND storage_type = ?
                """, (
                    nonce,
                    datetime.utcnow().isoformat(),
                    file_id,
                    StorageType.FILE.value
                ))
                conn.commit()
            
            # Record operation
            self._audit_operation(
                operation=StorageOperation.RESTORE,
                storage_type=StorageType.FILE,
                data_id=file_id,
                details={'from_backup': backup_path}
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error restoring file: {e}")
            return False
    
    def delete_file(self, file_id: str) -> bool:
        """
        Securely delete a file.
        
        Args:
            file_id: File ID to delete
        
        Returns:
            True if successful
        """
        try:
            file_path = self._get_file_path(file_id)
            
            # Securely delete file
            self._secure_delete_file(file_path)
            
            # Remove from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM secure_data
                    WHERE id = ? AND storage_type = ?
                """, (file_id, StorageType.FILE.value))
                conn.commit()
            
            # Record operation
            self._audit_operation(
                operation=StorageOperation.DELETE,
                storage_type=StorageType.FILE,
                data_id=file_id
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            return False
    
    def get_audit_log(
        self,
        storage_type: Optional[StorageType] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Get audit log entries.
        
        Args:
            storage_type: Filter by storage type
            start_date: Start date filter
            end_date: End date filter
        
        Returns:
            List of audit log entries
        """
        try:
            query = "SELECT * FROM storage_audit WHERE 1=1"
            params = []
            
            if storage_type:
                query += " AND storage_type = ?"
                params.append(storage_type.value)
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.isoformat())
            
            query += " ORDER BY timestamp DESC"
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, params)
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Error getting audit log: {e}")
            return [] 