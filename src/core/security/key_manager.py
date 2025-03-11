"""
Key Management System.

This module provides comprehensive key management capabilities for the AMPTALK system,
ensuring secure handling of cryptographic keys.

Author: AMPTALK Team
Date: 2024
"""

import os
import json
import base64
import logging
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.exceptions import InvalidKey

from src.core.utils.logging_config import get_logger

logger = get_logger(__name__)

class KeyType(Enum):
    """Types of cryptographic keys."""
    MASTER = "master"
    ENCRYPTION = "encryption"
    SIGNING = "signing"
    AUTHENTICATION = "authentication"
    BACKUP = "backup"

class KeyAlgorithm(Enum):
    """Supported key algorithms."""
    RSA = "rsa"
    AES = "aes"
    ED25519 = "ed25519"
    ECDSA = "ecdsa"

class KeyStatus(Enum):
    """Key lifecycle status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    EXPIRED = "expired"
    COMPROMISED = "compromised"
    ROTATED = "rotated"

@dataclass
class KeyMetadata:
    """Key metadata."""
    id: str
    type: KeyType
    algorithm: KeyAlgorithm
    created_at: datetime
    expires_at: datetime
    status: KeyStatus
    version: int
    usage_count: int
    last_used: Optional[datetime]
    last_rotated: Optional[datetime]
    tags: Dict[str, str]

@dataclass
class KeyMaterial:
    """Key material."""
    public_key: Optional[bytes]
    private_key: Optional[bytes]
    symmetric_key: Optional[bytes]
    initialization_vector: Optional[bytes]

class KeyManager:
    """
    Manages cryptographic keys for the system.
    
    Features:
    - Key generation and storage
    - Key rotation and expiry
    - Key backup and recovery
    - Key distribution
    - Key access control
    """
    
    def __init__(
        self,
        keys_dir: str = "security/keys",
        master_key: Optional[bytes] = None,
        rotation_days: int = 90,
        backup_enabled: bool = True
    ):
        """
        Initialize the key manager.
        
        Args:
            keys_dir: Directory for key storage
            master_key: Optional master key
            rotation_days: Days between key rotations
            backup_enabled: Whether to enable key backup
        """
        self.keys_dir = keys_dir
        self.rotation_days = rotation_days
        self.backup_enabled = backup_enabled
        
        # Initialize storage
        self.keys: Dict[str, KeyMetadata] = {}
        self.key_material: Dict[str, KeyMaterial] = {}
        
        # Initialize master key
        self.master_key = master_key or self._generate_master_key()
        
        # Initialize storage
        self._setup_storage()
        self._load_keys()
    
    def _setup_storage(self) -> None:
        """Set up key storage."""
        # Create directories
        os.makedirs(self.keys_dir, exist_ok=True)
        os.makedirs(os.path.join(self.keys_dir, "backup"), exist_ok=True)
        
        # Create subdirectories for key types
        for key_type in KeyType:
            os.makedirs(
                os.path.join(self.keys_dir, key_type.value),
                exist_ok=True
            )
        
        logger.info(f"Initialized key storage in {self.keys_dir}")
    
    def _generate_master_key(self) -> bytes:
        """Generate master key."""
        salt = os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000
        )
        key = kdf.derive(os.urandom(32))
        
        # Store salt
        with open(os.path.join(self.keys_dir, "salt"), "wb") as f:
            f.write(salt)
        
        return key
    
    def _load_keys(self) -> None:
        """Load keys from storage."""
        try:
            # Load metadata
            metadata_path = os.path.join(self.keys_dir, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    data = json.load(f)
                    for key_id, meta in data.items():
                        self.keys[key_id] = KeyMetadata(
                            id=meta["id"],
                            type=KeyType(meta["type"]),
                            algorithm=KeyAlgorithm(meta["algorithm"]),
                            created_at=datetime.fromisoformat(meta["created_at"]),
                            expires_at=datetime.fromisoformat(meta["expires_at"]),
                            status=KeyStatus(meta["status"]),
                            version=meta["version"],
                            usage_count=meta["usage_count"],
                            last_used=(
                                datetime.fromisoformat(meta["last_used"])
                                if meta.get("last_used") else None
                            ),
                            last_rotated=(
                                datetime.fromisoformat(meta["last_rotated"])
                                if meta.get("last_rotated") else None
                            ),
                            tags=meta["tags"]
                        )
            
            # Load key material
            for key_id, metadata in self.keys.items():
                key_path = os.path.join(
                    self.keys_dir,
                    metadata.type.value,
                    f"{key_id}.key"
                )
                
                if os.path.exists(key_path):
                    # Decrypt key material
                    with open(key_path, "rb") as f:
                        encrypted_data = f.read()
                    
                    decrypted_data = self._decrypt_key_material(encrypted_data)
                    key_data = json.loads(decrypted_data.decode())
                    
                    self.key_material[key_id] = KeyMaterial(
                        public_key=(
                            base64.b64decode(key_data["public_key"])
                            if key_data.get("public_key") else None
                        ),
                        private_key=(
                            base64.b64decode(key_data["private_key"])
                            if key_data.get("private_key") else None
                        ),
                        symmetric_key=(
                            base64.b64decode(key_data["symmetric_key"])
                            if key_data.get("symmetric_key") else None
                        ),
                        initialization_vector=(
                            base64.b64decode(key_data["initialization_vector"])
                            if key_data.get("initialization_vector") else None
                        )
                    )
            
            logger.info(f"Loaded {len(self.keys)} keys")
            
        except Exception as e:
            logger.error(f"Error loading keys: {e}")
            raise
    
    def _save_keys(self) -> None:
        """Save keys to storage."""
        try:
            # Save metadata
            metadata = {}
            for key_id, meta in self.keys.items():
                metadata[key_id] = {
                    "id": meta.id,
                    "type": meta.type.value,
                    "algorithm": meta.algorithm.value,
                    "created_at": meta.created_at.isoformat(),
                    "expires_at": meta.expires_at.isoformat(),
                    "status": meta.status.value,
                    "version": meta.version,
                    "usage_count": meta.usage_count,
                    "last_used": (
                        meta.last_used.isoformat()
                        if meta.last_used else None
                    ),
                    "last_rotated": (
                        meta.last_rotated.isoformat()
                        if meta.last_rotated else None
                    ),
                    "tags": meta.tags
                }
            
            metadata_path = os.path.join(self.keys_dir, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)
            
            # Save key material
            for key_id, material in self.key_material.items():
                metadata = self.keys[key_id]
                key_data = {
                    "public_key": (
                        base64.b64encode(material.public_key).decode()
                        if material.public_key else None
                    ),
                    "private_key": (
                        base64.b64encode(material.private_key).decode()
                        if material.private_key else None
                    ),
                    "symmetric_key": (
                        base64.b64encode(material.symmetric_key).decode()
                        if material.symmetric_key else None
                    ),
                    "initialization_vector": (
                        base64.b64encode(material.initialization_vector).decode()
                        if material.initialization_vector else None
                    )
                }
                
                # Encrypt key material
                encrypted_data = self._encrypt_key_material(
                    json.dumps(key_data).encode()
                )
                
                key_path = os.path.join(
                    self.keys_dir,
                    metadata.type.value,
                    f"{key_id}.key"
                )
                
                with open(key_path, "wb") as f:
                    f.write(encrypted_data)
            
            logger.info("Saved keys to storage")
            
        except Exception as e:
            logger.error(f"Error saving keys: {e}")
            raise
    
    def _encrypt_key_material(self, data: bytes) -> bytes:
        """Encrypt key material using master key."""
        fernet = Fernet(base64.b64encode(self.master_key))
        return fernet.encrypt(data)
    
    def _decrypt_key_material(self, encrypted_data: bytes) -> bytes:
        """Decrypt key material using master key."""
        fernet = Fernet(base64.b64encode(self.master_key))
        return fernet.decrypt(encrypted_data)
    
    def generate_key(
        self,
        key_type: KeyType,
        algorithm: KeyAlgorithm,
        expiry_days: Optional[int] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Generate a new key.
        
        Args:
            key_type: Type of key
            algorithm: Key algorithm
            expiry_days: Days until key expires
            tags: Optional key tags
        
        Returns:
            Key ID
        """
        try:
            # Generate key ID
            key_id = f"key_{datetime.utcnow().timestamp()}"
            
            # Generate key material
            if algorithm == KeyAlgorithm.RSA:
                private_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=2048
                )
                public_key = private_key.public_key()
                
                key_material = KeyMaterial(
                    public_key=public_key.public_bytes(
                        encoding=serialization.Encoding.DER,
                        format=serialization.PublicFormat.SubjectPublicKeyInfo
                    ),
                    private_key=private_key.private_bytes(
                        encoding=serialization.Encoding.DER,
                        format=serialization.PrivateFormat.PKCS8,
                        encryption_algorithm=serialization.NoEncryption()
                    ),
                    symmetric_key=None,
                    initialization_vector=None
                )
                
            elif algorithm == KeyAlgorithm.AES:
                key = AESGCM.generate_key(bit_length=256)
                iv = os.urandom(12)
                
                key_material = KeyMaterial(
                    public_key=None,
                    private_key=None,
                    symmetric_key=key,
                    initialization_vector=iv
                )
                
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            
            # Create metadata
            expiry = expiry_days or self.rotation_days
            metadata = KeyMetadata(
                id=key_id,
                type=key_type,
                algorithm=algorithm,
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(days=expiry),
                status=KeyStatus.ACTIVE,
                version=1,
                usage_count=0,
                last_used=None,
                last_rotated=None,
                tags=tags or {}
            )
            
            # Store key
            self.keys[key_id] = metadata
            self.key_material[key_id] = key_material
            
            # Save to storage
            self._save_keys()
            
            # Create backup
            if self.backup_enabled:
                self._backup_key(key_id)
            
            logger.info(f"Generated {algorithm.value} key: {key_id}")
            return key_id
            
        except Exception as e:
            logger.error(f"Error generating key: {e}")
            raise
    
    def get_key(
        self,
        key_id: str,
        include_private: bool = False
    ) -> Tuple[KeyMetadata, KeyMaterial]:
        """
        Get key metadata and material.
        
        Args:
            key_id: Key ID
            include_private: Whether to include private key material
        
        Returns:
            Tuple of (metadata, material)
        """
        try:
            metadata = self.keys.get(key_id)
            if not metadata:
                raise ValueError(f"Key not found: {key_id}")
            
            if metadata.status not in [KeyStatus.ACTIVE, KeyStatus.INACTIVE]:
                raise ValueError(f"Key is not available: {key_id}")
            
            material = self.key_material.get(key_id)
            if not material:
                raise ValueError(f"Key material not found: {key_id}")
            
            # Update usage statistics
            metadata.usage_count += 1
            metadata.last_used = datetime.utcnow()
            self._save_keys()
            
            # Remove private material if not requested
            if not include_private:
                material = KeyMaterial(
                    public_key=material.public_key,
                    private_key=None,
                    symmetric_key=None,
                    initialization_vector=material.initialization_vector
                )
            
            return metadata, material
            
        except Exception as e:
            logger.error(f"Error getting key: {e}")
            raise
    
    def rotate_key(
        self,
        key_id: str,
        expiry_days: Optional[int] = None
    ) -> str:
        """
        Rotate a key.
        
        Args:
            key_id: Key ID to rotate
            expiry_days: Days until new key expires
        
        Returns:
            New key ID
        """
        try:
            # Get existing key
            old_metadata = self.keys.get(key_id)
            if not old_metadata:
                raise ValueError(f"Key not found: {key_id}")
            
            # Generate new key
            new_key_id = self.generate_key(
                key_type=old_metadata.type,
                algorithm=old_metadata.algorithm,
                expiry_days=expiry_days,
                tags=old_metadata.tags
            )
            
            # Update old key
            old_metadata.status = KeyStatus.ROTATED
            old_metadata.last_rotated = datetime.utcnow()
            
            # Save changes
            self._save_keys()
            
            logger.info(f"Rotated key {key_id} to {new_key_id}")
            return new_key_id
            
        except Exception as e:
            logger.error(f"Error rotating key: {e}")
            raise
    
    def _backup_key(self, key_id: str) -> None:
        """Create key backup."""
        try:
            metadata = self.keys[key_id]
            material = self.key_material[key_id]
            
            backup_data = {
                "metadata": {
                    "id": metadata.id,
                    "type": metadata.type.value,
                    "algorithm": metadata.algorithm.value,
                    "created_at": metadata.created_at.isoformat(),
                    "expires_at": metadata.expires_at.isoformat(),
                    "status": metadata.status.value,
                    "version": metadata.version,
                    "tags": metadata.tags
                },
                "material": {
                    "public_key": (
                        base64.b64encode(material.public_key).decode()
                        if material.public_key else None
                    ),
                    "private_key": (
                        base64.b64encode(material.private_key).decode()
                        if material.private_key else None
                    ),
                    "symmetric_key": (
                        base64.b64encode(material.symmetric_key).decode()
                        if material.symmetric_key else None
                    ),
                    "initialization_vector": (
                        base64.b64encode(material.initialization_vector).decode()
                        if material.initialization_vector else None
                    )
                }
            }
            
            # Encrypt backup
            encrypted_data = self._encrypt_key_material(
                json.dumps(backup_data).encode()
            )
            
            # Save backup
            backup_path = os.path.join(
                self.keys_dir,
                "backup",
                f"{key_id}.backup"
            )
            
            with open(backup_path, "wb") as f:
                f.write(encrypted_data)
            
            logger.info(f"Created backup for key: {key_id}")
            
        except Exception as e:
            logger.error(f"Error creating key backup: {e}")
            raise
    
    def restore_key(self, key_id: str) -> bool:
        """
        Restore a key from backup.
        
        Args:
            key_id: Key ID to restore
        
        Returns:
            True if successful
        """
        try:
            backup_path = os.path.join(
                self.keys_dir,
                "backup",
                f"{key_id}.backup"
            )
            
            if not os.path.exists(backup_path):
                raise ValueError(f"Backup not found: {key_id}")
            
            # Read and decrypt backup
            with open(backup_path, "rb") as f:
                encrypted_data = f.read()
            
            decrypted_data = self._decrypt_key_material(encrypted_data)
            backup_data = json.loads(decrypted_data.decode())
            
            # Restore metadata
            meta = backup_data["metadata"]
            metadata = KeyMetadata(
                id=meta["id"],
                type=KeyType(meta["type"]),
                algorithm=KeyAlgorithm(meta["algorithm"]),
                created_at=datetime.fromisoformat(meta["created_at"]),
                expires_at=datetime.fromisoformat(meta["expires_at"]),
                status=KeyStatus.ACTIVE,
                version=meta["version"],
                usage_count=0,
                last_used=None,
                last_rotated=None,
                tags=meta["tags"]
            )
            
            # Restore material
            mat = backup_data["material"]
            material = KeyMaterial(
                public_key=(
                    base64.b64decode(mat["public_key"])
                    if mat.get("public_key") else None
                ),
                private_key=(
                    base64.b64decode(mat["private_key"])
                    if mat.get("private_key") else None
                ),
                symmetric_key=(
                    base64.b64decode(mat["symmetric_key"])
                    if mat.get("symmetric_key") else None
                ),
                initialization_vector=(
                    base64.b64decode(mat["initialization_vector"])
                    if mat.get("initialization_vector") else None
                )
            )
            
            # Store restored key
            self.keys[key_id] = metadata
            self.key_material[key_id] = material
            
            # Save to storage
            self._save_keys()
            
            logger.info(f"Restored key: {key_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring key: {e}")
            return False
    
    def list_keys(
        self,
        key_type: Optional[KeyType] = None,
        status: Optional[KeyStatus] = None,
        algorithm: Optional[KeyAlgorithm] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> List[KeyMetadata]:
        """
        List keys matching criteria.
        
        Args:
            key_type: Filter by key type
            status: Filter by status
            algorithm: Filter by algorithm
            tags: Filter by tags
        
        Returns:
            List of matching key metadata
        """
        try:
            results = []
            
            for metadata in self.keys.values():
                if key_type and metadata.type != key_type:
                    continue
                    
                if status and metadata.status != status:
                    continue
                    
                if algorithm and metadata.algorithm != algorithm:
                    continue
                    
                if tags:
                    match = True
                    for key, value in tags.items():
                        if metadata.tags.get(key) != value:
                            match = False
                            break
                    if not match:
                        continue
                
                results.append(metadata)
            
            return results
            
        except Exception as e:
            logger.error(f"Error listing keys: {e}")
            return []
    
    def update_key_status(
        self,
        key_id: str,
        status: KeyStatus
    ) -> bool:
        """
        Update key status.
        
        Args:
            key_id: Key ID
            status: New status
        
        Returns:
            True if successful
        """
        try:
            metadata = self.keys.get(key_id)
            if not metadata:
                raise ValueError(f"Key not found: {key_id}")
            
            metadata.status = status
            self._save_keys()
            
            logger.info(f"Updated key {key_id} status to {status.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating key status: {e}")
            return False
    
    def delete_key(self, key_id: str) -> bool:
        """
        Delete a key.
        
        Args:
            key_id: Key ID
        
        Returns:
            True if successful
        """
        try:
            metadata = self.keys.get(key_id)
            if not metadata:
                raise ValueError(f"Key not found: {key_id}")
            
            # Remove from storage
            key_path = os.path.join(
                self.keys_dir,
                metadata.type.value,
                f"{key_id}.key"
            )
            
            if os.path.exists(key_path):
                os.remove(key_path)
            
            # Remove from memory
            del self.keys[key_id]
            if key_id in self.key_material:
                del self.key_material[key_id]
            
            # Save changes
            self._save_keys()
            
            logger.info(f"Deleted key: {key_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting key: {e}")
            return False 