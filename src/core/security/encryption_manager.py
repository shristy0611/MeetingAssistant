"""
End-to-End Encryption Manager.

This module provides comprehensive encryption capabilities for the AMPTALK system,
ensuring secure data transmission and storage.

Author: AMPTALK Team
Date: 2024
"""

import os
import json
import base64
import logging
from enum import Enum
from typing import Dict, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime, timedelta

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidKey

from src.core.utils.logging_config import get_logger

logger = get_logger(__name__)

class EncryptionType(Enum):
    """Types of encryption."""
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"
    HYBRID = "hybrid"

class KeyType(Enum):
    """Types of encryption keys."""
    MASTER = "master"
    SESSION = "session"
    DATA = "data"
    BACKUP = "backup"

class EncryptionManager:
    """
    Manages end-to-end encryption for secure data handling.
    
    Features:
    - Symmetric encryption (AES-GCM)
    - Asymmetric encryption (RSA)
    - Hybrid encryption
    - Key rotation
    - Secure key storage
    """
    
    def __init__(
        self,
        keys_dir: str = "security/keys",
        key_size: int = 2048,
        rotation_days: int = 30,
        use_hardware_security: bool = False
    ):
        """
        Initialize the encryption manager.
        
        Args:
            keys_dir: Directory for key storage
            key_size: RSA key size in bits
            rotation_days: Days between key rotations
            use_hardware_security: Whether to use hardware security module
        """
        self.keys_dir = keys_dir
        self.key_size = key_size
        self.rotation_days = rotation_days
        self.use_hardware_security = use_hardware_security
        
        # Initialize key storage
        self._setup_key_storage()
        
        # Initialize encryption keys
        self.active_keys: Dict[KeyType, Dict] = {}
        self._initialize_keys()
    
    def _setup_key_storage(self) -> None:
        """Set up secure key storage."""
        os.makedirs(self.keys_dir, exist_ok=True)
        
        # Create subdirectories for different key types
        for key_type in KeyType:
            os.makedirs(
                os.path.join(self.keys_dir, key_type.value),
                exist_ok=True
            )
        
        logger.info(f"Initialized key storage in {self.keys_dir}")
    
    def _initialize_keys(self) -> None:
        """Initialize encryption keys."""
        try:
            for key_type in KeyType:
                key_path = os.path.join(
                    self.keys_dir,
                    key_type.value,
                    "active_key.json"
                )
                
                if os.path.exists(key_path):
                    # Load existing key
                    with open(key_path, 'r') as f:
                        key_data = json.load(f)
                        
                    if self._should_rotate_key(key_data['created_at']):
                        # Generate new key if rotation needed
                        self._generate_key_pair(key_type)
                    else:
                        self.active_keys[key_type] = key_data
                else:
                    # Generate new key pair
                    self._generate_key_pair(key_type)
            
            logger.info("Initialized encryption keys")
            
        except Exception as e:
            logger.error(f"Error initializing keys: {e}")
            raise
    
    def _should_rotate_key(self, created_at: str) -> bool:
        """Check if key should be rotated."""
        creation_date = datetime.fromisoformat(created_at)
        age = datetime.utcnow() - creation_date
        return age.days >= self.rotation_days
    
    def _generate_key_pair(self, key_type: KeyType) -> None:
        """Generate a new key pair."""
        try:
            # Generate RSA key pair
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=self.key_size
            )
            public_key = private_key.public_key()
            
            # Serialize keys
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            # Generate symmetric key
            symmetric_key = AESGCM.generate_key(bit_length=256)
            
            # Store key data
            key_data = {
                'type': key_type.value,
                'created_at': datetime.utcnow().isoformat(),
                'private_key': base64.b64encode(private_pem).decode(),
                'public_key': base64.b64encode(public_pem).decode(),
                'symmetric_key': base64.b64encode(symmetric_key).decode()
            }
            
            key_path = os.path.join(
                self.keys_dir,
                key_type.value,
                "active_key.json"
            )
            
            with open(key_path, 'w') as f:
                json.dump(key_data, f)
            
            self.active_keys[key_type] = key_data
            
            logger.info(f"Generated new {key_type.value} key pair")
            
        except Exception as e:
            logger.error(f"Error generating key pair: {e}")
            raise
    
    def encrypt_data(
        self,
        data: Union[str, bytes],
        encryption_type: EncryptionType = EncryptionType.HYBRID,
        key_type: KeyType = KeyType.DATA
    ) -> Tuple[bytes, bytes]:
        """
        Encrypt data using specified encryption type.
        
        Args:
            data: Data to encrypt
            encryption_type: Type of encryption to use
            key_type: Type of key to use
        
        Returns:
            Tuple of (encrypted_data, nonce/iv)
        """
        try:
            if isinstance(data, str):
                data = data.encode()
            
            if encryption_type == EncryptionType.SYMMETRIC:
                # Use AES-GCM
                key_bytes = base64.b64decode(
                    self.active_keys[key_type]['symmetric_key']
                )
                aesgcm = AESGCM(key_bytes)
                nonce = os.urandom(12)
                encrypted = aesgcm.encrypt(nonce, data, None)
                return encrypted, nonce
                
            elif encryption_type == EncryptionType.ASYMMETRIC:
                # Use RSA
                public_key = serialization.load_pem_public_key(
                    base64.b64decode(self.active_keys[key_type]['public_key'])
                )
                
                encrypted = public_key.encrypt(
                    data,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                return encrypted, b""
                
            else:  # HYBRID
                # Generate session key
                session_key = AESGCM.generate_key(bit_length=256)
                
                # Encrypt session key with RSA
                public_key = serialization.load_pem_public_key(
                    base64.b64decode(self.active_keys[key_type]['public_key'])
                )
                
                encrypted_session_key = public_key.encrypt(
                    session_key,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                
                # Encrypt data with session key
                aesgcm = AESGCM(session_key)
                nonce = os.urandom(12)
                encrypted_data = aesgcm.encrypt(nonce, data, None)
                
                # Combine encrypted session key and data
                return (
                    encrypted_session_key + encrypted_data,
                    nonce
                )
            
        except Exception as e:
            logger.error(f"Error encrypting data: {e}")
            raise
    
    def decrypt_data(
        self,
        encrypted_data: bytes,
        nonce: bytes,
        encryption_type: EncryptionType = EncryptionType.HYBRID,
        key_type: KeyType = KeyType.DATA
    ) -> bytes:
        """
        Decrypt data using specified encryption type.
        
        Args:
            encrypted_data: Data to decrypt
            nonce: Nonce/IV used for encryption
            encryption_type: Type of encryption used
            key_type: Type of key to use
        
        Returns:
            Decrypted data
        """
        try:
            if encryption_type == EncryptionType.SYMMETRIC:
                # Use AES-GCM
                key_bytes = base64.b64decode(
                    self.active_keys[key_type]['symmetric_key']
                )
                aesgcm = AESGCM(key_bytes)
                return aesgcm.decrypt(nonce, encrypted_data, None)
                
            elif encryption_type == EncryptionType.ASYMMETRIC:
                # Use RSA
                private_key = serialization.load_pem_private_key(
                    base64.b64decode(self.active_keys[key_type]['private_key']),
                    password=None
                )
                
                decrypted = private_key.decrypt(
                    encrypted_data,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                return decrypted
                
            else:  # HYBRID
                # Split encrypted session key and data
                session_key_size = self.key_size // 8
                encrypted_session_key = encrypted_data[:session_key_size]
                encrypted_data = encrypted_data[session_key_size:]
                
                # Decrypt session key
                private_key = serialization.load_pem_private_key(
                    base64.b64decode(self.active_keys[key_type]['private_key']),
                    password=None
                )
                
                session_key = private_key.decrypt(
                    encrypted_session_key,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                
                # Decrypt data with session key
                aesgcm = AESGCM(session_key)
                return aesgcm.decrypt(nonce, encrypted_data, None)
            
        except Exception as e:
            logger.error(f"Error decrypting data: {e}")
            raise
    
    def rotate_keys(self, key_type: Optional[KeyType] = None) -> None:
        """
        Rotate encryption keys.
        
        Args:
            key_type: Specific key type to rotate, or all if None
        """
        try:
            if key_type:
                self._generate_key_pair(key_type)
            else:
                for kt in KeyType:
                    self._generate_key_pair(kt)
            
            logger.info(
                f"Rotated {'all' if key_type is None else key_type.value} keys"
            )
            
        except Exception as e:
            logger.error(f"Error rotating keys: {e}")
            raise
    
    def export_public_key(self, key_type: KeyType = KeyType.DATA) -> str:
        """
        Export public key for sharing.
        
        Args:
            key_type: Type of key to export
        
        Returns:
            Base64 encoded public key
        """
        return self.active_keys[key_type]['public_key']
    
    def import_public_key(
        self,
        public_key: str,
        key_type: KeyType = KeyType.DATA
    ) -> None:
        """
        Import a public key.
        
        Args:
            public_key: Base64 encoded public key
            key_type: Type of key to import
        """
        try:
            # Verify key format
            key_bytes = base64.b64decode(public_key)
            serialization.load_pem_public_key(key_bytes)
            
            # Store key
            key_path = os.path.join(
                self.keys_dir,
                key_type.value,
                "imported_key.json"
            )
            
            with open(key_path, 'w') as f:
                json.dump({
                    'type': key_type.value,
                    'created_at': datetime.utcnow().isoformat(),
                    'public_key': public_key
                }, f)
            
            logger.info(f"Imported {key_type.value} public key")
            
        except Exception as e:
            logger.error(f"Error importing public key: {e}")
            raise 