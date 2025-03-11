"""
Consent Management System.

This module provides comprehensive consent management capabilities for the AMPTALK system,
ensuring proper handling of user consent for data processing.

Author: AMPTALK Team
Date: 2024
"""

import os
import json
import logging
import sqlite3
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from uuid import uuid4

from src.core.utils.logging_config import get_logger

logger = get_logger(__name__)

class ConsentStatus(Enum):
    """Status of user consent."""
    GRANTED = "granted"
    DENIED = "denied"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"

class ConsentPurpose(Enum):
    """Purposes for data processing."""
    ESSENTIAL = "essential"
    FUNCTIONAL = "functional"
    ANALYTICS = "analytics"
    MARKETING = "marketing"
    THIRD_PARTY = "third_party"

@dataclass
class ConsentRecord:
    """User consent record."""
    id: str
    user_id: str
    purpose: ConsentPurpose
    status: ConsentStatus
    version: str
    granted_at: Optional[datetime]
    expires_at: Optional[datetime]
    withdrawn_at: Optional[datetime]
    ip_address: Optional[str]
    user_agent: Optional[str]
    proof: str  # Hash of consent data
    metadata: Dict[str, Any]

class ConsentManager:
    """
    Manages user consent for data processing.
    
    Features:
    - Consent collection and storage
    - Consent versioning
    - Consent audit trails
    - Consent withdrawal
    - Purpose-based consent
    """
    
    def __init__(
        self,
        storage_dir: str = "privacy/consent",
        default_expiry_days: Optional[int] = None,
        require_proof: bool = True
    ):
        """
        Initialize the consent manager.
        
        Args:
            storage_dir: Directory for consent storage
            default_expiry_days: Default consent expiry period
            require_proof: Whether to require proof of consent
        """
        self.storage_dir = storage_dir
        self.default_expiry_days = default_expiry_days
        self.require_proof = require_proof
        
        # Initialize storage
        self._setup_storage()
    
    def _setup_storage(self) -> None:
        """Set up consent storage."""
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Initialize database
        db_path = os.path.join(self.storage_dir, "consent.db")
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS consent_records (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    purpose TEXT NOT NULL,
                    status TEXT NOT NULL,
                    version TEXT NOT NULL,
                    granted_at TIMESTAMP,
                    expires_at TIMESTAMP,
                    withdrawn_at TIMESTAMP,
                    ip_address TEXT,
                    user_agent TEXT,
                    proof TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            
            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_id
                ON consent_records(user_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_purpose
                ON consent_records(purpose)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_status
                ON consent_records(status)
            """)
            
            conn.commit()
        
        logger.info(f"Initialized consent storage in {self.storage_dir}")
    
    def _generate_proof(self, data: Dict[str, Any]) -> str:
        """Generate proof of consent."""
        import hashlib
        message = json.dumps(data, sort_keys=True).encode()
        return hashlib.sha256(message).hexdigest()
    
    def record_consent(
        self,
        user_id: str,
        purpose: ConsentPurpose,
        status: ConsentStatus,
        version: str,
        expiry_days: Optional[int] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Record user consent.
        
        Args:
            user_id: User identifier
            purpose: Purpose of consent
            status: Consent status
            version: Consent version
            expiry_days: Days until consent expires
            ip_address: User's IP address
            user_agent: User's browser agent
            metadata: Additional metadata
        
        Returns:
            Consent record ID
        """
        try:
            # Generate record ID
            record_id = str(uuid4())
            
            # Calculate timestamps
            now = datetime.utcnow()
            granted_at = now if status == ConsentStatus.GRANTED else None
            expires_at = (
                now + timedelta(days=expiry_days or self.default_expiry_days)
                if (expiry_days or self.default_expiry_days) and
                status == ConsentStatus.GRANTED
                else None
            )
            
            # Generate proof
            proof_data = {
                'id': record_id,
                'user_id': user_id,
                'purpose': purpose.value,
                'status': status.value,
                'version': version,
                'granted_at': granted_at.isoformat() if granted_at else None,
                'expires_at': expires_at.isoformat() if expires_at else None,
                'ip_address': ip_address,
                'user_agent': user_agent,
                'metadata': metadata or {}
            }
            proof = self._generate_proof(proof_data)
            
            # Store record
            db_path = os.path.join(self.storage_dir, "consent.db")
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO consent_records
                    (id, user_id, purpose, status, version,
                     granted_at, expires_at, withdrawn_at,
                     ip_address, user_agent, proof, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record_id,
                    user_id,
                    purpose.value,
                    status.value,
                    version,
                    granted_at.isoformat() if granted_at else None,
                    expires_at.isoformat() if expires_at else None,
                    None,  # withdrawn_at
                    ip_address,
                    user_agent,
                    proof,
                    json.dumps(metadata or {})
                ))
                conn.commit()
            
            logger.info(
                f"Recorded {status.value} consent for {user_id} "
                f"({purpose.value})"
            )
            return record_id
            
        except Exception as e:
            logger.error(f"Error recording consent: {e}")
            raise
    
    def withdraw_consent(
        self,
        record_id: str,
        reason: Optional[str] = None
    ) -> bool:
        """
        Withdraw previously granted consent.
        
        Args:
            record_id: Consent record ID
            reason: Optional withdrawal reason
        
        Returns:
            True if successful
        """
        try:
            db_path = os.path.join(self.storage_dir, "consent.db")
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                # Get current record
                cursor.execute("""
                    SELECT user_id, purpose, metadata
                    FROM consent_records
                    WHERE id = ? AND status = ?
                """, (record_id, ConsentStatus.GRANTED.value))
                
                result = cursor.fetchone()
                if not result:
                    return False
                
                user_id, purpose, metadata_str = result
                metadata = json.loads(metadata_str)
                
                # Update withdrawal info
                metadata['withdrawal_reason'] = reason
                withdrawn_at = datetime.utcnow()
                
                # Update record
                cursor.execute("""
                    UPDATE consent_records
                    SET status = ?, withdrawn_at = ?, metadata = ?
                    WHERE id = ?
                """, (
                    ConsentStatus.WITHDRAWN.value,
                    withdrawn_at.isoformat(),
                    json.dumps(metadata),
                    record_id
                ))
                conn.commit()
            
            logger.info(
                f"Withdrew consent {record_id} for {user_id} "
                f"({purpose})"
            )
            return True
            
        except Exception as e:
            logger.error(f"Error withdrawing consent: {e}")
            return False
    
    def check_consent(
        self,
        user_id: str,
        purpose: ConsentPurpose
    ) -> Optional[ConsentRecord]:
        """
        Check current consent status.
        
        Args:
            user_id: User identifier
            purpose: Purpose to check
        
        Returns:
            Current consent record if found
        """
        try:
            db_path = os.path.join(self.storage_dir, "consent.db")
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                # Get latest record
                cursor.execute("""
                    SELECT * FROM consent_records
                    WHERE user_id = ? AND purpose = ?
                    ORDER BY granted_at DESC
                    LIMIT 1
                """, (user_id, purpose.value))
                
                result = cursor.fetchone()
                if not result:
                    return None
                
                # Parse record
                (
                    id, user_id, purpose_str, status_str, version,
                    granted_at_str, expires_at_str, withdrawn_at_str,
                    ip_address, user_agent, proof, metadata_str
                ) = result
                
                record = ConsentRecord(
                    id=id,
                    user_id=user_id,
                    purpose=ConsentPurpose(purpose_str),
                    status=ConsentStatus(status_str),
                    version=version,
                    granted_at=(
                        datetime.fromisoformat(granted_at_str)
                        if granted_at_str else None
                    ),
                    expires_at=(
                        datetime.fromisoformat(expires_at_str)
                        if expires_at_str else None
                    ),
                    withdrawn_at=(
                        datetime.fromisoformat(withdrawn_at_str)
                        if withdrawn_at_str else None
                    ),
                    ip_address=ip_address,
                    user_agent=user_agent,
                    proof=proof,
                    metadata=json.loads(metadata_str)
                )
                
                # Check expiry
                if (
                    record.status == ConsentStatus.GRANTED and
                    record.expires_at and
                    record.expires_at <= datetime.utcnow()
                ):
                    # Update status to expired
                    cursor.execute("""
                        UPDATE consent_records
                        SET status = ?
                        WHERE id = ?
                    """, (ConsentStatus.EXPIRED.value, record.id))
                    conn.commit()
                    
                    record.status = ConsentStatus.EXPIRED
                
                return record
                
        except Exception as e:
            logger.error(f"Error checking consent: {e}")
            return None
    
    def get_consent_history(
        self,
        user_id: str,
        purpose: Optional[ConsentPurpose] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[ConsentRecord]:
        """
        Get consent history for a user.
        
        Args:
            user_id: User identifier
            purpose: Optional purpose filter
            start_date: Start date filter
            end_date: End date filter
        
        Returns:
            List of consent records
        """
        try:
            query = """
                SELECT * FROM consent_records
                WHERE user_id = ?
            """
            params = [user_id]
            
            if purpose:
                query += " AND purpose = ?"
                params.append(purpose.value)
            
            if start_date:
                query += " AND granted_at >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND granted_at <= ?"
                params.append(end_date.isoformat())
            
            query += " ORDER BY granted_at DESC"
            
            records = []
            db_path = os.path.join(self.storage_dir, "consent.db")
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                
                for result in cursor.fetchall():
                    (
                        id, user_id, purpose_str, status_str, version,
                        granted_at_str, expires_at_str, withdrawn_at_str,
                        ip_address, user_agent, proof, metadata_str
                    ) = result
                    
                    record = ConsentRecord(
                        id=id,
                        user_id=user_id,
                        purpose=ConsentPurpose(purpose_str),
                        status=ConsentStatus(status_str),
                        version=version,
                        granted_at=(
                            datetime.fromisoformat(granted_at_str)
                            if granted_at_str else None
                        ),
                        expires_at=(
                            datetime.fromisoformat(expires_at_str)
                            if expires_at_str else None
                        ),
                        withdrawn_at=(
                            datetime.fromisoformat(withdrawn_at_str)
                            if withdrawn_at_str else None
                        ),
                        ip_address=ip_address,
                        user_agent=user_agent,
                        proof=proof,
                        metadata=json.loads(metadata_str)
                    )
                    records.append(record)
            
            return records
            
        except Exception as e:
            logger.error(f"Error getting consent history: {e}")
            return []
    
    def verify_consent(
        self,
        record_id: str,
        original_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Verify consent record integrity.
        
        Args:
            record_id: Consent record ID
            original_data: Optional original consent data
        
        Returns:
            True if verified
        """
        try:
            db_path = os.path.join(self.storage_dir, "consent.db")
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM consent_records
                    WHERE id = ?
                """, (record_id,))
                
                result = cursor.fetchone()
                if not result:
                    return False
                
                (
                    id, user_id, purpose_str, status_str, version,
                    granted_at_str, expires_at_str, withdrawn_at_str,
                    ip_address, user_agent, stored_proof, metadata_str
                ) = result
                
                if original_data:
                    # Verify against original data
                    calculated_proof = self._generate_proof(original_data)
                    return calculated_proof == stored_proof
                
                # Verify stored data
                proof_data = {
                    'id': id,
                    'user_id': user_id,
                    'purpose': purpose_str,
                    'status': status_str,
                    'version': version,
                    'granted_at': granted_at_str,
                    'expires_at': expires_at_str,
                    'ip_address': ip_address,
                    'user_agent': user_agent,
                    'metadata': json.loads(metadata_str)
                }
                calculated_proof = self._generate_proof(proof_data)
                return calculated_proof == stored_proof
                
        except Exception as e:
            logger.error(f"Error verifying consent: {e}")
            return False
    
    def cleanup_expired_consent(self) -> int:
        """
        Clean up expired consent records.
        
        Returns:
            Number of records updated
        """
        try:
            db_path = os.path.join(self.storage_dir, "consent.db")
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                # Update expired records
                cursor.execute("""
                    UPDATE consent_records
                    SET status = ?
                    WHERE status = ?
                    AND expires_at IS NOT NULL
                    AND expires_at < ?
                """, (
                    ConsentStatus.EXPIRED.value,
                    ConsentStatus.GRANTED.value,
                    datetime.utcnow().isoformat()
                ))
                
                count = cursor.rowcount
                conn.commit()
            
            logger.info(f"Updated {count} expired consent records")
            return count
            
        except Exception as e:
            logger.error(f"Error cleaning up expired consent: {e}")
            return 0 