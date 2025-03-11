"""
Data Minimization System.

This module provides comprehensive data minimization capabilities for the AMPTALK system,
ensuring privacy-preserving data operations.

Author: AMPTALK Team
Date: 2024
"""

import os
import json
import hashlib
import logging
import sqlite3
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Union
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from src.core.utils.logging_config import get_logger

logger = get_logger(__name__)

class MinimizationType(Enum):
    """Types of data minimization."""
    ANONYMIZATION = "anonymization"
    PSEUDONYMIZATION = "pseudonymization"
    MASKING = "masking"
    AGGREGATION = "aggregation"
    FILTERING = "filtering"

class DataCategory(Enum):
    """Categories of sensitive data."""
    PII = "personally_identifiable_information"
    PHI = "protected_health_information"
    FINANCIAL = "financial_information"
    LOCATION = "location_information"
    BEHAVIORAL = "behavioral_information"
    BIOMETRIC = "biometric_information"

class RetentionPolicy(Enum):
    """Data retention policies."""
    PERMANENT = "permanent"
    TEMPORARY = "temporary"
    SESSION = "session"
    CUSTOM = "custom"

@dataclass
class DataField:
    """Data field information."""
    name: str
    category: DataCategory
    retention_policy: RetentionPolicy
    retention_days: Optional[int]
    minimization_type: MinimizationType
    is_required: bool
    validation_rules: Dict[str, Any]

@dataclass
class MinimizationRule:
    """Data minimization rule."""
    field: str
    category: DataCategory
    minimization_type: MinimizationType
    parameters: Dict[str, Any]
    exceptions: List[str]
    is_enabled: bool

class DataMinimizer:
    """
    Manages data minimization for privacy.
    
    Features:
    - Data collection minimization
    - Data retention policies
    - Data anonymization
    - Data pseudonymization
    - Data masking
    """
    
    def __init__(
        self,
        config_dir: str = "privacy/data_minimization",
        salt: Optional[bytes] = None,
        default_retention_days: int = 365,
        enable_encryption: bool = True
    ):
        """
        Initialize the data minimizer.
        
        Args:
            config_dir: Directory for configuration storage
            salt: Optional salt for hashing
            default_retention_days: Default data retention period
            enable_encryption: Whether to encrypt sensitive data
        """
        self.config_dir = config_dir
        self.salt = salt or os.urandom(16)
        self.default_retention_days = default_retention_days
        self.enable_encryption = enable_encryption
        
        # Initialize storage
        self.fields: Dict[str, DataField] = {}
        self.rules: Dict[str, MinimizationRule] = {}
        self.mappings: Dict[str, str] = {}  # For pseudonymization
        
        # Initialize encryption if enabled
        if enable_encryption:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=self.salt,
                iterations=100000
            )
            key = kdf.derive(os.urandom(32))
            self.cipher = Fernet(base64.b64encode(key))
        
        # Initialize storage
        self._setup_storage()
        self._load_config()
    
    def _setup_storage(self) -> None:
        """Set up configuration storage."""
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Initialize database
        db_path = os.path.join(self.config_dir, "minimization.db")
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pseudonyms (
                    original TEXT PRIMARY KEY,
                    pseudonym TEXT NOT NULL,
                    category TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS retention_tracking (
                    field TEXT NOT NULL,
                    data_id TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    expires_at TIMESTAMP,
                    PRIMARY KEY (field, data_id)
                )
            """)
            
            conn.commit()
        
        logger.info(f"Initialized data minimization storage in {self.config_dir}")
    
    def _load_config(self) -> None:
        """Load configuration from storage."""
        try:
            # Load field definitions
            fields_path = os.path.join(self.config_dir, "fields.json")
            if os.path.exists(fields_path):
                with open(fields_path, 'r') as f:
                    data = json.load(f)
                    for field_data in data:
                        field = DataField(
                            name=field_data['name'],
                            category=DataCategory(field_data['category']),
                            retention_policy=RetentionPolicy(
                                field_data['retention_policy']
                            ),
                            retention_days=field_data.get('retention_days'),
                            minimization_type=MinimizationType(
                                field_data['minimization_type']
                            ),
                            is_required=field_data['is_required'],
                            validation_rules=field_data['validation_rules']
                        )
                        self.fields[field.name] = field
            
            # Load minimization rules
            rules_path = os.path.join(self.config_dir, "rules.json")
            if os.path.exists(rules_path):
                with open(rules_path, 'r') as f:
                    data = json.load(f)
                    for rule_data in data:
                        rule = MinimizationRule(
                            field=rule_data['field'],
                            category=DataCategory(rule_data['category']),
                            minimization_type=MinimizationType(
                                rule_data['minimization_type']
                            ),
                            parameters=rule_data['parameters'],
                            exceptions=rule_data['exceptions'],
                            is_enabled=rule_data['is_enabled']
                        )
                        self.rules[rule.field] = rule
            
            # Load pseudonym mappings
            db_path = os.path.join(self.config_dir, "minimization.db")
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT original, pseudonym FROM pseudonyms")
                self.mappings.update(dict(cursor.fetchall()))
            
            logger.info("Loaded data minimization configuration")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def _save_config(self) -> None:
        """Save configuration to storage."""
        try:
            # Save field definitions
            fields_data = []
            for field in self.fields.values():
                fields_data.append({
                    'name': field.name,
                    'category': field.category.value,
                    'retention_policy': field.retention_policy.value,
                    'retention_days': field.retention_days,
                    'minimization_type': field.minimization_type.value,
                    'is_required': field.is_required,
                    'validation_rules': field.validation_rules
                })
            
            fields_path = os.path.join(self.config_dir, "fields.json")
            with open(fields_path, 'w') as f:
                json.dump(fields_data, f)
            
            # Save minimization rules
            rules_data = []
            for rule in self.rules.values():
                rules_data.append({
                    'field': rule.field,
                    'category': rule.category.value,
                    'minimization_type': rule.minimization_type.value,
                    'parameters': rule.parameters,
                    'exceptions': rule.exceptions,
                    'is_enabled': rule.is_enabled
                })
            
            rules_path = os.path.join(self.config_dir, "rules.json")
            with open(rules_path, 'w') as f:
                json.dump(rules_data, f)
            
            logger.info("Saved data minimization configuration")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise
    
    def register_field(
        self,
        name: str,
        category: DataCategory,
        minimization_type: MinimizationType,
        retention_policy: RetentionPolicy = RetentionPolicy.TEMPORARY,
        retention_days: Optional[int] = None,
        is_required: bool = False,
        validation_rules: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a data field for minimization.
        
        Args:
            name: Field name
            category: Data category
            minimization_type: Type of minimization
            retention_policy: Retention policy
            retention_days: Days to retain data
            is_required: Whether field is required
            validation_rules: Optional validation rules
        """
        try:
            field = DataField(
                name=name,
                category=category,
                retention_policy=retention_policy,
                retention_days=(
                    retention_days or
                    self.default_retention_days
                    if retention_policy != RetentionPolicy.PERMANENT
                    else None
                ),
                minimization_type=minimization_type,
                is_required=is_required,
                validation_rules=validation_rules or {}
            )
            
            self.fields[name] = field
            self._save_config()
            
            logger.info(f"Registered field: {name}")
            
        except Exception as e:
            logger.error(f"Error registering field: {e}")
            raise
    
    def add_minimization_rule(
        self,
        field: str,
        minimization_type: MinimizationType,
        parameters: Dict[str, Any],
        exceptions: Optional[List[str]] = None
    ) -> None:
        """
        Add a minimization rule.
        
        Args:
            field: Field name
            minimization_type: Type of minimization
            parameters: Rule parameters
            exceptions: Optional exceptions
        """
        try:
            if field not in self.fields:
                raise ValueError(f"Field not registered: {field}")
            
            rule = MinimizationRule(
                field=field,
                category=self.fields[field].category,
                minimization_type=minimization_type,
                parameters=parameters,
                exceptions=exceptions or [],
                is_enabled=True
            )
            
            self.rules[field] = rule
            self._save_config()
            
            logger.info(f"Added minimization rule for {field}")
            
        except Exception as e:
            logger.error(f"Error adding minimization rule: {e}")
            raise
    
    def minimize_data(
        self,
        data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Apply data minimization.
        
        Args:
            data: Data to minimize
            context: Optional context for rules
        
        Returns:
            Minimized data
        """
        try:
            result = {}
            
            for field, value in data.items():
                if field not in self.fields:
                    # Pass through unregistered fields
                    result[field] = value
                    continue
                
                field_def = self.fields[field]
                rule = self.rules.get(field)
                
                if not rule or not rule.is_enabled:
                    if field_def.is_required:
                        result[field] = value
                    continue
                
                # Check exceptions
                if context and any(
                    context.get(exc) for exc in rule.exceptions
                ):
                    result[field] = value
                    continue
                
                # Apply minimization
                if rule.minimization_type == MinimizationType.ANONYMIZATION:
                    result[field] = self._anonymize_value(value)
                    
                elif rule.minimization_type == MinimizationType.PSEUDONYMIZATION:
                    result[field] = self._pseudonymize_value(
                        value,
                        field_def.category
                    )
                    
                elif rule.minimization_type == MinimizationType.MASKING:
                    result[field] = self._mask_value(
                        value,
                        rule.parameters
                    )
                    
                elif rule.minimization_type == MinimizationType.AGGREGATION:
                    result[field] = self._aggregate_value(
                        value,
                        rule.parameters
                    )
                    
                elif rule.minimization_type == MinimizationType.FILTERING:
                    if self._should_include_value(
                        value,
                        rule.parameters
                    ):
                        result[field] = value
                
                # Track retention if needed
                if field_def.retention_policy != RetentionPolicy.PERMANENT:
                    self._track_retention(
                        field,
                        str(value),
                        field_def.retention_days
                    )
            
            return result
            
        except Exception as e:
            logger.error(f"Error minimizing data: {e}")
            raise
    
    def _anonymize_value(self, value: Any) -> str:
        """Anonymize a value using one-way hashing."""
        try:
            value_str = str(value)
            return hashlib.sha256(
                value_str.encode() + self.salt
            ).hexdigest()
            
        except Exception as e:
            logger.error(f"Error anonymizing value: {e}")
            raise
    
    def _pseudonymize_value(
        self,
        value: Any,
        category: DataCategory
    ) -> str:
        """
        Pseudonymize a value with consistent mapping.
        
        Args:
            value: Value to pseudonymize
            category: Data category
        
        Returns:
            Pseudonymized value
        """
        try:
            value_str = str(value)
            
            # Check existing mapping
            if value_str in self.mappings:
                return self.mappings[value_str]
            
            # Generate new pseudonym
            pseudonym = f"{category.value}_{os.urandom(8).hex()}"
            
            # Store mapping
            db_path = os.path.join(self.config_dir, "minimization.db")
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO pseudonyms
                    (original, pseudonym, category, created_at)
                    VALUES (?, ?, ?, ?)
                """, (
                    value_str,
                    pseudonym,
                    category.value,
                    datetime.utcnow().isoformat()
                ))
                conn.commit()
            
            self.mappings[value_str] = pseudonym
            return pseudonym
            
        except Exception as e:
            logger.error(f"Error pseudonymizing value: {e}")
            raise
    
    def _mask_value(
        self,
        value: Any,
        parameters: Dict[str, Any]
    ) -> str:
        """
        Mask a value.
        
        Args:
            value: Value to mask
            parameters: Masking parameters
        
        Returns:
            Masked value
        """
        try:
            value_str = str(value)
            mask_char = parameters.get('mask_char', '*')
            
            if 'prefix_length' in parameters:
                # Keep prefix
                prefix_len = parameters['prefix_length']
                return value_str[:prefix_len] + mask_char * (len(value_str) - prefix_len)
                
            elif 'suffix_length' in parameters:
                # Keep suffix
                suffix_len = parameters['suffix_length']
                return mask_char * (len(value_str) - suffix_len) + value_str[-suffix_len:]
                
            elif 'pattern' in parameters:
                # Apply pattern
                pattern = parameters['pattern']
                result = ''
                for i, c in enumerate(value_str):
                    if i < len(pattern):
                        result += c if pattern[i] == 'X' else mask_char
                    else:
                        result += mask_char
                return result
                
            else:
                # Mask entire value
                return mask_char * len(value_str)
            
        except Exception as e:
            logger.error(f"Error masking value: {e}")
            raise
    
    def _aggregate_value(
        self,
        value: Any,
        parameters: Dict[str, Any]
    ) -> Any:
        """
        Aggregate a value.
        
        Args:
            value: Value to aggregate
            parameters: Aggregation parameters
        
        Returns:
            Aggregated value
        """
        try:
            if not isinstance(value, (int, float)):
                return value
            
            method = parameters.get('method', 'round')
            
            if method == 'round':
                precision = parameters.get('precision', 0)
                return round(value, precision)
                
            elif method == 'floor':
                base = parameters.get('base', 1)
                return (value // base) * base
                
            elif method == 'ceiling':
                base = parameters.get('base', 1)
                return ((value + base - 1) // base) * base
                
            elif method == 'range':
                ranges = parameters.get('ranges', [])
                for start, end, label in ranges:
                    if start <= value <= end:
                        return label
                return 'other'
                
            else:
                raise ValueError(f"Unknown aggregation method: {method}")
            
        except Exception as e:
            logger.error(f"Error aggregating value: {e}")
            raise
    
    def _should_include_value(
        self,
        value: Any,
        parameters: Dict[str, Any]
    ) -> bool:
        """
        Check if value should be included.
        
        Args:
            value: Value to check
            parameters: Filter parameters
        
        Returns:
            Whether to include value
        """
        try:
            if 'min_value' in parameters and value < parameters['min_value']:
                return False
                
            if 'max_value' in parameters and value > parameters['max_value']:
                return False
                
            if 'allowed_values' in parameters:
                return value in parameters['allowed_values']
                
            if 'regex_pattern' in parameters:
                import re
                return bool(re.match(parameters['regex_pattern'], str(value)))
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking value inclusion: {e}")
            return False
    
    def _track_retention(
        self,
        field: str,
        data_id: str,
        retention_days: Optional[int]
    ) -> None:
        """Track data retention."""
        try:
            db_path = os.path.join(self.config_dir, "minimization.db")
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                # Calculate expiry
                created_at = datetime.utcnow()
                expires_at = (
                    created_at + timedelta(days=retention_days)
                    if retention_days else None
                )
                
                # Store tracking info
                cursor.execute("""
                    INSERT OR REPLACE INTO retention_tracking
                    (field, data_id, created_at, expires_at)
                    VALUES (?, ?, ?, ?)
                """, (
                    field,
                    data_id,
                    created_at.isoformat(),
                    expires_at.isoformat() if expires_at else None
                ))
                conn.commit()
            
        except Exception as e:
            logger.error(f"Error tracking retention: {e}")
            raise
    
    def cleanup_expired_data(self) -> None:
        """Clean up expired data."""
        try:
            db_path = os.path.join(self.config_dir, "minimization.db")
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                # Find expired data
                cursor.execute("""
                    SELECT field, data_id FROM retention_tracking
                    WHERE expires_at IS NOT NULL
                    AND expires_at < ?
                """, (datetime.utcnow().isoformat(),))
                
                expired = cursor.fetchall()
                
                # Remove expired data
                for field, data_id in expired:
                    # TODO: Implement actual data deletion
                    logger.info(f"Should delete {data_id} from {field}")
                
                # Remove tracking entries
                cursor.execute("""
                    DELETE FROM retention_tracking
                    WHERE expires_at IS NOT NULL
                    AND expires_at < ?
                """, (datetime.utcnow().isoformat(),))
                
                conn.commit()
            
            logger.info(f"Cleaned up {len(expired)} expired data items")
            
        except Exception as e:
            logger.error(f"Error cleaning up expired data: {e}")
            raise
    
    def get_field_info(self, field: str) -> Optional[DataField]:
        """Get field information."""
        return self.fields.get(field)
    
    def get_rule_info(self, field: str) -> Optional[MinimizationRule]:
        """Get rule information."""
        return self.rules.get(field)
    
    def list_fields(
        self,
        category: Optional[DataCategory] = None,
        minimization_type: Optional[MinimizationType] = None
    ) -> List[DataField]:
        """List registered fields matching criteria."""
        try:
            results = []
            
            for field in self.fields.values():
                if category and field.category != category:
                    continue
                    
                if minimization_type and field.minimization_type != minimization_type:
                    continue
                
                results.append(field)
            
            return results
            
        except Exception as e:
            logger.error(f"Error listing fields: {e}")
            return []
    
    def update_rule(
        self,
        field: str,
        parameters: Optional[Dict[str, Any]] = None,
        exceptions: Optional[List[str]] = None,
        is_enabled: Optional[bool] = None
    ) -> bool:
        """
        Update a minimization rule.
        
        Args:
            field: Field name
            parameters: New parameters
            exceptions: New exceptions
            is_enabled: New enabled status
        
        Returns:
            True if successful
        """
        try:
            rule = self.rules.get(field)
            if not rule:
                return False
            
            if parameters is not None:
                rule.parameters = parameters
            
            if exceptions is not None:
                rule.exceptions = exceptions
            
            if is_enabled is not None:
                rule.is_enabled = is_enabled
            
            self._save_config()
            return True
            
        except Exception as e:
            logger.error(f"Error updating rule: {e}")
            return False 