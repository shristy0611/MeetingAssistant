"""
Privacy Policy Generator.

This module provides comprehensive privacy policy generation capabilities for the AMPTALK system,
ensuring transparent and compliant data processing practices.

Author: AMPTALK Team
Date: 2024
"""

import os
import json
import logging
import sqlite3
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
import jinja2

from src.core.utils.logging_config import get_logger

logger = get_logger(__name__)

class PolicySection(Enum):
    """Sections of a privacy policy."""
    INTRODUCTION = "introduction"
    DATA_COLLECTION = "data_collection"
    DATA_USE = "data_use"
    DATA_SHARING = "data_sharing"
    DATA_SECURITY = "data_security"
    USER_RIGHTS = "user_rights"
    COOKIES = "cookies"
    CONTACT = "contact"
    UPDATES = "updates"

class PolicyLanguage(Enum):
    """Supported policy languages."""
    EN = "english"
    ES = "spanish"
    FR = "french"
    DE = "german"
    IT = "italian"
    PT = "portuguese"
    JA = "japanese"
    ZH = "chinese"

@dataclass
class PolicyVersion:
    """Privacy policy version information."""
    id: str
    version: str
    created_at: datetime
    effective_from: datetime
    language: PolicyLanguage
    content: Dict[PolicySection, str]
    metadata: Dict[str, Any]

class PolicyGenerator:
    """
    Generates and manages privacy policies.
    
    Features:
    - Dynamic policy generation
    - Version management
    - Multi-language support
    - Policy templates
    - Policy customization
    """
    
    def __init__(
        self,
        storage_dir: str = "privacy/policies",
        template_dir: str = "privacy/templates",
        default_language: PolicyLanguage = PolicyLanguage.EN
    ):
        """
        Initialize the policy generator.
        
        Args:
            storage_dir: Directory for policy storage
            template_dir: Directory for policy templates
            default_language: Default policy language
        """
        self.storage_dir = storage_dir
        self.template_dir = template_dir
        self.default_language = default_language
        
        # Initialize Jinja environment
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            autoescape=True,
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Initialize storage
        self._setup_storage()
    
    def _setup_storage(self) -> None:
        """Set up policy storage."""
        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(self.template_dir, exist_ok=True)
        
        # Initialize database
        db_path = os.path.join(self.storage_dir, "policies.db")
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS policy_versions (
                    id TEXT PRIMARY KEY,
                    version TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    effective_from TIMESTAMP NOT NULL,
                    language TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            
            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_version
                ON policy_versions(version)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_language
                ON policy_versions(language)
            """)
            
            conn.commit()
        
        logger.info(f"Initialized policy storage in {self.storage_dir}")
    
    def generate_policy(
        self,
        company_info: Dict[str, Any],
        data_practices: Dict[str, Any],
        version: str,
        language: Optional[PolicyLanguage] = None,
        effective_date: Optional[datetime] = None,
        custom_sections: Optional[Dict[PolicySection, str]] = None
    ) -> str:
        """
        Generate a privacy policy.
        
        Args:
            company_info: Company information
            data_practices: Data handling practices
            version: Policy version
            language: Policy language
            effective_date: Policy effective date
            custom_sections: Custom section content
        
        Returns:
            Policy ID
        """
        try:
            language = language or self.default_language
            effective_date = effective_date or datetime.utcnow()
            
            # Load base template
            template = self.jinja_env.get_template(
                f"policy_base_{language.value}.html"
            )
            
            # Prepare sections
            sections = {}
            for section in PolicySection:
                if custom_sections and section in custom_sections:
                    # Use custom content
                    sections[section] = custom_sections[section]
                else:
                    # Generate from template
                    section_template = self.jinja_env.get_template(
                        f"{section.value}_{language.value}.html"
                    )
                    sections[section] = section_template.render(
                        company=company_info,
                        practices=data_practices,
                        version=version,
                        effective_date=effective_date
                    )
            
            # Generate complete policy
            policy_content = template.render(sections=sections)
            
            # Create policy version
            policy_id = f"policy_{datetime.utcnow().timestamp()}"
            policy = PolicyVersion(
                id=policy_id,
                version=version,
                created_at=datetime.utcnow(),
                effective_from=effective_date,
                language=language,
                content=sections,
                metadata={
                    'company_info': company_info,
                    'data_practices': data_practices
                }
            )
            
            # Store policy
            self._save_policy(policy)
            
            # Save rendered policy
            output_path = os.path.join(
                self.storage_dir,
                f"{policy_id}.html"
            )
            with open(output_path, 'w') as f:
                f.write(policy_content)
            
            logger.info(
                f"Generated {language.value} policy version {version}"
            )
            return policy_id
            
        except Exception as e:
            logger.error(f"Error generating policy: {e}")
            raise
    
    def _save_policy(self, policy: PolicyVersion) -> None:
        """Save policy version to storage."""
        try:
            db_path = os.path.join(self.storage_dir, "policies.db")
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO policy_versions
                    (id, version, created_at, effective_from,
                     language, content, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    policy.id,
                    policy.version,
                    policy.created_at.isoformat(),
                    policy.effective_from.isoformat(),
                    policy.language.value,
                    json.dumps(
                        {k.value: v for k, v in policy.content.items()}
                    ),
                    json.dumps(policy.metadata)
                ))
                conn.commit()
            
        except Exception as e:
            logger.error(f"Error saving policy: {e}")
            raise
    
    def get_policy(
        self,
        policy_id: str
    ) -> Optional[PolicyVersion]:
        """
        Get a specific policy version.
        
        Args:
            policy_id: Policy ID
        
        Returns:
            Policy version if found
        """
        try:
            db_path = os.path.join(self.storage_dir, "policies.db")
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM policy_versions
                    WHERE id = ?
                """, (policy_id,))
                
                result = cursor.fetchone()
                if not result:
                    return None
                
                (
                    id, version, created_at_str, effective_from_str,
                    language_str, content_str, metadata_str
                ) = result
                
                return PolicyVersion(
                    id=id,
                    version=version,
                    created_at=datetime.fromisoformat(created_at_str),
                    effective_from=datetime.fromisoformat(
                        effective_from_str
                    ),
                    language=PolicyLanguage(language_str),
                    content={
                        PolicySection(k): v
                        for k, v in json.loads(content_str).items()
                    },
                    metadata=json.loads(metadata_str)
                )
                
        except Exception as e:
            logger.error(f"Error getting policy: {e}")
            return None
    
    def get_latest_policy(
        self,
        language: Optional[PolicyLanguage] = None
    ) -> Optional[PolicyVersion]:
        """
        Get latest policy version.
        
        Args:
            language: Optional language filter
        
        Returns:
            Latest policy version if found
        """
        try:
            query = """
                SELECT * FROM policy_versions
                WHERE effective_from <= ?
            """
            params = [datetime.utcnow().isoformat()]
            
            if language:
                query += " AND language = ?"
                params.append(language.value)
            
            query += " ORDER BY effective_from DESC LIMIT 1"
            
            db_path = os.path.join(self.storage_dir, "policies.db")
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                
                result = cursor.fetchone()
                if not result:
                    return None
                
                (
                    id, version, created_at_str, effective_from_str,
                    language_str, content_str, metadata_str
                ) = result
                
                return PolicyVersion(
                    id=id,
                    version=version,
                    created_at=datetime.fromisoformat(created_at_str),
                    effective_from=datetime.fromisoformat(
                        effective_from_str
                    ),
                    language=PolicyLanguage(language_str),
                    content={
                        PolicySection(k): v
                        for k, v in json.loads(content_str).items()
                    },
                    metadata=json.loads(metadata_str)
                )
                
        except Exception as e:
            logger.error(f"Error getting latest policy: {e}")
            return None
    
    def list_policy_versions(
        self,
        language: Optional[PolicyLanguage] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[PolicyVersion]:
        """
        List policy versions.
        
        Args:
            language: Optional language filter
            start_date: Start date filter
            end_date: End date filter
        
        Returns:
            List of policy versions
        """
        try:
            query = "SELECT * FROM policy_versions WHERE 1=1"
            params = []
            
            if language:
                query += " AND language = ?"
                params.append(language.value)
            
            if start_date:
                query += " AND effective_from >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND effective_from <= ?"
                params.append(end_date.isoformat())
            
            query += " ORDER BY effective_from DESC"
            
            versions = []
            db_path = os.path.join(self.storage_dir, "policies.db")
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                
                for result in cursor.fetchall():
                    (
                        id, version, created_at_str, effective_from_str,
                        language_str, content_str, metadata_str
                    ) = result
                    
                    policy = PolicyVersion(
                        id=id,
                        version=version,
                        created_at=datetime.fromisoformat(
                            created_at_str
                        ),
                        effective_from=datetime.fromisoformat(
                            effective_from_str
                        ),
                        language=PolicyLanguage(language_str),
                        content={
                            PolicySection(k): v
                            for k, v in json.loads(content_str).items()
                        },
                        metadata=json.loads(metadata_str)
                    )
                    versions.append(policy)
            
            return versions
            
        except Exception as e:
            logger.error(f"Error listing policy versions: {e}")
            return []
    
    def compare_versions(
        self,
        version1_id: str,
        version2_id: str
    ) -> Dict[PolicySection, Dict[str, Any]]:
        """
        Compare two policy versions.
        
        Args:
            version1_id: First version ID
            version2_id: Second version ID
        
        Returns:
            Dictionary of differences by section
        """
        try:
            # Get policy versions
            version1 = self.get_policy(version1_id)
            version2 = self.get_policy(version2_id)
            
            if not version1 or not version2:
                raise ValueError("Policy version not found")
            
            # Compare sections
            differences = {}
            for section in PolicySection:
                if (
                    section in version1.content and
                    section in version2.content
                ):
                    content1 = version1.content[section]
                    content2 = version2.content[section]
                    
                    if content1 != content2:
                        differences[section] = {
                            'old': content1,
                            'new': content2
                        }
            
            return differences
            
        except Exception as e:
            logger.error(f"Error comparing versions: {e}")
            return {}
    
    def translate_policy(
        self,
        policy_id: str,
        target_language: PolicyLanguage
    ) -> Optional[str]:
        """
        Translate a policy to another language.
        
        Args:
            policy_id: Source policy ID
            target_language: Target language
        
        Returns:
            New policy ID if successful
        """
        try:
            # Get source policy
            source = self.get_policy(policy_id)
            if not source:
                raise ValueError("Source policy not found")
            
            # Load translation template
            template = self.jinja_env.get_template(
                f"policy_base_{target_language.value}.html"
            )
            
            # Translate sections
            translated_sections = {}
            for section, content in source.content.items():
                section_template = self.jinja_env.get_template(
                    f"{section.value}_{target_language.value}.html"
                )
                translated_sections[section] = section_template.render(
                    **source.metadata
                )
            
            # Generate translated policy
            return self.generate_policy(
                company_info=source.metadata['company_info'],
                data_practices=source.metadata['data_practices'],
                version=source.version,
                language=target_language,
                effective_date=source.effective_from,
                custom_sections=translated_sections
            )
            
        except Exception as e:
            logger.error(f"Error translating policy: {e}")
            return None 