#!/usr/bin/env python3
"""
System Configuration Management Module.

This module provides a comprehensive system configuration management interface
for the AMPTALK system, allowing users to view and modify system settings.

Author: AMPTALK Team
Date: 2024
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.core.utils.logging_config import get_logger

logger = get_logger(__name__)

@dataclass
class ConfigSection:
    """Configuration section with metadata."""
    name: str
    description: str
    settings: Dict[str, Any]
    validators: Dict[str, callable]
    required: List[str]

class SystemConfig:
    """
    System configuration manager.
    
    Features:
    - Configuration file management
    - Section-based organization
    - Validation rules
    - Change tracking
    - Backup/restore
    - Access control
    """
    
    def __init__(
        self,
        config_dir: str = "config",
        backup_count: int = 5,
        auto_save: bool = True
    ):
        """
        Initialize system configuration.
        
        Args:
            config_dir: Configuration directory
            backup_count: Number of backups to maintain
            auto_save: Whether to auto-save changes
        """
        self.config_dir = Path(config_dir)
        self.backup_count = backup_count
        self.auto_save = auto_save
        
        # Initialize storage
        self._setup_storage()
        
        # Load sections
        self.sections: Dict[str, ConfigSection] = {}
        self._initialize_sections()
        
        # Track changes
        self.changes: List[Dict[str, Any]] = []
    
    def _setup_storage(self) -> None:
        """Set up configuration storage."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.config_dir / "system.json"
        self.backup_dir = self.config_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
    
    def _initialize_sections(self) -> None:
        """Initialize configuration sections."""
        # System section
        self.add_section(
            name="system",
            description="Core system settings",
            settings={
                "name": "AMPTALK",
                "version": "1.0.0",
                "environment": "development",
                "debug": False,
                "log_level": "INFO"
            },
            validators={
                "environment": lambda x: x in ["development", "testing", "production"],
                "log_level": lambda x: x in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            },
            required=["name", "version", "environment"]
        )
        
        # Performance section
        self.add_section(
            name="performance",
            description="Performance-related settings",
            settings={
                "max_workers": 4,
                "queue_size": 1000,
                "batch_size": 32,
                "timeout_seconds": 30
            },
            validators={
                "max_workers": lambda x: isinstance(x, int) and x > 0,
                "queue_size": lambda x: isinstance(x, int) and x > 0,
                "batch_size": lambda x: isinstance(x, int) and x > 0,
                "timeout_seconds": lambda x: isinstance(x, int) and x > 0
            },
            required=["max_workers", "timeout_seconds"]
        )
        
        # Security section
        self.add_section(
            name="security",
            description="Security-related settings",
            settings={
                "enable_encryption": True,
                "key_rotation_days": 30,
                "session_timeout_minutes": 60,
                "max_login_attempts": 3
            },
            validators={
                "key_rotation_days": lambda x: isinstance(x, int) and x > 0,
                "session_timeout_minutes": lambda x: isinstance(x, int) and x > 0,
                "max_login_attempts": lambda x: isinstance(x, int) and x > 0
            },
            required=["enable_encryption", "session_timeout_minutes"]
        )
        
        # Load saved configuration
        self.load()
    
    def add_section(
        self,
        name: str,
        description: str,
        settings: Dict[str, Any],
        validators: Optional[Dict[str, callable]] = None,
        required: Optional[List[str]] = None
    ) -> None:
        """
        Add configuration section.
        
        Args:
            name: Section name
            description: Section description
            settings: Default settings
            validators: Optional validation functions
            required: Required setting keys
        """
        self.sections[name] = ConfigSection(
            name=name,
            description=description,
            settings=settings.copy(),
            validators=validators or {},
            required=required or []
        )
    
    def get_section(self, name: str) -> Optional[ConfigSection]:
        """
        Get configuration section.
        
        Args:
            name: Section name
        
        Returns:
            Section if found
        """
        return self.sections.get(name)
    
    def get_setting(
        self,
        section: str,
        key: str,
        default: Any = None
    ) -> Any:
        """
        Get configuration setting.
        
        Args:
            section: Section name
            key: Setting key
            default: Default value
        
        Returns:
            Setting value
        """
        section_config = self.get_section(section)
        if not section_config:
            return default
        return section_config.settings.get(key, default)
    
    def update_setting(
        self,
        section: str,
        key: str,
        value: Any
    ) -> bool:
        """
        Update configuration setting.
        
        Args:
            section: Section name
            key: Setting key
            value: New value
        
        Returns:
            True if successful
        """
        try:
            section_config = self.get_section(section)
            if not section_config:
                raise ValueError(f"Section not found: {section}")
            
            # Validate value
            validator = section_config.validators.get(key)
            if validator and not validator(value):
                raise ValueError(
                    f"Invalid value for {section}.{key}: {value}"
                )
            
            # Update value
            old_value = section_config.settings.get(key)
            section_config.settings[key] = value
            
            # Track change
            self.changes.append({
                "timestamp": datetime.utcnow().isoformat(),
                "section": section,
                "key": key,
                "old_value": old_value,
                "new_value": value
            })
            
            # Auto-save if enabled
            if self.auto_save:
                self.save()
            
            logger.info(
                f"Updated {section}.{key}: {old_value} -> {value}"
            )
            return True
            
        except Exception as e:
            logger.error(f"Error updating setting: {e}")
            return False
    
    def validate_section(self, name: str) -> List[str]:
        """
        Validate configuration section.
        
        Args:
            name: Section name
        
        Returns:
            List of validation errors
        """
        errors = []
        section = self.get_section(name)
        if not section:
            return [f"Section not found: {name}"]
        
        # Check required settings
        for key in section.required:
            if key not in section.settings:
                errors.append(f"Missing required setting: {key}")
        
        # Validate values
        for key, value in section.settings.items():
            validator = section.validators.get(key)
            if validator and not validator(value):
                errors.append(
                    f"Invalid value for {key}: {value}"
                )
        
        return errors
    
    def load(self) -> bool:
        """
        Load configuration from file.
        
        Returns:
            True if successful
        """
        try:
            if not self.config_file.exists():
                return False
            
            with open(self.config_file) as f:
                data = json.load(f)
            
            # Update sections
            for name, settings in data.items():
                section = self.get_section(name)
                if section:
                    section.settings.update(settings)
            
            logger.info("Loaded configuration from file")
            return True
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return False
    
    def save(self) -> bool:
        """
        Save configuration to file.
        
        Returns:
            True if successful
        """
        try:
            # Create backup
            if self.config_file.exists():
                self._create_backup()
            
            # Save current config
            data = {
                name: section.settings
                for name, section in self.sections.items()
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info("Saved configuration to file")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def _create_backup(self) -> None:
        """Create configuration backup."""
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_dir / f"system_{timestamp}.json"
            
            # Copy current config
            import shutil
            shutil.copy2(self.config_file, backup_file)
            
            # Remove old backups
            backups = sorted(self.backup_dir.glob("system_*.json"))
            while len(backups) > self.backup_count:
                backups[0].unlink()
                backups = backups[1:]
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
    
    def restore_backup(self, timestamp: str) -> bool:
        """
        Restore configuration from backup.
        
        Args:
            timestamp: Backup timestamp
        
        Returns:
            True if successful
        """
        try:
            backup_file = self.backup_dir / f"system_{timestamp}.json"
            if not backup_file.exists():
                return False
            
            # Restore backup
            import shutil
            shutil.copy2(backup_file, self.config_file)
            
            # Reload configuration
            return self.load()
            
        except Exception as e:
            logger.error(f"Error restoring backup: {e}")
            return False
    
    def get_changes(
        self,
        section: Optional[str] = None,
        key: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get configuration changes.
        
        Args:
            section: Optional section filter
            key: Optional key filter
        
        Returns:
            List of changes
        """
        filtered = self.changes
        
        if section:
            filtered = [
                c for c in filtered
                if c['section'] == section
            ]
        
        if key:
            filtered = [
                c for c in filtered
                if c['key'] == key
            ]
        
        return filtered
    
    def reset_section(self, name: str) -> bool:
        """
        Reset section to defaults.
        
        Args:
            name: Section name
        
        Returns:
            True if successful
        """
        try:
            section = self.get_section(name)
            if not section:
                return False
            
            # Store current settings
            old_settings = section.settings.copy()
            
            # Reset to defaults
            self._initialize_sections()
            
            # Track changes
            for key, old_value in old_settings.items():
                new_value = section.settings.get(key)
                if new_value != old_value:
                    self.changes.append({
                        "timestamp": datetime.utcnow().isoformat(),
                        "section": name,
                        "key": key,
                        "old_value": old_value,
                        "new_value": new_value
                    })
            
            # Save if enabled
            if self.auto_save:
                self.save()
            
            logger.info(f"Reset section: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting section: {e}")
            return False 