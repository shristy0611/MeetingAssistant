"""
Version Management System.

This module handles version tracking, compatibility checks, and update management
for the AMPTALK system.

Author: AMPTALK Team
Date: 2024
"""

import os
import json
import hashlib
import logging
import semver
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from src.core.utils.logging_config import get_logger

logger = get_logger(__name__)

class ComponentType(Enum):
    """Types of system components."""
    CORE = "core"
    AGENT = "agent"
    MODEL = "model"
    CONFIG = "config"
    RESOURCE = "resource"

class UpdatePriority(Enum):
    """Update priority levels."""
    CRITICAL = "critical"  # Security fixes, major bug fixes
    HIGH = "high"         # Important features, performance improvements
    MEDIUM = "medium"     # Minor features, optimizations
    LOW = "low"          # Non-critical improvements

@dataclass
class VersionInfo:
    """Version information for a component."""
    version: str
    component_type: ComponentType
    dependencies: Dict[str, str]  # component_name: version_requirement
    checksum: str
    release_date: datetime
    min_system_version: str
    priority: UpdatePriority
    changelog: str
    rollback_version: Optional[str] = None

class VersionManager:
    """
    Manages system and component versions.
    
    Features:
    - Version tracking
    - Dependency resolution
    - Compatibility checking
    - Update validation
    """
    
    def __init__(self, version_file: str = "configs/versions.json"):
        """Initialize the version manager."""
        self.version_file = version_file
        self.current_versions: Dict[str, VersionInfo] = {}
        self.available_updates: Dict[str, VersionInfo] = {}
        self._load_versions()
    
    def _load_versions(self) -> None:
        """Load version information from file."""
        try:
            if os.path.exists(self.version_file):
                with open(self.version_file, 'r') as f:
                    data = json.load(f)
                    for component, info in data.items():
                        self.current_versions[component] = self._parse_version_info(info)
                logger.info(f"Loaded version information for {len(self.current_versions)} components")
        except Exception as e:
            logger.error(f"Error loading version information: {e}")
    
    def _parse_version_info(self, data: Dict) -> VersionInfo:
        """Parse version information from JSON data."""
        return VersionInfo(
            version=data["version"],
            component_type=ComponentType[data["component_type"].upper()],
            dependencies=data["dependencies"],
            checksum=data["checksum"],
            release_date=datetime.fromisoformat(data["release_date"]),
            min_system_version=data["min_system_version"],
            priority=UpdatePriority[data["priority"].upper()],
            changelog=data["changelog"],
            rollback_version=data.get("rollback_version")
        )
    
    def get_current_version(self, component: str) -> Optional[str]:
        """Get current version of a component."""
        if component in self.current_versions:
            return self.current_versions[component].version
        return None
    
    def check_compatibility(
        self,
        component: str,
        target_version: str
    ) -> Tuple[bool, List[str]]:
        """
        Check if a version upgrade is compatible.
        
        Args:
            component: Component name
            target_version: Target version to check
        
        Returns:
            Tuple of (is_compatible, list of incompatibility reasons)
        """
        if component not in self.current_versions:
            return False, ["Component not found"]
        
        current = self.current_versions[component]
        target = self.available_updates.get(f"{component}_{target_version}")
        
        if not target:
            return False, ["Target version not found"]
        
        incompatibilities = []
        
        # Check system version requirement
        if semver.compare(current.min_system_version, target.min_system_version) < 0:
            incompatibilities.append(
                f"System version {current.min_system_version} is lower than "
                f"required {target.min_system_version}"
            )
        
        # Check dependencies
        for dep, version_req in target.dependencies.items():
            if dep not in self.current_versions:
                incompatibilities.append(f"Missing dependency: {dep}")
                continue
            
            current_dep_version = self.current_versions[dep].version
            if not semver.match(current_dep_version, version_req):
                incompatibilities.append(
                    f"Dependency {dep} version {current_dep_version} "
                    f"does not match requirement {version_req}"
                )
        
        return len(incompatibilities) == 0, incompatibilities
    
    def validate_update(
        self,
        component: str,
        version: str,
        file_path: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate an update package.
        
        Args:
            component: Component name
            version: Version to validate
            file_path: Path to update package
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check if update exists
            update_key = f"{component}_{version}"
            if update_key not in self.available_updates:
                return False, "Update not found in available updates"
            
            update_info = self.available_updates[update_key]
            
            # Verify file exists
            if not os.path.exists(file_path):
                return False, "Update file not found"
            
            # Calculate checksum
            with open(file_path, 'rb') as f:
                checksum = hashlib.sha256(f.read()).hexdigest()
            
            # Verify checksum
            if checksum != update_info.checksum:
                return False, "Checksum verification failed"
            
            # Check compatibility
            is_compatible, incompatibilities = self.check_compatibility(
                component, version
            )
            if not is_compatible:
                return False, f"Incompatible update: {', '.join(incompatibilities)}"
            
            return True, None
            
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def prepare_rollback(self, component: str) -> Optional[str]:
        """
        Prepare rollback information for a component.
        
        Args:
            component: Component to prepare rollback for
        
        Returns:
            Rollback version if available
        """
        if component not in self.current_versions:
            return None
        
        current = self.current_versions[component]
        if not current.rollback_version:
            return None
        
        # Verify rollback version exists
        rollback_key = f"{component}_{current.rollback_version}"
        if rollback_key not in self.available_updates:
            return None
        
        return current.rollback_version
    
    def record_update(
        self,
        component: str,
        version: str,
        rollback_version: Optional[str] = None
    ) -> None:
        """
        Record a successful update.
        
        Args:
            component: Updated component
            version: New version
            rollback_version: Optional version to rollback to
        """
        update_key = f"{component}_{version}"
        if update_key not in self.available_updates:
            logger.error(f"Update {update_key} not found")
            return
        
        # Update current version
        update_info = self.available_updates[update_key]
        update_info.rollback_version = rollback_version
        self.current_versions[component] = update_info
        
        # Save to file
        self._save_versions()
    
    def _save_versions(self) -> None:
        """Save version information to file."""
        try:
            os.makedirs(os.path.dirname(self.version_file), exist_ok=True)
            with open(self.version_file, 'w') as f:
                data = {
                    component: {
                        "version": info.version,
                        "component_type": info.component_type.value,
                        "dependencies": info.dependencies,
                        "checksum": info.checksum,
                        "release_date": info.release_date.isoformat(),
                        "min_system_version": info.min_system_version,
                        "priority": info.priority.value,
                        "changelog": info.changelog,
                        "rollback_version": info.rollback_version
                    }
                    for component, info in self.current_versions.items()
                }
                json.dump(data, f, indent=2)
            logger.info("Saved version information")
        except Exception as e:
            logger.error(f"Error saving version information: {e}")
    
    def check_updates(self) -> Dict[str, List[VersionInfo]]:
        """
        Check for available updates.
        
        Returns:
            Dictionary of component -> list of available updates
        """
        updates: Dict[str, List[VersionInfo]] = {}
        
        for update_key, update_info in self.available_updates.items():
            component = update_key.rsplit('_', 1)[0]
            
            if component not in self.current_versions:
                continue
            
            current_version = self.current_versions[component].version
            if semver.compare(update_info.version, current_version) > 0:
                if component not in updates:
                    updates[component] = []
                updates[component].append(update_info)
        
        # Sort updates by priority and version
        for component in updates:
            updates[component].sort(
                key=lambda x: (x.priority.value, semver.VersionInfo.parse(x.version)),
                reverse=True
            )
        
        return updates 