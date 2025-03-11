"""
Update Management System.

This module handles the update process for AMPTALK components,
including downloading, validation, installation, and rollback.

Author: AMPTALK Team
Date: 2024
"""

import os
import shutil
import tempfile
import logging
import asyncio
import aiohttp
import tarfile
from enum import Enum
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path

from src.core.update_mechanism.version_manager import (
    VersionManager,
    VersionInfo,
    ComponentType,
    UpdatePriority
)
from src.core.utils.logging_config import get_logger

logger = get_logger(__name__)

class UpdateStatus(Enum):
    """Update process status."""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    VALIDATING = "validating"
    INSTALLING = "installing"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"

class UpdateManager:
    """
    Manages the update process for system components.
    
    Features:
    - Automatic update checking
    - Safe update installation
    - Rollback support
    - Progress tracking
    """
    
    def __init__(
        self,
        update_dir: str = "updates",
        backup_dir: str = "backups",
        update_url: str = "https://updates.amptalk.ai",
        check_interval: int = 3600  # 1 hour
    ):
        """
        Initialize the update manager.
        
        Args:
            update_dir: Directory for update files
            backup_dir: Directory for backups
            update_url: Base URL for updates
            check_interval: Interval between update checks in seconds
        """
        self.update_dir = update_dir
        self.backup_dir = backup_dir
        self.update_url = update_url
        self.check_interval = check_interval
        
        self.version_manager = VersionManager()
        self.current_updates: Dict[str, UpdateStatus] = {}
        self.progress_callbacks: List[Callable[[str, UpdateStatus, float], None]] = []
        
        # Create directories
        os.makedirs(update_dir, exist_ok=True)
        os.makedirs(backup_dir, exist_ok=True)
    
    async def start_update_checker(self) -> None:
        """Start periodic update checking."""
        while True:
            try:
                await self.check_for_updates()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in update checker: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def check_for_updates(self) -> Dict[str, List[VersionInfo]]:
        """
        Check for available updates.
        
        Returns:
            Dictionary of component -> list of available updates
        """
        try:
            async with aiohttp.ClientSession() as session:
                # Get update manifest
                async with session.get(f"{self.update_url}/manifest.json") as response:
                    if response.status != 200:
                        raise RuntimeError(f"Failed to get manifest: {response.status}")
                    manifest = await response.json()
                
                # Update available updates
                self.version_manager.available_updates.clear()
                for component, versions in manifest.items():
                    for version_info in versions:
                        update_key = f"{component}_{version_info['version']}"
                        self.version_manager.available_updates[update_key] = \
                            self.version_manager._parse_version_info(version_info)
                
                # Get available updates
                return self.version_manager.check_updates()
                
        except Exception as e:
            logger.error(f"Error checking for updates: {e}")
            return {}
    
    def register_progress_callback(
        self,
        callback: Callable[[str, UpdateStatus, float], None]
    ) -> None:
        """Register a callback for update progress."""
        self.progress_callbacks.append(callback)
    
    def _notify_progress(
        self,
        component: str,
        status: UpdateStatus,
        progress: float = 0.0
    ) -> None:
        """Notify progress callbacks."""
        self.current_updates[component] = status
        for callback in self.progress_callbacks:
            try:
                callback(component, status, progress)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")
    
    async def download_update(
        self,
        component: str,
        version: str
    ) -> Optional[str]:
        """
        Download an update package.
        
        Args:
            component: Component to update
            version: Version to download
        
        Returns:
            Path to downloaded file if successful
        """
        self._notify_progress(component, UpdateStatus.DOWNLOADING)
        
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tar.gz') as temp_file:
                async with aiohttp.ClientSession() as session:
                    url = f"{self.update_url}/{component}/{version}/update.tar.gz"
                    async with session.get(url) as response:
                        if response.status != 200:
                            raise RuntimeError(f"Download failed: {response.status}")
                        
                        # Download with progress tracking
                        total_size = int(response.headers.get('content-length', 0))
                        downloaded = 0
                        
                        async for chunk in response.content.iter_chunked(8192):
                            temp_file.write(chunk)
                            downloaded += len(chunk)
                            if total_size:
                                progress = downloaded / total_size
                                self._notify_progress(
                                    component,
                                    UpdateStatus.DOWNLOADING,
                                    progress
                                )
            
            return temp_file.name
            
        except Exception as e:
            logger.error(f"Error downloading update: {e}")
            self._notify_progress(component, UpdateStatus.FAILED)
            return None
    
    async def install_update(
        self,
        component: str,
        version: str,
        file_path: str
    ) -> bool:
        """
        Install an update package.
        
        Args:
            component: Component to update
            version: Version to install
            file_path: Path to update package
        
        Returns:
            True if successful
        """
        self._notify_progress(component, UpdateStatus.VALIDATING)
        
        try:
            # Validate update
            is_valid, error = self.version_manager.validate_update(
                component, version, file_path
            )
            if not is_valid:
                raise RuntimeError(f"Update validation failed: {error}")
            
            self._notify_progress(component, UpdateStatus.INSTALLING)
            
            # Create backup
            current_version = self.version_manager.get_current_version(component)
            if current_version:
                backup_path = self._create_backup(component)
                logger.info(f"Created backup at {backup_path}")
            
            # Extract update
            install_dir = os.path.join(self.update_dir, component)
            os.makedirs(install_dir, exist_ok=True)
            
            with tarfile.open(file_path, 'r:gz') as tar:
                tar.extractall(install_dir)
            
            # Record update
            self.version_manager.record_update(
                component,
                version,
                rollback_version=current_version
            )
            
            self._notify_progress(component, UpdateStatus.COMPLETED)
            return True
            
        except Exception as e:
            logger.error(f"Error installing update: {e}")
            self._notify_progress(component, UpdateStatus.FAILED)
            
            # Attempt rollback
            if current_version:
                await self.rollback_update(component)
            
            return False
        finally:
            # Clean up
            try:
                os.unlink(file_path)
            except Exception:
                pass
    
    def _create_backup(self, component: str) -> str:
        """
        Create a backup of a component.
        
        Args:
            component: Component to backup
        
        Returns:
            Path to backup file
        """
        component_dir = os.path.join(self.update_dir, component)
        if not os.path.exists(component_dir):
            raise RuntimeError(f"Component directory not found: {component_dir}")
        
        # Create backup archive
        backup_path = os.path.join(
            self.backup_dir,
            f"{component}_{self.version_manager.get_current_version(component)}.tar.gz"
        )
        
        with tarfile.open(backup_path, 'w:gz') as tar:
            tar.add(component_dir, arcname=component)
        
        return backup_path
    
    async def rollback_update(self, component: str) -> bool:
        """
        Rollback an update.
        
        Args:
            component: Component to rollback
        
        Returns:
            True if successful
        """
        try:
            # Get rollback version
            rollback_version = self.version_manager.prepare_rollback(component)
            if not rollback_version:
                raise RuntimeError("No rollback version available")
            
            # Find backup file
            backup_path = os.path.join(
                self.backup_dir,
                f"{component}_{rollback_version}.tar.gz"
            )
            if not os.path.exists(backup_path):
                raise RuntimeError(f"Backup not found: {backup_path}")
            
            # Remove current version
            component_dir = os.path.join(self.update_dir, component)
            shutil.rmtree(component_dir, ignore_errors=True)
            
            # Extract backup
            with tarfile.open(backup_path, 'r:gz') as tar:
                tar.extractall(self.update_dir)
            
            # Record rollback
            self.version_manager.record_update(
                component,
                rollback_version
            )
            
            self._notify_progress(component, UpdateStatus.ROLLED_BACK)
            return True
            
        except Exception as e:
            logger.error(f"Error rolling back update: {e}")
            self._notify_progress(component, UpdateStatus.FAILED)
            return False
    
    async def update_component(
        self,
        component: str,
        version: str
    ) -> bool:
        """
        Update a component to a specific version.
        
        Args:
            component: Component to update
            version: Target version
        
        Returns:
            True if successful
        """
        try:
            # Download update
            file_path = await self.download_update(component, version)
            if not file_path:
                return False
            
            # Install update
            return await self.install_update(component, version, file_path)
            
        except Exception as e:
            logger.error(f"Error updating component: {e}")
            self._notify_progress(component, UpdateStatus.FAILED)
            return False
    
    async def update_all(
        self,
        priority: Optional[UpdatePriority] = None
    ) -> Dict[str, bool]:
        """
        Update all components with available updates.
        
        Args:
            priority: Optional minimum priority level
        
        Returns:
            Dictionary of component -> success status
        """
        results = {}
        
        # Get available updates
        updates = await self.check_for_updates()
        
        # Update components
        for component, versions in updates.items():
            if not versions:
                continue
            
            # Filter by priority
            if priority:
                versions = [v for v in versions if v.priority.value <= priority.value]
                if not versions:
                    continue
            
            # Get latest version
            latest = versions[0]
            
            # Update component
            success = await self.update_component(component, latest.version)
            results[component] = success
        
        return results 