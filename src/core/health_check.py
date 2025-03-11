"""
Health Check Module for AMPTALK

This module provides health check capabilities for monitoring system components.
It allows registering and checking various health checks to ensure system reliability.
"""

import logging
import threading
import time
from typing import Dict, List, Any, Callable, Optional, Tuple
import json
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health status enum."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class HealthCheck:
    """
    A health check system to monitor component health.
    
    Features:
    - Register health checks for components
    - Run health checks on demand or periodically
    - Store health check history
    - Generate health check reports
    """
    
    def __init__(
        self,
        service_name: str = "amptalk",
        check_interval: float = 60.0,  # seconds
        history_size: int = 100,
        auto_check: bool = True
    ):
        """
        Initialize the HealthCheck system.
        
        Args:
            service_name: Name of the service being monitored
            check_interval: Interval for automatic health checks in seconds
            history_size: Number of health check results to keep in history
            auto_check: Whether to automatically run health checks periodically
        """
        self.service_name = service_name
        self.check_interval = check_interval
        self.history_size = history_size
        self.auto_check = auto_check
        
        # Initialize storage
        self.checks = {}  # name -> check_function
        self.results = {}  # name -> (status, message, timestamp)
        self.history = {}  # name -> list of (status, message, timestamp)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Start automatic health checks if enabled
        if self.auto_check:
            self._start_auto_checks()
        
        logger.info(f"HealthCheck initialized with check_interval={check_interval}s")
    
    def register_check(
        self,
        name: str,
        check_function: Callable[[], Tuple[bool, str]],
        description: str = ""
    ) -> None:
        """
        Register a health check.
        
        Args:
            name: Name of the health check
            check_function: Function that performs the check and returns (is_healthy, message)
            description: Description of the health check
        """
        with self._lock:
            self.checks[name] = {
                "function": check_function,
                "description": description,
                "added_at": datetime.now()
            }
            self.results[name] = (HealthStatus.UNKNOWN, "Not checked yet", datetime.now())
            self.history[name] = []
        
        logger.info(f"Registered health check: {name}")
    
    def unregister_check(self, name: str) -> bool:
        """
        Unregister a health check.
        
        Args:
            name: Name of the health check to unregister
            
        Returns:
            True if the check was unregistered, False if it wasn't found
        """
        with self._lock:
            if name in self.checks:
                del self.checks[name]
                del self.results[name]
                del self.history[name]
                logger.info(f"Unregistered health check: {name}")
                return True
            return False
    
    def run_check(self, name: str) -> Tuple[HealthStatus, str, datetime]:
        """
        Run a specific health check.
        
        Args:
            name: Name of the health check to run
            
        Returns:
            Tuple of (status, message, timestamp)
            
        Raises:
            KeyError: If the health check doesn't exist
        """
        with self._lock:
            if name not in self.checks:
                raise KeyError(f"Health check '{name}' not found")
            
            check_info = self.checks[name]
            check_function = check_info["function"]
            
            try:
                is_healthy, message = check_function()
                status = HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY
                timestamp = datetime.now()
                
                # Update result
                self.results[name] = (status, message, timestamp)
                
                # Add to history
                self.history[name].append((status, message, timestamp))
                
                # Trim history if needed
                if len(self.history[name]) > self.history_size:
                    self.history[name] = self.history[name][-self.history_size:]
                
                logger.debug(f"Health check '{name}' result: {status.value} - {message}")
                
                return status, message, timestamp
            except Exception as e:
                # Record exception as an unhealthy result
                status = HealthStatus.UNHEALTHY
                message = f"Error in health check: {str(e)}"
                timestamp = datetime.now()
                
                self.results[name] = (status, message, timestamp)
                self.history[name].append((status, message, timestamp))
                
                logger.error(f"Error running health check '{name}': {e}")
                
                return status, message, timestamp
    
    def run_all_checks(self) -> Dict[str, Tuple[HealthStatus, str, datetime]]:
        """
        Run all registered health checks.
        
        Returns:
            Dictionary mapping check names to (status, message, timestamp)
        """
        results = {}
        
        with self._lock:
            check_names = list(self.checks.keys())
        
        for name in check_names:
            try:
                status, message, timestamp = self.run_check(name)
                results[name] = (status, message, timestamp)
            except Exception as e:
                logger.error(f"Error running health check '{name}': {e}")
                results[name] = (
                    HealthStatus.UNHEALTHY,
                    f"Error running health check: {str(e)}",
                    datetime.now()
                )
        
        logger.info(f"Ran all health checks ({len(results)} total)")
        return results
    
    def get_overall_status(self) -> HealthStatus:
        """
        Get the overall health status of the system.
        
        Returns:
            Overall health status (worst status of all checks)
        """
        with self._lock:
            if not self.results:
                return HealthStatus.UNKNOWN
            
            # Initialize with the best possible status
            overall = HealthStatus.HEALTHY
            
            # Find the worst status
            for name, (status, _, _) in self.results.items():
                if status == HealthStatus.UNHEALTHY:
                    # If any check is unhealthy, the system is unhealthy
                    return HealthStatus.UNHEALTHY
                elif status == HealthStatus.DEGRADED and overall != HealthStatus.UNHEALTHY:
                    # If any check is degraded and we haven't found unhealthy, the system is degraded
                    overall = HealthStatus.DEGRADED
                elif status == HealthStatus.UNKNOWN and overall == HealthStatus.HEALTHY:
                    # If any check is unknown and all others are healthy, the system is unknown
                    overall = HealthStatus.UNKNOWN
            
            return overall
    
    def get_status_report(self) -> Dict[str, Any]:
        """
        Get a complete health status report.
        
        Returns:
            Dictionary with overall status and individual check results
        """
        with self._lock:
            overall = self.get_overall_status()
            
            checks_report = {}
            for name, (status, message, timestamp) in self.results.items():
                checks_report[name] = {
                    "status": status.value,
                    "message": message,
                    "timestamp": timestamp.isoformat(),
                    "description": self.checks[name]["description"] if name in self.checks else ""
                }
            
            return {
                "service": self.service_name,
                "timestamp": datetime.now().isoformat(),
                "status": overall.value,
                "checks": checks_report
            }
    
    def _start_auto_checks(self) -> None:
        """Start a background thread for automatic health checks."""
        def run_checks():
            while True:
                try:
                    self.run_all_checks()
                except Exception as e:
                    logger.error(f"Error in automatic health check: {e}")
                
                # Sleep until next check
                time.sleep(self.check_interval)
        
        thread = threading.Thread(
            target=run_checks,
            daemon=True,
            name="HealthCheckThread"
        )
        thread.start()
        logger.info(f"Started automatic health checks every {self.check_interval}s")
    
    def get_check_history(self, name: str, limit: int = None) -> List[Dict[str, Any]]:
        """
        Get history for a specific health check.
        
        Args:
            name: Name of the health check
            limit: Maximum number of history entries to return
            
        Returns:
            List of history entries (most recent first)
            
        Raises:
            KeyError: If the health check doesn't exist
        """
        with self._lock:
            if name not in self.history:
                raise KeyError(f"Health check '{name}' not found")
            
            # Get history and convert to dicts
            history = self.history[name]
            history_dicts = [
                {
                    "status": status.value,
                    "message": message,
                    "timestamp": timestamp.isoformat()
                }
                for status, message, timestamp in reversed(history)
            ]
            
            # Apply limit if specified
            if limit is not None and limit > 0:
                history_dicts = history_dicts[:limit]
            
            return history_dicts
    
    def clear_history(self, name: Optional[str] = None) -> None:
        """
        Clear history for a specific health check or all checks.
        
        Args:
            name: Name of the health check to clear history for, or None for all
            
        Raises:
            KeyError: If the specified health check doesn't exist
        """
        with self._lock:
            if name is not None:
                if name not in self.history:
                    raise KeyError(f"Health check '{name}' not found")
                self.history[name] = []
                logger.info(f"Cleared history for health check: {name}")
            else:
                for check_name in self.history:
                    self.history[check_name] = []
                logger.info("Cleared history for all health checks")


# Common health check functions

def create_cpu_health_check(
    warning_threshold: float = 80.0,
    critical_threshold: float = 90.0
) -> Callable[[], Tuple[bool, str]]:
    """
    Create a CPU usage health check function.
    
    Args:
        warning_threshold: CPU usage percentage that triggers a warning
        critical_threshold: CPU usage percentage that triggers a critical alert
        
    Returns:
        Health check function
    """
    try:
        import psutil
    except ImportError:
        def dummy_check():
            return False, "psutil not installed"
        return dummy_check
    
    def check_cpu():
        cpu_percent = psutil.cpu_percent(interval=0.1)
        if cpu_percent >= critical_threshold:
            return False, f"CPU usage is critical: {cpu_percent}%"
        elif cpu_percent >= warning_threshold:
            return True, f"CPU usage is high: {cpu_percent}%"
        else:
            return True, f"CPU usage is normal: {cpu_percent}%"
    
    return check_cpu

def create_memory_health_check(
    warning_threshold: float = 80.0,
    critical_threshold: float = 90.0
) -> Callable[[], Tuple[bool, str]]:
    """
    Create a memory usage health check function.
    
    Args:
        warning_threshold: Memory usage percentage that triggers a warning
        critical_threshold: Memory usage percentage that triggers a critical alert
        
    Returns:
        Health check function
    """
    try:
        import psutil
    except ImportError:
        def dummy_check():
            return False, "psutil not installed"
        return dummy_check
    
    def check_memory():
        memory = psutil.virtual_memory()
        if memory.percent >= critical_threshold:
            return False, f"Memory usage is critical: {memory.percent}%"
        elif memory.percent >= warning_threshold:
            return True, f"Memory usage is high: {memory.percent}%"
        else:
            return True, f"Memory usage is normal: {memory.percent}%"
    
    return check_memory

def create_disk_health_check(
    path: str = "/",
    warning_threshold: float = 80.0,
    critical_threshold: float = 90.0
) -> Callable[[], Tuple[bool, str]]:
    """
    Create a disk usage health check function.
    
    Args:
        path: Path to check disk usage for
        warning_threshold: Disk usage percentage that triggers a warning
        critical_threshold: Disk usage percentage that triggers a critical alert
        
    Returns:
        Health check function
    """
    try:
        import psutil
    except ImportError:
        def dummy_check():
            return False, "psutil not installed"
        return dummy_check
    
    def check_disk():
        disk = psutil.disk_usage(path)
        if disk.percent >= critical_threshold:
            return False, f"Disk usage is critical: {disk.percent}%"
        elif disk.percent >= warning_threshold:
            return True, f"Disk usage is high: {disk.percent}%"
        else:
            return True, f"Disk usage is normal: {disk.percent}%"
    
    return check_disk 