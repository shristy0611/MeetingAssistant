"""
Alert System Module for AMPTALK

This module provides alert generation, routing, and notification capabilities.
It allows the system to notify users and administrators about issues and events.
"""

import logging
import threading
import time
import queue
from typing import Dict, List, Any, Callable, Optional, Union
from enum import Enum
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class Alert:
    """Alert data class."""
    
    def __init__(
        self,
        title: str,
        message: str,
        severity: AlertSeverity,
        source: str,
        timestamp: Optional[datetime] = None,
        details: Optional[Dict[str, Any]] = None,
        alert_id: Optional[str] = None
    ):
        """
        Initialize an alert.
        
        Args:
            title: Short title describing the alert
            message: Detailed alert message
            severity: Alert severity level
            source: Component or system that generated the alert
            timestamp: When the alert was generated (defaults to now)
            details: Additional alert details
            alert_id: Unique alert ID (generated if not provided)
        """
        self.title = title
        self.message = message
        self.severity = severity
        self.source = source
        self.timestamp = timestamp or datetime.now()
        self.details = details or {}
        
        # Generate alert ID if not provided
        if alert_id is None:
            import uuid
            alert_id = str(uuid.uuid4())
        self.alert_id = alert_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "title": self.title,
            "message": self.message,
            "severity": self.severity.value,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Alert':
        """Create alert from dictionary."""
        severity = AlertSeverity(data["severity"])
        timestamp = datetime.fromisoformat(data["timestamp"])
        
        return cls(
            title=data["title"],
            message=data["message"],
            severity=severity,
            source=data["source"],
            timestamp=timestamp,
            details=data.get("details", {}),
            alert_id=data.get("alert_id")
        )
    
    def __str__(self) -> str:
        """String representation of alert."""
        return f"[{self.severity.value.upper()}] {self.title}: {self.message}"


class AlertNotifier:
    """Base class for alert notifiers."""
    
    def __init__(self, name: str):
        """
        Initialize an alert notifier.
        
        Args:
            name: Name of the notifier
        """
        self.name = name
    
    def notify(self, alert: Alert) -> bool:
        """
        Send notification for an alert.
        
        Args:
            alert: Alert to notify about
            
        Returns:
            True if notification was sent successfully, False otherwise
        """
        raise NotImplementedError("Subclasses must implement notify()")


class LoggingNotifier(AlertNotifier):
    """Alert notifier that logs alerts."""
    
    def __init__(self, name: str = "logging"):
        """Initialize a logging notifier."""
        super().__init__(name)
        self.logger = logging.getLogger(f"alert.{name}")
    
    def notify(self, alert: Alert) -> bool:
        """Log an alert."""
        if alert.severity == AlertSeverity.CRITICAL:
            self.logger.critical(str(alert))
        elif alert.severity == AlertSeverity.ERROR:
            self.logger.error(str(alert))
        elif alert.severity == AlertSeverity.WARNING:
            self.logger.warning(str(alert))
        else:
            self.logger.info(str(alert))
        
        return True


class EmailNotifier(AlertNotifier):
    """Alert notifier that sends email notifications."""
    
    def __init__(
        self,
        name: str = "email",
        smtp_server: str = "localhost",
        smtp_port: int = 25,
        sender: str = "alerts@example.com",
        recipients: List[str] = None,
        use_ssl: bool = False,
        username: Optional[str] = None,
        password: Optional[str] = None,
        min_severity: AlertSeverity = AlertSeverity.WARNING
    ):
        """
        Initialize an email notifier.
        
        Args:
            name: Name of the notifier
            smtp_server: SMTP server hostname
            smtp_port: SMTP server port
            sender: Sender email address
            recipients: List of recipient email addresses
            use_ssl: Whether to use SSL for SMTP connection
            username: SMTP username if authentication is required
            password: SMTP password if authentication is required
            min_severity: Minimum severity level to send email notifications for
        """
        super().__init__(name)
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender = sender
        self.recipients = recipients or []
        self.use_ssl = use_ssl
        self.username = username
        self.password = password
        self.min_severity = min_severity
    
    def notify(self, alert: Alert) -> bool:
        """
        Send email notification for an alert.
        
        Args:
            alert: Alert to notify about
            
        Returns:
            True if email was sent successfully, False otherwise
        """
        # Only notify for alerts with severity >= min_severity
        if self._get_severity_level(alert.severity) < self._get_severity_level(self.min_severity):
            return True
        
        if not self.recipients:
            logger.warning("No recipients configured for EmailNotifier")
            return False
        
        try:
            # This is just a stub implementation
            # In a real implementation, you would use smtplib to send an email
            logger.info(
                f"Would send email: To: {', '.join(self.recipients)}, "
                f"Subject: [{alert.severity.value.upper()}] {alert.title}, "
                f"Body: {alert.message}"
            )
            
            # Simulating successful email sending
            return True
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
            return False
    
    def _get_severity_level(self, severity: AlertSeverity) -> int:
        """Get numeric severity level."""
        levels = {
            AlertSeverity.INFO: 0,
            AlertSeverity.WARNING: 1,
            AlertSeverity.ERROR: 2,
            AlertSeverity.CRITICAL: 3
        }
        return levels.get(severity, 0)


class WebhookNotifier(AlertNotifier):
    """Alert notifier that sends webhook notifications."""
    
    def __init__(
        self,
        name: str = "webhook",
        webhook_url: str = "",
        headers: Optional[Dict[str, str]] = None,
        min_severity: AlertSeverity = AlertSeverity.WARNING
    ):
        """
        Initialize a webhook notifier.
        
        Args:
            name: Name of the notifier
            webhook_url: Webhook URL to send notifications to
            headers: HTTP headers to include in webhook requests
            min_severity: Minimum severity level to send webhook notifications for
        """
        super().__init__(name)
        self.webhook_url = webhook_url
        self.headers = headers or {}
        self.min_severity = min_severity
    
    def notify(self, alert: Alert) -> bool:
        """
        Send webhook notification for an alert.
        
        Args:
            alert: Alert to notify about
            
        Returns:
            True if webhook notification was sent successfully, False otherwise
        """
        # Only notify for alerts with severity >= min_severity
        if self._get_severity_level(alert.severity) < self._get_severity_level(self.min_severity):
            return True
        
        if not self.webhook_url:
            logger.warning("No webhook URL configured for WebhookNotifier")
            return False
        
        try:
            # This is just a stub implementation
            # In a real implementation, you would use requests to send a webhook
            logger.info(
                f"Would send webhook to {self.webhook_url}: "
                f"{json.dumps(alert.to_dict(), indent=2)}"
            )
            
            # Simulating successful webhook sending
            return True
        except Exception as e:
            logger.error(f"Error sending webhook notification: {e}")
            return False
    
    def _get_severity_level(self, severity: AlertSeverity) -> int:
        """Get numeric severity level."""
        levels = {
            AlertSeverity.INFO: 0,
            AlertSeverity.WARNING: 1,
            AlertSeverity.ERROR: 2,
            AlertSeverity.CRITICAL: 3
        }
        return levels.get(severity, 0)


class AlertSystem:
    """
    A system for generating and routing alerts.
    
    Features:
    - Generate alerts with different severity levels
    - Route alerts to different notifiers
    - Store alert history
    - Filter alerts based on severity and source
    """
    
    def __init__(
        self,
        service_name: str = "amptalk",
        max_history_size: int = 1000,
        default_notifiers: Optional[List[AlertNotifier]] = None
    ):
        """
        Initialize the AlertSystem.
        
        Args:
            service_name: Name of the service
            max_history_size: Maximum number of alerts to keep in history
            default_notifiers: Default notifiers to use for all alerts
        """
        self.service_name = service_name
        self.max_history_size = max_history_size
        
        # Initialize notifiers
        self.notifiers = {}
        if default_notifiers:
            for notifier in default_notifiers:
                self.add_notifier(notifier)
        else:
            # Add default logging notifier
            self.add_notifier(LoggingNotifier())
        
        # Alert history
        self.history = []
        
        # Alert queue and worker thread
        self.alert_queue = queue.Queue()
        self._start_worker_thread()
        
        logger.info(f"AlertSystem initialized with {len(self.notifiers)} notifiers")
    
    def add_notifier(self, notifier: AlertNotifier) -> None:
        """
        Add a notifier to the alert system.
        
        Args:
            notifier: Notifier to add
        """
        self.notifiers[notifier.name] = notifier
        logger.info(f"Added notifier: {notifier.name}")
    
    def remove_notifier(self, name: str) -> bool:
        """
        Remove a notifier from the alert system.
        
        Args:
            name: Name of the notifier to remove
            
        Returns:
            True if the notifier was removed, False if it wasn't found
        """
        if name in self.notifiers:
            del self.notifiers[name]
            logger.info(f"Removed notifier: {name}")
            return True
        return False
    
    def alert(
        self,
        title: str,
        message: str,
        severity: Union[AlertSeverity, str],
        source: str,
        details: Optional[Dict[str, Any]] = None
    ) -> Alert:
        """
        Generate an alert.
        
        Args:
            title: Short title describing the alert
            message: Detailed alert message
            severity: Alert severity level
            source: Component or system that generated the alert
            details: Additional alert details
            
        Returns:
            The generated Alert object
        """
        # Convert string severity to AlertSeverity enum
        if isinstance(severity, str):
            try:
                severity = AlertSeverity(severity.lower())
            except ValueError:
                logger.warning(f"Invalid severity: {severity}, using WARNING")
                severity = AlertSeverity.WARNING
        
        # Create alert
        alert = Alert(
            title=title,
            message=message,
            severity=severity,
            source=source,
            details=details
        )
        
        # Queue alert for processing
        self.alert_queue.put(alert)
        
        # Log alert
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL
        }.get(severity, logging.INFO)
        
        logger.log(log_level, f"Alert: {alert}")
        
        return alert
    
    def _process_alert(self, alert: Alert) -> None:
        """
        Process an alert (called by worker thread).
        
        Args:
            alert: Alert to process
        """
        # Add to history
        self.history.append(alert)
        
        # Trim history if needed
        if len(self.history) > self.max_history_size:
            self.history = self.history[-self.max_history_size:]
        
        # Notify all notifiers
        for name, notifier in self.notifiers.items():
            try:
                success = notifier.notify(alert)
                if not success:
                    logger.warning(f"Notifier '{name}' failed to send notification")
            except Exception as e:
                logger.error(f"Error in notifier '{name}': {e}")
    
    def _start_worker_thread(self) -> None:
        """Start a worker thread to process alerts from the queue."""
        def worker():
            while True:
                try:
                    # Get alert from queue
                    alert = self.alert_queue.get()
                    
                    # Process alert
                    self._process_alert(alert)
                    
                    # Mark task as done
                    self.alert_queue.task_done()
                except Exception as e:
                    logger.error(f"Error in alert worker thread: {e}")
        
        thread = threading.Thread(
            target=worker,
            daemon=True,
            name="AlertWorker"
        )
        thread.start()
        logger.info("Started alert worker thread")
    
    def get_history(
        self,
        limit: Optional[int] = None,
        min_severity: Optional[AlertSeverity] = None,
        source: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get alert history.
        
        Args:
            limit: Maximum number of alerts to return
            min_severity: Minimum severity level to include
            source: Filter alerts by source
            
        Returns:
            List of alert dictionaries (most recent first)
        """
        # Filter alerts
        filtered_alerts = self.history
        
        if min_severity:
            min_level = self._get_severity_level(min_severity)
            filtered_alerts = [
                a for a in filtered_alerts
                if self._get_severity_level(a.severity) >= min_level
            ]
        
        if source:
            filtered_alerts = [a for a in filtered_alerts if a.source == source]
        
        # Sort by timestamp (most recent first)
        sorted_alerts = sorted(
            filtered_alerts,
            key=lambda a: a.timestamp,
            reverse=True
        )
        
        # Apply limit
        if limit is not None:
            sorted_alerts = sorted_alerts[:limit]
        
        # Convert to dictionaries
        return [alert.to_dict() for alert in sorted_alerts]
    
    def clear_history(self) -> None:
        """Clear alert history."""
        self.history = []
        logger.info("Cleared alert history")
    
    def _get_severity_level(self, severity: AlertSeverity) -> int:
        """Get numeric severity level."""
        levels = {
            AlertSeverity.INFO: 0,
            AlertSeverity.WARNING: 1,
            AlertSeverity.ERROR: 2,
            AlertSeverity.CRITICAL: 3
        }
        return levels.get(severity, 0)


# Convenience functions for creating alerts

def create_info_alert(
    alert_system: AlertSystem,
    title: str,
    message: str,
    source: str,
    details: Optional[Dict[str, Any]] = None
) -> Alert:
    """
    Create an info alert.
    
    Args:
        alert_system: Alert system to use
        title: Alert title
        message: Alert message
        source: Alert source
        details: Additional details
        
    Returns:
        Created alert
    """
    return alert_system.alert(
        title=title,
        message=message,
        severity=AlertSeverity.INFO,
        source=source,
        details=details
    )

def create_warning_alert(
    alert_system: AlertSystem,
    title: str,
    message: str,
    source: str,
    details: Optional[Dict[str, Any]] = None
) -> Alert:
    """
    Create a warning alert.
    
    Args:
        alert_system: Alert system to use
        title: Alert title
        message: Alert message
        source: Alert source
        details: Additional details
        
    Returns:
        Created alert
    """
    return alert_system.alert(
        title=title,
        message=message,
        severity=AlertSeverity.WARNING,
        source=source,
        details=details
    )

def create_error_alert(
    alert_system: AlertSystem,
    title: str,
    message: str,
    source: str,
    details: Optional[Dict[str, Any]] = None
) -> Alert:
    """
    Create an error alert.
    
    Args:
        alert_system: Alert system to use
        title: Alert title
        message: Alert message
        source: Alert source
        details: Additional details
        
    Returns:
        Created alert
    """
    return alert_system.alert(
        title=title,
        message=message,
        severity=AlertSeverity.ERROR,
        source=source,
        details=details
    )

def create_critical_alert(
    alert_system: AlertSystem,
    title: str,
    message: str,
    source: str,
    details: Optional[Dict[str, Any]] = None
) -> Alert:
    """
    Create a critical alert.
    
    Args:
        alert_system: Alert system to use
        title: Alert title
        message: Alert message
        source: Alert source
        details: Additional details
        
    Returns:
        Created alert
    """
    return alert_system.alert(
        title=title,
        message=message,
        severity=AlertSeverity.CRITICAL,
        source=source,
        details=details
    ) 