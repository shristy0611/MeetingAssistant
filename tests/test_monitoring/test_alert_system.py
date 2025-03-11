"""
Tests for the AlertSystem class.
"""

import pytest
import time
from unittest.mock import MagicMock, patch
import logging
from datetime import datetime

from src.core.alert_system import (
    AlertSystem, Alert, AlertSeverity,
    AlertNotifier, LoggingNotifier, EmailNotifier, WebhookNotifier
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO)

class TestAlertSystem:
    """Tests for the AlertSystem class."""
    
    @pytest.fixture
    def alert_system(self):
        """Create an AlertSystem instance for testing."""
        alert_sys = AlertSystem(
            service_name="test-service",
            max_history_size=5,
            default_notifiers=[]  # No default notifiers for testing
        )
        return alert_sys
    
    def test_init(self, alert_system):
        """Test initialization."""
        assert alert_system.service_name == "test-service"
        assert alert_system.max_history_size == 5
        assert alert_system.notifiers == {}
        assert alert_system.history == []
    
    def test_add_remove_notifier(self, alert_system):
        """Test adding and removing notifiers."""
        # Create mock notifiers
        notifier1 = MagicMock(spec=AlertNotifier)
        notifier1.name = "notifier1"
        
        notifier2 = MagicMock(spec=AlertNotifier)
        notifier2.name = "notifier2"
        
        # Add notifiers
        alert_system.add_notifier(notifier1)
        alert_system.add_notifier(notifier2)
        
        # Check they were added
        assert len(alert_system.notifiers) == 2
        assert "notifier1" in alert_system.notifiers
        assert "notifier2" in alert_system.notifiers
        assert alert_system.notifiers["notifier1"] == notifier1
        assert alert_system.notifiers["notifier2"] == notifier2
        
        # Remove a notifier
        result = alert_system.remove_notifier("notifier1")
        assert result is True
        
        # Check it was removed
        assert len(alert_system.notifiers) == 1
        assert "notifier1" not in alert_system.notifiers
        assert "notifier2" in alert_system.notifiers
        
        # Try to remove a non-existent notifier
        result = alert_system.remove_notifier("non_existent")
        assert result is False
    
    def test_alert(self, alert_system):
        """Test generating alerts."""
        # Create a mock notifier
        notifier = MagicMock(spec=AlertNotifier)
        notifier.name = "test_notifier"
        notifier.notify.return_value = True
        
        # Add the notifier
        alert_system.add_notifier(notifier)
        
        # Generate an alert
        alert = alert_system.alert(
            title="Test Alert",
            message="This is a test alert",
            severity=AlertSeverity.WARNING,
            source="test_source",
            details={"key": "value"}
        )
        
        # Check alert properties
        assert isinstance(alert, Alert)
        assert alert.title == "Test Alert"
        assert alert.message == "This is a test alert"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.source == "test_source"
        assert alert.details == {"key": "value"}
        
        # Wait a bit for the alert to be processed by the worker thread
        time.sleep(0.1)
        
        # Check if notifier was called
        notifier.notify.assert_called_once()
        
        # Check alert in the call
        called_alert = notifier.notify.call_args[0][0]
        assert called_alert.title == "Test Alert"
        assert called_alert.message == "This is a test alert"
        
        # Check history
        assert len(alert_system.history) == 1
    
    def test_alert_with_string_severity(self, alert_system):
        """Test generating alerts with string severity."""
        # Generate an alert with string severity
        alert = alert_system.alert(
            title="Test Alert",
            message="This is a test alert",
            severity="warning",  # String instead of enum
            source="test_source"
        )
        
        # Check severity was converted to enum
        assert alert.severity == AlertSeverity.WARNING
        
        # Try with invalid severity
        alert = alert_system.alert(
            title="Test Alert",
            message="This is a test alert",
            severity="invalid",  # Invalid severity
            source="test_source"
        )
        
        # Should default to WARNING
        assert alert.severity == AlertSeverity.WARNING
    
    def test_get_history(self, alert_system):
        """Test getting alert history."""
        # Generate some alerts
        alert_system.alert("Alert 1", "Message 1", AlertSeverity.INFO, "source1")
        alert_system.alert("Alert 2", "Message 2", AlertSeverity.WARNING, "source2")
        alert_system.alert("Alert 3", "Message 3", AlertSeverity.ERROR, "source1")
        alert_system.alert("Alert 4", "Message 4", AlertSeverity.CRITICAL, "source2")
        
        # Wait for alerts to be processed
        time.sleep(0.1)
        
        # Get all history
        history = alert_system.get_history()
        assert len(history) == 4
        
        # Check history format
        assert isinstance(history[0], dict)
        assert "alert_id" in history[0]
        assert "title" in history[0]
        assert "message" in history[0]
        assert "severity" in history[0]
        assert "source" in history[0]
        assert "timestamp" in history[0]
        assert "details" in history[0]
        
        # Get history with limit
        history = alert_system.get_history(limit=2)
        assert len(history) == 2
        
        # Get history with min_severity
        history = alert_system.get_history(min_severity=AlertSeverity.ERROR)
        assert len(history) == 2
        assert history[0]["severity"] in ["error", "critical"]
        assert history[1]["severity"] in ["error", "critical"]
        
        # Get history with source
        history = alert_system.get_history(source="source1")
        assert len(history) == 2
        assert history[0]["source"] == "source1"
        assert history[1]["source"] == "source1"
        
        # Combine filters
        history = alert_system.get_history(
            limit=1,
            min_severity=AlertSeverity.ERROR,
            source="source1"
        )
        assert len(history) == 1
        assert history[0]["severity"] == "error"
        assert history[0]["source"] == "source1"
    
    def test_clear_history(self, alert_system):
        """Test clearing alert history."""
        # Generate some alerts
        alert_system.alert("Alert 1", "Message 1", AlertSeverity.INFO, "source1")
        alert_system.alert("Alert 2", "Message 2", AlertSeverity.WARNING, "source2")
        
        # Wait for alerts to be processed
        time.sleep(0.1)
        
        # Check history
        assert len(alert_system.history) == 2
        
        # Clear history
        alert_system.clear_history()
        
        # Check history is empty
        assert len(alert_system.history) == 0
    
    def test_history_size_limit(self, alert_system):
        """Test that history size is limited."""
        # Generate more alerts than the history size
        for i in range(10):
            alert_system.alert(
                f"Alert {i}",
                f"Message {i}",
                AlertSeverity.INFO,
                "source"
            )
        
        # Wait for alerts to be processed
        time.sleep(0.1)
        
        # Check history size
        assert len(alert_system.history) == 5  # max_history_size is 5
        
        # Check most recent alerts are kept
        history = alert_system.get_history()
        assert history[0]["title"] == "Alert 9"
        assert history[1]["title"] == "Alert 8"


class TestNotifiers:
    """Tests for alert notifiers."""
    
    def test_logging_notifier(self):
        """Test the LoggingNotifier."""
        # Create a LoggingNotifier
        notifier = LoggingNotifier(name="test_logger")
        
        # Check name
        assert notifier.name == "test_logger"
        
        # Create alerts of different severities
        info_alert = Alert(
            title="Info Alert",
            message="This is an info alert",
            severity=AlertSeverity.INFO,
            source="test_source"
        )
        
        warning_alert = Alert(
            title="Warning Alert",
            message="This is a warning alert",
            severity=AlertSeverity.WARNING,
            source="test_source"
        )
        
        error_alert = Alert(
            title="Error Alert",
            message="This is an error alert",
            severity=AlertSeverity.ERROR,
            source="test_source"
        )
        
        critical_alert = Alert(
            title="Critical Alert",
            message="This is a critical alert",
            severity=AlertSeverity.CRITICAL,
            source="test_source"
        )
        
        # Mock the logger
        notifier.logger = MagicMock()
        
        # Send alerts
        notifier.notify(info_alert)
        notifier.notify(warning_alert)
        notifier.notify(error_alert)
        notifier.notify(critical_alert)
        
        # Check that logger was called with correct method for each severity
        notifier.logger.info.assert_called_once()
        notifier.logger.warning.assert_called_once()
        notifier.logger.error.assert_called_once()
        notifier.logger.critical.assert_called_once()
    
    def test_email_notifier(self):
        """Test the EmailNotifier."""
        # Create an EmailNotifier
        notifier = EmailNotifier(
            name="test_email",
            smtp_server="smtp.example.com",
            smtp_port=587,
            sender="alerts@example.com",
            recipients=["admin@example.com"],
            use_ssl=True,
            username="user",
            password="pass",
            min_severity=AlertSeverity.WARNING
        )
        
        # Check configuration
        assert notifier.name == "test_email"
        assert notifier.smtp_server == "smtp.example.com"
        assert notifier.smtp_port == 587
        assert notifier.sender == "alerts@example.com"
        assert notifier.recipients == ["admin@example.com"]
        assert notifier.use_ssl is True
        assert notifier.username == "user"
        assert notifier.password == "pass"
        assert notifier.min_severity == AlertSeverity.WARNING
        
        # Create alerts of different severities
        info_alert = Alert(
            title="Info Alert",
            message="This is an info alert",
            severity=AlertSeverity.INFO,
            source="test_source"
        )
        
        warning_alert = Alert(
            title="Warning Alert",
            message="This is a warning alert",
            severity=AlertSeverity.WARNING,
            source="test_source"
        )
        
        # INFO alert should not trigger a notification (below min_severity)
        with patch('logging.Logger.info') as mock_log:
            notifier.notify(info_alert)
            mock_log.assert_not_called()
        
        # WARNING alert should trigger a notification
        with patch('logging.Logger.info') as mock_log:
            notifier.notify(warning_alert)
            mock_log.assert_called_once()
    
    def test_webhook_notifier(self):
        """Test the WebhookNotifier."""
        # Create a WebhookNotifier
        notifier = WebhookNotifier(
            name="test_webhook",
            webhook_url="https://example.com/webhook",
            headers={"Authorization": "Bearer token"},
            min_severity=AlertSeverity.ERROR
        )
        
        # Check configuration
        assert notifier.name == "test_webhook"
        assert notifier.webhook_url == "https://example.com/webhook"
        assert notifier.headers == {"Authorization": "Bearer token"}
        assert notifier.min_severity == AlertSeverity.ERROR
        
        # Create alerts of different severities
        warning_alert = Alert(
            title="Warning Alert",
            message="This is a warning alert",
            severity=AlertSeverity.WARNING,
            source="test_source"
        )
        
        error_alert = Alert(
            title="Error Alert",
            message="This is an error alert",
            severity=AlertSeverity.ERROR,
            source="test_source"
        )
        
        # WARNING alert should not trigger a notification (below min_severity)
        with patch('logging.Logger.info') as mock_log:
            notifier.notify(warning_alert)
            mock_log.assert_not_called()
        
        # ERROR alert should trigger a notification
        with patch('logging.Logger.info') as mock_log:
            notifier.notify(error_alert)
            mock_log.assert_called_once() 