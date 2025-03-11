"""
Tests for the HealthCheck class.
"""

import pytest
import time
from unittest.mock import MagicMock, patch
import logging
from datetime import datetime, timedelta

from src.core.health_check import HealthCheck, HealthStatus

# Configure logging for tests
logging.basicConfig(level=logging.INFO)

class TestHealthCheck:
    """Tests for the HealthCheck class."""
    
    @pytest.fixture
    def health_check(self):
        """Create a HealthCheck instance for testing."""
        # Create with auto_check disabled to avoid background thread in tests
        health = HealthCheck(
            service_name="test-service",
            check_interval=0.1,
            history_size=5,
            auto_check=False
        )
        return health
    
    def test_init(self, health_check):
        """Test initialization."""
        assert health_check.service_name == "test-service"
        assert health_check.check_interval == 0.1
        assert health_check.history_size == 5
        assert health_check.auto_check is False
        
        assert health_check.checks == {}
        assert health_check.results == {}
        assert health_check.history == {}
    
    def test_register_unregister_check(self, health_check):
        """Test registering and unregistering health checks."""
        # Create a mock health check function
        check_function = MagicMock(return_value=(True, "All good"))
        
        # Register the check
        health_check.register_check(
            name="test_check",
            check_function=check_function,
            description="Test health check"
        )
        
        # Check it was registered correctly
        assert "test_check" in health_check.checks
        assert health_check.checks["test_check"]["function"] == check_function
        assert health_check.checks["test_check"]["description"] == "Test health check"
        
        # Check initial result
        assert "test_check" in health_check.results
        status, message, timestamp = health_check.results["test_check"]
        assert status == HealthStatus.UNKNOWN
        assert message == "Not checked yet"
        assert isinstance(timestamp, datetime)
        
        # Check history
        assert "test_check" in health_check.history
        assert health_check.history["test_check"] == []
        
        # Unregister the check
        result = health_check.unregister_check("test_check")
        assert result is True
        
        # Check it was unregistered
        assert "test_check" not in health_check.checks
        assert "test_check" not in health_check.results
        assert "test_check" not in health_check.history
        
        # Try to unregister a non-existent check
        result = health_check.unregister_check("non_existent")
        assert result is False
    
    def test_run_check(self, health_check):
        """Test running a health check."""
        # Create a mock health check function
        check_function = MagicMock(return_value=(True, "All good"))
        
        # Register the check
        health_check.register_check(
            name="test_check",
            check_function=check_function,
            description="Test health check"
        )
        
        # Run the check
        status, message, timestamp = health_check.run_check("test_check")
        
        # Check results
        assert status == HealthStatus.HEALTHY
        assert message == "All good"
        assert isinstance(timestamp, datetime)
        
        # Check that the check function was called
        check_function.assert_called_once()
        
        # Check that the result was stored
        assert health_check.results["test_check"] == (status, message, timestamp)
        
        # Check that the history was updated
        assert len(health_check.history["test_check"]) == 1
        assert health_check.history["test_check"][0] == (status, message, timestamp)
        
        # Try to run a non-existent check
        with pytest.raises(KeyError):
            health_check.run_check("non_existent")
    
    def test_run_check_unhealthy(self, health_check):
        """Test running a health check that returns unhealthy."""
        # Create a mock health check function
        check_function = MagicMock(return_value=(False, "Something is wrong"))
        
        # Register the check
        health_check.register_check(
            name="test_check",
            check_function=check_function,
            description="Test health check"
        )
        
        # Run the check
        status, message, timestamp = health_check.run_check("test_check")
        
        # Check results
        assert status == HealthStatus.UNHEALTHY
        assert message == "Something is wrong"
        assert isinstance(timestamp, datetime)
    
    def test_run_check_exception(self, health_check):
        """Test running a health check that raises an exception."""
        # Create a mock health check function that raises an exception
        def check_function():
            raise ValueError("Test exception")
        
        # Register the check
        health_check.register_check(
            name="test_check",
            check_function=check_function,
            description="Test health check"
        )
        
        # Run the check
        status, message, timestamp = health_check.run_check("test_check")
        
        # Check results
        assert status == HealthStatus.UNHEALTHY
        assert "Error in health check: Test exception" in message
        assert isinstance(timestamp, datetime)
    
    def test_run_all_checks(self, health_check):
        """Test running all health checks."""
        # Create mock health check functions
        check1 = MagicMock(return_value=(True, "Service 1 OK"))
        check2 = MagicMock(return_value=(False, "Service 2 failed"))
        
        # Register the checks
        health_check.register_check("service1", check1, "Service 1 health check")
        health_check.register_check("service2", check2, "Service 2 health check")
        
        # Run all checks
        results = health_check.run_all_checks()
        
        # Check results
        assert len(results) == 2
        assert "service1" in results
        assert "service2" in results
        
        service1_status, service1_message, service1_timestamp = results["service1"]
        assert service1_status == HealthStatus.HEALTHY
        assert service1_message == "Service 1 OK"
        assert isinstance(service1_timestamp, datetime)
        
        service2_status, service2_message, service2_timestamp = results["service2"]
        assert service2_status == HealthStatus.UNHEALTHY
        assert service2_message == "Service 2 failed"
        assert isinstance(service2_timestamp, datetime)
        
        # Check that the check functions were called
        check1.assert_called_once()
        check2.assert_called_once()
    
    def test_get_overall_status(self, health_check):
        """Test getting the overall health status."""
        # No checks, should be UNKNOWN
        assert health_check.get_overall_status() == HealthStatus.UNKNOWN
        
        # Add a healthy check
        health_check.register_check("service1", lambda: (True, "OK"), "Service 1")
        health_check.run_check("service1")
        assert health_check.get_overall_status() == HealthStatus.HEALTHY
        
        # Add an unhealthy check
        health_check.register_check("service2", lambda: (False, "Failed"), "Service 2")
        health_check.run_check("service2")
        assert health_check.get_overall_status() == HealthStatus.UNHEALTHY
        
        # Fix the unhealthy check
        health_check.checks["service2"]["function"] = lambda: (True, "Fixed")
        health_check.run_check("service2")
        assert health_check.get_overall_status() == HealthStatus.HEALTHY
    
    def test_get_status_report(self, health_check):
        """Test getting the health status report."""
        # Create mock health check functions
        check1 = MagicMock(return_value=(True, "Service 1 OK"))
        check2 = MagicMock(return_value=(False, "Service 2 failed"))
        
        # Register the checks
        health_check.register_check("service1", check1, "Service 1 health check")
        health_check.register_check("service2", check2, "Service 2 health check")
        
        # Run all checks
        health_check.run_all_checks()
        
        # Get status report
        report = health_check.get_status_report()
        
        # Check report structure
        assert "service" in report
        assert report["service"] == "test-service"
        assert "timestamp" in report
        assert isinstance(datetime.fromisoformat(report["timestamp"]), datetime)
        assert "status" in report
        assert report["status"] == "unhealthy"  # Overall status is UNHEALTHY
        assert "checks" in report
        
        # Check checks
        checks = report["checks"]
        assert len(checks) == 2
        assert "service1" in checks
        assert "service2" in checks
        
        # Check service1
        service1 = checks["service1"]
        assert service1["status"] == "healthy"
        assert service1["message"] == "Service 1 OK"
        assert isinstance(datetime.fromisoformat(service1["timestamp"]), datetime)
        assert service1["description"] == "Service 1 health check"
        
        # Check service2
        service2 = checks["service2"]
        assert service2["status"] == "unhealthy"
        assert service2["message"] == "Service 2 failed"
        assert isinstance(datetime.fromisoformat(service2["timestamp"]), datetime)
        assert service2["description"] == "Service 2 health check"
    
    def test_history_management(self, health_check):
        """Test history management."""
        # Create health check with small history size
        health_check = HealthCheck(
            service_name="test-service",
            history_size=3,
            auto_check=False
        )
        
        # Register a check
        health_check.register_check("test", lambda: (True, "OK"), "Test check")
        
        # Run the check multiple times
        for i in range(5):
            health_check.run_check("test")
        
        # Check that history was trimmed to the specified size
        assert len(health_check.history["test"]) == 3
        
        # Get check history
        history = health_check.get_check_history("test")
        
        # Check history
        assert len(history) == 3
        assert isinstance(history[0], dict)
        assert "status" in history[0]
        assert "message" in history[0]
        assert "timestamp" in history[0]
        
        # Get history with limit
        history = health_check.get_check_history("test", limit=2)
        assert len(history) == 2
        
        # Try to get history for non-existent check
        with pytest.raises(KeyError):
            health_check.get_check_history("non_existent")
        
        # Clear history for specific check
        health_check.clear_history("test")
        assert health_check.history["test"] == []
        
        # Run the check again
        health_check.run_check("test")
        assert len(health_check.history["test"]) == 1
        
        # Register another check
        health_check.register_check("test2", lambda: (True, "OK"), "Test check 2")
        health_check.run_check("test2")
        
        # Clear all history
        health_check.clear_history()
        assert health_check.history["test"] == []
        assert health_check.history["test2"] == []
        
        # Try to clear history for non-existent check
        with pytest.raises(KeyError):
            health_check.clear_history("non_existent") 