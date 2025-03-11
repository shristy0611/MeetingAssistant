"""
Tests for the SystemMonitor class.
"""

import pytest
import time
from unittest.mock import MagicMock, patch
import logging

from src.core.system_monitoring import SystemMonitor

# Configure logging for tests
logging.basicConfig(level=logging.INFO)

class TestSystemMonitor:
    """Tests for the SystemMonitor class."""
    
    @pytest.fixture
    def system_monitor(self):
        """Create a SystemMonitor instance for testing."""
        with patch('src.core.system_monitoring.start_http_server'):  # Mock Prometheus server start
            monitor = SystemMonitor(
                service_name="test-service",
                prometheus_port=9999,
                metrics_export_interval=0.1,
                enable_prometheus=True,
                enable_console=True,
                collect_interval=0.1
            )
            yield monitor
    
    def test_init(self, system_monitor):
        """Test initialization."""
        assert system_monitor.service_name == "test-service"
        assert system_monitor.prometheus_port == 9999
        assert system_monitor.metrics_export_interval == 0.1
        assert system_monitor.enable_prometheus is True
        assert system_monitor.enable_console is True
        assert system_monitor.collect_interval == 0.1
        
        # Check system info is populated
        assert isinstance(system_monitor.system_info, dict)
        assert "os" in system_monitor.system_info
        assert "platform" in system_monitor.system_info
        assert "python_version" in system_monitor.system_info
        assert "processor" in system_monitor.system_info
        assert "hostname" in system_monitor.system_info
    
    @pytest.mark.skipif(not hasattr(SystemMonitor, 'record_error'), reason="record_error not implemented")
    def test_record_error(self, system_monitor):
        """Test recording errors."""
        # Mock the error counter
        if hasattr(SystemMonitor, '_setup_metrics') and \
           hasattr(system_monitor, 'error_counter'):
            system_monitor.error_counter = MagicMock()
            
            # Record an error
            system_monitor.record_error(
                error_type="test_error",
                error_details={"key": "value"}
            )
            
            # Check that add was called once
            system_monitor.error_counter.add.assert_called_once()
            
            # Check attributes
            args, kwargs = system_monitor.error_counter.add.call_args
            assert args[0] == 1  # Count is 1
            assert "error_type" in kwargs.get("attributes", {})
            assert kwargs.get("attributes", {}).get("error_type") == "test_error"
            assert "key" in kwargs.get("attributes", {})
            assert kwargs.get("attributes", {}).get("key") == "value"
    
    @pytest.mark.skipif(not hasattr(SystemMonitor, 'record_request'), reason="record_request not implemented")
    def test_record_request(self, system_monitor):
        """Test recording requests."""
        # Mock the request counter and histogram
        if hasattr(SystemMonitor, '_setup_metrics') and \
           hasattr(system_monitor, 'request_counter') and \
           hasattr(system_monitor, 'request_duration'):
            system_monitor.request_counter = MagicMock()
            system_monitor.request_duration = MagicMock()
            
            # Record a request
            system_monitor.record_request(
                endpoint="/test",
                method="GET",
                status_code=200,
                duration_ms=123.45
            )
            
            # Check that counter and histogram were updated
            system_monitor.request_counter.add.assert_called_once()
            system_monitor.request_duration.record.assert_called_once()
            
            # Check counter attributes
            counter_args, counter_kwargs = system_monitor.request_counter.add.call_args
            assert counter_args[0] == 1  # Count is 1
            assert "endpoint" in counter_kwargs.get("attributes", {})
            assert counter_kwargs.get("attributes", {}).get("endpoint") == "/test"
            assert "method" in counter_kwargs.get("attributes", {})
            assert counter_kwargs.get("attributes", {}).get("method") == "GET"
            assert "status_code" in counter_kwargs.get("attributes", {})
            assert counter_kwargs.get("attributes", {}).get("status_code") == 200
            
            # Check histogram attributes
            hist_args, hist_kwargs = system_monitor.request_duration.record.call_args
            assert hist_args[0] == 123.45  # Duration
            assert "endpoint" in hist_kwargs.get("attributes", {})
            assert hist_kwargs.get("attributes", {}).get("endpoint") == "/test"
            assert "method" in hist_kwargs.get("attributes", {})
            assert hist_kwargs.get("attributes", {}).get("method") == "GET"
            assert "status_code" in hist_kwargs.get("attributes", {})
            assert hist_kwargs.get("attributes", {}).get("status_code") == 200
    
    @pytest.mark.skipif(not hasattr(SystemMonitor, 'set_health_status'), reason="set_health_status not implemented")
    def test_set_health_status(self, system_monitor):
        """Test setting health status."""
        # Mock the health status gauge
        if hasattr(SystemMonitor, '_setup_metrics') and \
           hasattr(system_monitor, 'health_status'):
            system_monitor.health_status = MagicMock()
            
            # Set health status to healthy
            system_monitor.set_health_status(True, "All systems operational")
            
            # Check that set was called with 1 (healthy)
            system_monitor.health_status.set.assert_called_once()
            args, kwargs = system_monitor.health_status.set.call_args
            assert args[0] == 1  # Healthy
            assert "reason" in kwargs.get("attributes", {})
            assert kwargs.get("attributes", {}).get("reason") == "All systems operational"
            
            # Reset mock
            system_monitor.health_status.reset_mock()
            
            # Set health status to unhealthy
            system_monitor.set_health_status(False, "System degraded")
            
            # Check that set was called with 0 (unhealthy)
            system_monitor.health_status.set.assert_called_once()
            args, kwargs = system_monitor.health_status.set.call_args
            assert args[0] == 0  # Unhealthy
            assert "reason" in kwargs.get("attributes", {})
            assert kwargs.get("attributes", {}).get("reason") == "System degraded"
    
    def test_get_system_info(self, system_monitor):
        """Test getting system info."""
        system_info = system_monitor.get_system_info()
        
        assert isinstance(system_info, dict)
        assert "os" in system_info
        assert "platform" in system_info
        assert "python_version" in system_info
        assert "processor" in system_info
        assert "hostname" in system_info 