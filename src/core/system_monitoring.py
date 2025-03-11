"""
System Monitoring Module for AMPTALK

This module provides comprehensive system monitoring capabilities using OpenTelemetry
for metrics collection, with support for Prometheus and Grafana visualization.
"""

import logging
import threading
import time
from typing import Dict, Any, List, Optional
import platform
import os
import psutil

# OpenTelemetry imports
try:
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import (
        ConsoleMetricExporter,
        PeriodicExportingMetricReader,
    )
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from prometheus_client import start_http_server
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

logger = logging.getLogger(__name__)

class SystemMonitor:
    """
    A comprehensive system monitoring component using OpenTelemetry.
    
    Features:
    - Performance metrics collection
    - Resource utilization tracking
    - Error monitoring
    - Health checks
    - Prometheus integration
    """
    
    def __init__(
        self,
        service_name: str = "amptalk",
        prometheus_port: int = 8000,
        metrics_export_interval: float = 5.0,  # seconds
        enable_prometheus: bool = True,
        enable_console: bool = False,
        collect_interval: float = 1.0,  # seconds
    ):
        """
        Initialize the SystemMonitor.
        
        Args:
            service_name: Name of the service being monitored
            prometheus_port: Port to expose Prometheus metrics on
            metrics_export_interval: Interval for exporting metrics in seconds
            enable_prometheus: Whether to enable Prometheus exporter
            enable_console: Whether to enable console exporter
            collect_interval: Interval for collecting system metrics in seconds
        """
        self.service_name = service_name
        self.prometheus_port = prometheus_port
        self.metrics_export_interval = metrics_export_interval
        self.enable_prometheus = enable_prometheus
        self.enable_console = enable_console
        self.collect_interval = collect_interval
        
        # System info
        self.system_info = {
            "os": platform.system(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "hostname": platform.node(),
        }
        
        # Initialize metrics only if OpenTelemetry is available
        if OPENTELEMETRY_AVAILABLE:
            self._setup_metrics()
            
            # Start metrics collection
            self._start_metrics_collection()
        else:
            logger.warning(
                "OpenTelemetry not available. Install with: "
                "pip install opentelemetry-api opentelemetry-sdk "
                "opentelemetry-exporter-prometheus prometheus-client psutil"
            )
    
    def _setup_metrics(self):
        """Set up OpenTelemetry metrics collection."""
        # Create readers for metrics export
        readers = []
        
        if self.enable_prometheus:
            # Start Prometheus HTTP server
            start_http_server(port=self.prometheus_port)
            logger.info(f"Prometheus metrics server started on port {self.prometheus_port}")
            
            # Create Prometheus metric reader
            prometheus_reader = PrometheusMetricReader()
            readers.append(prometheus_reader)
        
        if self.enable_console:
            # Create console metric reader
            console_reader = PeriodicExportingMetricReader(
                ConsoleMetricExporter(),
                export_interval_millis=int(self.metrics_export_interval * 1000)
            )
            readers.append(console_reader)
        
        # Create meter provider with readers
        provider = MeterProvider(metric_readers=readers)
        
        # Set global meter provider
        metrics.set_meter_provider(provider)
        
        # Get meter
        self.meter = metrics.get_meter(self.service_name)
        
        # Create metrics
        self._create_metrics()
        
        logger.info("OpenTelemetry metrics initialized")
    
    def _create_metrics(self):
        """Create various metrics for monitoring."""
        # System metrics
        self.cpu_usage = self.meter.create_gauge(
            name="system.cpu.usage",
            description="CPU usage percentage",
            unit="%"
        )
        
        self.memory_usage = self.meter.create_gauge(
            name="system.memory.usage",
            description="Memory usage percentage",
            unit="%"
        )
        
        self.memory_available = self.meter.create_gauge(
            name="system.memory.available",
            description="Available memory",
            unit="bytes"
        )
        
        self.disk_usage = self.meter.create_gauge(
            name="system.disk.usage",
            description="Disk usage percentage",
            unit="%"
        )
        
        # Application metrics
        self.active_threads = self.meter.create_gauge(
            name="app.threads.active",
            description="Number of active threads",
            unit="1"
        )
        
        self.open_file_descriptors = self.meter.create_gauge(
            name="app.fd.open",
            description="Number of open file descriptors",
            unit="1"
        )
        
        # Error metrics
        self.error_counter = self.meter.create_counter(
            name="app.errors",
            description="Counter for application errors",
            unit="1"
        )
        
        # Health metrics
        self.health_status = self.meter.create_gauge(
            name="app.health.status",
            description="Health status (1=healthy, 0=unhealthy)",
            unit="1"
        )
        
        # Request metrics
        self.request_counter = self.meter.create_counter(
            name="app.requests",
            description="Counter for requests",
            unit="1"
        )
        
        self.request_duration = self.meter.create_histogram(
            name="app.request.duration",
            description="Request duration",
            unit="ms"
        )
    
    def _start_metrics_collection(self):
        """Start a background thread for collecting system metrics."""
        def collect_metrics():
            while True:
                try:
                    # Collect system metrics
                    self._collect_system_metrics()
                    
                    # Sleep until next collection
                    time.sleep(self.collect_interval)
                except Exception as e:
                    logger.error(f"Error collecting metrics: {e}")
        
        # Start collection thread
        thread = threading.Thread(
            target=collect_metrics,
            daemon=True,
            name="MetricsCollection"
        )
        thread.start()
        logger.info("Started metrics collection thread")
    
    def _collect_system_metrics(self):
        """Collect and record system metrics."""
        if not OPENTELEMETRY_AVAILABLE:
            return
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)
        self.cpu_usage.set(cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.memory_usage.set(memory.percent)
        self.memory_available.set(memory.available)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        self.disk_usage.set(disk.percent)
        
        # Thread count
        self.active_threads.set(threading.active_count())
        
        # File descriptors (Unix-like systems only)
        if hasattr(os, 'getpid') and hasattr(psutil.Process, 'open_files'):
            try:
                process = psutil.Process(os.getpid())
                self.open_file_descriptors.set(len(process.open_files()))
            except (psutil.Error, OSError) as e:
                logger.debug(f"Could not get open file count: {e}")
        
        # Set health status (default to healthy)
        self.health_status.set(1)
    
    def record_error(self, error_type: str, error_details: Optional[Dict[str, Any]] = None):
        """
        Record an application error.
        
        Args:
            error_type: Type of error
            error_details: Additional details about the error
        """
        if not OPENTELEMETRY_AVAILABLE:
            return
        
        attributes = {"error_type": error_type}
        if error_details:
            for key, value in error_details.items():
                if isinstance(value, (str, int, float, bool)):
                    attributes[key] = value
        
        self.error_counter.add(1, attributes)
        logger.info(f"Recorded error: {error_type}")
    
    def record_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        duration_ms: float
    ):
        """
        Record a request.
        
        Args:
            endpoint: Request endpoint
            method: HTTP method
            status_code: Response status code
            duration_ms: Request duration in milliseconds
        """
        if not OPENTELEMETRY_AVAILABLE:
            return
        
        attributes = {
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code
        }
        
        self.request_counter.add(1, attributes)
        self.request_duration.record(duration_ms, attributes)
        
        logger.debug(f"Recorded request: {method} {endpoint} {status_code} {duration_ms}ms")
    
    def set_health_status(self, healthy: bool, reason: Optional[str] = None):
        """
        Set the health status of the application.
        
        Args:
            healthy: Whether the application is healthy
            reason: Reason for the health status
        """
        if not OPENTELEMETRY_AVAILABLE:
            return
        
        attributes = {}
        if reason:
            attributes["reason"] = reason
        
        self.health_status.set(1 if healthy else 0, attributes)
        logger.info(f"Health status set to {'healthy' if healthy else 'unhealthy'}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information.
        
        Returns:
            Dictionary with system information
        """
        return self.system_info 