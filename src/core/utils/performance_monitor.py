"""
Performance Monitoring Module for the AMPTALK Multi-Agent Framework.

This module provides real-time performance monitoring capabilities using 
OpenTelemetry and Prometheus. It allows tracking metrics such as message 
processing times, queue sizes, memory usage, and agent health information.

Author: AMPTALK Team
Date: 2024
"""

import time
import logging
import asyncio
import threading
import platform
import psutil
from typing import Dict, List, Any, Optional, Set, Union, Callable, Coroutine
from enum import Enum
from contextlib import contextmanager
import gc
from datetime import datetime

# OpenTelemetry imports
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from prometheus_client import start_http_server

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics that can be tracked."""
    
    COUNTER = "counter"           # Continuously increasing counter
    GAUGE = "gauge"               # Value that can go up and down
    HISTOGRAM = "histogram"       # Distribution of values
    UPDOWNCOUNTER = "updowncounter"  # Counter that can go up and down


class PerformanceMonitor:
    """
    Performance monitoring system for the AMPTALK agent framework.
    
    This class provides real-time monitoring capabilities for agents, tracking
    metrics such as message throughput, processing times, memory usage, and more.
    """
    
    def __init__(self, 
                name: str = "amptalk_monitor", 
                enable_prometheus: bool = True,
                prometheus_port: int = 8000,
                collection_interval_seconds: float = 5.0):
        """
        Initialize the performance monitor.
        
        Args:
            name: Name for this monitor instance
            enable_prometheus: Whether to expose metrics via Prometheus endpoint
            prometheus_port: Port for the Prometheus HTTP server
            collection_interval_seconds: Interval for collecting metrics
        """
        self.name = name
        self.enable_prometheus = enable_prometheus
        self.prometheus_port = prometheus_port
        self.collection_interval_seconds = collection_interval_seconds
        
        # Initialize metrics
        self._initialize_metrics()
        
        # Track monitored agents
        self.monitored_agents: Dict[str, Any] = {}
        
        # Flag for the collection loop
        self.is_collecting = False
        self.collection_task = None
        
        logger.info(f"Initialized {self.name} performance monitor")
    
    def _initialize_metrics(self):
        """Initialize the OpenTelemetry metrics infrastructure."""
        # Set up the metrics provider
        readers = []
        
        # Add Prometheus exporter if enabled
        if self.enable_prometheus:
            readers.append(PrometheusMetricReader())
        
        # Set up the meter provider with configured readers
        self.meter_provider = MeterProvider(metric_readers=readers)
        metrics.set_meter_provider(self.meter_provider)
        
        # Create a meter
        self.meter = metrics.get_meter(self.name)
        
        # Define basic metrics
        self._setup_metrics()
        
        # Start the Prometheus HTTP server if enabled
        if self.enable_prometheus:
            try:
                start_http_server(port=self.prometheus_port)
                logger.info(f"Started Prometheus metrics server on port {self.prometheus_port}")
            except Exception as e:
                logger.error(f"Failed to start Prometheus metrics server: {str(e)}")
    
    def _setup_metrics(self):
        """Set up the metrics to be tracked."""
        # System-level metrics
        self.system_cpu_usage = self.meter.create_gauge(
            name="system_cpu_usage_percent",
            description="System CPU usage percentage",
            unit="percent"
        )
        
        self.system_memory_usage = self.meter.create_gauge(
            name="system_memory_usage_percent",
            description="System memory usage percentage",
            unit="percent"
        )
        
        self.process_cpu_usage = self.meter.create_gauge(
            name="process_cpu_usage_percent",
            description="Process CPU usage percentage",
            unit="percent"
        )
        
        self.process_memory_usage = self.meter.create_gauge(
            name="process_memory_usage_bytes",
            description="Process memory usage in bytes",
            unit="bytes"
        )
        
        # Agent-level metrics
        self.agent_message_received = self.meter.create_counter(
            name="agent_messages_received_total",
            description="Total number of messages received by an agent",
            unit="1"
        )
        
        self.agent_message_processed = self.meter.create_counter(
            name="agent_messages_processed_total",
            description="Total number of messages processed by an agent",
            unit="1"
        )
        
        self.agent_message_sent = self.meter.create_counter(
            name="agent_messages_sent_total",
            description="Total number of messages sent by an agent",
            unit="1"
        )
        
        self.agent_message_failed = self.meter.create_counter(
            name="agent_messages_failed_total",
            description="Total number of messages that failed processing",
            unit="1"
        )
        
        self.agent_processing_time = self.meter.create_histogram(
            name="agent_message_processing_time_seconds",
            description="Time taken to process a message in seconds",
            unit="seconds"
        )
        
        self.agent_queue_size = self.meter.create_gauge(
            name="agent_queue_size",
            description="Current size of an agent's message queue",
            unit="1"
        )
        
        self.agent_is_running = self.meter.create_gauge(
            name="agent_is_running",
            description="Whether an agent is currently running (1) or not (0)",
            unit="1"
        )
        
        self.agent_uptime = self.meter.create_gauge(
            name="agent_uptime_seconds",
            description="How long an agent has been running in seconds",
            unit="seconds"
        )
        
        self.agent_retry_attempts = self.meter.create_counter(
            name="agent_retry_attempts_total",
            description="Total number of retry attempts made by an agent",
            unit="1"
        )
        
        self.agent_retry_success = self.meter.create_counter(
            name="agent_retry_success_total",
            description="Total number of successful retry attempts",
            unit="1"
        )
        
        # State manager metrics
        self.state_manager_items = self.meter.create_gauge(
            name="state_manager_items",
            description="Number of items in a state manager",
            unit="1"
        )
        
        self.state_manager_size = self.meter.create_gauge(
            name="state_manager_size_bytes",
            description="Size of state data in bytes",
            unit="bytes"
        )
    
    async def start_collecting(self):
        """Start the metrics collection loop."""
        if self.is_collecting:
            logger.warning(f"{self.name} is already collecting metrics")
            return
        
        self.is_collecting = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        logger.info(f"{self.name} started collecting metrics")
    
    async def stop_collecting(self):
        """Stop the metrics collection loop."""
        if not self.is_collecting:
            logger.warning(f"{self.name} is not collecting metrics")
            return
        
        self.is_collecting = False
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
            self.collection_task = None
        
        logger.info(f"{self.name} stopped collecting metrics")
    
    async def _collection_loop(self):
        """Main loop for collecting metrics."""
        try:
            while self.is_collecting:
                await self._collect_system_metrics()
                await self._collect_agent_metrics()
                await asyncio.sleep(self.collection_interval_seconds)
        except asyncio.CancelledError:
            logger.info(f"{self.name} collection loop cancelled")
        except Exception as e:
            logger.error(f"Error in {self.name} collection loop: {str(e)}")
            self.is_collecting = False
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            # System CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.system_cpu_usage.set(cpu_percent)
            
            # System memory usage
            memory = psutil.virtual_memory()
            self.system_memory_usage.set(memory.percent)
            
            # Process metrics
            process = psutil.Process()
            
            # Process CPU usage
            process_cpu = process.cpu_percent(interval=0.1) / psutil.cpu_count()
            self.process_cpu_usage.set(process_cpu)
            
            # Process memory usage
            process_memory = process.memory_info().rss
            self.process_memory_usage.set(process_memory)
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")
    
    async def _collect_agent_metrics(self):
        """Collect metrics from all monitored agents."""
        for agent_id, agent in list(self.monitored_agents.items()):
            try:
                await self._collect_agent_specific_metrics(agent)
            except Exception as e:
                logger.error(f"Error collecting metrics for agent {agent.name}: {str(e)}")
    
    async def _collect_agent_specific_metrics(self, agent):
        """Collect metrics for a specific agent."""
        try:
            # Get agent status info
            status = await agent.get_status()
            
            # Basic agent metrics
            agent_attrs = {"agent_id": agent.agent_id, "agent_name": agent.name}
            
            # Update running status
            self.agent_is_running.set(1 if status.get("is_running", False) else 0, agent_attrs)
            
            # Queue size
            queue_size = status.get("queue_size", 0)
            self.agent_queue_size.set(queue_size, agent_attrs)
            
            # Extract stats from status
            stats = status.get("stats", {})
            
            # Message stats
            msg_stats = stats.get("messages", {})
            self.agent_message_received.add(
                msg_stats.get("received", 0) - self.get_last_value(agent.agent_id, "received", 0),
                agent_attrs
            )
            self.agent_message_processed.add(
                msg_stats.get("processed", 0) - self.get_last_value(agent.agent_id, "processed", 0),
                agent_attrs
            )
            self.agent_message_sent.add(
                msg_stats.get("sent", 0) - self.get_last_value(agent.agent_id, "sent", 0),
                agent_attrs
            )
            self.agent_message_failed.add(
                msg_stats.get("failed", 0) - self.get_last_value(agent.agent_id, "failed", 0),
                agent_attrs
            )
            
            # Update last values
            self.update_last_value(agent.agent_id, "received", msg_stats.get("received", 0))
            self.update_last_value(agent.agent_id, "processed", msg_stats.get("processed", 0))
            self.update_last_value(agent.agent_id, "sent", msg_stats.get("sent", 0))
            self.update_last_value(agent.agent_id, "failed", msg_stats.get("failed", 0))
            
            # Processing time
            timing_stats = stats.get("timing", {})
            avg_processing_time = timing_stats.get("avg_processing_time", 0)
            if avg_processing_time > 0:
                self.agent_processing_time.record(avg_processing_time, agent_attrs)
            
            # Uptime
            uptime = timing_stats.get("uptime", 0)
            if uptime is not None:
                self.agent_uptime.set(uptime, agent_attrs)
            
            # Retry stats
            retry_stats = stats.get("errors", {}).get("retries", {})
            self.agent_retry_attempts.add(
                retry_stats.get("total", 0) - self.get_last_value(agent.agent_id, "retry_total", 0),
                agent_attrs
            )
            self.agent_retry_success.add(
                retry_stats.get("successful", 0) - self.get_last_value(agent.agent_id, "retry_success", 0),
                agent_attrs
            )
            
            # Update last values
            self.update_last_value(agent.agent_id, "retry_total", retry_stats.get("total", 0))
            self.update_last_value(agent.agent_id, "retry_success", retry_stats.get("successful", 0))
            
            # State manager metrics if available
            state_manager_stats = status.get("state_manager", {})
            if state_manager_stats:
                # Add state manager specific attributes
                sm_attrs = agent_attrs.copy()
                sm_attrs["storage_type"] = state_manager_stats.get("storage_type", "unknown")
                
                # Items in state manager
                items = state_manager_stats.get("memory_cache_items", 0)
                self.state_manager_items.set(items, sm_attrs)
                
                # Memory usage
                memory_stats = state_manager_stats.get("memory_optimizer", {})
                current_size_mb = memory_stats.get("current_size_mb", 0)
                if current_size_mb:
                    # Convert MB to bytes
                    current_size_bytes = current_size_mb * 1024 * 1024
                    self.state_manager_size.set(current_size_bytes, sm_attrs)
                    
                # File stats if available
                if "file_count" in state_manager_stats:
                    sm_attrs["file_count"] = state_manager_stats.get("file_count", 0)
                if "total_file_size_mb" in state_manager_stats:
                    file_size_bytes = state_manager_stats.get("total_file_size_mb", 0) * 1024 * 1024
                    self.state_manager_size.set(file_size_bytes, {**sm_attrs, "storage": "file"})
        
        except Exception as e:
            logger.error(f"Error collecting metrics for agent {agent.name}: {str(e)}")
    
    def add_agent(self, agent):
        """
        Add an agent to be monitored.
        
        Args:
            agent: The agent to monitor
        """
        if agent.agent_id in self.monitored_agents:
            logger.warning(f"Agent {agent.name} is already being monitored")
            return
        
        self.monitored_agents[agent.agent_id] = agent
        
        # Initialize last values for this agent
        self.initialize_last_values(agent.agent_id)
        
        logger.info(f"Added agent {agent.name} to performance monitoring")
    
    def remove_agent(self, agent_id: str):
        """
        Remove an agent from monitoring.
        
        Args:
            agent_id: ID of the agent to remove
        """
        if agent_id not in self.monitored_agents:
            logger.warning(f"Agent {agent_id} is not being monitored")
            return
        
        agent = self.monitored_agents.pop(agent_id)
        
        # Clean up last values
        self.cleanup_last_values(agent_id)
        
        logger.info(f"Removed agent {agent.name} from performance monitoring")
    
    # Helper methods for tracking last values to calculate delta
    def initialize_last_values(self, agent_id: str):
        """Initialize last values for a new agent."""
        if not hasattr(self, '_last_values'):
            self._last_values = {}
        
        self._last_values[agent_id] = {
            "received": 0,
            "processed": 0,
            "sent": 0,
            "failed": 0,
            "retry_total": 0,
            "retry_success": 0
        }
    
    def update_last_value(self, agent_id: str, key: str, value: Any):
        """Update a last value for an agent."""
        if not hasattr(self, '_last_values'):
            self._last_values = {}
        
        if agent_id not in self._last_values:
            self.initialize_last_values(agent_id)
        
        self._last_values[agent_id][key] = value
    
    def get_last_value(self, agent_id: str, key: str, default: Any = None):
        """Get a last value for an agent."""
        if not hasattr(self, '_last_values'):
            return default
        
        if agent_id not in self._last_values:
            return default
        
        return self._last_values[agent_id].get(key, default)
    
    def cleanup_last_values(self, agent_id: str):
        """Clean up last values when an agent is removed."""
        if hasattr(self, '_last_values') and agent_id in self._last_values:
            del self._last_values[agent_id]
    
    @contextmanager
    def measure_execution_time(self, agent_id: str, operation: str):
        """
        Context manager to measure execution time of an operation.
        
        Args:
            agent_id: ID of the agent
            operation: Name of the operation being measured
        """
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            # Record the duration
            if agent_id in self.monitored_agents:
                agent = self.monitored_agents[agent_id]
                agent_attrs = {"agent_id": agent_id, "agent_name": agent.name, "operation": operation}
                self.agent_processing_time.record(duration, agent_attrs)


# Global instance for easy access
_global_monitor: Optional[PerformanceMonitor] = None

def get_monitor(create_if_none: bool = True, 
               enable_prometheus: bool = True,
               prometheus_port: int = 8000) -> Optional[PerformanceMonitor]:
    """
    Get the global PerformanceMonitor instance.
    
    Args:
        create_if_none: Whether to create a new instance if none exists
        enable_prometheus: Whether to enable Prometheus export
        prometheus_port: Port for Prometheus HTTP server
        
    Returns:
        The global PerformanceMonitor instance
    """
    global _global_monitor
    
    if _global_monitor is None and create_if_none:
        _global_monitor = PerformanceMonitor(
            enable_prometheus=enable_prometheus,
            prometheus_port=prometheus_port
        )
    
    return _global_monitor


async def setup_performance_monitoring(enable_prometheus: bool = True,
                                     prometheus_port: int = 8000) -> PerformanceMonitor:
    """
    Set up performance monitoring and return the monitor instance.
    
    Args:
        enable_prometheus: Whether to enable Prometheus export
        prometheus_port: Port for Prometheus HTTP server
        
    Returns:
        Configured PerformanceMonitor instance
    """
    monitor = get_monitor(
        create_if_none=True,
        enable_prometheus=enable_prometheus,
        prometheus_port=prometheus_port
    )
    
    # Start collection
    await monitor.start_collecting()
    
    return monitor 