"""
Resource Management Module for AMPTALK.

This module implements state-of-the-art resource management techniques for edge deployment,
including memory optimization, CPU/GPU utilization, power consumption monitoring,
thermal management, and intelligent resource scheduling.

Author: AMPTALK Team
Date: 2024
"""

import os
import time
import logging
import json
import threading
import asyncio
from enum import Enum
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass
from pathlib import Path

import psutil
import numpy as np
import torch
import torch.cuda as cuda
from opentelemetry import metrics
from opentelemetry.metrics import Meter, Counter, UpDownCounter, Histogram

from src.core.utils.logging_config import get_logger

# Configure logger
logger = get_logger("amptalk.core.resource_management")

# Create metrics
meter = metrics.get_meter("resource_management")


class ResourceType(Enum):
    """Types of resources to manage."""
    
    MEMORY = "memory"
    CPU = "cpu"
    GPU = "gpu"
    POWER = "power"
    THERMAL = "thermal"


class OptimizationStrategy(Enum):
    """Resource optimization strategies."""
    
    DYNAMIC = "dynamic"           # Dynamic resource allocation
    PREDICTIVE = "predictive"     # Predictive resource management
    ADAPTIVE = "adaptive"         # Adaptive resource scaling
    COOPERATIVE = "cooperative"   # Cooperative resource sharing


@dataclass
class ResourceLimits:
    """Resource limits configuration."""
    
    max_memory_mb: int = 0        # Maximum memory usage in MB (0 = unlimited)
    max_cpu_percent: float = 0    # Maximum CPU usage percentage (0 = unlimited)
    max_gpu_memory_mb: int = 0    # Maximum GPU memory in MB (0 = unlimited)
    max_power_watts: float = 0    # Maximum power consumption in watts (0 = unlimited)
    max_temp_celsius: float = 0   # Maximum temperature in Celsius (0 = unlimited)


@dataclass
class ResourceMetrics:
    """Resource usage metrics."""
    
    memory_used_mb: float = 0
    memory_percent: float = 0
    cpu_percent: float = 0
    gpu_memory_mb: float = 0
    gpu_utilization: float = 0
    power_watts: float = 0
    temperature_celsius: float = 0


class ResourceManager:
    """
    Resource management for edge deployment.
    
    This class provides comprehensive resource management capabilities including:
    - Memory optimization and monitoring
    - CPU/GPU utilization tracking and optimization
    - Power consumption monitoring and management
    - Thermal monitoring and management
    - Intelligent resource scheduling
    """
    
    def __init__(
        self,
        limits: Optional[ResourceLimits] = None,
        strategy: OptimizationStrategy = OptimizationStrategy.ADAPTIVE,
        monitoring_interval: float = 1.0,
        callback: Optional[Callable[[ResourceMetrics], None]] = None
    ):
        """
        Initialize the resource manager.
        
        Args:
            limits: Resource limits configuration
            strategy: Resource optimization strategy
            monitoring_interval: Interval for resource monitoring in seconds
            callback: Optional callback for resource metrics
        """
        self.limits = limits or ResourceLimits()
        self.strategy = strategy
        self.monitoring_interval = monitoring_interval
        self.callback = callback
        
        # Initialize metrics
        self._init_metrics()
        
        # Initialize monitoring
        self._stop_monitoring = threading.Event()
        self._monitoring_thread = None
        
        # Initialize resource trackers
        self.current_metrics = ResourceMetrics()
        self.historical_metrics: List[ResourceMetrics] = []
        
        # Initialize GPU if available
        self.has_gpu = torch.cuda.is_available()
        if self.has_gpu:
            self.gpu_device = torch.device("cuda")
            self.gpu_properties = torch.cuda.get_device_properties(0)
        
        logger.info(f"Initialized ResourceManager with strategy {strategy.value}")
        
        # Start monitoring if interval > 0
        if monitoring_interval > 0:
            self.start_monitoring()
    
    def _init_metrics(self) -> None:
        """Initialize OpenTelemetry metrics."""
        self.memory_usage = meter.create_up_down_counter(
            name="memory_usage_mb",
            description="Memory usage in MB",
            unit="MB"
        )
        
        self.cpu_usage = meter.create_histogram(
            name="cpu_usage_percent",
            description="CPU usage percentage",
            unit="%"
        )
        
        self.gpu_memory = meter.create_up_down_counter(
            name="gpu_memory_mb",
            description="GPU memory usage in MB",
            unit="MB"
        )
        
        self.power_usage = meter.create_histogram(
            name="power_usage_watts",
            description="Power consumption in watts",
            unit="W"
        )
        
        self.temperature = meter.create_histogram(
            name="temperature_celsius",
            description="Temperature in Celsius",
            unit="°C"
        )
    
    def start_monitoring(self) -> None:
        """Start resource monitoring thread."""
        if self._monitoring_thread is None:
            self._stop_monitoring.clear()
            self._monitoring_thread = threading.Thread(
                target=self._monitor_resources,
                daemon=True
            )
            self._monitoring_thread.start()
            logger.info("Started resource monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring thread."""
        if self._monitoring_thread is not None:
            self._stop_monitoring.set()
            self._monitoring_thread.join()
            self._monitoring_thread = None
            logger.info("Stopped resource monitoring")
    
    def _monitor_resources(self) -> None:
        """Resource monitoring loop."""
        while not self._stop_monitoring.is_set():
            try:
                # Update metrics
                self.current_metrics = self._collect_metrics()
                self.historical_metrics.append(self.current_metrics)
                
                # Check resource limits
                self._check_limits()
                
                # Apply optimization strategy
                self._optimize_resources()
                
                # Call callback if provided
                if self.callback:
                    self.callback(self.current_metrics)
                
                # Sleep for interval
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
    
    def _collect_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics."""
        metrics = ResourceMetrics()
        
        # Memory metrics
        process = psutil.Process()
        memory_info = process.memory_info()
        metrics.memory_used_mb = memory_info.rss / (1024 * 1024)
        metrics.memory_percent = process.memory_percent()
        
        # CPU metrics
        metrics.cpu_percent = psutil.cpu_percent()
        
        # GPU metrics
        if self.has_gpu:
            metrics.gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            metrics.gpu_utilization = torch.cuda.utilization()
        
        # Power metrics (if available)
        try:
            sensors = psutil.sensors_power()
            if sensors:
                metrics.power_watts = sensors.power_now
        except (AttributeError, NotImplementedError):
            pass
        
        # Temperature metrics (if available)
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # Use average of CPU temperatures
                cpu_temps = [temp.current for temp in temps.get('coretemp', [])]
                if cpu_temps:
                    metrics.temperature_celsius = sum(cpu_temps) / len(cpu_temps)
        except (AttributeError, NotImplementedError):
            pass
        
        # Update OpenTelemetry metrics
        self._update_telemetry(metrics)
        
        return metrics
    
    def _update_telemetry(self, metrics: ResourceMetrics) -> None:
        """Update OpenTelemetry metrics."""
        self.memory_usage.add(metrics.memory_used_mb)
        self.cpu_usage.record(metrics.cpu_percent)
        
        if self.has_gpu:
            self.gpu_memory.add(metrics.gpu_memory_mb)
        
        if metrics.power_watts > 0:
            self.power_usage.record(metrics.power_watts)
        
        if metrics.temperature_celsius > 0:
            self.temperature.record(metrics.temperature_celsius)
    
    def _check_limits(self) -> None:
        """Check if resource usage exceeds limits."""
        metrics = self.current_metrics
        limits = self.limits
        
        if limits.max_memory_mb > 0 and metrics.memory_used_mb > limits.max_memory_mb:
            logger.warning(f"Memory usage ({metrics.memory_used_mb:.1f} MB) exceeds limit ({limits.max_memory_mb} MB)")
            self._handle_memory_limit()
        
        if limits.max_cpu_percent > 0 and metrics.cpu_percent > limits.max_cpu_percent:
            logger.warning(f"CPU usage ({metrics.cpu_percent:.1f}%) exceeds limit ({limits.max_cpu_percent}%)")
            self._handle_cpu_limit()
        
        if (limits.max_gpu_memory_mb > 0 and self.has_gpu and 
            metrics.gpu_memory_mb > limits.max_gpu_memory_mb):
            logger.warning(f"GPU memory ({metrics.gpu_memory_mb:.1f} MB) exceeds limit ({limits.max_gpu_memory_mb} MB)")
            self._handle_gpu_limit()
        
        if limits.max_power_watts > 0 and metrics.power_watts > limits.max_power_watts:
            logger.warning(f"Power usage ({metrics.power_watts:.1f} W) exceeds limit ({limits.max_power_watts} W)")
            self._handle_power_limit()
        
        if limits.max_temp_celsius > 0 and metrics.temperature_celsius > limits.max_temp_celsius:
            logger.warning(f"Temperature ({metrics.temperature_celsius:.1f}°C) exceeds limit ({limits.max_temp_celsius}°C)")
            self._handle_thermal_limit()
    
    def _optimize_resources(self) -> None:
        """Apply resource optimization strategy."""
        if self.strategy == OptimizationStrategy.DYNAMIC:
            self._apply_dynamic_optimization()
        elif self.strategy == OptimizationStrategy.PREDICTIVE:
            self._apply_predictive_optimization()
        elif self.strategy == OptimizationStrategy.ADAPTIVE:
            self._apply_adaptive_optimization()
        elif self.strategy == OptimizationStrategy.COOPERATIVE:
            self._apply_cooperative_optimization()
    
    def _apply_dynamic_optimization(self) -> None:
        """Apply dynamic resource optimization."""
        # Implement dynamic resource allocation based on current usage
        pass
    
    def _apply_predictive_optimization(self) -> None:
        """Apply predictive resource optimization."""
        # Implement predictive resource management using historical data
        pass
    
    def _apply_adaptive_optimization(self) -> None:
        """Apply adaptive resource optimization."""
        # Implement adaptive resource scaling based on workload
        pass
    
    def _apply_cooperative_optimization(self) -> None:
        """
        Apply cooperative resource optimization.
        
        This method implements intelligent resource scheduling:
        1. Workload distribution based on resource availability
        2. Priority-based resource allocation
        3. Dynamic resource sharing
        4. Load balancing
        """
        try:
            metrics = self.current_metrics
            
            # Calculate resource utilization percentages
            cpu_util = metrics.cpu_percent / 100.0
            mem_util = metrics.memory_percent / 100.0
            gpu_util = metrics.gpu_utilization / 100.0 if self.has_gpu else 0
            
            # 1. Resource Availability Assessment
            resource_status = {
                "cpu": {
                    "utilization": cpu_util,
                    "available": 1 - cpu_util,
                    "critical": cpu_util > 0.9  # 90% threshold
                },
                "memory": {
                    "utilization": mem_util,
                    "available": 1 - mem_util,
                    "critical": mem_util > 0.9
                },
                "gpu": {
                    "utilization": gpu_util,
                    "available": 1 - gpu_util,
                    "critical": gpu_util > 0.9
                } if self.has_gpu else None
            }
            
            # 2. Priority-based Resource Allocation
            # Identify critical resources
            critical_resources = [
                resource for resource, status in resource_status.items()
                if status and status["critical"]
            ]
            
            if critical_resources:
                logger.warning(f"Critical resource utilization: {', '.join(critical_resources)}")
                
                # Apply emergency resource management
                for resource in critical_resources:
                    if resource == "cpu":
                        self._handle_cpu_limit()
                    elif resource == "memory":
                        self._handle_memory_limit()
                    elif resource == "gpu" and self.has_gpu:
                        self._handle_gpu_limit()
            
            # 3. Dynamic Resource Sharing
            # Calculate balanced resource distribution
            total_resources = sum(
                status["utilization"] 
                for status in resource_status.values() 
                if status is not None
            )
            num_resources = sum(1 for status in resource_status.values() if status is not None)
            
            if num_resources > 0:
                avg_utilization = total_resources / num_resources
                
                # Log resource distribution
                logger.info(
                    f"Resource distribution - "
                    f"CPU: {cpu_util:.2%}, "
                    f"Memory: {mem_util:.2%}"
                    + (f", GPU: {gpu_util:.2%}" if self.has_gpu else "")
                )
                
                # Identify overutilized resources
                overutilized = [
                    resource for resource, status in resource_status.items()
                    if status and status["utilization"] > avg_utilization * 1.2  # 20% above average
                ]
                
                if overutilized:
                    logger.warning(f"Overutilized resources: {', '.join(overutilized)}")
            
            # 4. Load Balancing
            # Track historical load distribution
            self.historical_metrics.append(metrics)
            if len(self.historical_metrics) > 100:  # Keep last 100 readings
                self.historical_metrics.pop(0)
            
            # Calculate load trends
            if len(self.historical_metrics) >= 5:
                recent_metrics = self.historical_metrics[-5:]
                
                # CPU trend
                cpu_trend = sum(
                    b.cpu_percent - a.cpu_percent 
                    for a, b in zip(recent_metrics[:-1], recent_metrics[1:])
                ) / 4
                
                # Memory trend
                mem_trend = sum(
                    b.memory_percent - a.memory_percent 
                    for a, b in zip(recent_metrics[:-1], recent_metrics[1:])
                ) / 4
                
                # GPU trend if available
                if self.has_gpu:
                    gpu_trend = sum(
                        b.gpu_utilization - a.gpu_utilization 
                        for a, b in zip(recent_metrics[:-1], recent_metrics[1:])
                    ) / 4
                    
                    logger.info(
                        f"Load trends - CPU: {cpu_trend:+.1f}%/sample, "
                        f"Memory: {mem_trend:+.1f}%/sample, "
                        f"GPU: {gpu_trend:+.1f}%/sample"
                    )
                else:
                    logger.info(
                        f"Load trends - CPU: {cpu_trend:+.1f}%/sample, "
                        f"Memory: {mem_trend:+.1f}%/sample"
                    )
            
        except Exception as e:
            logger.error(f"Error in cooperative optimization: {e}")
    
    def _handle_memory_limit(self) -> None:
        """Handle memory limit exceeded."""
        if self.has_gpu:
            # Clear GPU cache
            torch.cuda.empty_cache()
        
        # Trigger garbage collection
        import gc
        gc.collect()
    
    def _handle_cpu_limit(self) -> None:
        """Handle CPU limit exceeded."""
        # Implement CPU throttling or workload reduction
        pass
    
    def _handle_gpu_limit(self) -> None:
        """Handle GPU limit exceeded."""
        if self.has_gpu:
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            # Move less critical operations to CPU
            pass
    
    def _handle_power_limit(self) -> None:
        """
        Handle power limit exceeded.
        
        Implements power management strategies:
        1. Reduce CPU frequency if supported
        2. Throttle GPU operations if available
        3. Adjust workload scheduling
        4. Log power consumption patterns
        """
        try:
            # Get current metrics for decision making
            metrics = self.current_metrics
            current_power = metrics.power_watts
            target_power = self.limits.max_power_watts
            
            # Calculate required reduction
            power_reduction_needed = current_power - target_power
            
            # 1. CPU Power Management
            if hasattr(psutil, "cpu_freq"):
                try:
                    cpu_freq = psutil.cpu_freq()
                    if cpu_freq and cpu_freq.max > cpu_freq.min:
                        # Calculate new frequency target (reduce by up to 20%)
                        current_freq = cpu_freq.current
                        min_freq = cpu_freq.min
                        reduction_factor = max(0.8, 1 - (power_reduction_needed / current_power))
                        target_freq = max(min_freq, current_freq * reduction_factor)
                        
                        # Attempt to set new frequency (requires appropriate permissions)
                        if hasattr(psutil, "cpu_freq") and callable(getattr(psutil.cpu_freq(), "set")):
                            psutil.cpu_freq().set(target_freq)
                            logger.info(f"Reduced CPU frequency to {target_freq:.1f} MHz")
                except Exception as e:
                    logger.debug(f"Could not adjust CPU frequency: {e}")
            
            # 2. GPU Power Management
            if self.has_gpu:
                try:
                    # Reduce GPU power limit if supported
                    if hasattr(torch.cuda, "get_device_properties"):
                        device_props = torch.cuda.get_device_properties(0)
                        if hasattr(device_props, "max_power"):
                            current_gpu_power = device_props.max_power
                            # Reduce GPU power limit by up to 20%
                            reduction_factor = max(0.8, 1 - (power_reduction_needed / current_power))
                            new_gpu_power = current_gpu_power * reduction_factor
                            
                            # Set new power limit if supported
                            if hasattr(torch.cuda, "set_power_limit"):
                                torch.cuda.set_power_limit(0, new_gpu_power)
                                logger.info(f"Reduced GPU power limit to {new_gpu_power:.1f}W")
                    
                    # Clear GPU cache
                    torch.cuda.empty_cache()
                except Exception as e:
                    logger.debug(f"Could not adjust GPU power: {e}")
            
            # 3. Workload Scheduling
            # Implement cooperative scheduling to reduce power consumption
            if self.strategy == OptimizationStrategy.COOPERATIVE:
                self._apply_cooperative_optimization()
            
            # 4. Log power consumption patterns
            self.historical_metrics.append(self.current_metrics)
            if len(self.historical_metrics) > 100:  # Keep last 100 readings
                self.historical_metrics.pop(0)
            
            logger.warning(
                f"Applied power management strategies. Current: {current_power:.1f}W, "
                f"Target: {target_power:.1f}W"
            )
            
        except Exception as e:
            logger.error(f"Error in power limit handling: {e}")
    
    def _handle_thermal_limit(self) -> None:
        """
        Handle thermal limit exceeded.
        
        Implements thermal management strategies:
        1. Increase cooling if supported
        2. Reduce processing intensity
        3. Throttle components based on temperature
        4. Monitor thermal zones
        """
        try:
            # Get current metrics for decision making
            metrics = self.current_metrics
            current_temp = metrics.temperature_celsius
            target_temp = self.limits.max_temp_celsius
            
            # Calculate temperature excess
            temp_excess = current_temp - target_temp
            
            # 1. Cooling Management
            try:
                # Check available thermal zones
                thermal_zones = psutil.sensors_temperatures()
                if thermal_zones:
                    # Log thermal zone information
                    for zone_name, entries in thermal_zones.items():
                        for entry in entries:
                            logger.info(
                                f"Thermal zone {zone_name}: "
                                f"Current: {entry.current}°C, "
                                f"High: {entry.high if hasattr(entry, 'high') else 'N/A'}°C, "
                                f"Critical: {entry.critical if hasattr(entry, 'critical') else 'N/A'}°C"
                            )
            except Exception as e:
                logger.debug(f"Could not read thermal zones: {e}")
            
            # 2. Processing Intensity Reduction
            # Calculate reduction factor based on temperature excess
            reduction_factor = max(0.6, 1 - (temp_excess / target_temp))  # Reduce by up to 40%
            
            # CPU Throttling
            if hasattr(psutil, "cpu_freq"):
                try:
                    cpu_freq = psutil.cpu_freq()
                    if cpu_freq and cpu_freq.max > cpu_freq.min:
                        target_freq = max(
                            cpu_freq.min,
                            cpu_freq.current * reduction_factor
                        )
                        if hasattr(psutil.cpu_freq(), "set"):
                            psutil.cpu_freq().set(target_freq)
                            logger.info(f"Reduced CPU frequency to {target_freq:.1f} MHz")
                except Exception as e:
                    logger.debug(f"Could not adjust CPU frequency: {e}")
            
            # 3. GPU Thermal Management
            if self.has_gpu:
                try:
                    # Get GPU temperature if available
                    gpu_temp = torch.cuda.temperature()
                    
                    # Apply GPU-specific thermal management
                    if gpu_temp > target_temp:
                        # Clear GPU cache
                        torch.cuda.empty_cache()
                        
                        # Reduce GPU clock if supported
                        if hasattr(torch.cuda, "get_device_properties"):
                            device_props = torch.cuda.get_device_properties(0)
                            if hasattr(device_props, "max_clock_rate"):
                                current_clock = device_props.max_clock_rate
                                new_clock = int(current_clock * reduction_factor)
                                
                                # Set new clock rate if supported
                                if hasattr(torch.cuda, "set_clock_rate"):
                                    torch.cuda.set_clock_rate(0, new_clock)
                                    logger.info(f"Reduced GPU clock to {new_clock} MHz")
                except Exception as e:
                    logger.debug(f"Could not adjust GPU settings: {e}")
            
            # 4. Monitor and Log
            # Add current metrics to history
            self.historical_metrics.append(self.current_metrics)
            if len(self.historical_metrics) > 100:  # Keep last 100 readings
                self.historical_metrics.pop(0)
            
            # Calculate temperature trend
            if len(self.historical_metrics) >= 3:
                recent_temps = [m.temperature_celsius for m in self.historical_metrics[-3:]]
                temp_trend = sum(b - a for a, b in zip(recent_temps[:-1], recent_temps[1:])) / 2
                logger.info(f"Temperature trend: {temp_trend:+.1f}°C/sample")
            
            logger.warning(
                f"Applied thermal management strategies. Current: {current_temp:.1f}°C, "
                f"Target: {target_temp:.1f}°C"
            )
            
        except Exception as e:
            logger.error(f"Error in thermal limit handling: {e}")
    
    def get_current_metrics(self) -> ResourceMetrics:
        """Get current resource metrics."""
        return self.current_metrics
    
    def get_historical_metrics(self) -> List[ResourceMetrics]:
        """Get historical resource metrics."""
        return self.historical_metrics
    
    def clear_historical_metrics(self) -> None:
        """Clear historical metrics."""
        self.historical_metrics.clear()
    
    def save_metrics(self, path: str) -> None:
        """
        Save metrics to a file.
        
        Args:
            path: Path to save metrics
        """
        metrics_data = {
            "current": self.current_metrics.__dict__,
            "historical": [m.__dict__ for m in self.historical_metrics]
        }
        
        with open(path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        logger.info(f"Saved metrics to {path}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()


def create_resource_manager(
    max_memory_mb: int = 0,
    max_cpu_percent: float = 0,
    max_gpu_memory_mb: int = 0,
    max_power_watts: float = 0,
    max_temp_celsius: float = 0,
    strategy: str = "adaptive",
    monitoring_interval: float = 1.0,
    callback: Optional[Callable[[ResourceMetrics], None]] = None
) -> ResourceManager:
    """
    Create a resource manager with specified limits.
    
    Args:
        max_memory_mb: Maximum memory usage in MB (0 = unlimited)
        max_cpu_percent: Maximum CPU usage percentage (0 = unlimited)
        max_gpu_memory_mb: Maximum GPU memory in MB (0 = unlimited)
        max_power_watts: Maximum power consumption in watts (0 = unlimited)
        max_temp_celsius: Maximum temperature in Celsius (0 = unlimited)
        strategy: Resource optimization strategy
        monitoring_interval: Interval for resource monitoring in seconds
        callback: Optional callback for resource metrics
        
    Returns:
        Configured ResourceManager instance
    """
    limits = ResourceLimits(
        max_memory_mb=max_memory_mb,
        max_cpu_percent=max_cpu_percent,
        max_gpu_memory_mb=max_gpu_memory_mb,
        max_power_watts=max_power_watts,
        max_temp_celsius=max_temp_celsius
    )
    
    return ResourceManager(
        limits=limits,
        strategy=OptimizationStrategy[strategy.upper()],
        monitoring_interval=monitoring_interval,
        callback=callback
    ) 