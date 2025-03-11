# Resource Management

## Overview

The AMPTALK Resource Management module provides comprehensive resource monitoring and optimization capabilities for edge deployments. This module is designed to efficiently manage system resources including memory, CPU/GPU utilization, power consumption, and thermal characteristics.

## Key Features

- **Resource Monitoring**:
  - Real-time memory usage tracking
  - CPU/GPU utilization monitoring
  - Power consumption measurement
  - Thermal monitoring
  - Historical metrics collection

- **Resource Optimization**:
  - Dynamic resource allocation
  - Predictive resource management
  - Adaptive resource scaling
  - Cooperative resource sharing

- **Resource Limits**:
  - Configurable memory limits
  - CPU/GPU usage thresholds
  - Power consumption limits
  - Temperature thresholds
  - Automatic limit handling

## Installation

The resource management module is part of the AMPTALK package. To use it, ensure you have the required dependencies:

```bash
pip install torch psutil opentelemetry-api opentelemetry-sdk
```

## Usage

### Basic Example

```python
from src.core.resource_management import create_resource_manager

# Create resource manager with limits
manager = create_resource_manager(
    max_memory_mb=1000,
    max_cpu_percent=80,
    max_gpu_memory_mb=1000,
    max_power_watts=100,
    max_temp_celsius=80,
    strategy="adaptive"
)

# Start monitoring
manager.start_monitoring()

# ... your application code ...

# Stop monitoring
manager.stop_monitoring()

# Save metrics
manager.save_metrics("resource_metrics.json")
```

### Using Context Manager

```python
from src.core.resource_management import ResourceManager, ResourceLimits

limits = ResourceLimits(
    max_memory_mb=1000,
    max_cpu_percent=80
)

with ResourceManager(limits=limits) as manager:
    # Resources are monitored within this block
    metrics = manager.get_current_metrics()
    print(f"Memory usage: {metrics.memory_used_mb:.1f} MB")
```

### Custom Metrics Callback

```python
def metrics_callback(metrics):
    print(f"Memory: {metrics.memory_used_mb:.1f} MB")
    print(f"CPU: {metrics.cpu_percent:.1f}%")
    print(f"GPU Memory: {metrics.gpu_memory_mb:.1f} MB")

manager = create_resource_manager(
    callback=metrics_callback,
    monitoring_interval=1.0
)
```

### Using the Demo Script

```bash
# Basic usage
python examples/resource_management_demo.py

# With workload simulation
python examples/resource_management_demo.py --simulate-load

# Custom configuration
python examples/resource_management_demo.py \
    --strategy adaptive \
    --max-memory-mb 2000 \
    --max-cpu-percent 90 \
    --duration 300
```

## Resource Optimization Strategies

### 1. Dynamic Strategy

The dynamic strategy adjusts resource allocation in real-time based on current usage:

```python
manager = create_resource_manager(
    strategy="dynamic",
    max_memory_mb=1000
)
```

### 2. Predictive Strategy

The predictive strategy uses historical data to anticipate resource needs:

```python
manager = create_resource_manager(
    strategy="predictive",
    monitoring_interval=1.0
)
```

### 3. Adaptive Strategy

The adaptive strategy scales resources based on workload patterns:

```python
manager = create_resource_manager(
    strategy="adaptive",
    max_cpu_percent=80
)
```

### 4. Cooperative Strategy

The cooperative strategy enables resource sharing between components:

```python
manager = create_resource_manager(
    strategy="cooperative",
    max_gpu_memory_mb=1000
)
```

## Resource Metrics

The module collects comprehensive resource metrics:

```python
metrics = manager.get_current_metrics()

# Memory metrics
print(f"Memory used: {metrics.memory_used_mb:.1f} MB")
print(f"Memory percentage: {metrics.memory_percent:.1f}%")

# CPU metrics
print(f"CPU usage: {metrics.cpu_percent:.1f}%")

# GPU metrics (if available)
print(f"GPU memory: {metrics.gpu_memory_mb:.1f} MB")
print(f"GPU utilization: {metrics.gpu_utilization:.1f}%")

# Power and thermal metrics
print(f"Power usage: {metrics.power_watts:.1f}W")
print(f"Temperature: {metrics.temperature_celsius:.1f}°C")
```

## Integration with OpenTelemetry

The module integrates with OpenTelemetry for metrics collection:

```python
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter

# Configure metrics
metrics.set_meter_provider(MeterProvider())
meter = metrics.get_meter("resource_management")

# Create resource manager
manager = create_resource_manager()
```

## Best Practices

1. **Resource Monitoring**:
   - Set appropriate monitoring intervals
   - Use callbacks for real-time monitoring
   - Save metrics for analysis
   - Monitor trends over time

2. **Resource Limits**:
   - Set conservative resource limits
   - Monitor limit violations
   - Implement graceful degradation
   - Adjust limits based on device capabilities

3. **Optimization Strategies**:
   - Choose strategies based on workload
   - Combine multiple strategies when needed
   - Monitor strategy effectiveness
   - Adjust parameters based on results

4. **Integration Tips**:
   - Start monitoring early
   - Use context managers for safety
   - Implement proper cleanup
   - Save metrics regularly

## Limitations

- Some metrics may not be available on all platforms
- Power and thermal metrics require hardware support
- GPU metrics require CUDA support
- Resource limits may not be enforceable on all systems

## References

1. "Efficient Resource Management for Edge Computing" (2024)
2. "Adaptive Resource Allocation in Edge Environments" (2023)
3. "Power-Aware Computing for Edge Devices" (2024)
4. "Thermal Management in Edge Computing" (2023)
5. "OpenTelemetry for Resource Monitoring" (2024) 