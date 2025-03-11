# AMPTALK Performance Monitoring

The AMPTALK framework includes a comprehensive real-time performance monitoring system based on OpenTelemetry and Prometheus. This document explains the monitoring system, its features, and how to use it in your applications.

## Overview

The performance monitoring system in AMPTALK provides:

- Real-time visibility into system and agent performance
- Metrics collection for CPU, memory, message processing, and agent-specific metrics
- Prometheus integration for metrics visualization
- Support for performance profiling and bottleneck identification
- Seamless integration with the agent framework and orchestrator

## Key Components

### PerformanceMonitor

The `PerformanceMonitor` class in `src/core/utils/performance_monitor.py` is the main component responsible for:

- Collecting system-level metrics (CPU, memory)
- Tracking agent-specific metrics (queue size, processing time)
- Exposing metrics via Prometheus
- Providing context managers for performance measurement

### Agent Integration

Agents can be monitored automatically by:

1. Enabling performance monitoring at the Orchestrator level
2. Explicitly registering agents with a PerformanceMonitor instance

### Orchestrator Integration

The Orchestrator can initialize and manage performance monitoring for all registered agents.

## Metrics Collected

The monitoring system collects the following metrics:

### System Metrics
- CPU usage (percentage)
- Memory usage (percentage)
- Process memory usage (bytes)
- Available system memory (bytes)

### Agent Metrics
- Message queue size
- Message processing rate
- Message processing time
- Agent running status
- Message count by type
- Error count

### State Manager Metrics
- State size
- State operations (read/write)
- State operation time

## Usage

### Basic Setup

To enable performance monitoring in your AMPTALK application:

```python
from src.core.utils.performance_monitor import setup_performance_monitoring

# Setup monitoring
monitor = await setup_performance_monitoring(
    enable_prometheus=True,
    prometheus_port=8000,
    collection_interval_seconds=1.0
)

# Create orchestrator with monitoring enabled
orchestrator = Orchestrator(
    name="MyOrchestrator",
    enable_performance_monitoring=True
)

# The orchestrator will automatically register agents with the monitor
```

### Manual Agent Registration

To manually monitor specific agents:

```python
from src.core.utils.performance_monitor import get_monitor

# Get the global monitor instance
monitor = get_monitor(create_if_none=True)

# Register an agent
monitor.add_agent(my_agent)

# Remove an agent when done
monitor.remove_agent(my_agent.agent_id)
```

### Measuring Custom Operations

To measure the performance of specific operations:

```python
from src.core.utils.performance_monitor import get_monitor

monitor = get_monitor()

# Use the context manager to measure operation time
with monitor.measure_execution_time(agent_id, "custom_operation"):
    # Perform the operation
    result = await perform_complex_operation()
```

## Visualization

### Prometheus

The metrics are exposed in Prometheus format at `http://localhost:<port>/metrics` where `<port>` is the configured port (default: 8000).

### Grafana Integration

For a comprehensive dashboard, you can configure Grafana to use Prometheus as a data source:

1. Add the Prometheus endpoint as a data source in Grafana
2. Import or create dashboards to visualize:
   - System resource utilization
   - Agent performance
   - Message processing metrics
   - Error rates

## Example Dashboard Panels

- System Resources
  - CPU Usage
  - Memory Usage
  - Available Memory

- Agent Performance
  - Message Queue Size
  - Messages Processed per Second
  - Average Processing Time

- Message Stats
  - Message Count by Type
  - Message Size Distribution
  - Error Rate

## Performance Impact

The monitoring system is designed to have minimal impact on performance:

- Metrics are collected asynchronously
- Collection interval is configurable
- Agents continue to process messages during metrics collection
- Prometheus HTTP server runs in a separate thread

## Example

See `examples/performance_monitoring_demo.py` for a complete demonstration of the performance monitoring system in action.

## Configuration Options

The `PerformanceMonitor` constructor accepts the following parameters:

- `name`: Name of the monitor instance
- `enable_prometheus`: Whether to expose metrics via Prometheus
- `prometheus_port`: Port to expose Prometheus metrics on
- `collection_interval_seconds`: How often to collect metrics
- `include_system_metrics`: Whether to collect system-level metrics
- `include_agent_metrics`: Whether to collect agent-specific metrics

## Best Practices

1. **Set appropriate collection intervals**: For development and debugging, use shorter intervals (0.5-1s) for more granular data. For production, use longer intervals (5-10s) to reduce overhead.

2. **Monitor relevant metrics**: Focus on metrics that matter for your specific use case to avoid information overload.

3. **Set up alerts**: Configure alerts in Prometheus/Grafana for critical conditions like high memory usage or increasing queue sizes.

4. **Use tags for filtering**: When visualizing metrics, use the provided tags (agent_id, agent_name, etc.) to filter and group data.

5. **Cleanup**: Always call `stop_collecting()` on the monitor when shutting down your application.

## Troubleshooting

- If metrics aren't appearing, check that the Prometheus server is running and the port is not blocked.
- If agent metrics are missing, ensure that agents are properly registered with the monitor.
- For high CPU usage, consider increasing the collection interval.

## Future Enhancements

Planned enhancements to the monitoring system include:

- Distributed tracing with OpenTelemetry
- More detailed message-level metrics
- Custom metric definitions
- Automatic anomaly detection
- Integration with more visualization tools 