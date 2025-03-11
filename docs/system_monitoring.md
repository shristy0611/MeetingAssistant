# System Monitoring

This document describes the System Monitoring components of the AMPTALK system, which provide comprehensive monitoring, health checking, and alerting capabilities.

## Overview

The System Monitoring components consist of three main modules:

1. **SystemMonitor**: Collects and exposes performance metrics using OpenTelemetry
2. **HealthCheck**: Monitors system health with customizable health checks
3. **AlertSystem**: Generates and routes alerts to various notification channels

Together, these components enable real-time monitoring and observability of the AMPTALK system, helping to ensure reliability and performance.

## Components

### SystemMonitor

The SystemMonitor is built on OpenTelemetry and provides performance metrics collection and exposure capabilities.

Key features:
- Performance metrics collection
- Resource utilization tracking (CPU, memory, disk)
- Request monitoring
- Error tracking
- Prometheus integration
- Configurable metrics export

Example usage:
```python
from core.system_monitoring import SystemMonitor

# Initialize the system monitor
monitor = SystemMonitor(
    service_name="amptalk",
    prometheus_port=8000,
    enable_prometheus=True
)

# Record a request
monitor.record_request(
    endpoint="/api/transcription",
    method="POST",
    status_code=200,
    duration_ms=157.3
)

# Record an error
monitor.record_error(
    error_type="database_connection",
    error_details={"database": "main", "host": "localhost"}
)

# Set health status
monitor.set_health_status(True, "All systems operational")
```

### HealthCheck

The HealthCheck system provides a flexible way to register and run health checks for various components of the system.

Key features:
- Register custom health checks
- Automatic periodic health checks
- Health check history
- Detailed health status reports
- Overall system health status

Example usage:
```python
from core.health_check import HealthCheck, create_cpu_health_check, create_memory_health_check

# Initialize the health check system
health_check = HealthCheck(
    service_name="amptalk",
    check_interval=60.0  # Run health checks every 60 seconds
)

# Register some built-in health checks
health_check.register_check(
    name="cpu",
    check_function=create_cpu_health_check(
        warning_threshold=75.0,
        critical_threshold=90.0
    ),
    description="CPU usage health check"
)

health_check.register_check(
    name="memory",
    check_function=create_memory_health_check(
        warning_threshold=75.0,
        critical_threshold=90.0
    ),
    description="Memory usage health check"
)

# Register a custom health check
def check_database_connection():
    # Custom logic to check database connection
    # Return (is_healthy, message)
    return True, "Database connection is healthy"

health_check.register_check(
    name="database",
    check_function=check_database_connection,
    description="Database connection health check"
)

# Run all health checks
results = health_check.run_all_checks()

# Get overall health status
status = health_check.get_overall_status()

# Get detailed health status report
report = health_check.get_status_report()
```

### AlertSystem

The AlertSystem provides a flexible way to generate and route alerts to various notification channels.

Key features:
- Multiple severity levels (INFO, WARNING, ERROR, CRITICAL)
- Multiple notification channels (logging, email, webhook)
- Alert history
- Filtering by severity and source
- Custom notifiers

Example usage:
```python
from core.alert_system import AlertSystem, AlertSeverity
from core.alert_system import LoggingNotifier, EmailNotifier, WebhookNotifier
from core.alert_system import create_warning_alert, create_error_alert

# Initialize the alert system
alert_system = AlertSystem(
    service_name="amptalk",
    default_notifiers=[
        LoggingNotifier(),
        EmailNotifier(
            recipients=["admin@example.com"],
            min_severity=AlertSeverity.WARNING
        ),
        WebhookNotifier(
            webhook_url="https://example.com/webhook",
            min_severity=AlertSeverity.ERROR
        )
    ]
)

# Generate alerts using convenience functions
create_warning_alert(
    alert_system=alert_system,
    title="High CPU Usage",
    message="CPU usage has exceeded 80%",
    source="system_monitor",
    details={"cpu_usage": 83.5}
)

create_error_alert(
    alert_system=alert_system,
    title="Database Connection Failed",
    message="Failed to connect to the database",
    source="database",
    details={"database": "main", "host": "localhost"}
)

# Or directly using the alert method
alert_system.alert(
    title="API Rate Limit Exceeded",
    message="Too many requests to the API",
    severity=AlertSeverity.WARNING,
    source="api_gateway",
    details={"rate": 120, "limit": 100}
)

# Get alert history
history = alert_system.get_history(
    limit=10,
    min_severity=AlertSeverity.WARNING,
    source="database"
)
```

## Integration

These components are designed to work together to provide comprehensive monitoring and observability:

1. **SystemMonitor** collects performance metrics and exposes them for visualization
2. **HealthCheck** uses the collected metrics to determine system health
3. **AlertSystem** generates alerts based on health check results and other events

The system is designed to be highly configurable and extensible to meet specific needs.

## Deployment

### Metrics Visualization

To visualize metrics collected by SystemMonitor, you can use:

1. **Prometheus**: For metrics collection and storage
2. **Grafana**: For metrics visualization and dashboards

Example Prometheus configuration:
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'amptalk'
    static_configs:
      - targets: ['localhost:8000']
```

### Health Status API

You can expose the health status through an API endpoint:

```python
@app.get("/health")
def health_status():
    return health_check.get_status_report()
```

This provides a standardized way for external systems to check the health of the AMPTALK system.

### Alert Notifications

The AlertSystem can be configured to send notifications through various channels:

1. **Logging**: For local development and debugging
2. **Email**: For important alerts to administrators
3. **Webhook**: For integration with external systems like Slack, PagerDuty, etc.

## Requirements

- Python 3.8+
- OpenTelemetry libraries
- Prometheus client (for Prometheus integration)
- psutil (for system metrics)

## Best Practices

1. **Metrics Collection**
   - Collect only metrics that provide value
   - Use meaningful metric names and labels
   - Prefer gauges for values that can go up and down
   - Use counters for events and accumulating values

2. **Health Checks**
   - Keep health checks lightweight
   - Make health checks specific to single components
   - Include useful messages in health check results
   - Set appropriate warning and critical thresholds

3. **Alerts**
   - Use appropriate severity levels
   - Include enough context in alerts
   - Set up appropriate notification channels
   - Avoid alert storms with proper thresholds 