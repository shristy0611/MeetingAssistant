"""
System Monitoring Demo for AMPTALK

This script demonstrates the use of the system monitoring components,
including metrics collection, health checks, and alerting.
"""

import time
import logging
import json
import random
import os
from typing import Dict, Any
from pathlib import Path

from core.system_monitoring import SystemMonitor
from core.health_check import HealthCheck, create_cpu_health_check, create_memory_health_check, create_disk_health_check
from core.alert_system import AlertSystem, LoggingNotifier, EmailNotifier, WebhookNotifier
from core.alert_system import create_info_alert, create_warning_alert, create_error_alert, create_critical_alert

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MonitoringDemo")

class MonitoringDemo:
    """
    Demonstrates the integrated use of system monitoring components.
    """
    
    def __init__(self):
        """Initialize the monitoring demo."""
        logger.info("Initializing Monitoring Demo")
        
        # Create directory for demo data
        os.makedirs("./data/monitoring", exist_ok=True)
        
        # Initialize system monitor
        self.system_monitor = SystemMonitor(
            service_name="amptalk-demo",
            prometheus_port=8000,
            enable_prometheus=True,
            enable_console=True
        )
        
        # Initialize health check system
        self.health_check = HealthCheck(
            service_name="amptalk-demo",
            check_interval=10.0,  # 10 seconds
            auto_check=True
        )
        
        # Initialize alert system
        self.alert_system = AlertSystem(
            service_name="amptalk-demo",
            default_notifiers=[
                LoggingNotifier(),
                # In a real application, you would configure these with real endpoints
                # EmailNotifier(recipients=["admin@example.com"]),
                # WebhookNotifier(webhook_url="https://example.com/webhook")
            ]
        )
        
        # Register health checks
        self._register_health_checks()
        
        logger.info("Monitoring Demo initialized")
    
    def _register_health_checks(self):
        """Register health checks with the health check system."""
        # Register system health checks
        self.health_check.register_check(
            name="cpu",
            check_function=create_cpu_health_check(warning_threshold=75.0, critical_threshold=90.0),
            description="CPU usage health check"
        )
        
        self.health_check.register_check(
            name="memory",
            check_function=create_memory_health_check(warning_threshold=75.0, critical_threshold=90.0),
            description="Memory usage health check"
        )
        
        self.health_check.register_check(
            name="disk",
            check_function=create_disk_health_check(warning_threshold=75.0, critical_threshold=90.0),
            description="Disk usage health check"
        )
        
        # Register custom application health check
        def check_database():
            # Simulate a database health check
            is_healthy = random.random() > 0.1  # 90% chance of being healthy
            message = "Database connection is healthy" if is_healthy else "Database connection failed"
            return is_healthy, message
        
        self.health_check.register_check(
            name="database",
            check_function=check_database,
            description="Database connection health check"
        )
        
        logger.info("Registered health checks")
    
    def simulate_requests(self, num_requests: int = 10):
        """
        Simulate HTTP requests to demonstrate request metrics.
        
        Args:
            num_requests: Number of requests to simulate
        """
        logger.info(f"Simulating {num_requests} HTTP requests")
        
        endpoints = ["/api/users", "/api/items", "/api/orders", "/api/auth"]
        methods = ["GET", "POST", "PUT", "DELETE"]
        status_codes = [200, 200, 200, 200, 201, 400, 404, 500]  # Weighted towards success
        
        for i in range(num_requests):
            # Simulate a request
            endpoint = random.choice(endpoints)
            method = random.choice(methods)
            status_code = random.choice(status_codes)
            duration_ms = random.uniform(10.0, 500.0)  # Between 10ms and 500ms
            
            # Record the request in the system monitor
            self.system_monitor.record_request(
                endpoint=endpoint,
                method=method,
                status_code=status_code,
                duration_ms=duration_ms
            )
            
            # If this was an error response, also record an error
            if status_code >= 400:
                error_type = "client_error" if status_code < 500 else "server_error"
                self.system_monitor.record_error(
                    error_type=error_type,
                    error_details={
                        "endpoint": endpoint,
                        "method": method,
                        "status_code": status_code
                    }
                )
                
                # Also generate an alert for server errors
                if status_code >= 500:
                    create_error_alert(
                        alert_system=self.alert_system,
                        title=f"Server Error on {endpoint}",
                        message=f"Server error occurred for {method} {endpoint} with status code {status_code}",
                        source="web_server",
                        details={
                            "endpoint": endpoint,
                            "method": method,
                            "status_code": status_code,
                            "duration_ms": duration_ms
                        }
                    )
            
            # Small delay to simulate real-time
            time.sleep(0.1)
        
        logger.info("Finished simulating HTTP requests")
    
    def simulate_system_events(self, num_events: int = 5):
        """
        Simulate system events to demonstrate alerts.
        
        Args:
            num_events: Number of events to simulate
        """
        logger.info(f"Simulating {num_events} system events")
        
        event_types = [
            ("info", "System started", "System successfully started"),
            ("warning", "High CPU usage", "CPU usage exceeded 80%"),
            ("error", "Database connection failed", "Failed to connect to the database"),
            ("critical", "Disk full", "Disk space is critically low"),
            ("info", "Cache cleared", "Cache was successfully cleared"),
            ("warning", "Memory usage high", "Memory usage exceeded 80%"),
            ("error", "API rate limit exceeded", "Too many requests to external API"),
            ("critical", "Service unavailable", "Critical service is down")
        ]
        
        for i in range(num_events):
            # Simulate an event
            severity, title_template, message_template = random.choice(event_types)
            
            # Add some randomness to the message
            title = f"{title_template} #{i+1}"
            message = f"{message_template} at {time.strftime('%H:%M:%S')}"
            
            # Generate an alert
            if severity == "info":
                create_info_alert(self.alert_system, title, message, "system")
            elif severity == "warning":
                create_warning_alert(self.alert_system, title, message, "system")
            elif severity == "error":
                create_error_alert(self.alert_system, title, message, "system")
            elif severity == "critical":
                create_critical_alert(self.alert_system, title, message, "system")
            
            # Small delay
            time.sleep(0.5)
        
        logger.info("Finished simulating system events")
    
    def show_health_status(self):
        """Show the current health status."""
        # Run all health checks
        self.health_check.run_all_checks()
        
        # Get status report
        status_report = self.health_check.get_status_report()
        
        logger.info(f"Health Status: {status_report['status']}")
        for name, check in status_report["checks"].items():
            logger.info(f"  {name}: {check['status']} - {check['message']}")
    
    def show_alert_history(self):
        """Show the alert history."""
        # Get alert history
        alerts = self.alert_system.get_history(limit=10)
        
        logger.info(f"Alert History (last {len(alerts)} alerts):")
        for alert in alerts:
            logger.info(f"  [{alert['severity'].upper()}] {alert['title']} - {alert['message']}")
    
    def run_demo(self):
        """Run the monitoring demo."""
        logger.info("Starting Monitoring Demo")
        
        # System info
        system_info = self.system_monitor.get_system_info()
        logger.info(f"System Info: {json.dumps(system_info, indent=2)}")
        
        # Simulate some requests
        self.simulate_requests(20)
        
        # Show health status
        self.show_health_status()
        
        # Simulate some system events
        self.simulate_system_events(10)
        
        # Show alert history
        self.show_alert_history()
        
        # Keep running for a while to collect metrics
        logger.info("Demo is running. Press Ctrl+C to stop...")
        try:
            while True:
                # Simulate more requests every 5 seconds
                time.sleep(5)
                self.simulate_requests(5)
                
                # Show health status every 30 seconds
                if int(time.time()) % 30 == 0:
                    self.show_health_status()
                
                # Simulate system events occasionally
                if random.random() < 0.2:  # 20% chance
                    self.simulate_system_events(2)
        except KeyboardInterrupt:
            logger.info("Demo stopped by user")

def main():
    """Run the monitoring demo."""
    demo = MonitoringDemo()
    demo.run_demo()

if __name__ == "__main__":
    main() 