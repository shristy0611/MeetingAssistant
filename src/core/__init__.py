"""
Core functionality for the AMPTALK multi-agent system.

This package contains the core framework and utilities for the multi-agent system.
"""

from src.core.agent import Agent, AgentStatus, Message, MessagePriority
from src.core.orchestrator import Orchestrator

# Data Flow Management Components
try:
    from src.core.buffer_manager import BufferManager
    from src.core.data_persistence import DataPersistence
    from src.core.stream_processor import StreamProcessor
except ImportError:
    # Some components may have dependencies that aren't installed
    pass

# System Monitoring Components
try:
    from src.core.system_monitoring import SystemMonitor
    from src.core.health_check import HealthCheck, HealthStatus
    from src.core.alert_system import AlertSystem, Alert, AlertSeverity
except ImportError:
    # Some components may have dependencies that aren't installed
    pass

__all__ = [
    "Agent", 
    "AgentStatus", 
    "Message", 
    "MessagePriority", 
    "Orchestrator",
    # Data Flow Management
    "BufferManager",
    "DataPersistence",
    "StreamProcessor",
    # System Monitoring
    "SystemMonitor",
    "HealthCheck",
    "HealthStatus",
    "AlertSystem",
    "Alert",
    "AlertSeverity",
] 