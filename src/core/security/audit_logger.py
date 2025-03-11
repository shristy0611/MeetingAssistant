"""
Audit Logging System.

This module provides comprehensive audit logging capabilities for the AMPTALK system,
ensuring security event tracking and compliance.

Author: AMPTALK Team
Date: 2024
"""

import os
import json
import hmac
import hashlib
import logging
import sqlite3
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass

from src.core.utils.logging_config import get_logger

logger = get_logger(__name__)

class AuditEventType(Enum):
    """Types of audit events."""
    AUTH = "authentication"
    ACCESS = "access_control"
    DATA = "data_operation"
    CONFIG = "configuration"
    SECURITY = "security"
    SYSTEM = "system"
    USER = "user_management"

class AuditEventSeverity(Enum):
    """Severity levels for audit events."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class AuditEvent:
    """Audit event information."""
    id: str
    timestamp: datetime
    event_type: AuditEventType
    severity: AuditEventSeverity
    user_id: Optional[str]
    resource: str
    action: str
    status: str
    details: Dict[str, Any]
    metadata: Dict[str, Any]
    ip_address: Optional[str]
    user_agent: Optional[str]
    checksum: str

class AuditLogger:
    """
    Manages audit logging for the system.
    
    Features:
    - Comprehensive event logging
    - Tamper-evident logs
    - Log rotation and archival
    - Log analysis tools
    - Compliance reporting
    """
    
    def __init__(
        self,
        logs_dir: str = "security/audit_logs",
        db_path: Optional[str] = None,
        hmac_key: Optional[str] = None,
        retention_days: int = 365,
        max_log_size_mb: int = 100
    ):
        """
        Initialize the audit logger.
        
        Args:
            logs_dir: Directory for log storage
            db_path: Path to SQLite database
            hmac_key: Key for log integrity
            retention_days: Days to retain logs
            max_log_size_mb: Maximum log file size in MB
        """
        self.logs_dir = logs_dir
        self.db_path = db_path or os.path.join(logs_dir, "audit.db")
        self.hmac_key = hmac_key or os.urandom(32).hex()
        self.retention_days = retention_days
        self.max_log_size = max_log_size_mb * 1024 * 1024
        
        # Initialize storage
        self._setup_storage()
    
    def _setup_storage(self) -> None:
        """Set up log storage."""
        # Create directories
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(os.path.join(self.logs_dir, "archive"), exist_ok=True)
        
        # Initialize database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    user_id TEXT,
                    resource TEXT NOT NULL,
                    action TEXT NOT NULL,
                    status TEXT NOT NULL,
                    details TEXT NOT NULL,
                    metadata TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    checksum TEXT NOT NULL
                )
            """)
            
            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON audit_events(timestamp)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_event_type
                ON audit_events(event_type)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_id
                ON audit_events(user_id)
            """)
            
            conn.commit()
        
        logger.info(f"Initialized audit logging in {self.logs_dir}")
    
    def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """Calculate HMAC for data integrity."""
        message = json.dumps(data, sort_keys=True).encode()
        return hmac.new(
            self.hmac_key.encode(),
            message,
            hashlib.sha256
        ).hexdigest()
    
    def log_event(
        self,
        event_type: AuditEventType,
        severity: AuditEventSeverity,
        resource: str,
        action: str,
        status: str,
        details: Dict[str, Any],
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> str:
        """
        Log an audit event.
        
        Args:
            event_type: Type of event
            severity: Event severity
            resource: Resource affected
            action: Action performed
            status: Operation status
            details: Event details
            user_id: Optional user ID
            metadata: Optional metadata
            ip_address: Optional IP address
            user_agent: Optional user agent
        
        Returns:
            Event ID
        """
        try:
            # Generate event ID
            event_id = f"event_{datetime.utcnow().timestamp()}"
            
            # Create event data
            event_data = {
                'id': event_id,
                'timestamp': datetime.utcnow().isoformat(),
                'event_type': event_type.value,
                'severity': severity.value,
                'user_id': user_id,
                'resource': resource,
                'action': action,
                'status': status,
                'details': details,
                'metadata': metadata or {},
                'ip_address': ip_address,
                'user_agent': user_agent
            }
            
            # Calculate checksum
            checksum = self._calculate_checksum(event_data)
            event_data['checksum'] = checksum
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO audit_events
                    (id, timestamp, event_type, severity, user_id,
                     resource, action, status, details, metadata,
                     ip_address, user_agent, checksum)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event_id,
                    event_data['timestamp'],
                    event_data['event_type'],
                    event_data['severity'],
                    event_data['user_id'],
                    event_data['resource'],
                    event_data['action'],
                    event_data['status'],
                    json.dumps(event_data['details']),
                    json.dumps(event_data['metadata']),
                    event_data['ip_address'],
                    event_data['user_agent'],
                    checksum
                ))
                conn.commit()
            
            # Write to file
            self._write_to_file(event_data)
            
            return event_id
            
        except Exception as e:
            logger.error(f"Error logging audit event: {e}")
            raise
    
    def _write_to_file(self, event_data: Dict[str, Any]) -> None:
        """Write event to log file."""
        try:
            # Get current log file
            current_file = os.path.join(
                self.logs_dir,
                f"audit_{datetime.utcnow():%Y%m%d}.log"
            )
            
            # Check if rotation needed
            if (
                os.path.exists(current_file) and
                os.path.getsize(current_file) >= self.max_log_size
            ):
                self._rotate_logs()
            
            # Write event
            with open(current_file, 'a') as f:
                json.dump(event_data, f)
                f.write('\n')
            
        except Exception as e:
            logger.error(f"Error writing to log file: {e}")
            raise
    
    def _rotate_logs(self) -> None:
        """Rotate log files."""
        try:
            # Get current log file
            current_date = datetime.utcnow()
            current_file = os.path.join(
                self.logs_dir,
                f"audit_{current_date:%Y%m%d}.log"
            )
            
            if os.path.exists(current_file):
                # Create archive name
                archive_name = os.path.join(
                    self.logs_dir,
                    "archive",
                    f"audit_{current_date:%Y%m%d_%H%M%S}.log.gz"
                )
                
                # Compress and move
                import gzip
                with open(current_file, 'rb') as f_in:
                    with gzip.open(archive_name, 'wb') as f_out:
                        f_out.write(f_in.read())
                
                # Remove original
                os.remove(current_file)
            
            # Clean old archives
            self._cleanup_old_logs()
            
        except Exception as e:
            logger.error(f"Error rotating logs: {e}")
            raise
    
    def _cleanup_old_logs(self) -> None:
        """Clean up old log files."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
            
            # Clean database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM audit_events
                    WHERE timestamp < ?
                """, (cutoff_date.isoformat(),))
                conn.commit()
            
            # Clean archive files
            archive_dir = os.path.join(self.logs_dir, "archive")
            for filename in os.listdir(archive_dir):
                if filename.endswith('.log.gz'):
                    file_date = datetime.strptime(
                        filename[6:14],
                        '%Y%m%d'
                    )
                    if file_date < cutoff_date:
                        os.remove(os.path.join(archive_dir, filename))
            
        except Exception as e:
            logger.error(f"Error cleaning old logs: {e}")
            raise
    
    def get_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_type: Optional[AuditEventType] = None,
        severity: Optional[AuditEventSeverity] = None,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        limit: int = 100
    ) -> List[AuditEvent]:
        """
        Get audit events.
        
        Args:
            start_date: Start date filter
            end_date: End date filter
            event_type: Event type filter
            severity: Severity filter
            user_id: User ID filter
            resource: Resource filter
            limit: Maximum events to return
        
        Returns:
            List of audit events
        """
        try:
            query = "SELECT * FROM audit_events WHERE 1=1"
            params = []
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.isoformat())
            
            if event_type:
                query += " AND event_type = ?"
                params.append(event_type.value)
            
            if severity:
                query += " AND severity = ?"
                params.append(severity.value)
            
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            
            if resource:
                query += " AND resource = ?"
                params.append(resource)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            events = []
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, params)
                
                for row in cursor:
                    event = AuditEvent(
                        id=row['id'],
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        event_type=AuditEventType(row['event_type']),
                        severity=AuditEventSeverity(row['severity']),
                        user_id=row['user_id'],
                        resource=row['resource'],
                        action=row['action'],
                        status=row['status'],
                        details=json.loads(row['details']),
                        metadata=json.loads(row['metadata']),
                        ip_address=row['ip_address'],
                        user_agent=row['user_agent'],
                        checksum=row['checksum']
                    )
                    
                    # Verify integrity
                    calculated_checksum = self._calculate_checksum({
                        'id': event.id,
                        'timestamp': event.timestamp.isoformat(),
                        'event_type': event.event_type.value,
                        'severity': event.severity.value,
                        'user_id': event.user_id,
                        'resource': event.resource,
                        'action': event.action,
                        'status': event.status,
                        'details': event.details,
                        'metadata': event.metadata,
                        'ip_address': event.ip_address,
                        'user_agent': event.user_agent
                    })
                    
                    if calculated_checksum == event.checksum:
                        events.append(event)
                    else:
                        logger.warning(
                            f"Integrity check failed for event {event.id}"
                        )
            
            return events
            
        except Exception as e:
            logger.error(f"Error getting audit events: {e}")
            return []
    
    def generate_report(
        self,
        start_date: datetime,
        end_date: datetime,
        report_type: str = "summary"
    ) -> Dict[str, Any]:
        """
        Generate audit report.
        
        Args:
            start_date: Report start date
            end_date: Report end date
            report_type: Type of report
        
        Returns:
            Report data
        """
        try:
            events = self.get_events(
                start_date=start_date,
                end_date=end_date,
                limit=10000  # High limit for reports
            )
            
            if report_type == "summary":
                return self._generate_summary_report(events)
            elif report_type == "detailed":
                return self._generate_detailed_report(events)
            else:
                raise ValueError(f"Unknown report type: {report_type}")
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return {}
    
    def _generate_summary_report(
        self,
        events: List[AuditEvent]
    ) -> Dict[str, Any]:
        """Generate summary report."""
        try:
            summary = {
                'total_events': len(events),
                'event_types': {},
                'severities': {},
                'users': set(),
                'resources': set(),
                'status_counts': {},
                'hourly_distribution': [0] * 24
            }
            
            for event in events:
                # Count event types
                event_type = event.event_type.value
                summary['event_types'][event_type] = (
                    summary['event_types'].get(event_type, 0) + 1
                )
                
                # Count severities
                severity = event.severity.value
                summary['severities'][severity] = (
                    summary['severities'].get(severity, 0) + 1
                )
                
                # Track users and resources
                if event.user_id:
                    summary['users'].add(event.user_id)
                summary['resources'].add(event.resource)
                
                # Count status
                summary['status_counts'][event.status] = (
                    summary['status_counts'].get(event.status, 0) + 1
                )
                
                # Track hourly distribution
                hour = event.timestamp.hour
                summary['hourly_distribution'][hour] += 1
            
            # Convert sets to counts
            summary['unique_users'] = len(summary['users'])
            summary['unique_resources'] = len(summary['resources'])
            del summary['users']
            del summary['resources']
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary report: {e}")
            return {}
    
    def _generate_detailed_report(
        self,
        events: List[AuditEvent]
    ) -> Dict[str, Any]:
        """Generate detailed report."""
        try:
            report = {
                'metadata': {
                    'generated_at': datetime.utcnow().isoformat(),
                    'total_events': len(events)
                },
                'events': []
            }
            
            for event in events:
                report['events'].append({
                    'id': event.id,
                    'timestamp': event.timestamp.isoformat(),
                    'event_type': event.event_type.value,
                    'severity': event.severity.value,
                    'user_id': event.user_id,
                    'resource': event.resource,
                    'action': event.action,
                    'status': event.status,
                    'details': event.details,
                    'metadata': event.metadata,
                    'ip_address': event.ip_address,
                    'user_agent': event.user_agent
                })
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating detailed report: {e}")
            return {} 