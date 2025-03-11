"""
Data Subject Rights Management System.

This module provides comprehensive data subject rights management capabilities for the AMPTALK system,
ensuring compliance with privacy regulations.

Author: AMPTALK Team
Date: 2024
"""

import os
import json
import logging
import sqlite3
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from uuid import uuid4

from src.core.utils.logging_config import get_logger

logger = get_logger(__name__)

class RightType(Enum):
    """Types of data subject rights."""
    ACCESS = "access"
    RECTIFICATION = "rectification"
    ERASURE = "erasure"
    PORTABILITY = "portability"
    RESTRICTION = "restriction"
    OBJECTION = "objection"
    AUTOMATED_DECISION = "automated_decision"

class RequestStatus(Enum):
    """Status of rights request."""
    SUBMITTED = "submitted"
    VALIDATING = "validating"
    PROCESSING = "processing"
    COMPLETED = "completed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"

class RequestPriority(Enum):
    """Priority levels for requests."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

@dataclass
class RightsRequest:
    """Data subject rights request."""
    id: str
    user_id: str
    right_type: RightType
    status: RequestStatus
    priority: RequestPriority
    submitted_at: datetime
    deadline: datetime
    description: str
    proof_of_identity: Optional[str]
    data_scope: List[str]
    response: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]

class RightsManager:
    """
    Manages data subject rights requests.
    
    Features:
    - Rights request management
    - Request workflow automation
    - Response generation
    - Compliance tracking
    - Documentation and reporting
    """
    
    def __init__(
        self,
        storage_dir: str = "privacy/rights",
        default_deadline_days: int = 30,
        require_identity_proof: bool = True
    ):
        """
        Initialize the rights manager.
        
        Args:
            storage_dir: Directory for request storage
            default_deadline_days: Default deadline for requests
            require_identity_proof: Whether to require proof of identity
        """
        self.storage_dir = storage_dir
        self.default_deadline_days = default_deadline_days
        self.require_identity_proof = require_identity_proof
        
        # Initialize storage
        self._setup_storage()
    
    def _setup_storage(self) -> None:
        """Set up request storage."""
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Initialize database
        db_path = os.path.join(self.storage_dir, "rights.db")
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rights_requests (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    right_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    submitted_at TIMESTAMP NOT NULL,
                    deadline TIMESTAMP NOT NULL,
                    description TEXT NOT NULL,
                    proof_of_identity TEXT,
                    data_scope TEXT NOT NULL,
                    response TEXT,
                    metadata TEXT
                )
            """)
            
            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_id
                ON rights_requests(user_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_status
                ON rights_requests(status)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_deadline
                ON rights_requests(deadline)
            """)
            
            conn.commit()
        
        logger.info(f"Initialized rights storage in {self.storage_dir}")
    
    def submit_request(
        self,
        user_id: str,
        right_type: RightType,
        description: str,
        data_scope: List[str],
        proof_of_identity: Optional[str] = None,
        priority: RequestPriority = RequestPriority.MEDIUM,
        deadline_days: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Submit a rights request.
        
        Args:
            user_id: User identifier
            right_type: Type of right being requested
            description: Request description
            data_scope: Scope of data involved
            proof_of_identity: Optional identity proof
            priority: Request priority
            deadline_days: Days until deadline
            metadata: Additional metadata
        
        Returns:
            Request ID
        """
        try:
            # Validate identity proof
            if (
                self.require_identity_proof and
                not proof_of_identity and
                right_type != RightType.ACCESS
            ):
                raise ValueError("Proof of identity required")
            
            # Generate request ID
            request_id = str(uuid4())
            
            # Calculate timestamps
            submitted_at = datetime.utcnow()
            deadline = submitted_at + timedelta(
                days=deadline_days or self.default_deadline_days
            )
            
            # Create request
            request = RightsRequest(
                id=request_id,
                user_id=user_id,
                right_type=right_type,
                status=RequestStatus.SUBMITTED,
                priority=priority,
                submitted_at=submitted_at,
                deadline=deadline,
                description=description,
                proof_of_identity=proof_of_identity,
                data_scope=data_scope,
                response=None,
                metadata=metadata or {}
            )
            
            # Store request
            self._save_request(request)
            
            logger.info(
                f"Submitted {right_type.value} request for {user_id}: "
                f"{request_id}"
            )
            return request_id
            
        except Exception as e:
            logger.error(f"Error submitting request: {e}")
            raise
    
    def _save_request(self, request: RightsRequest) -> None:
        """Save request to storage."""
        try:
            db_path = os.path.join(self.storage_dir, "rights.db")
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO rights_requests
                    (id, user_id, right_type, status, priority,
                     submitted_at, deadline, description,
                     proof_of_identity, data_scope, response, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    request.id,
                    request.user_id,
                    request.right_type.value,
                    request.status.value,
                    request.priority.value,
                    request.submitted_at.isoformat(),
                    request.deadline.isoformat(),
                    request.description,
                    request.proof_of_identity,
                    json.dumps(request.data_scope),
                    json.dumps(request.response) if request.response else None,
                    json.dumps(request.metadata)
                ))
                conn.commit()
            
        except Exception as e:
            logger.error(f"Error saving request: {e}")
            raise
    
    def get_request(self, request_id: str) -> Optional[RightsRequest]:
        """
        Get a specific request.
        
        Args:
            request_id: Request ID
        
        Returns:
            Request if found
        """
        try:
            db_path = os.path.join(self.storage_dir, "rights.db")
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM rights_requests
                    WHERE id = ?
                """, (request_id,))
                
                result = cursor.fetchone()
                if not result:
                    return None
                
                (
                    id, user_id, right_type_str, status_str,
                    priority_str, submitted_at_str, deadline_str,
                    description, proof_of_identity, data_scope_str,
                    response_str, metadata_str
                ) = result
                
                return RightsRequest(
                    id=id,
                    user_id=user_id,
                    right_type=RightType(right_type_str),
                    status=RequestStatus(status_str),
                    priority=RequestPriority(priority_str),
                    submitted_at=datetime.fromisoformat(submitted_at_str),
                    deadline=datetime.fromisoformat(deadline_str),
                    description=description,
                    proof_of_identity=proof_of_identity,
                    data_scope=json.loads(data_scope_str),
                    response=(
                        json.loads(response_str)
                        if response_str else None
                    ),
                    metadata=json.loads(metadata_str)
                )
                
        except Exception as e:
            logger.error(f"Error getting request: {e}")
            return None
    
    def update_request_status(
        self,
        request_id: str,
        status: RequestStatus,
        response: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update request status.
        
        Args:
            request_id: Request ID
            status: New status
            response: Optional response data
        
        Returns:
            True if successful
        """
        try:
            request = self.get_request(request_id)
            if not request:
                return False
            
            # Update request
            request.status = status
            if response:
                request.response = response
            
            # Save changes
            self._save_request(request)
            
            logger.info(
                f"Updated request {request_id} status to {status.value}"
            )
            return True
            
        except Exception as e:
            logger.error(f"Error updating request status: {e}")
            return False
    
    def list_requests(
        self,
        user_id: Optional[str] = None,
        right_type: Optional[RightType] = None,
        status: Optional[RequestStatus] = None,
        priority: Optional[RequestPriority] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[RightsRequest]:
        """
        List rights requests.
        
        Args:
            user_id: Optional user filter
            right_type: Optional right type filter
            status: Optional status filter
            priority: Optional priority filter
            start_date: Start date filter
            end_date: End date filter
        
        Returns:
            List of matching requests
        """
        try:
            query = "SELECT * FROM rights_requests WHERE 1=1"
            params = []
            
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            
            if right_type:
                query += " AND right_type = ?"
                params.append(right_type.value)
            
            if status:
                query += " AND status = ?"
                params.append(status.value)
            
            if priority:
                query += " AND priority = ?"
                params.append(priority.value)
            
            if start_date:
                query += " AND submitted_at >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND submitted_at <= ?"
                params.append(end_date.isoformat())
            
            query += " ORDER BY submitted_at DESC"
            
            requests = []
            db_path = os.path.join(self.storage_dir, "rights.db")
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                
                for result in cursor.fetchall():
                    (
                        id, user_id, right_type_str, status_str,
                        priority_str, submitted_at_str, deadline_str,
                        description, proof_of_identity, data_scope_str,
                        response_str, metadata_str
                    ) = result
                    
                    request = RightsRequest(
                        id=id,
                        user_id=user_id,
                        right_type=RightType(right_type_str),
                        status=RequestStatus(status_str),
                        priority=RequestPriority(priority_str),
                        submitted_at=datetime.fromisoformat(
                            submitted_at_str
                        ),
                        deadline=datetime.fromisoformat(deadline_str),
                        description=description,
                        proof_of_identity=proof_of_identity,
                        data_scope=json.loads(data_scope_str),
                        response=(
                            json.loads(response_str)
                            if response_str else None
                        ),
                        metadata=json.loads(metadata_str)
                    )
                    requests.append(request)
            
            return requests
            
        except Exception as e:
            logger.error(f"Error listing requests: {e}")
            return []
    
    def check_overdue_requests(self) -> List[RightsRequest]:
        """
        Check for overdue requests.
        
        Returns:
            List of overdue requests
        """
        try:
            now = datetime.utcnow()
            
            # Get requests past deadline
            return self.list_requests(
                status=RequestStatus.PROCESSING,
                end_date=now
            )
            
        except Exception as e:
            logger.error(f"Error checking overdue requests: {e}")
            return []
    
    def generate_compliance_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """
        Generate compliance report.
        
        Args:
            start_date: Report start date
            end_date: Report end date
        
        Returns:
            Report data
        """
        try:
            # Get requests in date range
            requests = self.list_requests(
                start_date=start_date,
                end_date=end_date
            )
            
            # Calculate statistics
            total_requests = len(requests)
            completed_requests = sum(
                1 for r in requests
                if r.status == RequestStatus.COMPLETED
            )
            rejected_requests = sum(
                1 for r in requests
                if r.status == RequestStatus.REJECTED
            )
            overdue_requests = sum(
                1 for r in requests
                if r.deadline < datetime.utcnow() and
                r.status not in [
                    RequestStatus.COMPLETED,
                    RequestStatus.REJECTED,
                    RequestStatus.CANCELLED
                ]
            )
            
            avg_completion_time = timedelta()
            if completed_requests > 0:
                completion_times = [
                    r.response['completed_at'] - r.submitted_at
                    for r in requests
                    if r.status == RequestStatus.COMPLETED and
                    r.response and
                    'completed_at' in r.response
                ]
                if completion_times:
                    avg_completion_time = sum(
                        completion_times,
                        timedelta()
                    ) / len(completion_times)
            
            # Group by type
            requests_by_type = {}
            for right_type in RightType:
                type_requests = [
                    r for r in requests
                    if r.right_type == right_type
                ]
                if type_requests:
                    requests_by_type[right_type.value] = len(type_requests)
            
            return {
                'period': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                },
                'statistics': {
                    'total_requests': total_requests,
                    'completed_requests': completed_requests,
                    'rejected_requests': rejected_requests,
                    'overdue_requests': overdue_requests,
                    'completion_rate': (
                        completed_requests / total_requests
                        if total_requests > 0 else 0
                    ),
                    'avg_completion_time': str(avg_completion_time)
                },
                'requests_by_type': requests_by_type,
                'requests': [
                    {
                        'id': r.id,
                        'user_id': r.user_id,
                        'right_type': r.right_type.value,
                        'status': r.status.value,
                        'priority': r.priority.value,
                        'submitted_at': r.submitted_at.isoformat(),
                        'deadline': r.deadline.isoformat(),
                        'description': r.description,
                        'data_scope': r.data_scope,
                        'response': r.response
                    }
                    for r in requests
                ]
            }
            
        except Exception as e:
            logger.error(f"Error generating compliance report: {e}")
            return {} 