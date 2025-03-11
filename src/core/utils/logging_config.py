"""
Logging Configuration Module for AMPTALK.

This module provides a standardized logging configuration for the AMPTALK system,
including structured logging, rotating file handlers, and configurable log levels.

Author: AMPTALK Team
Date: 2024
"""

import os
import sys
import logging
import logging.handlers
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, List


# Define log levels with names and values
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}


class StructuredLogFormatter(logging.Formatter):
    """
    Custom formatter that outputs logs in a structured JSON format.
    
    This formatter enriches log messages with contextual information
    and outputs them in a standardized JSON format for easier parsing
    and analysis.
    """
    
    def __init__(self, include_extra: bool = True):
        """
        Initialize the formatter.
        
        Args:
            include_extra: Whether to include extra fields from the log record
        """
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record as a JSON string.
        
        Args:
            record: The log record to format
            
        Returns:
            JSON string representation of the log entry
        """
        # Base log data
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process_id": record.process,
            "thread_id": record.thread
        }
        
        # Include exception info if available
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }
        
        # Include any extra fields added to the record
        if self.include_extra and hasattr(record, "extra"):
            log_data.update(record.extra)
        
        # Include any arbitrary fields added to the record
        for key, value in record.__dict__.items():
            if key not in ["args", "created", "exc_info", "exc_text", "filename", 
                          "funcName", "levelname", "levelno", "lineno", "module", 
                          "msecs", "message", "msg", "name", "pathname", "process", 
                          "processName", "relativeCreated", "stack_info", "thread", 
                          "threadName", "extra"]:
                if isinstance(value, (str, int, float, bool, type(None))):
                    log_data[key] = value
        
        return json.dumps(log_data)


class PrettyLogFormatter(logging.Formatter):
    """
    Formatter for human-readable logs with color coding.
    
    This formatter outputs logs in a human-friendly format with
    colored output for different log levels when viewed in a
    terminal that supports ANSI color codes.
    """
    
    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[41m",  # Red background
        "RESET": "\033[0m"       # Reset
    }
    
    def __init__(self, use_colors: bool = True, 
                format_string: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"):
        """
        Initialize the formatter.
        
        Args:
            use_colors: Whether to use ANSI color codes
            format_string: Format string for the log message
        """
        super().__init__(format_string)
        self.use_colors = use_colors
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record with optional color coding.
        
        Args:
            record: The log record to format
            
        Returns:
            Formatted log message
        """
        formatted_message = super().format(record)
        
        if self.use_colors and record.levelname in self.COLORS:
            return f"{self.COLORS[record.levelname]}{formatted_message}{self.COLORS['RESET']}"
        
        return formatted_message


def configure_logging(
    app_name: str = "amptalk",
    log_level: Union[str, int] = "INFO",
    log_dir: Optional[str] = None,
    console_output: bool = True,
    file_output: bool = True,
    structured_logs: bool = True,
    max_file_size_mb: int = 10,
    backup_count: int = 5,
    module_levels: Optional[Dict[str, str]] = None
) -> None:
    """
    Configure the logging system for the AMPTALK application.
    
    Args:
        app_name: Name of the application used for log file names
        log_level: Default log level for all loggers
        log_dir: Directory to store log files, if None uses './logs'
        console_output: Whether to output logs to the console
        file_output: Whether to output logs to files
        structured_logs: Whether to use structured JSON logs for file output
        max_file_size_mb: Maximum size of each log file in MB
        backup_count: Number of backup log files to keep
        module_levels: Dict mapping module names to specific log levels
    """
    # Convert string log level to int if needed
    if isinstance(log_level, str):
        log_level = LOG_LEVELS.get(log_level.upper(), logging.INFO)
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set up console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_formatter = PrettyLogFormatter(
            use_colors=sys.stdout.isatty()  # Only use colors if stdout is a terminal
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # Set up file handlers
    if file_output:
        # Create log directory if it doesn't exist
        if log_dir is None:
            log_dir = os.path.join(os.getcwd(), "logs")
        
        os.makedirs(log_dir, exist_ok=True)
        
        # Regular log file
        log_file = os.path.join(log_dir, f"{app_name}.log")
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count
        )
        file_handler.setLevel(log_level)
        
        if structured_logs:
            file_formatter = StructuredLogFormatter()
        else:
            file_formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
            )
        
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Error log file (ERROR and above only)
        error_log_file = os.path.join(log_dir, f"{app_name}_error.log")
        error_file_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count
        )
        error_file_handler.setLevel(logging.ERROR)
        error_file_handler.setFormatter(file_formatter)
        root_logger.addHandler(error_file_handler)
    
    # Set specific log levels for modules if provided
    if module_levels:
        for module_name, level in module_levels.items():
            module_logger = logging.getLogger(module_name)
            if isinstance(level, str):
                level = LOG_LEVELS.get(level.upper(), logging.INFO)
            module_logger.setLevel(level)
    
    # Log the configuration
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized: level={logging.getLevelName(log_level)}, "
               f"console={console_output}, file={file_output}")
    if file_output:
        logger.info(f"Log files will be stored in: {log_dir}")


def get_logger(name: str, extra: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    Get a logger with optional default extra context.
    
    Args:
        name: Name of the logger
        extra: Optional dictionary of extra fields to add to all log records
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    if extra:
        # Use a filter to add extra context to all log records
        class ContextFilter(logging.Filter):
            def filter(self, record):
                record.extra = extra
                return True
        
        logger.addFilter(ContextFilter())
    
    return logger


class LoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that can add structured context to log messages.
    
    This adapter allows for dynamically adding contextual information to log
    messages, such as request IDs, user information, or other metadata.
    """
    
    def __init__(self, logger: logging.Logger, extra: Optional[Dict[str, Any]] = None):
        """
        Initialize the adapter with a logger and optional extra context.
        
        Args:
            logger: The logger to adapt
            extra: Dictionary of extra context to add to all log messages
        """
        super().__init__(logger, extra or {})
    
    def process(self, msg, kwargs):
        """
        Process the log message and arguments.
        
        Args:
            msg: The log message
            kwargs: The logging keyword arguments
            
        Returns:
            Tuple of (modified message, modified kwargs)
        """
        # Combine permanent extra with any extra provided for this log call
        if 'extra' in kwargs:
            kwargs['extra'] = {**self.extra, **kwargs['extra']}
        else:
            kwargs['extra'] = self.extra
        
        return msg, kwargs
    
    def with_context(self, **context) -> 'LoggerAdapter':
        """
        Create a new logger adapter with additional context.
        
        Args:
            **context: Context key-value pairs to add
            
        Returns:
            A new logger adapter with the combined context
        """
        new_extra = {**self.extra, **context}
        return LoggerAdapter(self.logger, new_extra)


def create_audit_logger(
    log_dir: Optional[str] = None,
    app_name: str = "amptalk"
) -> logging.Logger:
    """
    Create a specialized logger for security and audit events.
    
    This logger writes to a separate file for security-related events
    and ensures these logs are always written regardless of log level.
    
    Args:
        log_dir: Directory to store the audit log, if None uses './logs'
        app_name: Name of the application used for log file names
        
    Returns:
        Configured audit logger
    """
    if log_dir is None:
        log_dir = os.path.join(os.getcwd(), "logs")
    
    os.makedirs(log_dir, exist_ok=True)
    
    audit_log_file = os.path.join(log_dir, f"{app_name}_audit.log")
    
    # Create a logger specifically for audit events
    audit_logger = logging.getLogger("amptalk.audit")
    audit_logger.setLevel(logging.INFO)
    
    # Ensure the audit logger doesn't propagate to the root logger
    audit_logger.propagate = False
    
    # Remove any existing handlers
    for handler in audit_logger.handlers[:]:
        audit_logger.removeHandler(handler)
    
    # Create a handler that writes to the audit log file
    audit_handler = logging.handlers.RotatingFileHandler(
        audit_log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=10
    )
    
    # Use structured formatter for audit logs
    audit_formatter = StructuredLogFormatter()
    audit_handler.setFormatter(audit_formatter)
    
    audit_logger.addHandler(audit_handler)
    
    return audit_logger 