"""
Data Minimization Demo.

This script demonstrates the usage of the data minimization system for privacy-preserving
data operations in the AMPTALK system.

Author: AMPTALK Team
Date: 2024
"""

import os
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, Any

from src.core.privacy.data_minimizer import (
    DataMinimizer,
    DataCategory,
    MinimizationType,
    RetentionPolicy
)
from src.core.utils.logging_config import get_logger

logger = get_logger(__name__)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Data minimization demo"
    )
    
    parser.add_argument(
        "--output-dir",
        default="output/data_minimization",
        help="Output directory for demo results"
    )
    
    parser.add_argument(
        "--config-dir",
        default="privacy/data_minimization",
        help="Configuration directory"
    )
    
    parser.add_argument(
        "--retention-days",
        type=int,
        default=30,
        help="Default retention period in days"
    )
    
    parser.add_argument(
        "--demo-type",
        choices=["pii", "financial", "location", "all"],
        default="all",
        help="Type of data to demonstrate"
    )
    
    return parser.parse_args()

def setup_demo_fields(minimizer: DataMinimizer) -> None:
    """Set up demo fields with minimization rules."""
    # PII fields
    minimizer.register_field(
        name="full_name",
        category=DataCategory.PII,
        minimization_type=MinimizationType.PSEUDONYMIZATION,
        retention_policy=RetentionPolicy.TEMPORARY,
        is_required=True
    )
    
    minimizer.add_minimization_rule(
        field="full_name",
        minimization_type=MinimizationType.PSEUDONYMIZATION,
        parameters={}
    )
    
    minimizer.register_field(
        name="email",
        category=DataCategory.PII,
        minimization_type=MinimizationType.MASKING,
        retention_policy=RetentionPolicy.TEMPORARY
    )
    
    minimizer.add_minimization_rule(
        field="email",
        minimization_type=MinimizationType.MASKING,
        parameters={
            "pattern": "XXXXX@XXX.XXX",
            "mask_char": "*"
        }
    )
    
    # Financial fields
    minimizer.register_field(
        name="credit_card",
        category=DataCategory.FINANCIAL,
        minimization_type=MinimizationType.MASKING,
        retention_policy=RetentionPolicy.SESSION
    )
    
    minimizer.add_minimization_rule(
        field="credit_card",
        minimization_type=MinimizationType.MASKING,
        parameters={
            "prefix_length": 0,
            "suffix_length": 4,
            "mask_char": "X"
        }
    )
    
    minimizer.register_field(
        name="income",
        category=DataCategory.FINANCIAL,
        minimization_type=MinimizationType.AGGREGATION,
        retention_policy=RetentionPolicy.TEMPORARY
    )
    
    minimizer.add_minimization_rule(
        field="income",
        minimization_type=MinimizationType.AGGREGATION,
        parameters={
            "method": "range",
            "ranges": [
                (0, 50000, "0-50k"),
                (50001, 100000, "50k-100k"),
                (100001, 200000, "100k-200k"),
                (200001, float("inf"), "200k+")
            ]
        }
    )
    
    # Location fields
    minimizer.register_field(
        name="coordinates",
        category=DataCategory.LOCATION,
        minimization_type=MinimizationType.AGGREGATION,
        retention_policy=RetentionPolicy.TEMPORARY
    )
    
    minimizer.add_minimization_rule(
        field="coordinates",
        minimization_type=MinimizationType.AGGREGATION,
        parameters={
            "method": "round",
            "precision": 2
        }
    )
    
    minimizer.register_field(
        name="ip_address",
        category=DataCategory.LOCATION,
        minimization_type=MinimizationType.ANONYMIZATION,
        retention_policy=RetentionPolicy.TEMPORARY
    )
    
    minimizer.add_minimization_rule(
        field="ip_address",
        minimization_type=MinimizationType.ANONYMIZATION,
        parameters={}
    )

def generate_demo_data(demo_type: str) -> Dict[str, Any]:
    """Generate sample data for demonstration."""
    data = {}
    
    if demo_type in ["pii", "all"]:
        data.update({
            "full_name": "John Smith",
            "email": "john.smith@example.com",
            "phone": "+1-555-123-4567"
        })
    
    if demo_type in ["financial", "all"]:
        data.update({
            "credit_card": "4532015112830366",
            "income": 175000,
            "account_number": "123456789"
        })
    
    if demo_type in ["location", "all"]:
        data.update({
            "coordinates": (37.7749, -122.4194),
            "ip_address": "192.168.1.100",
            "address": "123 Main St, San Francisco, CA"
        })
    
    return data

def save_results(
    original_data: Dict[str, Any],
    minimized_data: Dict[str, Any],
    output_dir: str
) -> None:
    """Save demo results."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"demo_results_{timestamp}.json")
    
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "original_data": original_data,
        "minimized_data": minimized_data
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved results to {output_file}")

def main() -> None:
    """Run the data minimization demo."""
    args = parse_args()
    
    try:
        # Initialize minimizer
        minimizer = DataMinimizer(
            config_dir=args.config_dir,
            default_retention_days=args.retention_days
        )
        
        # Set up demo fields and rules
        setup_demo_fields(minimizer)
        
        # Generate and minimize data
        original_data = generate_demo_data(args.demo_type)
        logger.info("Original data:")
        logger.info(json.dumps(original_data, indent=2))
        
        minimized_data = minimizer.minimize_data(original_data)
        logger.info("\nMinimized data:")
        logger.info(json.dumps(minimized_data, indent=2))
        
        # Save results
        save_results(
            original_data,
            minimized_data,
            args.output_dir
        )
        
        # Demonstrate retention cleanup
        minimizer.cleanup_expired_data()
        
    except Exception as e:
        logger.error(f"Error running demo: {e}")
        raise

if __name__ == "__main__":
    main() 