"""
Main entry point for the AMPTALK system.

This module provides the entry point for the AMPTALK meeting
transcription and analysis system, initializing the necessary
components and starting the system.
"""

import argparse
import asyncio
import logging
import os
import sys
from typing import Dict, List, Optional

from src.core import Orchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("AMPTALK")


async def main(args: argparse.Namespace) -> None:
    """
    Main entry point for the AMPTALK system.
    
    Args:
        args: Command-line arguments
    """
    logger.info(f"Starting AMPTALK system in {args.mode} mode")
    
    # Create the orchestrator
    orchestrator = Orchestrator()
    
    try:
        # In the future, we'll dynamically load agents based on configuration
        # For now, just log that we would initialize agents here
        logger.info("Initializing agents...")
        
        # Show configuration
        logger.info(f"Language: {args.language}")
        logger.info(f"Model size: {args.model_size}")
        logger.info(f"Input: {args.input if args.input else 'microphone'}")
        
        # Start the orchestrator
        logger.info("Starting orchestrator...")
        await orchestrator.start()
        
        # In a real implementation, we would wait for completion
        # or user input to stop the system
        logger.info("System running. Press Ctrl+C to stop.")
        
        # For now, we'll just wait a bit to demonstrate
        await asyncio.sleep(10)
        
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
    finally:
        # Stop the orchestrator
        logger.info("Stopping orchestrator...")
        await orchestrator.stop()
        logger.info("AMPTALK system stopped")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="AMPTALK Meeting Transcription System")
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["dev", "test", "prod"],
        default="dev",
        help="Operation mode",
    )
    
    parser.add_argument(
        "--language",
        type=str,
        choices=["en", "ja", "auto"],
        default="auto",
        help="Processing language (en=English, ja=Japanese, auto=Automatic detection)",
    )
    
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["tiny", "base", "small", "medium", "large"],
        default="base",
        help="Whisper model size to use",
    )
    
    parser.add_argument(
        "--input",
        type=str,
        help="Input audio file (omit to use microphone)",
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        
    return args


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args)) 