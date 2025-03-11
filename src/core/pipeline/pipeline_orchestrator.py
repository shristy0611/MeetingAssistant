"""
Pipeline Orchestrator for AMPTALK.

This module provides sophisticated pipeline orchestration capabilities for managing
complex workflows between multiple agents. It implements state-of-the-art practices
for dynamic resource allocation, load balancing, and error handling.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
import numpy as np

from src.core.agent import Agent
from src.core.message import Message
from src.utils.performance_monitor import PerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class PipelineStage:
    """Configuration for a pipeline stage."""
    
    agent_id: str
    next_stages: List[str] = field(default_factory=list)
    retry_policy: Dict[str, Any] = field(default_factory=lambda: {
        "max_retries": 3,
        "base_delay": 1.0,
        "max_delay": 30.0,
        "exponential_base": 2
    })
    timeout: float = 30.0  # seconds
    priority: int = 1
    resource_requirements: Dict[str, float] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Configuration for the pipeline orchestrator."""
    
    # Pipeline structure
    stages: Dict[str, PipelineStage] = field(default_factory=dict)
    
    # Resource management
    enable_load_balancing: bool = True
    resource_monitoring_interval: float = 1.0  # seconds
    max_concurrent_tasks: int = 10
    
    # Performance optimization
    enable_dynamic_batching: bool = True
    batch_size_range: Tuple[int, int] = (1, 32)
    batch_timeout: float = 0.1  # seconds
    
    # Error handling
    error_recovery_enabled: bool = True
    error_threshold: float = 0.1  # Error rate threshold for circuit breaking
    circuit_breaker_reset_time: float = 60.0  # seconds
    
    # Monitoring
    enable_performance_monitoring: bool = True
    metrics_collection_interval: float = 5.0  # seconds


class PipelineOrchestrator:
    """
    Orchestrates complex pipelines of agents with dynamic resource allocation
    and sophisticated error handling.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the pipeline orchestrator.
        
        Args:
            config: Configuration for the pipeline, or None to use defaults
        """
        self.config = config or PipelineConfig()
        
        # Pipeline state
        self.stages: Dict[str, PipelineStage] = {}
        self.agents: Dict[str, Agent] = {}
        self.running = False
        
        # Task management
        self.tasks: List[asyncio.Task] = []
        self.active_tasks: Dict[str, Set[asyncio.Task]] = {}
        self.task_queue = asyncio.PriorityQueue()
        
        # Performance monitoring
        self.metrics = PerformanceMetrics()
        self.stage_metrics: Dict[str, Dict[str, float]] = {}
        
        # Resource management
        self.resource_usage: Dict[str, float] = {}
        self.stage_load: Dict[str, float] = {}
        
        # Error tracking
        self.error_counts: Dict[str, int] = {}
        self.circuit_breakers: Dict[str, bool] = {}
        self.last_error_time: Dict[str, float] = {}
        
        logger.info("Pipeline orchestrator initialized")
    
    def add_stage(self, stage: PipelineStage) -> None:
        """
        Add a stage to the pipeline.
        
        Args:
            stage: The pipeline stage configuration
        """
        if stage.agent_id in self.stages:
            logger.warning(f"Stage {stage.agent_id} already exists in pipeline")
            return
        
        self.stages[stage.agent_id] = stage
        self.active_tasks[stage.agent_id] = set()
        self.error_counts[stage.agent_id] = 0
        self.circuit_breakers[stage.agent_id] = False
        self.stage_metrics[stage.agent_id] = {
            "throughput": 0.0,
            "latency": 0.0,
            "error_rate": 0.0
        }
        
        logger.info(f"Added pipeline stage: {stage.agent_id}")
    
    def register_agent(self, agent: Agent) -> None:
        """
        Register an agent with the pipeline.
        
        Args:
            agent: The agent to register
        """
        if agent.agent_id in self.agents:
            logger.warning(f"Agent {agent.agent_id} already registered")
            return
        
        self.agents[agent.agent_id] = agent
        logger.info(f"Registered agent: {agent.agent_id}")
    
    async def start(self) -> None:
        """Start the pipeline orchestrator."""
        if self.running:
            logger.warning("Pipeline orchestrator already running")
            return
        
        self.running = True
        logger.info("Starting pipeline orchestrator")
        
        # Start monitoring tasks
        if self.config.enable_performance_monitoring:
            self.tasks.append(asyncio.create_task(self._monitor_performance()))
        
        if self.config.enable_load_balancing:
            self.tasks.append(asyncio.create_task(self._balance_load()))
        
        # Start task processor
        self.tasks.append(asyncio.create_task(self._process_tasks()))
        
        logger.info("Pipeline orchestrator started")
    
    async def stop(self) -> None:
        """Stop the pipeline orchestrator."""
        if not self.running:
            logger.warning("Pipeline orchestrator not running")
            return
        
        self.running = False
        logger.info("Stopping pipeline orchestrator")
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        self.tasks = []
        
        logger.info("Pipeline orchestrator stopped")
    
    async def process_message(self, message: Message, stage_id: str) -> None:
        """
        Process a message through a pipeline stage.
        
        Args:
            message: The message to process
            stage_id: ID of the stage to process the message
        """
        if stage_id not in self.stages:
            logger.error(f"Invalid stage ID: {stage_id}")
            return
        
        if self.circuit_breakers[stage_id]:
            logger.warning(f"Circuit breaker open for stage {stage_id}")
            return
        
        stage = self.stages[stage_id]
        agent = self.agents.get(stage.agent_id)
        
        if not agent:
            logger.error(f"No agent found for stage {stage_id}")
            return
        
        # Create and queue the task
        task = asyncio.create_task(self._execute_stage(message, stage, agent))
        self.active_tasks[stage_id].add(task)
        
        try:
            await task
        except Exception as e:
            logger.error(f"Error in stage {stage_id}: {str(e)}")
            self._handle_stage_error(stage_id)
        finally:
            self.active_tasks[stage_id].remove(task)
    
    async def _execute_stage(self, message: Message, stage: PipelineStage, agent: Agent) -> None:
        """
        Execute a pipeline stage.
        
        Args:
            message: The message to process
            stage: The pipeline stage configuration
            agent: The agent to execute the stage
        """
        start_time = time.time()
        
        try:
            # Process the message
            result = await agent.process_message(message)
            
            # Update metrics
            processing_time = time.time() - start_time
            self._update_stage_metrics(stage.agent_id, processing_time, success=True)
            
            # Forward results to next stages
            if result:
                for next_stage_id in stage.next_stages:
                    if next_stage_id in self.stages:
                        await self.process_message(result, next_stage_id)
            
        except Exception as e:
            logger.error(f"Stage execution error in {stage.agent_id}: {str(e)}")
            self._update_stage_metrics(stage.agent_id, time.time() - start_time, success=False)
            raise
    
    def _update_stage_metrics(self, stage_id: str, processing_time: float, success: bool) -> None:
        """
        Update performance metrics for a stage.
        
        Args:
            stage_id: ID of the stage
            processing_time: Time taken to process the message
            success: Whether the processing was successful
        """
        metrics = self.stage_metrics[stage_id]
        
        # Update throughput (exponential moving average)
        alpha = 0.1
        metrics["throughput"] = (1 - alpha) * metrics["throughput"] + alpha * (1.0 / processing_time)
        
        # Update latency (exponential moving average)
        metrics["latency"] = (1 - alpha) * metrics["latency"] + alpha * processing_time
        
        # Update error rate
        if not success:
            self.error_counts[stage_id] += 1
        total_processed = sum(1 for tasks in self.active_tasks.values() for _ in tasks)
        if total_processed > 0:
            metrics["error_rate"] = self.error_counts[stage_id] / total_processed
    
    def _handle_stage_error(self, stage_id: str) -> None:
        """
        Handle an error in a pipeline stage.
        
        Args:
            stage_id: ID of the stage where the error occurred
        """
        self.error_counts[stage_id] += 1
        self.last_error_time[stage_id] = time.time()
        
        # Check if we need to open the circuit breaker
        metrics = self.stage_metrics[stage_id]
        if metrics["error_rate"] > self.config.error_threshold:
            logger.warning(f"Opening circuit breaker for stage {stage_id}")
            self.circuit_breakers[stage_id] = True
            
            # Schedule circuit breaker reset
            asyncio.create_task(self._reset_circuit_breaker(stage_id))
    
    async def _reset_circuit_breaker(self, stage_id: str) -> None:
        """
        Reset a circuit breaker after a delay.
        
        Args:
            stage_id: ID of the stage to reset
        """
        await asyncio.sleep(self.config.circuit_breaker_reset_time)
        self.circuit_breakers[stage_id] = False
        logger.info(f"Reset circuit breaker for stage {stage_id}")
    
    async def _monitor_performance(self) -> None:
        """Monitor pipeline performance metrics."""
        while self.running:
            try:
                # Collect metrics for each stage
                for stage_id, metrics in self.stage_metrics.items():
                    self.metrics.record_metric(f"{stage_id}_throughput", metrics["throughput"])
                    self.metrics.record_metric(f"{stage_id}_latency", metrics["latency"])
                    self.metrics.record_metric(f"{stage_id}_error_rate", metrics["error_rate"])
                
                # Collect overall pipeline metrics
                total_throughput = sum(m["throughput"] for m in self.stage_metrics.values())
                avg_latency = np.mean([m["latency"] for m in self.stage_metrics.values()])
                
                self.metrics.record_metric("pipeline_throughput", total_throughput)
                self.metrics.record_metric("pipeline_latency", avg_latency)
                
                await asyncio.sleep(self.config.metrics_collection_interval)
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {str(e)}")
                await asyncio.sleep(1.0)
    
    async def _balance_load(self) -> None:
        """Balance load across pipeline stages."""
        while self.running:
            try:
                # Calculate current load for each stage
                for stage_id, tasks in self.active_tasks.items():
                    self.stage_load[stage_id] = len(tasks)
                
                # Adjust resource allocation based on load
                total_load = sum(self.stage_load.values())
                if total_load > 0:
                    for stage_id, load in self.stage_load.items():
                        relative_load = load / total_load
                        self._adjust_resources(stage_id, relative_load)
                
                await asyncio.sleep(self.config.resource_monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in load balancing: {str(e)}")
                await asyncio.sleep(1.0)
    
    def _adjust_resources(self, stage_id: str, relative_load: float) -> None:
        """
        Adjust resource allocation for a stage based on its load.
        
        Args:
            stage_id: ID of the stage
            relative_load: The stage's load relative to total pipeline load
        """
        stage = self.stages[stage_id]
        
        # Adjust maximum concurrent tasks based on load
        max_tasks = max(1, int(self.config.max_concurrent_tasks * relative_load))
        current_tasks = len(self.active_tasks[stage_id])
        
        if current_tasks > max_tasks:
            logger.info(f"Reducing concurrent tasks for stage {stage_id}")
            # Let existing tasks complete naturally
        elif current_tasks < max_tasks:
            logger.info(f"Increasing concurrent tasks for stage {stage_id}")
            # New tasks will be created as needed
    
    async def _process_tasks(self) -> None:
        """Process tasks from the task queue."""
        while self.running:
            try:
                # Get next task from queue
                priority, task = await self.task_queue.get()
                
                # Execute the task
                try:
                    await task
                except Exception as e:
                    logger.error(f"Error processing task: {str(e)}")
                finally:
                    self.task_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in task processor: {str(e)}")
                await asyncio.sleep(0.1) 