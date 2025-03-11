#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Whisper Model Pruning Script

This script implements state-of-the-art pruning techniques for Whisper models,
including Outlier Weighted Layerwise Sparsity (OWL).

Author: AMPTALK Team
Date: 2024
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

import torch
import torch.nn as nn
import torch.nn.utils.prune as torch_prune
import numpy as np
import torch_pruning as tp
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperConfig,
)
from datasets import load_dataset, Audio
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from tqdm import tqdm
from jiwer import wer
import matplotlib.pyplot as plt
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from tensorboardX import SummaryWriter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
console = Console()

class WhisperModelPruner:
    """
    A class for pruning Whisper models using various techniques including OWL.
    """
    
    def __init__(self, config_path: str):
        """Initialize the pruner with configuration parameters.
        
        Args:
            config_path: Path to the JSON configuration file
        """
        self.config = self._load_config(config_path)
        self.device = self._get_device()
        self.tensorboard_writer = None
        
        if self.config["output"]["log_to_tensorboard"]:
            tb_dir = os.path.join(
                self.config["output"]["dir"], "tensorboard", 
                f"whisper_{self.config['pruning']['method']}_{time.strftime('%Y%m%d_%H%M%S')}"
            )
            os.makedirs(tb_dir, exist_ok=True)
            self.tensorboard_writer = SummaryWriter(tb_dir)
        
        # Initialize model and processor
        console.print(f"[bold green]Loading Whisper model: {self.config['model']['name']}[/bold green]")
        self.model = self._load_model()
        self.processor = WhisperProcessor.from_pretrained(
            self.config["model"]["name"], cache_dir=self.config["model"]["cache_dir"]
        )
        
        # Metrics tracking
        self.baseline_metrics = {}
        self.pruned_metrics = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Loaded configuration as a dictionary
        """
        with open(config_path, "r") as f:
            config = json.load(f)
        return config
    
    def _get_device(self) -> torch.device:
        """Determine the appropriate device based on config and availability.
        
        Returns:
            torch.device: The device to use for computations
        """
        if self.config["hardware"]["device"] == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and self.config["hardware"]["use_mps"]:
                device = torch.device("mps")
                logger.info("Using Apple Metal Performance Shaders (MPS)")
            else:
                device = torch.device("cpu")
                logger.info(f"Using CPU with {self.config['hardware']['cpu_threads']} threads")
                torch.set_num_threads(self.config["hardware"]["cpu_threads"])
        else:
            device = torch.device(self.config["hardware"]["device"])
        
        return device
    
    def _load_model(self) -> WhisperForConditionalGeneration:
        """Load the Whisper model based on configuration.
        
        Returns:
            Loaded Whisper model
        """
        model = WhisperForConditionalGeneration.from_pretrained(
            self.config["model"]["name"], 
            cache_dir=self.config["model"]["cache_dir"],
            torch_dtype=torch.float16 if self.config["hardware"]["precision"] == "float16" else torch.float32,
            low_cpu_mem_usage=True,
        )
        model.to(self.device)
        
        if self.config["optimization"]["compile"]["enabled"] and hasattr(torch, "compile"):
            try:
                model = torch.compile(
                    model, 
                    mode=self.config["optimization"]["compile"]["mode"],
                    fullgraph=False
                )
                logger.info("Model successfully compiled with torch.compile")
            except Exception as e:
                logger.warning(f"Failed to compile model: {e}")
        
        return model
    
    def analyze_model(self) -> Dict[str, Any]:
        """Analyze the model architecture and parameter distribution.
        
        Returns:
            Dictionary with model analysis information
        """
        console.print("[bold blue]Analyzing model architecture...[/bold blue]")
        
        total_params = 0
        module_params = {}
        
        # Analyze parameter distribution
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Only leaf modules
                num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                if num_params > 0:
                    module_params[name] = num_params
                    total_params += num_params
        
        # Sort modules by parameter count
        sorted_modules = sorted(module_params.items(), key=lambda x: x[1], reverse=True)
        top_modules = sorted_modules[:10]
        
        # Calculate layer statistics for encoder and decoder
        encoder_params = sum(p.numel() for name, p in self.model.named_parameters() 
                         if "encoder" in name and p.requires_grad)
        decoder_params = sum(p.numel() for name, p in self.model.named_parameters() 
                         if "decoder" in name and p.requires_grad)
        
        analysis = {
            "total_parameters": total_params,
            "encoder_parameters": encoder_params,
            "decoder_parameters": decoder_params,
            "encoder_percentage": encoder_params / total_params * 100,
            "decoder_percentage": decoder_params / total_params * 100,
            "top_modules": top_modules
        }
        
        # Print analysis results
        console.print(f"Total trainable parameters: [bold]{total_params:,}[/bold]")
        console.print(f"Encoder parameters: [bold]{encoder_params:,}[/bold] ({encoder_params / total_params * 100:.2f}%)")
        console.print(f"Decoder parameters: [bold]{decoder_params:,}[/bold] ({decoder_params / total_params * 100:.2f}%)")
        console.print("\nTop 10 modules by parameter count:")
        
        for i, (name, count) in enumerate(top_modules, 1):
            console.print(f"{i}. [cyan]{name}[/cyan]: {count:,} ({count / total_params * 100:.2f}%)")
        
        return analysis
    
    def evaluate_model(self, model: Optional[WhisperForConditionalGeneration] = None) -> Dict[str, float]:
        """Evaluate the model on test datasets.
        
        Args:
            model: Model to evaluate, uses self.model if None
            
        Returns:
            Dictionary of evaluation metrics
        """
        if model is None:
            model = self.model
        
        model.eval()
        metrics = {}
        
        with torch.no_grad():
            for dataset_config in self.config["evaluation"]["datasets"]:
                dataset_name = dataset_config["name"]
                console.print(f"\n[bold yellow]Evaluating on {dataset_name}...[/bold yellow]")
                
                # Load dataset
                if dataset_name == "librispeech":
                    dataset = load_dataset(
                        dataset_name,
                        dataset_config.get("subset", "clean"),
                        split=dataset_config["split"],
                        streaming=False
                    )
                elif dataset_name == "common_voice":
                    dataset = load_dataset(
                        dataset_name,
                        dataset_config.get("language", "en"),
                        split=dataset_config["split"],
                        streaming=False
                    )
                else:
                    logger.warning(f"Unknown dataset: {dataset_name}")
                    continue
                
                # Limit samples for faster evaluation
                if "max_samples" in dataset_config:
                    dataset = dataset.select(range(min(dataset_config["max_samples"], len(dataset))))
                
                # Resample to 16kHz if needed
                dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
                
                # Process audio samples and calculate metrics
                all_references = []
                all_predictions = []
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    transient=True,
                ) as progress:
                    task = progress.add_task(f"Transcribing {dataset_name} samples...", total=len(dataset))
                    
                    for sample in dataset:
                        # Get audio input
                        audio_input = sample["audio"]["array"]
                        input_features = self.processor(
                            audio_input, 
                            sampling_rate=16000, 
                            return_tensors="pt"
                        ).input_features.to(self.device)
                        
                        # Get prediction
                        predicted_ids = model.generate(input_features)
                        transcription = self.processor.batch_decode(
                            predicted_ids, skip_special_tokens=True
                        )[0]
                        
                        # Reference text
                        reference = sample["text" if "text" in sample else "sentence"]
                        
                        all_references.append(reference)
                        all_predictions.append(transcription)
                        
                        progress.update(task, advance=1)
                
                # Calculate WER
                dataset_wer = wer(all_references, all_predictions)
                metrics[f"{dataset_name}_wer"] = dataset_wer
                
                console.print(f"[green]WER on {dataset_name}: {dataset_wer:.4f}[/green]")
                
                # Character error rate
                if "character_error_rate" in self.config["evaluation"]["metrics"]:
                    import editdistance
                    total_cer = 0
                    total_chars = 0
                    
                    for ref, pred in zip(all_references, all_predictions):
                        distance = editdistance.eval(ref, pred)
                        total_chars += len(ref)
                        total_cer += distance
                    
                    cer = total_cer / total_chars if total_chars > 0 else 0
                    metrics[f"{dataset_name}_cer"] = cer
                    console.print(f"[green]CER on {dataset_name}: {cer:.4f}[/green]")
        
        return metrics
    
    def _compute_outlier_weights(self, 
                                 model: nn.Module, 
                                 module_name: str, 
                                 weight: torch.Tensor) -> torch.Tensor:
        """Compute Outlier Weighting (OWL) for model pruning.
        
        Args:
            model: The model to compute outlier weights for
            module_name: Name of the module being pruned
            weight: Weight tensor to analyze
            
        Returns:
            Tensor of outlier weights of same shape as input weight
        """
        # Flatten the weight tensor
        weight_flat = weight.view(-1).abs()
        
        # Sort the absolute values
        sorted_weights, _ = torch.sort(weight_flat)
        
        # Compute median and lambda-scaled median for outlier detection
        lambda_val = self.config["pruning"]["outlier_hyper_lambda"]
        m_val = self.config["pruning"]["outlier_hyper_m"]
        median_val = sorted_weights[len(sorted_weights) // 2]
        threshold = lambda_val * median_val
        
        # Create mask for outliers (weights above threshold)
        outlier_mask = (weight.abs() > threshold).float()
        
        # Apply exponential scaling for outliers (OWL technique)
        outlier_weights = torch.ones_like(weight)
        outlier_weights = outlier_weights + (m_val - 1) * outlier_mask
        
        if self.tensorboard_writer:
            if "encoder.layers.0" in module_name or "decoder.layers.0" in module_name:
                # Log histogram of first layer for visualization
                self.tensorboard_writer.add_histogram(
                    f"{module_name}_weight_distribution", weight.detach().view(-1).cpu(), 
                    global_step=0
                )
                self.tensorboard_writer.add_histogram(
                    f"{module_name}_outlier_weights", outlier_weights.detach().view(-1).cpu(), 
                    global_step=0
                )
        
        return outlier_weights
    
    def _magnitude_pruning_with_owl(self, model: nn.Module, sparsity: float) -> nn.Module:
        """Apply magnitude-based pruning with Outlier Weighted Layerwise Sparsity (OWL).
        
        Args:
            model: The model to prune
            sparsity: Target sparsity ratio (0.0 to 1.0)
            
        Returns:
            Pruned model
        """
        pruned_model = model
        
        # Get target modules from config
        target_module_types = [nn.Linear]
        excluded_modules = self.config["pruning"]["excluded_modules"]
        
        # Collect all prunable parameters
        parameters_to_prune = []
        
        for name, module in pruned_model.named_modules():
            if isinstance(module, tuple(target_module_types)):
                if not any(excluded in name for excluded in excluded_modules):
                    if hasattr(module, "weight"):
                        parameters_to_prune.append((module, "weight"))
        
        # Apply OWL pruning to each parameter
        for module, param_name in parameters_to_prune:
            # Get the full module name
            for full_name, mod in pruned_model.named_modules():
                if mod is module:
                    module_name = full_name
                    break
            
            # Get the weight tensor
            weight = getattr(module, param_name)
            
            # Compute outlier weights
            if self.config["pruning"]["apply_outlier_weighting"]:
                outlier_weights = self._compute_outlier_weights(pruned_model, module_name, weight)
                
                # Apply custom pruning with outlier weighting
                custom_pruning = tp.importance.MagnitudeImportance(p=2)
                importance_scores = custom_pruning(weight)
                importance_scores = importance_scores * outlier_weights
                
                # Create pruning mask (keep top values)
                num_params = weight.numel()
                num_to_keep = int(num_params * (1 - sparsity))
                
                # Flatten tensors for selection
                flat_importance = importance_scores.view(-1)
                _, indices = torch.topk(flat_importance, k=num_to_keep)
                mask = torch.zeros_like(flat_importance)
                mask[indices] = 1.0
                mask = mask.view_as(weight)
                
                # Apply custom pruning mask
                with torch.no_grad():
                    pruned_weight = weight * mask
                    module.weight.copy_(pruned_weight)
            else:
                # Standard magnitude pruning without OWL
                torch_prune.l1_unstructured(module, name=param_name, amount=sparsity)
        
        return pruned_model
    
    def prune_model(self) -> WhisperForConditionalGeneration:
        """Prune the model using the configured method.
        
        Returns:
            Pruned model instance
        """
        # First, analyze the model
        self.analyze_model()
        
        # Evaluate baseline model
        console.print("\n[bold green]Evaluating baseline model...[/bold green]")
        self.baseline_metrics = self.evaluate_model()
        
        # Begin pruning
        console.print(f"\n[bold magenta]Pruning model with {self.config['pruning']['method']} method...[/bold magenta]")
        console.print(f"Target sparsity: [bold]{self.config['pruning']['sparsity'] * 100:.1f}%[/bold]")
        
        # Create a copy of the model for pruning
        pruned_model = self.model
        
        if self.config["pruning"]["method"] == "magnitude":
            if self.config["pruning"]["apply_outlier_weighting"]:
                console.print("[bold cyan]Using Outlier Weighted Layerwise Sparsity (OWL)[/bold cyan]")
            pruned_model = self._magnitude_pruning_with_owl(
                pruned_model, 
                self.config["pruning"]["sparsity"]
            )
        else:
            logger.error(f"Unsupported pruning method: {self.config['pruning']['method']}")
            return self.model
        
        # Calculate sparsity achieved
        total_zeros = 0
        total_elements = 0
        
        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.Linear):
                if hasattr(module, "weight"):
                    tensor = module.weight.data
                    zeros = torch.sum(tensor == 0).item()
                    elements = tensor.numel()
                    total_zeros += zeros
                    total_elements += elements
        
        achieved_sparsity = total_zeros / total_elements if total_elements > 0 else 0
        console.print(f"Achieved sparsity: [bold]{achieved_sparsity * 100:.2f}%[/bold]")
        
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar("pruning/sparsity", achieved_sparsity, 0)
        
        # Evaluate pruned model
        console.print("\n[bold green]Evaluating pruned model...[/bold green]")
        self.pruned_metrics = self.evaluate_model(pruned_model)
        
        # Compare baseline and pruned metrics
        self._print_metrics_comparison()
        
        # Save pruned model if configured
        if self.config["output"]["save_model"]:
            self._save_pruned_model(pruned_model, achieved_sparsity)
        
        return pruned_model
    
    def _print_metrics_comparison(self) -> None:
        """Print a comparison of baseline and pruned model metrics."""
        console.print("\n[bold blue]Metrics Comparison:[/bold blue]")
        
        for metric in self.baseline_metrics:
            baseline_value = self.baseline_metrics[metric]
            pruned_value = self.pruned_metrics.get(metric, 0)
            rel_change = (pruned_value - baseline_value) / baseline_value * 100 if baseline_value != 0 else 0
            
            if "wer" in metric or "cer" in metric:  # Lower is better for error rates
                change_color = "green" if pruned_value <= baseline_value else "red"
                change_symbol = "↓" if pruned_value <= baseline_value else "↑"
            else:  # Higher is better for other metrics
                change_color = "green" if pruned_value >= baseline_value else "red"
                change_symbol = "↑" if pruned_value >= baseline_value else "↓"
            
            console.print(
                f"{metric}: {baseline_value:.4f} → {pruned_value:.4f} "
                f"[{change_color}]{change_symbol} {abs(rel_change):.2f}%[/{change_color}]"
            )
            
            if self.tensorboard_writer:
                self.tensorboard_writer.add_scalar(f"metrics/{metric}_baseline", baseline_value, 0)
                self.tensorboard_writer.add_scalar(f"metrics/{metric}_pruned", pruned_value, 0)
                self.tensorboard_writer.add_scalar(f"metrics/{metric}_change", rel_change, 0)
    
    def _save_pruned_model(self, model: WhisperForConditionalGeneration, sparsity: float) -> None:
        """Save the pruned model to disk.
        
        Args:
            model: The pruned model to save
            sparsity: Achieved sparsity level
        """
        # Create output directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(
            self.config["output"]["dir"],
            f"whisper_{self.config['pruning']['method']}_{sparsity:.2f}_{timestamp}"
        )
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        console.print(f"\n[bold green]Saving pruned model to {save_dir}[/bold green]")
        model.save_pretrained(save_dir)
        self.processor.save_pretrained(save_dir)
        
        # Save configuration
        with open(os.path.join(save_dir, "pruning_config.json"), "w") as f:
            json.dump(self.config, f, indent=2)
        
        # Save metrics comparison
        metrics_data = {
            "baseline": self.baseline_metrics,
            "pruned": self.pruned_metrics,
            "sparsity": sparsity
        }
        with open(os.path.join(save_dir, "metrics.json"), "w") as f:
            json.dump(metrics_data, f, indent=2)
        
        # Export to ONNX if requested
        if self.config["output"]["export_onnx"]:
            try:
                console.print("[bold yellow]Exporting to ONNX format...[/bold yellow]")
                onnx_path = os.path.join(save_dir, "model.onnx")
                ort_model = ORTModelForSpeechSeq2Seq.from_pretrained(save_dir)
                ort_model.save_pretrained(os.path.join(save_dir, "onnx"))
                console.print(f"[green]ONNX model saved to {os.path.join(save_dir, 'onnx')}[/green]")
            except Exception as e:
                logger.error(f"Failed to export to ONNX: {e}")
        
        console.print("[bold green]Model saving complete![/bold green]")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Whisper Model Pruning")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration JSON file"
    )
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    with open(args.config, "r") as f:
        config = json.load(f)
    
    os.makedirs(config["output"]["dir"], exist_ok=True)
    os.makedirs(config["model"]["cache_dir"], exist_ok=True)
    
    # Run pruning
    pruner = WhisperModelPruner(args.config)
    pruned_model = pruner.prune_model()
    
    console.print("\n[bold green]Pruning process completed![/bold green]")


if __name__ == "__main__":
    main() 