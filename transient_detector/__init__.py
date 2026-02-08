"""
Specialized Transient Detector Module

A modular system for training YOLO-based anomaly detection on
astronomical transients and unusual objects.

This module is separate from the main AstroLens classification system
and focuses on specialized anomaly detection with reduced false positives.
"""

from .pipeline import TransientPipeline, PipelinePhase, PipelineState
from .data_collector import TransientDataCollector
from .trainer import YOLOTrainer

__all__ = [
    "TransientPipeline",
    "PipelinePhase", 
    "PipelineState",
    "TransientDataCollector",
    "YOLOTrainer",
]
