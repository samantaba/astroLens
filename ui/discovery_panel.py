"""
Discovery Panel - Real-time Anomaly Hunting Dashboard

A mesmerizing, premium visualization of the autonomous discovery loop.
Shows live progress, statistics, candidates, and logs.
"""

from __future__ import annotations

import atexit
import json
import os
import signal
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QScrollArea, QGridLayout, QProgressBar, QTextEdit,
    QSplitter, QGroupBox, QSpinBox, QDoubleSpinBox, QCheckBox,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont, QColor, QPainter, QBrush, QPen, QLinearGradient

# Global registry of active discovery processes for cleanup
_active_discovery_processes = []

def _cleanup_discovery_processes():
    """Cleanup all active discovery processes on exit."""
    for proc in _active_discovery_processes:
        try:
            if proc and proc.poll() is None:
                # Send SIGTERM first, then SIGKILL if needed
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)
        except Exception:
            pass
    _active_discovery_processes.clear()

# Register cleanup on interpreter exit
atexit.register(_cleanup_discovery_processes)


# Path to state files
DATA_DIR = Path(__file__).parent.parent.parent / "astrolens_artifacts" / "data"
STATE_FILE = DATA_DIR / "discovery_state.json"
CANDIDATES_FILE = DATA_DIR / "anomaly_candidates.json"
LOG_FILE = DATA_DIR / "discovery_loop.log"


class PulsingOrb(QFrame):
    """Animated pulsing orb indicator."""
    
    def __init__(self, color: str = "#5b8def", size: int = 16):
        super().__init__()
        self.color = QColor(color)
        self.pulse_value = 0.3
        self.direction = 1
        self.setFixedSize(size, size)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._animate)
    
    def start(self):
        self.timer.start(50)
    
    def stop(self):
        self.timer.stop()
        self.pulse_value = 0.3
        self.update()
    
    def _animate(self):
        self.pulse_value += 0.05 * self.direction
        if self.pulse_value >= 1.0:
            self.direction = -1
        elif self.pulse_value <= 0.3:
            self.direction = 1
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Outer glow
        glow_color = QColor(self.color)
        glow_color.setAlphaF(self.pulse_value * 0.3)
        painter.setBrush(QBrush(glow_color))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(0, 0, self.width(), self.height())
        
        # Inner solid
        inner_size = int(self.width() * 0.6)
        offset = (self.width() - inner_size) // 2
        inner_color = QColor(self.color)
        inner_color.setAlphaF(0.8 + self.pulse_value * 0.2)
        painter.setBrush(QBrush(inner_color))
        painter.drawEllipse(offset, offset, inner_size, inner_size)


class StatCard(QFrame):
    """Animated statistic card with gradient background."""
    
    def __init__(self, title: str, value: str = "0", icon: str = ""):
        super().__init__()
        self.setObjectName("statCard")
        self.setMinimumSize(140, 100)
        
        self.setStyleSheet("""
            #statCard {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(20, 24, 32, 0.8), stop:1 rgba(30, 40, 55, 0.6));
                border: 1px solid rgba(60, 80, 120, 0.3);
                border-radius: 16px;
            }
            #statCard:hover {
                border-color: rgba(91, 141, 239, 0.5);
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)
        
        # Title
        title_label = QLabel(title.upper())
        title_label.setStyleSheet("""
            font-size: 10px;
            font-weight: 600;
            color: #4a5568;
            letter-spacing: 1.5px;
        """)
        layout.addWidget(title_label)
        
        # Value
        self.value_label = QLabel(value)
        self.value_label.setStyleSheet("""
            font-size: 32px;
            font-weight: 300;
            color: #c8d0e0;
        """)
        layout.addWidget(self.value_label)
        
        layout.addStretch()
    
    def set_value(self, value: str):
        self.value_label.setText(value)
    
    def set_highlight(self, active: bool = True):
        if active:
            self.value_label.setStyleSheet("""
                font-size: 32px;
                font-weight: 300;
                color: #5b8def;
            """)
        else:
            self.value_label.setStyleSheet("""
                font-size: 32px;
                font-weight: 300;
                color: #c8d0e0;
            """)


class ThresholdGauge(QFrame):
    """Visual gauge for OOD threshold."""
    
    def __init__(self):
        super().__init__()
        self.threshold = 3.0
        self.highest_ood = 0.0
        self.setMinimumHeight(80)
        self.setStyleSheet("""
            background: rgba(14, 18, 24, 0.6);
            border: 1px solid rgba(40, 50, 70, 0.5);
            border-radius: 12px;
        """)
    
    def set_values(self, threshold: float, highest_ood: float):
        self.threshold = threshold
        self.highest_ood = highest_ood
        self.update()
    
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w, h = self.width(), self.height()
        margin = 20
        bar_height = 12
        bar_y = h // 2 - bar_height // 2
        bar_width = w - 2 * margin
        
        # Background bar
        painter.setBrush(QBrush(QColor("#1a2030")))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(margin, bar_y, bar_width, bar_height, 6, 6)
        
        # Scale: 0 to 5 (typical OOD range)
        max_scale = 5.0
        
        # Threshold marker
        threshold_x = margin + int((self.threshold / max_scale) * bar_width)
        painter.setPen(QPen(QColor("#f87171"), 2))
        painter.drawLine(threshold_x, bar_y - 8, threshold_x, bar_y + bar_height + 8)
        
        # Highest OOD marker (gradient fill up to this point)
        if self.highest_ood > 0:
            ood_width = int((min(self.highest_ood, max_scale) / max_scale) * bar_width)
            gradient = QLinearGradient(margin, 0, margin + ood_width, 0)
            gradient.setColorAt(0, QColor("#34d399"))
            gradient.setColorAt(0.7, QColor("#fbbf24"))
            gradient.setColorAt(1, QColor("#f87171") if self.highest_ood > self.threshold else QColor("#fbbf24"))
            painter.setBrush(QBrush(gradient))
            painter.drawRoundedRect(margin, bar_y + 2, ood_width, bar_height - 4, 4, 4)
        
        # Labels
        painter.setPen(QPen(QColor("#7a8599")))
        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)
        
        painter.drawText(margin, h - 8, f"0")
        painter.drawText(w - margin - 20, h - 8, f"{max_scale:.0f}")
        
        # Threshold label
        painter.setPen(QPen(QColor("#f87171")))
        painter.drawText(threshold_x - 15, bar_y - 12, f"T:{self.threshold:.1f}")


class ModelImprovementCard(QFrame):
    """Beautiful card showing model training progress and improvement."""
    
    def __init__(self):
        super().__init__()
        self.accuracy = 0.0
        self.initial_accuracy = 0.0
        self.improvement = 0.0
        self.labeled_count = 0
        self.training_runs = 0
        self.training_history = []
        
        self.setMinimumHeight(180)
        self.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(16, 20, 28, 0.9), stop:1 rgba(24, 32, 44, 0.8));
                border: 1px solid rgba(52, 211, 153, 0.2);
                border-radius: 16px;
            }
        """)
        
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)
        
        # Header row
        header = QHBoxLayout()
        title = QLabel("🎓 Model Training Progress")
        title.setStyleSheet("""
            font-size: 14px;
            font-weight: 600;
            color: #34d399;
            letter-spacing: 0.5px;
        """)
        header.addWidget(title)
        header.addStretch()
        
        self.runs_label = QLabel("0 runs")
        self.runs_label.setStyleSheet("color: #7a8599; font-size: 12px;")
        header.addWidget(self.runs_label)
        layout.addLayout(header)
        
        # Main metrics row
        metrics = QHBoxLayout()
        metrics.setSpacing(24)
        
        # Current Accuracy
        acc_section = QVBoxLayout()
        acc_section.setSpacing(2)
        self.accuracy_label = QLabel("--")
        self.accuracy_label.setStyleSheet("""
            font-size: 36px;
            font-weight: 300;
            color: #5b8def;
        """)
        acc_section.addWidget(self.accuracy_label)
        acc_title = QLabel("ACCURACY")
        acc_title.setStyleSheet("font-size: 10px; color: #4a5568; letter-spacing: 1px;")
        acc_section.addWidget(acc_title)
        metrics.addLayout(acc_section)
        
        # Improvement
        imp_section = QVBoxLayout()
        imp_section.setSpacing(2)
        self.improvement_label = QLabel("+0%")
        self.improvement_label.setStyleSheet("""
            font-size: 28px;
            font-weight: 500;
            color: #34d399;
        """)
        imp_section.addWidget(self.improvement_label)
        imp_title = QLabel("IMPROVEMENT")
        imp_title.setStyleSheet("font-size: 10px; color: #4a5568; letter-spacing: 1px;")
        imp_section.addWidget(imp_title)
        metrics.addLayout(imp_section)
        
        # Labeled Anomalies
        labeled_section = QVBoxLayout()
        labeled_section.setSpacing(2)
        self.labeled_label = QLabel("0")
        self.labeled_label.setStyleSheet("""
            font-size: 28px;
            font-weight: 300;
            color: #fbbf24;
        """)
        labeled_section.addWidget(self.labeled_label)
        labeled_title = QLabel("LABELED")
        labeled_title.setStyleSheet("font-size: 10px; color: #4a5568; letter-spacing: 1px;")
        labeled_section.addWidget(labeled_title)
        metrics.addLayout(labeled_section)
        
        metrics.addStretch()
        layout.addLayout(metrics)
        
        # Progress bar showing improvement visually
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background: rgba(30, 40, 55, 0.8);
                border-radius: 4px;
                border: none;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #34d399, stop:0.5 #5b8def, stop:1 #a78bfa);
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Training history preview
        self.history_label = QLabel("")
        self.history_label.setStyleSheet("font-size: 11px; color: #5a6577;")
        self.history_label.setWordWrap(True)
        layout.addWidget(self.history_label)
    
    def update_metrics(self, accuracy: float, initial_accuracy: float, 
                       improvement: float, labeled_count: int, 
                       training_runs: int, training_history: list):
        """Update all metrics display."""
        self.accuracy = accuracy
        self.initial_accuracy = initial_accuracy
        self.improvement = improvement
        self.labeled_count = labeled_count
        self.training_runs = training_runs
        self.training_history = training_history
        
        # Update labels
        if accuracy > 0:
            self.accuracy_label.setText(f"{accuracy:.1%}")
        else:
            self.accuracy_label.setText("--")
        
        # Format improvement with sign and color
        if improvement > 0:
            self.improvement_label.setText(f"+{improvement:.1f}%")
            self.improvement_label.setStyleSheet("""
                font-size: 28px;
                font-weight: 500;
                color: #34d399;
            """)
        elif improvement < 0:
            self.improvement_label.setText(f"{improvement:.1f}%")
            self.improvement_label.setStyleSheet("""
                font-size: 28px;
                font-weight: 500;
                color: #f87171;
            """)
        else:
            self.improvement_label.setText("0%")
            self.improvement_label.setStyleSheet("""
                font-size: 28px;
                font-weight: 500;
                color: #7a8599;
            """)
        
        self.labeled_label.setText(str(labeled_count))
        self.runs_label.setText(f"{training_runs} run{'s' if training_runs != 1 else ''}")
        
        # Update progress bar (0% = 50% baseline, show improvement)
        # Scale: -20% to +20% improvement maps to 0-100 on progress bar
        progress_value = int(50 + improvement * 2.5)
        progress_value = max(0, min(100, progress_value))
        self.progress_bar.setValue(progress_value)
        
        # Show training history summary
        if training_history:
            recent = training_history[-3:]  # Last 3 runs
            history_parts = []
            for run in recent:
                dataset = run.get("dataset", "?")[:8]
                acc = run.get("accuracy_after", 0)
                imp = run.get("improvement_pct", 0)
                sign = "+" if imp > 0 else ""
                history_parts.append(f"{dataset}: {acc:.0%} ({sign}{imp:.1f}%)")
            self.history_label.setText("Recent: " + " → ".join(history_parts))
        else:
            self.history_label.setText("No training runs yet. Model will improve as discovery continues.")


class DiscoveryPanel(QWidget):
    """
    Premium discovery dashboard with real-time visualization.
    """
    
    # Signals
    discovery_started = pyqtSignal()
    discovery_stopped = pyqtSignal()
    anomaly_found = pyqtSignal(dict)
    log_signal = pyqtSignal(str)  # Thread-safe logging signal
    
    def __init__(self):
        super().__init__()
        self.discovery_process: Optional[subprocess.Popen] = None
        self.is_running = False
        self.log_thread = None
        
        # Connect thread-safe log signal
        self.log_signal.connect(self._append_log)
        
        self._setup_ui()
        self._setup_refresh_timer()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(24)
        
        # Header with controls
        header = self._create_header()
        layout.addLayout(header)
        
        # Main content splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background: rgba(40, 50, 70, 0.3);
                width: 2px;
            }
        """)
        
        # Left side - Stats & Visualization
        left_panel = self._create_stats_panel()
        splitter.addWidget(left_panel)
        
        # Right side - Log & Candidates
        right_panel = self._create_activity_panel()
        splitter.addWidget(right_panel)
        
        splitter.setSizes([500, 400])
        layout.addWidget(splitter, 1)
    
    def _create_header(self) -> QHBoxLayout:
        header = QHBoxLayout()
        
        # Title section
        title_section = QVBoxLayout()
        title_section.setSpacing(4)
        
        title_row = QHBoxLayout()
        self.status_orb = PulsingOrb("#4a5568", 12)
        title_row.addWidget(self.status_orb)
        
        title = QLabel("Discovery Loop")
        title.setStyleSheet("""
            font-size: 28px;
            font-weight: 300;
            color: #c8d0e0;
            letter-spacing: -0.5px;
        """)
        title_row.addWidget(title)
        title_row.addStretch()
        title_section.addLayout(title_row)
        
        self.status_label = QLabel("Idle - Ready to hunt for anomalies")
        self.status_label.setStyleSheet("font-size: 13px; color: #4a5568;")
        title_section.addWidget(self.status_label)
        
        header.addLayout(title_section)
        header.addStretch()
        
        # Controls
        controls = QHBoxLayout()
        controls.setSpacing(12)
        
        # Threshold presets - easy quick selection
        threshold_frame = QFrame()
        threshold_frame.setStyleSheet("""
            QFrame {
                background: rgba(20, 24, 32, 0.6);
                border: 1px solid rgba(40, 50, 70, 0.4);
                border-radius: 6px;
                padding: 4px;
            }
        """)
        threshold_layout = QHBoxLayout(threshold_frame)
        threshold_layout.setContentsMargins(6, 4, 6, 4)
        threshold_layout.setSpacing(4)
        
        threshold_label = QLabel("Threshold:")
        threshold_label.setStyleSheet("color: #7a8599; font-size: 11px; border: none;")
        threshold_layout.addWidget(threshold_label)
        
        # Note: Threshold is adaptive during discovery - starts at spinbox value
        # and automatically adjusts based on detection results
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.1, 5.0)
        self.threshold_spin.setValue(0.5)  # Default to more sensitive
        self.threshold_spin.setSingleStep(0.1)
        self.threshold_spin.setFixedWidth(60)
        self.threshold_spin.setStyleSheet("""
            QDoubleSpinBox {
                background: rgba(30, 40, 55, 0.8);
                border: 1px solid rgba(60, 70, 90, 0.5);
                border-radius: 4px;
                padding: 2px 4px;
                color: #c8d0e0;
                font-size: 11px;
            }
        """)
        self.threshold_spin.valueChanged.connect(self._on_threshold_changed)
        threshold_layout.addWidget(self.threshold_spin)
        
        controls.addWidget(threshold_frame)
        
        # Mode selection
        self.aggressive_check = QCheckBox("Aggressive")
        self.aggressive_check.setStyleSheet("color: #7a8599; font-size: 11px;")
        self.aggressive_check.setToolTip("30-sec cycles, 50 images")
        controls.addWidget(self.aggressive_check)
        
        self.turbo_check = QCheckBox("Turbo")
        self.turbo_check.setStyleSheet("color: #fbbf24; font-size: 11px;")
        self.turbo_check.setToolTip("Maximum speed - 5-sec cycles, 100 images")
        controls.addWidget(self.turbo_check)
        
        # Start/Stop button
        self.start_btn = QPushButton("Start Discovery")
        self.start_btn.setObjectName("primary")
        self.start_btn.setStyleSheet("""
            QPushButton#primary {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(52, 211, 153, 0.85), stop:1 rgba(78, 205, 196, 0.9));
                border: 1px solid rgba(52, 211, 153, 0.3);
                color: #ffffff;
                font-weight: 600;
                padding: 12px 24px;
                border-radius: 10px;
                min-width: 140px;
            }
            QPushButton#primary:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(78, 205, 196, 0.95), stop:1 rgba(52, 211, 153, 1));
            }
        """)
        self.start_btn.clicked.connect(self._toggle_discovery)
        controls.addWidget(self.start_btn)
        
        # Verify button - opens verification panel
        self.verify_btn = QPushButton("🔍 Verify")
        self.verify_btn.setToolTip("Cross-reference anomalies against astronomical catalogs")
        self.verify_btn.setStyleSheet("""
            QPushButton {
                background: rgba(91, 141, 239, 0.2);
                border: 1px solid rgba(91, 141, 239, 0.4);
                border-radius: 10px;
                padding: 12px 20px;
                color: #5b8def;
                font-weight: 500;
            }
            QPushButton:hover {
                background: rgba(91, 141, 239, 0.3);
            }
        """)
        self.verify_btn.clicked.connect(self._open_verification)
        controls.addWidget(self.verify_btn)
        
        header.addLayout(controls)
        
        return header
    
    def _open_verification(self):
        """Request parent to open verification panel."""
        # Navigate to verification tab via parent control center
        parent = self.parent()
        while parent:
            if hasattr(parent, 'show_verification'):
                parent.show_verification()
                break
            parent = parent.parent()
    
    def _create_stats_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 16, 0)
        layout.setSpacing(20)
        
        # Stats grid
        stats_grid = QGridLayout()
        stats_grid.setSpacing(16)
        
        self.stat_cycles = StatCard("Cycles", "0")
        self.stat_downloaded = StatCard("Downloaded", "0")
        self.stat_analyzed = StatCard("Analyzed", "0")
        self.stat_anomalies = StatCard("Anomalies", "0")
        self.stat_near_misses = StatCard("Near Misses", "0")
        self.stat_finetunes = StatCard("Fine-Tunes", "0")
        
        stats_grid.addWidget(self.stat_cycles, 0, 0)
        stats_grid.addWidget(self.stat_downloaded, 0, 1)
        stats_grid.addWidget(self.stat_analyzed, 0, 2)
        stats_grid.addWidget(self.stat_anomalies, 1, 0)
        stats_grid.addWidget(self.stat_near_misses, 1, 1)
        stats_grid.addWidget(self.stat_finetunes, 1, 2)
        
        layout.addLayout(stats_grid)
        
        # Threshold gauge
        gauge_group = QGroupBox("OOD Detection Threshold")
        gauge_layout = QVBoxLayout(gauge_group)
        
        self.threshold_gauge = ThresholdGauge()
        gauge_layout.addWidget(self.threshold_gauge)
        
        gauge_info = QHBoxLayout()
        gauge_info.addWidget(QLabel("Lower threshold = more sensitive"))
        gauge_info.addStretch()
        self.highest_ood_label = QLabel("Highest OOD: --")
        self.highest_ood_label.setStyleSheet("color: #fbbf24; font-weight: 500;")
        gauge_info.addWidget(self.highest_ood_label)
        gauge_layout.addLayout(gauge_info)
        
        layout.addWidget(gauge_group)
        
        # Model Training Progress (new beautiful visualization)
        self.model_improvement_card = ModelImprovementCard()
        layout.addWidget(self.model_improvement_card)
        
        # Progress info
        progress_group = QGroupBox("Current Cycle")
        progress_layout = QVBoxLayout(progress_group)
        
        self.cycle_progress = QProgressBar()
        self.cycle_progress.setTextVisible(False)
        self.cycle_progress.setMaximum(100)
        progress_layout.addWidget(self.cycle_progress)
        
        self.cycle_info = QLabel("Waiting to start...")
        self.cycle_info.setStyleSheet("color: #7a8599; font-size: 12px;")
        progress_layout.addWidget(self.cycle_info)
        
        layout.addWidget(progress_group)
        
        layout.addStretch()
        
        return panel
    
    def _create_activity_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 0, 0, 0)
        layout.setSpacing(12)
        
        # Candidates section - Clean table-like design
        candidates_group = QGroupBox("Recent Anomaly Candidates")
        candidates_group.setStyleSheet("""
            QGroupBox {
                font-weight: 600;
                color: #e2e8f0;
                border: 1px solid rgba(40, 50, 70, 0.5);
                border-radius: 8px;
                margin-top: 8px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
            }
        """)
        candidates_layout = QVBoxLayout(candidates_group)
        candidates_layout.setSpacing(4)
        
        # Simple text-based candidates list (much faster than cards)
        self.candidates_list = QTextEdit()
        self.candidates_list.setReadOnly(True)
        self.candidates_list.setStyleSheet("""
            QTextEdit {
                background: rgba(15, 20, 30, 0.9);
                border: none;
                border-radius: 6px;
                font-family: 'SF Mono', 'Menlo', monospace;
                font-size: 11px;
                color: #94a3b8;
                padding: 8px;
                selection-background-color: rgba(91, 141, 239, 0.3);
            }
        """)
        self.candidates_list.setMaximumHeight(180)
        self.candidates_list.setPlaceholderText("No candidates yet. Start discovery to hunt for anomalies...")
        candidates_layout.addWidget(self.candidates_list)
        
        # Quick actions for candidates
        actions_row = QHBoxLayout()
        self.clear_candidates_btn = QPushButton("Clear")
        self.clear_candidates_btn.setFixedWidth(60)
        self.clear_candidates_btn.setStyleSheet("""
            QPushButton {
                background: rgba(100, 100, 100, 0.3);
                border: 1px solid rgba(100, 100, 100, 0.4);
                color: #888;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 10px;
            }
            QPushButton:hover {
                background: rgba(100, 100, 100, 0.5);
            }
        """)
        self.clear_candidates_btn.clicked.connect(self._clear_candidates)
        actions_row.addWidget(self.clear_candidates_btn)
        
        self.candidates_count_label = QLabel("0 candidates")
        self.candidates_count_label.setStyleSheet("color: #64748b; font-size: 10px;")
        actions_row.addWidget(self.candidates_count_label)
        actions_row.addStretch()
        
        candidates_layout.addLayout(actions_row)
        layout.addWidget(candidates_group)
        
        # Live log - Compact
        log_group = QGroupBox("Activity Log")
        log_group.setStyleSheet("""
            QGroupBox {
                font-weight: 600;
                color: #e2e8f0;
                border: 1px solid rgba(40, 50, 70, 0.5);
                border-radius: 8px;
                margin-top: 8px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
            }
        """)
        log_layout = QVBoxLayout(log_group)
        
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setStyleSheet("""
            QTextEdit {
                background: rgba(10, 13, 18, 0.9);
                border: none;
                border-radius: 6px;
                font-family: 'SF Mono', 'Menlo', monospace;
                font-size: 10px;
                color: #64748b;
                padding: 8px;
            }
        """)
        log_layout.addWidget(self.log_view)
        
        layout.addWidget(log_group, 1)
        
        return panel
    
    def _clear_candidates(self):
        """Clear candidates list and file."""
        self.candidates_list.clear()
        self.candidates_count_label.setText("0 candidates")
        try:
            if CANDIDATES_FILE.exists():
                CANDIDATES_FILE.write_text("[]")
        except Exception:
            pass
    
    def _setup_refresh_timer(self):
        """Setup timer to refresh stats - slower to reduce CPU."""
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self._refresh_data)
        self.refresh_timer.start(3000)  # Every 3 seconds (was 2)
    
    def _on_threshold_changed(self, value: float):
        """Update gauge immediately when threshold spinbox changes."""
        if not self.is_running:
            # Only update if discovery is not running (otherwise state file is source of truth)
            self.threshold_gauge.set_values(value, self.threshold_gauge.highest_ood)
    
    def _toggle_discovery(self):
        if self.is_running:
            self._stop_discovery()
        else:
            self._start_discovery()
    
    def _start_discovery(self):
        """Start the discovery loop in background."""
        self.is_running = True
        self.status_orb.color = QColor("#34d399")
        self.status_orb.start()
        self.status_label.setText("Running - Actively hunting for anomalies...")
        self.status_label.setStyleSheet("font-size: 13px; color: #34d399;")
        
        self.start_btn.setText("Stop Discovery")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(248, 113, 113, 0.85), stop:1 rgba(239, 68, 68, 0.9));
                border: 1px solid rgba(248, 113, 113, 0.3);
                color: #ffffff;
                font-weight: 600;
                padding: 12px 24px;
                border-radius: 10px;
                min-width: 140px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(239, 68, 68, 0.95), stop:1 rgba(248, 113, 113, 1));
            }
        """)
        
        # Build command
        script_path = Path(__file__).parent.parent / "scripts" / "discovery_loop.py"
        cmd = [
            sys.executable,
            str(script_path),
            "--threshold", str(self.threshold_spin.value()),
        ]
        
        if self.turbo_check.isChecked():
            cmd.append("--turbo")
        elif self.aggressive_check.isChecked():
            cmd.append("--aggressive")
        
        # Start process - DO NOT use start_new_session to ensure we can track and kill it
        try:
            self.discovery_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                # Do NOT use start_new_session - we need to be able to kill this process
            )
            
            # Register for cleanup
            _active_discovery_processes.append(self.discovery_process)
            
            # Start log reader thread
            self.log_thread = threading.Thread(target=self._read_process_output, daemon=True)
            self.log_thread.start()
            
            self._log("Discovery loop started")
            self._log(f"Process ID: {self.discovery_process.pid}")
            self.discovery_started.emit()
            
        except Exception as e:
            self._log(f"Failed to start: {e}")
            import traceback
            self._log(traceback.format_exc())
            self._stop_discovery()
    
    def _stop_discovery(self):
        """Stop the discovery loop with proper cleanup."""
        self.is_running = False
        self.status_orb.stop()
        self.status_orb.color = QColor("#4a5568")
        self.status_label.setText("Stopping...")
        self.status_label.setStyleSheet("font-size: 13px; color: #fbbf24;")
        
        self.start_btn.setText("Start Discovery")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(52, 211, 153, 0.85), stop:1 rgba(78, 205, 196, 0.9));
                border: 1px solid rgba(52, 211, 153, 0.3);
                color: #ffffff;
                font-weight: 600;
                padding: 12px 24px;
                border-radius: 10px;
                min-width: 140px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(78, 205, 196, 0.95), stop:1 rgba(52, 211, 153, 1));
            }
        """)
        
        if self.discovery_process:
            try:
                # Send SIGTERM first for graceful shutdown
                self.discovery_process.terminate()
                try:
                    # Wait up to 5 seconds for graceful shutdown
                    self.discovery_process.wait(timeout=5)
                    self._log("Discovery loop stopped gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't stop
                    self._log("Process not responding, force killing...")
                    self.discovery_process.kill()
                    self.discovery_process.wait(timeout=2)
                    self._log("Discovery loop force stopped")
            except Exception as e:
                self._log(f"Error stopping process: {e}")
            finally:
                # Remove from global registry
                if self.discovery_process in _active_discovery_processes:
                    _active_discovery_processes.remove(self.discovery_process)
                self.discovery_process = None
        
        self.status_label.setText("Stopped - Discovery paused")
        self.status_label.setStyleSheet("font-size: 13px; color: #4a5568;")
        self.discovery_stopped.emit()
    
    def _read_process_output(self):
        """Read output from discovery process (runs in background thread)."""
        if not self.discovery_process:
            return
        
        try:
            for line in self.discovery_process.stdout:
                if line.strip():
                    # Use signal for thread-safe UI update
                    self.log_signal.emit(line.strip())
        except Exception:
            pass
    
    def _log(self, message: str):
        """Add message to log view (thread-safe via signal)."""
        self.log_signal.emit(message)
    
    def _append_log(self, message: str):
        """Actually append to log view (called on main thread via signal)."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_view.append(f"[{timestamp}] {message}")
        # Auto-scroll to bottom
        scrollbar = self.log_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def _refresh_data(self):
        """Refresh stats from state file."""
        self._load_stats()
        self._load_candidates()
    
    def _load_stats(self):
        """Load stats from state file."""
        try:
            if STATE_FILE.exists():
                with open(STATE_FILE) as f:
                    stats = json.load(f)
                
                self.stat_cycles.set_value(str(stats.get("cycles_completed", 0)))
                self.stat_downloaded.set_value(str(stats.get("total_downloaded", 0)))
                self.stat_analyzed.set_value(str(stats.get("total_analyzed", 0)))
                self.stat_anomalies.set_value(str(stats.get("anomalies_found", 0)))
                self.stat_near_misses.set_value(str(stats.get("near_misses", 0)))
                self.stat_finetunes.set_value(str(stats.get("finetune_runs", 0)))
                
                # Highlight anomalies if found
                if stats.get("anomalies_found", 0) > 0:
                    self.stat_anomalies.set_highlight(True)
                
                # Highlight fine-tunes if any
                if stats.get("finetune_runs", 0) > 0:
                    self.stat_finetunes.set_highlight(True)
                
                # Update gauge - use spinbox value for threshold if discovery is not running
                # When running, use the actual threshold from state file
                if self.is_running:
                    threshold = stats.get("current_threshold", self.threshold_spin.value())
                else:
                    threshold = self.threshold_spin.value()
                highest = stats.get("highest_ood_score", 0)
                self.threshold_gauge.set_values(threshold, highest)
                self.highest_ood_label.setText(f"Highest OOD: {highest:.3f}")
                
                # Also update the spinbox to show actual running threshold
                if self.is_running and stats.get("current_threshold"):
                    self.threshold_spin.blockSignals(True)
                    self.threshold_spin.setValue(stats.get("current_threshold", 3.0))
                    self.threshold_spin.blockSignals(False)
                
                # Cycle info
                last_cycle = stats.get("last_cycle_at", "")
                if last_cycle:
                    self.cycle_info.setText(f"Last cycle: {last_cycle[:19]}")
                
                # Update Model Improvement Card
                self.model_improvement_card.update_metrics(
                    accuracy=stats.get("model_accuracy", 0.0),
                    initial_accuracy=stats.get("initial_accuracy", 0.0),
                    improvement=stats.get("total_improvement_pct", 0.0),
                    labeled_count=stats.get("labeled_anomalies_downloaded", 0),
                    training_runs=stats.get("finetune_runs", 0),
                    training_history=stats.get("training_history", []),
                )
                
        except Exception as e:
            pass  # Silently handle missing/invalid state
    
    def _load_candidates(self):
        """Load candidates from file - efficient text-based display."""
        try:
            if CANDIDATES_FILE.exists():
                with open(CANDIDATES_FILE) as f:
                    candidates = json.load(f)
                
                if not candidates:
                    self.candidates_list.clear()
                    self.candidates_count_label.setText("0 candidates")
                    return
                
                # Build text display (much faster than widget cards)
                lines = []
                recent = candidates[-15:]  # Last 15
                for c in reversed(recent):  # Most recent first
                    ood = c.get("ood_score", 0)
                    cls = c.get("classification", "?")[:12]
                    conf = c.get("confidence", 0) * 100
                    source = c.get("source", "?")[:8]
                    time_str = c.get("detected_at", "")
                    time_short = time_str[11:16] if len(time_str) > 16 else time_str[:5]
                    
                    # Color code by OOD score
                    if ood > 0.7:
                        color = "#f87171"  # Red - high anomaly
                    elif ood > 0.4:
                        color = "#fbbf24"  # Yellow - medium
                    else:
                        color = "#94a3b8"  # Gray - low
                    
                    line = f'<span style="color:{color}">●</span> {cls:<12} OOD:{ood:.2f} {conf:>4.0f}% {source:<8} {time_short}'
                    lines.append(line)
                
                self.candidates_list.setHtml(
                    f'<pre style="margin:0; line-height:1.6;">{"<br>".join(lines)}</pre>'
                )
                self.candidates_count_label.setText(f"{len(candidates)} candidates")
                        
        except Exception:
            pass
    
    def _on_candidate_clicked(self, path: str):
        """Handle candidate click - could open viewer."""
        self._log(f"Selected candidate: {Path(path).name}")
    
    def closeEvent(self, event):
        """Cleanup on close - ensure discovery process is terminated."""
        if self.is_running:
            self._stop_discovery()
        self.refresh_timer.stop()
        
        # Double-check process is dead
        if self.discovery_process and self.discovery_process.poll() is None:
            try:
                self.discovery_process.kill()
                self.discovery_process.wait(timeout=2)
            except Exception:
                pass
        
        super().closeEvent(event)

