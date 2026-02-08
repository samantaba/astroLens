"""
Transient Detector Panel

UI panel for the specialized transient detection pipeline.
Shows progress of all phases and allows starting/stopping.

Uses file-based state refresh to avoid Qt threading issues.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QScrollArea, QProgressBar, QTextEdit, QGroupBox,
)
from PyQt5.QtCore import Qt, QTimer


# State file location
STATE_FILE = Path(__file__).parent.parent.parent / "astrolens_artifacts" / "transient_data" / "pipeline_state.json"


class PhaseCard(QFrame):
    """Card displaying a single phase with its subtasks."""
    
    def __init__(self, phase_data: dict):
        super().__init__()
        self.phase_id = phase_data.get("id", "")
        
        status = phase_data.get("status", "pending")
        progress = phase_data.get("progress", 0)
        
        # Style based on status
        bg_color = {
            "pending": "30, 35, 45",
            "in_progress": "35, 45, 60",
            "completed": "30, 50, 40",
            "failed": "50, 35, 35",
        }.get(status, "30, 35, 45")
        
        border_color = {
            "pending": "50, 55, 70",
            "in_progress": "70, 130, 200",
            "completed": "60, 180, 100",
            "failed": "200, 80, 80",
        }.get(status, "50, 55, 70")
        
        self.setStyleSheet(f"""
            QFrame {{
                background: rgba({bg_color}, 0.7);
                border: 1px solid rgba({border_color}, 0.6);
                border-radius: 12px;
                padding: 16px;
                margin: 4px;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        
        # Header
        header = QHBoxLayout()
        
        # Status icon
        icon = {"pending": "○", "in_progress": "◐", "completed": "✓", "failed": "✗"}.get(status, "○")
        icon_color = {"pending": "#7a8599", "in_progress": "#5b8def", "completed": "#34d399", "failed": "#f87171"}.get(status, "#7a8599")
        
        icon_label = QLabel(icon)
        icon_label.setStyleSheet(f"font-size: 20px; color: {icon_color};")
        header.addWidget(icon_label)
        
        # Phase name
        name_label = QLabel(phase_data.get("name", "Phase"))
        name_label.setStyleSheet("font-size: 16px; font-weight: 600; color: #e8ecf4;")
        header.addWidget(name_label)
        
        header.addStretch()
        
        # Progress percentage
        pct_label = QLabel(f"{progress:.0f}%")
        pct_label.setStyleSheet(f"font-size: 14px; color: {icon_color}; font-weight: 500;")
        header.addWidget(pct_label)
        
        layout.addLayout(header)
        
        # Description
        desc_label = QLabel(phase_data.get("description", ""))
        desc_label.setStyleSheet("font-size: 11px; color: #8a94a6;")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        # Progress bar
        progress_bar = QProgressBar()
        progress_bar.setValue(int(progress))
        progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background: rgba(20, 25, 35, 0.5);
                border: none;
                border-radius: 4px;
                height: 8px;
            }}
            QProgressBar::chunk {{
                background: {icon_color};
                border-radius: 4px;
            }}
        """)
        progress_bar.setTextVisible(False)
        layout.addWidget(progress_bar)
        
        # Subtasks
        subtasks = phase_data.get("subtasks", [])
        if subtasks:
            subtasks_frame = QFrame()
            subtasks_frame.setStyleSheet("background: transparent; border: none;")
            subtasks_layout = QVBoxLayout(subtasks_frame)
            subtasks_layout.setContentsMargins(16, 8, 0, 0)
            subtasks_layout.setSpacing(6)
            
            for st in subtasks:
                st_widget = self._create_subtask_widget(st)
                subtasks_layout.addWidget(st_widget)
            
            layout.addWidget(subtasks_frame)
    
    def _create_subtask_widget(self, subtask: dict) -> QWidget:
        """Create widget for a single subtask."""
        widget = QFrame()
        widget.setStyleSheet("background: transparent; border: none;")
        
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        status = subtask.get("status", "pending")
        
        # Status icon
        icon = {"pending": "○", "in_progress": "◐", "completed": "✓", "failed": "✗"}.get(status, "○")
        icon_color = {"pending": "#6b7280", "in_progress": "#3b82f6", "completed": "#10b981", "failed": "#ef4444"}.get(status, "#6b7280")
        
        icon_label = QLabel(icon)
        icon_label.setStyleSheet(f"font-size: 12px; color: {icon_color};")
        icon_label.setFixedWidth(20)
        layout.addWidget(icon_label)
        
        # Name
        name_label = QLabel(subtask.get("name", ""))
        name_label.setStyleSheet(f"font-size: 12px; color: {'#c8d0e0' if status != 'completed' else '#8a94a6'};")
        layout.addWidget(name_label)
        
        layout.addStretch()
        
        # Progress count
        current = subtask.get("current_count", 0)
        target = subtask.get("target_count", 0)
        if target > 0:
            count_label = QLabel(f"{current}/{target}")
            count_label.setStyleSheet(f"font-size: 11px; color: {icon_color};")
            layout.addWidget(count_label)
        
        return widget


class TransientPanel(QWidget):
    """
    Main panel for transient detection pipeline.
    
    Shows all phases and their progress with real-time updates.
    Uses file-based polling instead of callbacks to avoid Qt threading issues.
    """
    
    def __init__(self):
        super().__init__()
        self._pipeline_process = None
        self._is_running = False
        
        self._setup_ui()
        self._load_initial_state()
        self._setup_refresh_timer()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        
        # Header
        header_layout = QHBoxLayout()
        
        title = QLabel("🔬 Specialized Transient Detector")
        title.setStyleSheet("font-size: 24px; font-weight: 300; color: #e8ecf4;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Control buttons
        self.start_btn = QPushButton("▶ Start Pipeline")
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background: rgba(52, 211, 153, 0.2);
                border: 1px solid rgba(52, 211, 153, 0.4);
                border-radius: 8px;
                padding: 10px 20px;
                color: #34d399;
                font-size: 13px;
                font-weight: 500;
            }
            QPushButton:hover { background: rgba(52, 211, 153, 0.3); }
            QPushButton:disabled { opacity: 0.5; }
        """)
        self.start_btn.clicked.connect(self._start_pipeline)
        header_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("■ Stop")
        self.stop_btn.setCursor(Qt.PointingHandCursor)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background: rgba(248, 113, 113, 0.2);
                border: 1px solid rgba(248, 113, 113, 0.4);
                border-radius: 8px;
                padding: 10px 20px;
                color: #f87171;
                font-size: 13px;
                font-weight: 500;
            }
            QPushButton:hover { background: rgba(248, 113, 113, 0.3); }
            QPushButton:disabled { opacity: 0.5; }
        """)
        self.stop_btn.clicked.connect(self._stop_pipeline)
        self.stop_btn.setEnabled(False)
        header_layout.addWidget(self.stop_btn)
        
        # Reset button
        self.reset_btn = QPushButton("↺ Reset")
        self.reset_btn.setCursor(Qt.PointingHandCursor)
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background: rgba(100, 100, 120, 0.2);
                border: 1px solid rgba(100, 100, 120, 0.4);
                border-radius: 8px;
                padding: 10px 20px;
                color: #8a94a6;
                font-size: 13px;
                font-weight: 500;
            }
            QPushButton:hover { background: rgba(100, 100, 120, 0.3); }
        """)
        self.reset_btn.clicked.connect(self._reset_pipeline)
        header_layout.addWidget(self.reset_btn)
        
        layout.addLayout(header_layout)
        
        # Overall progress
        self.overall_progress = QProgressBar()
        self.overall_progress.setStyleSheet("""
            QProgressBar {
                background: rgba(30, 40, 55, 0.6);
                border: 1px solid rgba(60, 70, 90, 0.4);
                border-radius: 8px;
                height: 24px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3b82f6, stop:1 #8b5cf6);
                border-radius: 6px;
            }
        """)
        self.overall_progress.setFormat("Overall Progress: %p%")
        layout.addWidget(self.overall_progress)
        
        # Status label
        self.status_label = QLabel("Pipeline Status: Not started")
        self.status_label.setStyleSheet("font-size: 13px; color: #8a94a6;")
        layout.addWidget(self.status_label)
        
        # Phases scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background: rgba(30, 40, 55, 0.3);
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: rgba(100, 110, 130, 0.5);
                border-radius: 4px;
            }
        """)
        
        self.phases_container = QWidget()
        self.phases_layout = QVBoxLayout(self.phases_container)
        self.phases_layout.setSpacing(12)
        self.phases_layout.addStretch()
        
        scroll.setWidget(self.phases_container)
        layout.addWidget(scroll, 1)
        
        # Next steps
        self.next_steps_group = QGroupBox("Next Steps")
        self.next_steps_group.setStyleSheet("""
            QGroupBox {
                background: rgba(30, 40, 55, 0.4);
                border: 1px solid rgba(60, 70, 90, 0.3);
                border-radius: 8px;
                padding: 16px;
                margin-top: 12px;
                font-size: 13px;
                font-weight: 500;
                color: #a78bfa;
            }
            QGroupBox::title {
                padding: 0 8px;
            }
        """)
        
        self.next_steps_layout = QVBoxLayout(self.next_steps_group)
        self.next_steps_label = QLabel("• Start the pipeline to begin data collection")
        self.next_steps_label.setStyleSheet("color: #c8d0e0; font-size: 12px;")
        self.next_steps_label.setWordWrap(True)
        self.next_steps_layout.addWidget(self.next_steps_label)
        
        layout.addWidget(self.next_steps_group)
        
        # Activity log
        log_group = QGroupBox("Activity Log")
        log_group.setStyleSheet("""
            QGroupBox {
                background: rgba(20, 25, 35, 0.4);
                border: 1px solid rgba(40, 50, 65, 0.3);
                border-radius: 8px;
                padding: 12px;
                margin-top: 8px;
                font-size: 12px;
                color: #7a8599;
            }
        """)
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background: rgba(15, 20, 28, 0.6);
                border: none;
                border-radius: 6px;
                color: #8a94a6;
                font-family: 'Menlo', 'Monaco', monospace;
                font-size: 11px;
                padding: 8px;
            }
        """)
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_group)
    
    def _load_initial_state(self):
        """Load initial state from file or show default."""
        self._refresh_state()
    
    def _setup_refresh_timer(self):
        """Setup timer for UI updates - reads state file periodically."""
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self._refresh_state)
        self.refresh_timer.start(2000)  # Every 2 seconds
    
    def _refresh_state(self):
        """Refresh state from file - thread-safe since we only read."""
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, "r") as f:
                    data = json.load(f)
                self._update_ui_from_state(data)
            except Exception as e:
                pass  # File might be being written
        else:
            # Show default state
            self._update_ui_from_state(None)
    
    def _update_ui_from_state(self, state: Optional[dict]):
        """Update UI with state dictionary."""
        if not state:
            self.overall_progress.setValue(0)
            self.status_label.setText("<span style='color:#7a8599'>○ Pipeline Ready</span>")
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.reset_btn.setEnabled(False)
            self.next_steps_label.setText("• Start the pipeline to begin data collection")
            # Clear phase cards
            self._clear_phase_cards()
            return
        
        # Overall progress
        total_progress = state.get("total_progress", 0)
        self.overall_progress.setValue(int(min(total_progress, 100)))
        
        # Status
        is_complete = state.get("is_complete", False)
        is_running = state.get("is_running", False)
        current_phase = state.get("current_phase", "")
        
        if is_complete:
            status_text = "✓ Pipeline Complete! Click Reset to start fresh."
            status_color = "#34d399"
        elif is_running:
            status_text = f"◐ Running: {current_phase}"
            status_color = "#5b8def"
        else:
            status_text = "○ Pipeline Ready"
            status_color = "#7a8599"
        
        self.status_label.setText(f"<span style='color:{status_color}'>{status_text}</span>")
        
        # Buttons logic
        has_process = self._pipeline_process is not None
        
        # Start: enabled when not running (allow restart after complete via reset)
        self.start_btn.setEnabled(not is_running and not is_complete)
        
        # Stop: enabled when running or process exists
        self.stop_btn.setEnabled(is_running or has_process)
        
        # Reset: always enabled except when actively running
        self.reset_btn.setEnabled(not is_running)
        
        self._is_running = is_running
        
        # Update phases - clear existing first
        self._clear_phase_cards()
        
        # Add phase cards
        for phase_data in state.get("phases", []):
            card = PhaseCard(phase_data)
            self.phases_layout.insertWidget(self.phases_layout.count() - 1, card)
        
        # Next steps
        next_steps = state.get("next_steps", ["Start the pipeline to begin"])
        self.next_steps_label.setText("\n".join(f"• {step}" for step in next_steps))
    
    def _clear_phase_cards(self):
        """Clear all phase cards from the layout."""
        while self.phases_layout.count() > 1:
            child = self.phases_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
    
    def _start_pipeline(self):
        """Start the pipeline as a subprocess."""
        import subprocess
        import sys
        
        # Run the pipeline in a separate process
        script_path = Path(__file__).parent.parent / "transient_detector" / "run_pipeline.py"
        
        # Create a simple runner script if it doesn't exist
        if not script_path.exists():
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text('''#!/usr/bin/env python3
"""Run the transient pipeline as a standalone process."""
import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])
from transient_detector.pipeline import TransientPipeline
pipeline = TransientPipeline()
pipeline.start()
# Keep running until complete
import time
while pipeline.state.is_running:
    time.sleep(1)
''')
        
        try:
            self._pipeline_process = subprocess.Popen(
                [sys.executable, str(script_path)],
                cwd=str(Path(__file__).parent.parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self._is_running = True
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self._log("Pipeline started in background process")
        except Exception as e:
            self._log(f"Failed to start pipeline: {e}")
    
    def _stop_pipeline(self):
        """Stop the pipeline."""
        # Update state file to signal stop
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, "r") as f:
                    data = json.load(f)
                data["is_running"] = False
                with open(STATE_FILE, "w") as f:
                    json.dump(data, f, indent=2)
            except:
                pass
        
        # Kill subprocess if running
        if self._pipeline_process:
            try:
                self._pipeline_process.terminate()
            except:
                pass
            self._pipeline_process = None
        
        # Also try to kill any orphan pipeline processes
        import subprocess as sp
        try:
            sp.run(["pkill", "-f", "run_pipeline"], capture_output=True)
        except:
            pass
        
        self._is_running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._log("Pipeline stopped")
    
    def _reset_pipeline(self):
        """Reset the pipeline state to start fresh."""
        self._stop_pipeline()
        
        # Delete state file
        if STATE_FILE.exists():
            try:
                STATE_FILE.unlink()
                self._log("State file deleted")
            except Exception as e:
                self._log(f"Could not delete state file: {e}")
        
        # Reset internal state
        self._pipeline_process = None
        self._is_running = False
        
        # Clear UI immediately
        self.overall_progress.setValue(0)
        self.status_label.setText("<span style='color:#7a8599'>○ Pipeline Ready</span>")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.reset_btn.setEnabled(True)
        
        # Clear phase cards
        self._clear_phase_cards()
        
        self._log("Pipeline reset - ready to start fresh")
        self.next_steps_label.setText("• Click 'Start Pipeline' to begin data collection")
    
    def _log(self, message: str):
        """Add message to log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
