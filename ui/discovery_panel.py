"""
Discovery Panel - Real-time Anomaly Hunting Dashboard

A mesmerizing, premium visualization of the autonomous discovery loop.
Shows live progress, statistics, candidates, and logs.
"""

from __future__ import annotations

import json
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


class CandidateCard(QFrame):
    """Card showing an anomaly candidate."""
    
    clicked = pyqtSignal(str)
    
    def __init__(self, candidate: dict):
        super().__init__()
        self.candidate = candidate
        self.setCursor(Qt.PointingHandCursor)
        
        self.setStyleSheet("""
            QFrame {
                background: rgba(20, 24, 32, 0.7);
                border: 1px solid rgba(91, 141, 239, 0.3);
                border-radius: 12px;
            }
            QFrame:hover {
                border-color: rgba(91, 141, 239, 0.6);
                background: rgba(30, 40, 55, 0.6);
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)
        
        # Class label
        class_label = QLabel(candidate.get("classification", "Unknown"))
        class_label.setStyleSheet("font-size: 14px; font-weight: 500; color: #c8d0e0;")
        layout.addWidget(class_label)
        
        # OOD score
        ood = candidate.get("ood_score", 0)
        ood_label = QLabel(f"OOD: {ood:.3f}")
        ood_label.setStyleSheet("font-size: 12px; color: #f87171; font-weight: 600;")
        layout.addWidget(ood_label)
        
        # Source & time
        source = candidate.get("source", "unknown")
        detected = candidate.get("detected_at", "")[:16]
        info = QLabel(f"{source} · {detected}")
        info.setStyleSheet("font-size: 10px; color: #4a5568;")
        layout.addWidget(info)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.candidate.get("image_path", ""))


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
        
        # Settings
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.5, 5.0)
        self.threshold_spin.setValue(3.0)
        self.threshold_spin.setSingleStep(0.1)
        self.threshold_spin.setPrefix("Threshold: ")
        self.threshold_spin.setStyleSheet("""
            QDoubleSpinBox {
                background: rgba(20, 24, 32, 0.8);
                border: 1px solid rgba(40, 50, 70, 0.5);
                border-radius: 8px;
                padding: 8px 12px;
                color: #a0aec0;
                min-width: 120px;
            }
        """)
        controls.addWidget(self.threshold_spin)
        
        self.aggressive_check = QCheckBox("Aggressive")
        self.aggressive_check.setStyleSheet("color: #7a8599;")
        self.aggressive_check.setToolTip("30-sec cycles, 50 images")
        controls.addWidget(self.aggressive_check)
        
        self.turbo_check = QCheckBox("Turbo")
        self.turbo_check.setStyleSheet("color: #fbbf24;")
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
        
        header.addLayout(controls)
        
        return header
    
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
        self.stat_duplicates = StatCard("Duplicates", "0")
        self.stat_anomalies = StatCard("Anomalies", "0")
        self.stat_near_misses = StatCard("Near Misses", "0")
        
        stats_grid.addWidget(self.stat_cycles, 0, 0)
        stats_grid.addWidget(self.stat_downloaded, 0, 1)
        stats_grid.addWidget(self.stat_analyzed, 0, 2)
        stats_grid.addWidget(self.stat_duplicates, 1, 0)
        stats_grid.addWidget(self.stat_anomalies, 1, 1)
        stats_grid.addWidget(self.stat_near_misses, 1, 2)
        
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
        layout.setSpacing(16)
        
        # Candidates section
        candidates_group = QGroupBox("Anomaly Candidates")
        candidates_layout = QVBoxLayout(candidates_group)
        
        self.candidates_scroll = QScrollArea()
        self.candidates_scroll.setWidgetResizable(True)
        self.candidates_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.candidates_scroll.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
        """)
        
        self.candidates_container = QWidget()
        self.candidates_layout = QVBoxLayout(self.candidates_container)
        self.candidates_layout.setSpacing(8)
        self.candidates_layout.addStretch()
        
        self.candidates_scroll.setWidget(self.candidates_container)
        candidates_layout.addWidget(self.candidates_scroll)
        
        layout.addWidget(candidates_group, 1)
        
        # Live log
        log_group = QGroupBox("Activity Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setStyleSheet("""
            QTextEdit {
                background: rgba(10, 13, 18, 0.8);
                border: 1px solid rgba(40, 50, 70, 0.4);
                border-radius: 8px;
                font-family: 'SF Mono', 'Fira Code', monospace;
                font-size: 11px;
                color: #7a8599;
                padding: 12px;
            }
        """)
        self.log_view.setMaximumHeight(200)
        log_layout.addWidget(self.log_view)
        
        layout.addWidget(log_group)
        
        return panel
    
    def _setup_refresh_timer(self):
        """Setup timer to refresh stats."""
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self._refresh_data)
        self.refresh_timer.start(2000)  # Every 2 seconds
    
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
        
        # Start process in a completely detached way to prevent UI crashes from killing it
        try:
            # Use start_new_session to detach from parent process group
            self.discovery_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,  # Detach from parent
            )
            
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
        """Stop the discovery loop."""
        self.is_running = False
        self.status_orb.stop()
        self.status_orb.color = QColor("#4a5568")
        self.status_label.setText("Stopped - Discovery paused")
        self.status_label.setStyleSheet("font-size: 13px; color: #4a5568;")
        
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
            self.discovery_process.terminate()
            self.discovery_process = None
        
        self._log("Discovery loop stopped")
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
                self.stat_duplicates.set_value(str(stats.get("duplicates_skipped", 0)))
                self.stat_anomalies.set_value(str(stats.get("anomalies_found", 0)))
                self.stat_near_misses.set_value(str(stats.get("near_misses", 0)))
                
                # Highlight anomalies if found
                if stats.get("anomalies_found", 0) > 0:
                    self.stat_anomalies.set_highlight(True)
                
                # Update gauge
                threshold = stats.get("current_threshold", 3.0)
                highest = stats.get("highest_ood_score", 0)
                self.threshold_gauge.set_values(threshold, highest)
                self.highest_ood_label.setText(f"Highest OOD: {highest:.3f}")
                
                # Cycle info
                last_cycle = stats.get("last_cycle_at", "")
                if last_cycle:
                    self.cycle_info.setText(f"Last cycle: {last_cycle[:19]}")
                
        except Exception as e:
            pass  # Silently handle missing/invalid state
    
    def _load_candidates(self):
        """Load candidates from file."""
        try:
            if CANDIDATES_FILE.exists():
                with open(CANDIDATES_FILE) as f:
                    candidates = json.load(f)
                
                # Clear existing
                while self.candidates_layout.count() > 1:
                    item = self.candidates_layout.takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()
                
                # Add new candidates
                for candidate in candidates[-10:]:  # Last 10
                    card = CandidateCard(candidate)
                    card.clicked.connect(self._on_candidate_clicked)
                    self.candidates_layout.insertWidget(
                        self.candidates_layout.count() - 1,
                        card
                    )
                    
                    # Emit signal for new anomalies
                    if not candidate.get("is_confirmed"):
                        self.anomaly_found.emit(candidate)
                        
        except Exception:
            pass
    
    def _on_candidate_clicked(self, path: str):
        """Handle candidate click - could open viewer."""
        self._log(f"Selected candidate: {Path(path).name}")
    
    def closeEvent(self, event):
        """Cleanup on close."""
        if self.is_running:
            self._stop_discovery()
        self.refresh_timer.stop()
        super().closeEvent(event)

