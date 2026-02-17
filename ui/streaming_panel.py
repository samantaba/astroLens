"""
Streaming Discovery Panel - Multi-Day Autonomous Discovery Dashboard

Controls the streaming discovery engine from the UI:
- Start/stop streaming with configurable duration
- Real-time progress monitoring
- Access daily reports
- View self-correction log
- Top candidate display
"""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QScrollArea, QGridLayout, QProgressBar, QTextEdit,
    QSpinBox, QComboBox, QGroupBox, QFileDialog, QSizePolicy,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPainter, QBrush


# Paths
_ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "astrolens_artifacts"
_DATA_DIR = _ARTIFACTS_DIR / "data"
_STREAMING_STATE = _DATA_DIR / "streaming_state.json"
_DISCOVERY_STATE = _DATA_DIR / "discovery_state.json"
_CANDIDATES_FILE = _DATA_DIR / "anomaly_candidates.json"
_STREAMING_LOG = _DATA_DIR / "streaming_discovery.log"
_REPORTS_DIR = _ARTIFACTS_DIR / "streaming_reports" / "daily"

# Track active processes for cleanup
_active_processes = []


def _cleanup():
    for proc in _active_processes:
        try:
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        except Exception:
            pass
    _active_processes.clear()


atexit.register(_cleanup)


class _LiveDot(QFrame):
    """Animated status dot."""

    def __init__(self, size=12, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._color = QColor("#4a5568")
        self._alpha = 1.0
        self._dir = -1

    def set_active(self, active: bool):
        self._color = QColor("#3fb950") if active else QColor("#4a5568")
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        c = QColor(self._color)
        c.setAlphaF(self._alpha)
        p.setBrush(QBrush(c))
        p.setPen(Qt.NoPen)
        p.drawEllipse(1, 1, self.width() - 2, self.height() - 2)


class StreamingPanel(QWidget):
    """
    Multi-day streaming discovery control panel.

    Features:
    - Start/stop with day selector and mode
    - Real-time progress bars and stats
    - Daily report access
    - Self-correction log viewer
    - Top candidates table
    """

    anomaly_found = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._process: Optional[subprocess.Popen] = None
        self._setup_ui()

        # Refresh timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(5000)  # Every 5 seconds

        # Initial refresh
        QTimer.singleShot(500, self._refresh)

    # ── UI setup ─────────────────────────────────────────────────────────

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(0, 0, 0, 0)

        # ── Header ───────────────────────────────────────────────────────
        header = QHBoxLayout()

        title_col = QVBoxLayout()
        title_row = QHBoxLayout()
        self._status_dot = _LiveDot(size=10)
        title_row.addWidget(self._status_dot)
        title_label = QLabel("Streaming Discovery")
        title_label.setStyleSheet(
            "font-size: 18px; font-weight: 600; color: #c8d0e0;"
        )
        title_row.addWidget(title_label)
        title_row.addStretch()
        title_col.addLayout(title_row)

        self._status_label = QLabel("Idle")
        self._status_label.setStyleSheet(
            "font-size: 12px; color: #4a5568; margin-left: 22px;"
        )
        title_col.addWidget(self._status_label)
        header.addLayout(title_col)
        header.addStretch()

        layout.addLayout(header)

        # ── Controls ─────────────────────────────────────────────────────
        controls = QFrame()
        controls.setStyleSheet("""
            QFrame {
                background: rgba(22, 27, 34, 0.6);
                border: 1px solid rgba(48, 54, 61, 0.6);
                border-radius: 8px;
            }
        """)
        ctrl_layout = QHBoxLayout(controls)
        ctrl_layout.setContentsMargins(16, 12, 16, 12)

        # Days selector
        days_label = QLabel("Duration:")
        days_label.setStyleSheet("font-size: 12px; color: #8b949e; border: none;")
        ctrl_layout.addWidget(days_label)

        self._days_spin = QSpinBox()
        self._days_spin.setRange(1, 30)
        self._days_spin.setValue(7)
        self._days_spin.setSuffix(" days")
        self._days_spin.setStyleSheet("""
            QSpinBox {
                background: #0d1117;
                border: 1px solid #30363d;
                border-radius: 4px;
                padding: 4px 8px;
                color: #c9d1d9;
                font-size: 12px;
            }
        """)
        ctrl_layout.addWidget(self._days_spin)

        # Mode selector
        mode_label = QLabel("Mode:")
        mode_label.setStyleSheet("font-size: 12px; color: #8b949e; border: none;")
        ctrl_layout.addWidget(mode_label)

        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Normal", "Aggressive", "Turbo"])
        self._mode_combo.setStyleSheet("""
            QComboBox {
                background: #0d1117;
                border: 1px solid #30363d;
                border-radius: 4px;
                padding: 4px 8px;
                color: #c9d1d9;
                font-size: 12px;
                min-width: 100px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background: #161b22;
                border: 1px solid #30363d;
                color: #c9d1d9;
                selection-background-color: rgba(88, 166, 255, 0.2);
            }
        """)
        ctrl_layout.addWidget(self._mode_combo)

        ctrl_layout.addStretch()

        # Start/Stop button
        self._start_btn = QPushButton("Start Streaming")
        self._start_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(35, 134, 54, 0.9), stop:1 rgba(46, 160, 67, 0.9));
                border: none;
                border-radius: 6px;
                padding: 8px 20px;
                color: white;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: rgba(46, 160, 67, 1);
            }
            QPushButton:disabled {
                background: rgba(48, 54, 61, 0.5);
                color: #484f58;
            }
        """)
        self._start_btn.clicked.connect(self._toggle_streaming)
        ctrl_layout.addWidget(self._start_btn)

        layout.addWidget(controls)

        # ── Progress ─────────────────────────────────────────────────────
        progress_frame = QFrame()
        progress_frame.setStyleSheet("""
            QFrame {
                background: rgba(22, 27, 34, 0.4);
                border: 1px solid rgba(48, 54, 61, 0.4);
                border-radius: 8px;
            }
        """)
        prog_layout = QVBoxLayout(progress_frame)
        prog_layout.setContentsMargins(16, 12, 16, 12)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFormat("Day %v of %m")
        self._progress_bar.setStyleSheet("""
            QProgressBar {
                background: #161b22;
                border: 1px solid #30363d;
                border-radius: 4px;
                height: 22px;
                text-align: center;
                color: #c9d1d9;
                font-size: 11px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(88, 166, 255, 0.6), stop:1 rgba(163, 113, 247, 0.6));
                border-radius: 3px;
            }
        """)
        prog_layout.addWidget(self._progress_bar)

        layout.addWidget(progress_frame)

        # ── Stats grid ───────────────────────────────────────────────────
        stats_grid = QGridLayout()
        stats_grid.setSpacing(8)

        self._stat_cards = {}
        stat_defs = [
            ("images", "Images Analyzed", "0"),
            ("anomalies", "Anomalies Found", "0"),
            ("yolo_detected", "YOLO Detections", "0"),
            ("yolo_scanned", "YOLO Scanned", "0"),
            ("rate", "Anomaly Rate", "0.00%"),
            ("threshold", "Threshold", "3.000"),
            ("corrections", "Self-Corrections", "0"),
            ("strategy", "Strategy", "Normal"),
        ]

        for i, (key, label, default) in enumerate(stat_defs):
            card = self._create_stat_card(label, default)
            row, col = divmod(i, 4)
            stats_grid.addWidget(card, row, col)
            self._stat_cards[key] = card

        layout.addLayout(stats_grid)

        # ── Live Log + Candidates ────────────────────────────────────────
        _group_ss = """
            QGroupBox {
                font-size: 13px;
                font-weight: 600;
                color: #c8d0e0;
                border: 1px solid rgba(48, 54, 61, 0.4);
                border-radius: 8px;
                padding-top: 24px;
                margin-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }
        """
        _mono_ss = """
            QTextEdit {
                background: #0d1117;
                border: none;
                color: #c9d1d9;
                font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
                font-size: 11px;
            }
        """

        # Live log (the main feedback -- reads streaming log file)
        log_group = QGroupBox("Live Output")
        log_group.setStyleSheet(_group_ss)
        log_layout = QVBoxLayout(log_group)

        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setStyleSheet(_mono_ss)
        self._log_text.setMinimumHeight(140)
        log_layout.addWidget(self._log_text)
        self._log_file_pos = 0  # Track read position

        layout.addWidget(log_group)

        # Bottom row: candidates + right sidebar
        bottom = QHBoxLayout()

        # Top candidates
        candidates_group = QGroupBox("Top Candidates")
        candidates_group.setStyleSheet(_group_ss)
        cand_layout = QVBoxLayout(candidates_group)

        self._candidates_text = QTextEdit()
        self._candidates_text.setReadOnly(True)
        self._candidates_text.setStyleSheet(_mono_ss)
        cand_layout.addWidget(self._candidates_text)
        bottom.addWidget(candidates_group, 2)

        # Reports + Corrections
        right_col = QVBoxLayout()

        # Reports section
        reports_group = QGroupBox("Reports")
        reports_group.setStyleSheet(_group_ss)
        rep_layout = QVBoxLayout(reports_group)

        self._reports_list = QTextEdit()
        self._reports_list.setReadOnly(True)
        self._reports_list.setMaximumHeight(120)
        self._reports_list.setStyleSheet(_mono_ss)
        rep_layout.addWidget(self._reports_list)

        open_reports_btn = QPushButton("Open Reports Folder")
        open_reports_btn.setStyleSheet("""
            QPushButton {
                background: rgba(88, 166, 255, 0.1);
                border: 1px solid rgba(88, 166, 255, 0.3);
                border-radius: 4px;
                padding: 6px 12px;
                color: #58a6ff;
                font-size: 12px;
            }
            QPushButton:hover {
                background: rgba(88, 166, 255, 0.2);
            }
        """)
        open_reports_btn.clicked.connect(self._open_reports_folder)
        rep_layout.addWidget(open_reports_btn)

        right_col.addWidget(reports_group)

        # Self-correction log
        corrections_group = QGroupBox("Self-Corrections")
        corrections_group.setStyleSheet(_group_ss)
        corr_layout = QVBoxLayout(corrections_group)

        self._corrections_text = QTextEdit()
        self._corrections_text.setReadOnly(True)
        self._corrections_text.setStyleSheet(_mono_ss)
        corr_layout.addWidget(self._corrections_text)

        right_col.addWidget(corrections_group)

        bottom.addLayout(right_col, 1)

        layout.addLayout(bottom, 1)

    def _create_stat_card(self, label: str, default: str) -> QFrame:
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background: rgba(22, 27, 34, 0.5);
                border: 1px solid rgba(48, 54, 61, 0.4);
                border-radius: 6px;
            }
        """)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(12, 10, 12, 10)
        card_layout.setSpacing(4)

        value_label = QLabel(default)
        value_label.setObjectName("value")
        value_label.setAlignment(Qt.AlignCenter)
        value_label.setStyleSheet(
            "font-size: 22px; font-weight: 700; color: #58a6ff; border: none;"
        )
        card_layout.addWidget(value_label)

        name_label = QLabel(label)
        name_label.setAlignment(Qt.AlignCenter)
        name_label.setStyleSheet(
            "font-size: 10px; color: #8b949e; text-transform: uppercase; "
            "letter-spacing: 0.5px; border: none;"
        )
        card_layout.addWidget(name_label)

        return card

    def _update_stat(self, key: str, value: str, color: str = "#58a6ff"):
        card = self._stat_cards.get(key)
        if card:
            val = card.findChild(QLabel, "value")
            if val:
                val.setText(str(value))
                val.setStyleSheet(
                    f"font-size: 22px; font-weight: 700; color: {color}; border: none;"
                )

    # ── Start/Stop ───────────────────────────────────────────────────────

    def _toggle_streaming(self):
        if self._process and self._process.poll() is None:
            self._stop_streaming()
        else:
            self._start_streaming()

    def _start_streaming(self, reset: bool = False):
        days = self._days_spin.value()
        mode = self._mode_combo.currentText().lower()

        cmd = [
            sys.executable,
            "scripts/streaming_discovery.py",
            "--days", str(days),
        ]
        if mode == "aggressive":
            cmd.append("--aggressive")
        elif mode == "turbo":
            cmd.append("--turbo")
        if reset:
            cmd.append("--reset")

        project_root = Path(__file__).parent.parent

        # Reset log file position so we read from the start of new output
        self._log_file_pos = 0

        self._process = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            stdout=subprocess.DEVNULL,  # Output goes to log file
            stderr=subprocess.DEVNULL,
        )
        _active_processes.append(self._process)

        self._start_btn.setText("Stop Streaming")
        self._start_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(218, 54, 51, 0.9), stop:1 rgba(248, 81, 73, 0.9));
                border: none;
                border-radius: 6px;
                padding: 8px 20px;
                color: white;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: rgba(248, 81, 73, 1);
            }
        """)
        self._status_dot.set_active(True)
        self._status_label.setText(f"Running ({mode}, {days} days)")
        self._status_label.setStyleSheet(
            "font-size: 12px; color: #3fb950; margin-left: 22px;"
        )
        self._days_spin.setEnabled(False)
        self._mode_combo.setEnabled(False)

    def _stop_streaming(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()

        self._process = None
        self._start_btn.setText("Start Streaming")
        self._start_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(35, 134, 54, 0.9), stop:1 rgba(46, 160, 67, 0.9));
                border: none;
                border-radius: 6px;
                padding: 8px 20px;
                color: white;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: rgba(46, 160, 67, 1);
            }
        """)
        self._status_dot.set_active(False)
        self._status_label.setText("Stopped")
        self._status_label.setStyleSheet(
            "font-size: 12px; color: #f0883e; margin-left: 22px;"
        )
        self._days_spin.setEnabled(True)
        self._mode_combo.setEnabled(True)

    # ── Refresh from state files ─────────────────────────────────────────

    _last_ext_mtime: float = 0.0

    def _is_externally_running(self) -> bool:
        """Check if a streaming process is running externally (not started by this UI).
        Verifies the file is *actively* being updated (mtime changed between checks)."""
        try:
            if _DISCOVERY_STATE.exists():
                import time as _time
                current_mtime = _DISCOVERY_STATE.stat().st_mtime
                age = _time.time() - current_mtime
                # File must be fresh AND its mtime must have changed since last check
                if age < 60 and self._last_ext_mtime > 0 and current_mtime != self._last_ext_mtime:
                    self._last_ext_mtime = current_mtime
                    return True
                self._last_ext_mtime = current_mtime
        except Exception:
            pass
        return False

    def _refresh(self):
        """Refresh stats from state files and live log."""
        # Check if our own process is still running
        if self._process and self._process.poll() is not None:
            self._stop_streaming()

        # Detect externally running process (started from terminal)
        external_running = self._is_externally_running()
        if external_running and not (self._process and self._process.poll() is None):
            # External process detected -- show running state
            self._start_btn.setText("External Process Running")
            self._start_btn.setEnabled(False)
            self._status_dot.set_active(True)
            self._days_spin.setEnabled(False)
            self._mode_combo.setEnabled(False)
        elif not external_running and not (self._process and self._process.poll() is None):
            self._start_btn.setText("Start Streaming")
            self._start_btn.setEnabled(True)
            self._days_spin.setEnabled(True)
            self._mode_combo.setEnabled(True)

        # Read streaming state
        streaming = self._read_json(_STREAMING_STATE)
        discovery = self._read_json(_DISCOVERY_STATE)
        candidates = self._read_json(_CANDIDATES_FILE)
        if isinstance(candidates, dict):
            candidates = []

        # Update progress bar -- use target from UI if state has 0
        current_day = streaming.get("current_day", 0)
        target_days = streaming.get("target_days", 0) or self._days_spin.value()
        self._progress_bar.setMaximum(max(target_days, 1))
        self._progress_bar.setValue(current_day)
        self._progress_bar.setFormat(f"Day {current_day} of {target_days}")

        # Update stats from discovery state
        total_images = discovery.get("total_analyzed", 0)
        total_anomalies = discovery.get("anomalies_found", 0)
        threshold = discovery.get("current_threshold", 3.0)
        total_corrections = streaming.get("total_corrections", 0)
        strategy = streaming.get("current_strategy", "normal")

        anomaly_rate = (
            f"{total_anomalies / total_images * 100:.2f}%"
            if total_images > 0
            else "0.00%"
        )

        # YOLO stats
        yolo_confirmed = streaming.get("yolo_confirmations", 0)
        yolo_scanned = streaming.get("yolo_images_scanned", 0)

        self._update_stat("images", f"{total_images:,}")
        self._update_stat(
            "anomalies", str(total_anomalies),
            "#3fb950" if total_anomalies > 0 else "#58a6ff",
        )
        self._update_stat(
            "yolo_detected", str(yolo_confirmed),
            "#da3633" if yolo_confirmed > 0 else "#8b949e",
        )
        self._update_stat("yolo_scanned", str(yolo_scanned), "#a371f7")
        self._update_stat("rate", anomaly_rate)
        self._update_stat("threshold", f"{threshold:.3f}")
        self._update_stat("corrections", str(total_corrections))
        self._update_stat("strategy", strategy.title())

        # Update status label with live info when running (own or external)
        is_running = (self._process and self._process.poll() is None) or self._is_externally_running()
        if is_running:
            cycles = discovery.get("cycles_completed", 0)
            self._status_label.setText(
                f"Running | Cycle {cycles} | "
                f"{total_images} images | {total_anomalies} anomalies | "
                f"YOLO: {yolo_confirmed}/{yolo_scanned}"
            )
            self._status_label.setStyleSheet(
                "font-size: 12px; color: #3fb950; margin-left: 22px;"
            )

        # Update candidates (show YOLO tag for confirmed)
        if candidates:
            sorted_cands = sorted(
                candidates, key=lambda c: c.get("ood_score", 0), reverse=True
            )
            lines = []
            for i, c in enumerate(sorted_cands[:8], 1):
                score = c.get("ood_score", 0)
                cls = c.get("classification", "?")[:15]
                src = c.get("source", "?")[:10]
                yolo = " [YOLO]" if c.get("yolo_confirmed") else ""
                lines.append(
                    f"  {i}. OOD={score:.4f}  {cls:<15}  [{src}]{yolo}"
                )
            self._candidates_text.setPlainText("\n".join(lines))
        else:
            self._candidates_text.setPlainText(
                "  No candidates yet.\n\n"
                "  Start streaming to begin discovery."
            )

        # Update live log from streaming log file
        self._refresh_log()

        # Update reports list
        self._refresh_reports()

        # Update corrections log
        snapshots = streaming.get("daily_snapshots", [])
        correction_lines = []
        for snap in snapshots:
            for c in snap.get("corrections_applied", []):
                correction_lines.append(f"  Day {snap.get('day', '?')}: {c}")
        if correction_lines:
            self._corrections_text.setPlainText("\n".join(correction_lines))
        else:
            self._corrections_text.setPlainText("  No corrections applied yet.")

    def _refresh_log(self):
        """Read new lines from the streaming log file and display them."""
        if not _STREAMING_LOG.exists():
            if not (self._process and self._process.poll() is None):
                self._log_text.setPlainText(
                    "  No streaming log yet. Start streaming to see live output."
                )
            return

        try:
            file_size = _STREAMING_LOG.stat().st_size
            if file_size == self._log_file_pos:
                return  # No new data

            # Read last 8KB (most recent output)
            read_from = max(0, file_size - 8192)
            with open(_STREAMING_LOG, "r", errors="replace") as f:
                f.seek(read_from)
                if read_from > 0:
                    f.readline()  # Skip partial line
                content = f.read()

            self._log_file_pos = file_size

            # Show last ~60 lines
            lines = content.strip().split("\n")
            display = "\n".join(lines[-60:])
            self._log_text.setPlainText(display)

            # Auto-scroll to bottom
            scrollbar = self._log_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
        except Exception:
            pass

    def _refresh_reports(self):
        """List available reports."""
        _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        reports = sorted(_REPORTS_DIR.glob("*.html"), reverse=True)
        if reports:
            lines = []
            for r in reports[:10]:
                lines.append(f"  {r.name}")
            self._reports_list.setPlainText("\n".join(lines))
        else:
            self._reports_list.setPlainText("  No reports generated yet.")

    def _open_reports_folder(self):
        """Open the reports folder in the system file browser."""
        _REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        # Check for latest report to open
        reports = sorted(_REPORTS_DIR.glob("*.html"), reverse=True)
        if reports:
            webbrowser.open(f"file://{reports[0]}")
        else:
            # Open folder instead
            import platform
            if platform.system() == "Darwin":
                subprocess.Popen(["open", str(_REPORTS_DIR)])
            elif platform.system() == "Linux":
                subprocess.Popen(["xdg-open", str(_REPORTS_DIR)])
            else:
                subprocess.Popen(["explorer", str(_REPORTS_DIR)])

    def _read_json(self, path: Path) -> dict:
        """Safely read a JSON file."""
        if path.exists():
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
