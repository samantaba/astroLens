"""
Verification Panel - Cross-Reference Anomalies Against Astronomical Catalogs

A premium interface for:
- Cross-referencing anomalies against SIMBAD and NED
- Viewing known vs unknown objects
- Human verification workflow
- OOD detector calibration
- False positive/negative tracking
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QScrollArea, QGridLayout, QProgressBar, QTextEdit,
    QSplitter, QGroupBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QComboBox, QSpinBox, QMessageBox,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QUrl
from PyQt5.QtGui import QFont, QColor, QDesktopServices

import httpx

# Path to artifacts
ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "astrolens_artifacts"
CROSS_REF_RESULTS = ARTIFACTS_DIR / "data" / "cross_reference_results.json"


class CrossRefWorker(QThread):
    """Background worker for cross-reference operations."""
    
    progress = pyqtSignal(int, int, dict)  # current, total, result
    finished = pyqtSignal(dict)  # stats
    error = pyqtSignal(str)
    log = pyqtSignal(str)
    
    def __init__(self, api_base: str, limit: int = 100, radius_arcsec: int = 60):
        super().__init__()
        self.api_base = api_base
        self.limit = limit
        self.radius_arcsec = radius_arcsec
        self._stop = False
    
    def stop(self):
        self._stop = True
    
    def run(self):
        try:
            # Get anomaly candidates from API
            self.log.emit("Fetching anomaly candidates...")
            response = httpx.get(
                f"{self.api_base}/candidates",
                params={"limit": self.limit},
                timeout=30.0,
            )
            response.raise_for_status()
            anomalies = response.json()
            
            if not anomalies:
                self.log.emit("No anomaly candidates found")
                self.finished.emit({"total": 0, "known": 0, "unknown": 0})
                return
            
            self.log.emit(f"Found {len(anomalies)} candidates to check")
            self.log.emit(f"Using search radius: {self.radius_arcsec}″")
            
            # Use CatalogCrossReference directly with custom radius
            from catalog.cross_reference import CatalogCrossReference
            xref = CatalogCrossReference(search_radius_arcsec=self.radius_arcsec)
            
            stats = {"total": len(anomalies), "known": 0, "unknown": 0, "errors": 0, "published": 0}
            
            for i, anomaly in enumerate(anomalies):
                if self._stop:
                    self.log.emit("Stopped by user")
                    break
                
                image_id = anomaly.get("id")
                filepath = anomaly.get("filepath", "")
                
                try:
                    # Cross-reference directly with custom radius
                    result = xref.cross_reference(
                        image_id=image_id,
                        image_path=filepath,
                        force=True,  # Re-query with new radius
                    )
                    
                    # Convert to dict for signal
                    result_dict = {
                        "image_id": result.image_id,
                        "is_known": result.is_known,
                        "is_published": result.is_published,
                        "status": result.status,
                        "total_matches": len(result.matches),
                        "primary_match": None,
                    }
                    
                    if result.primary_match:
                        result_dict["primary_match"] = {
                            "object_name": result.primary_match.object_name,
                            "object_type": result.primary_match.object_type,
                            "distance_arcsec": result.primary_match.distance_arcsec,
                        }
                    
                    if result.is_known:
                        stats["known"] += 1
                    else:
                        stats["unknown"] += 1
                    
                    if result.is_published:
                        stats["published"] += 1
                    
                    self.progress.emit(i + 1, len(anomalies), result_dict)
                    
                    status_icon = "✓ KNOWN" if result.is_known else "★ UNKNOWN"
                    match_info = ""
                    if result.primary_match:
                        match_info = f" → {result.primary_match.object_name}"
                    self.log.emit(
                        f"[{i+1}/{len(anomalies)}] {status_icon}: "
                        f"{Path(filepath).name}{match_info}"
                    )
                    
                    # Small delay to not hammer servers
                    import time
                    time.sleep(0.3)
                    
                except Exception as e:
                    stats["errors"] += 1
                    self.log.emit(f"Error checking {image_id}: {e}")
            
            self.finished.emit(stats)
            
        except Exception as e:
            self.error.emit(str(e))


class CalibrateWorker(QThread):
    """Background worker for OOD calibration based on score distribution."""
    
    log = pyqtSignal(str)
    finished = pyqtSignal(bool, str)  # success, message
    
    def __init__(self, api_base: str = "http://localhost:8000"):
        super().__init__()
        self.api_base = api_base
    
    def run(self):
        try:
            import numpy as np
            
            self.log.emit("Starting OOD threshold calibration...")
            
            # Get all images with OOD scores
            response = httpx.get(
                f"{self.api_base}/images",
                params={"limit": 5000},
                timeout=30.0,
            )
            response.raise_for_status()
            images = response.json()
            
            # Extract scores
            scores = [img.get("ood_score", 0) or 0 for img in images if img.get("ood_score")]
            
            if not scores:
                self.finished.emit(False, "No OOD scores found - run Batch Analyze first")
                return
            
            self.log.emit(f"Found {len(scores)} images with OOD scores")
            
            # Calculate statistics
            scores_arr = np.array(scores)
            mean_score = np.mean(scores_arr)
            std_score = np.std(scores_arr)
            p90 = np.percentile(scores_arr, 90)
            p95 = np.percentile(scores_arr, 95)
            
            self.log.emit(f"Score statistics:")
            self.log.emit(f"  Mean: {mean_score:.3f}")
            self.log.emit(f"  Std: {std_score:.3f}")
            self.log.emit(f"  Range: [{min(scores):.3f}, {max(scores):.3f}]")
            self.log.emit(f"  90th percentile: {p90:.3f}")
            self.log.emit(f"  95th percentile: {p95:.3f}")
            
            # Suggest optimal threshold (90th percentile = top 10% are anomalies)
            optimal_threshold = p90
            
            self.log.emit(f"\nRecommended threshold: {optimal_threshold:.3f}")
            self.log.emit(f"This would mark ~{int(len(scores) * 0.1)} images as anomalies")
            
            # Count current anomalies
            current_anomalies = sum(1 for img in images if img.get("is_anomaly"))
            self.log.emit(f"Current anomalies: {current_anomalies}")
            
            self.finished.emit(True, f"Calibration complete!\nRecommended threshold: {optimal_threshold:.3f}\nCurrent anomalies: {current_anomalies}")
                
        except Exception as e:
            self.finished.emit(False, str(e))


class StatCard(QFrame):
    """A stat display card."""
    
    def __init__(self, title: str, value: str = "0", color: str = "#5b8def"):
        super().__init__()
        self.setStyleSheet(f"""
            QFrame {{
                background: rgba(20, 27, 38, 0.6);
                border: 1px solid rgba(40, 50, 70, 0.4);
                border-radius: 12px;
                padding: 16px;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        
        self.value_label = QLabel(value)
        self.value_label.setStyleSheet(f"""
            font-size: 32px;
            font-weight: 300;
            color: {color};
        """)
        layout.addWidget(self.value_label)
        
        title_label = QLabel(title.upper())
        title_label.setStyleSheet("font-size: 10px; color: #4a5568; letter-spacing: 1px;")
        layout.addWidget(title_label)
    
    def set_value(self, value: str):
        self.value_label.setText(value)


class ResultRow(QFrame):
    """A single cross-reference result row."""
    
    verify_clicked = pyqtSignal(int, str)  # image_id, label
    view_clicked = pyqtSignal(str)  # url
    image_clicked = pyqtSignal(str)  # filepath
    
    def __init__(self, result: dict):
        super().__init__()
        self.result = result
        self.image_id = result.get("image_id", 0)
        self.filepath = result.get("image_path", "")
        
        is_known = result.get("is_known", False)
        
        self.setStyleSheet(f"""
            QFrame {{
                background: rgba({'30, 40, 55' if is_known else '25, 50, 40'}, 0.5);
                border: 1px solid rgba({'60, 70, 90' if is_known else '50, 150, 100'}, 0.3);
                border-radius: 8px;
                padding: 8px 12px;
                margin: 2px 0;
            }}
            QFrame:hover {{
                background: rgba({'40, 50, 65' if is_known else '35, 60, 50'}, 0.7);
            }}
        """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(12)
        
        # Status icon
        status_icon = QLabel("✓" if is_known else "★")
        status_icon.setStyleSheet(f"""
            font-size: 16px;
            color: {'#7a8599' if is_known else '#34d399'};
        """)
        status_icon.setFixedWidth(24)
        layout.addWidget(status_icon)
        
        # Image name (clickable button to open)
        name = Path(self.filepath).name if self.filepath else f"Image {self.image_id}"
        name_btn = QPushButton(name)
        name_btn.setToolTip(f"Click to open: {self.filepath}" if self.filepath else "No file path")
        name_btn.setCursor(Qt.PointingHandCursor)
        name_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                color: #c8d0e0;
                font-size: 12px;
                text-align: left;
                padding: 0;
            }
            QPushButton:hover {
                color: #5b8def;
                text-decoration: underline;
            }
        """)
        name_btn.clicked.connect(self._open_image)
        name_btn.setMinimumWidth(200)
        layout.addWidget(name_btn)
        
        # Status
        status = result.get("status", "unknown")
        status_label = QLabel(status.upper())
        status_label.setStyleSheet(f"""
            color: {'#7a8599' if is_known else '#34d399'};
            font-size: 11px;
            font-weight: 500;
        """)
        status_label.setFixedWidth(80)
        layout.addWidget(status_label)
        
        # Match info
        primary = result.get("primary_match")
        if primary:
            match_text = f"{primary.get('object_name', '')} ({primary.get('object_type', '')})"
            match_label = QLabel(match_text)
            match_label.setStyleSheet("color: #5b8def; font-size: 11px;")
            match_label.setMinimumWidth(200)
            layout.addWidget(match_label)
            
            # View in catalog button
            url = primary.get("url", "")
            if url:
                view_btn = QPushButton("View")
                view_btn.setStyleSheet("""
                    QPushButton {
                        background: rgba(91, 141, 239, 0.2);
                        border: 1px solid rgba(91, 141, 239, 0.4);
                        border-radius: 4px;
                        padding: 4px 8px;
                        color: #5b8def;
                        font-size: 10px;
                    }
                    QPushButton:hover {
                        background: rgba(91, 141, 239, 0.3);
                    }
                """)
                view_btn.clicked.connect(lambda: self.view_clicked.emit(url))
                layout.addWidget(view_btn)
        else:
            no_match = QLabel("No catalog match")
            no_match.setStyleSheet("color: #34d399; font-size: 11px; font-style: italic;")
            layout.addWidget(no_match)
        
        layout.addStretch()
        
        # Human verification buttons
        human_label = result.get("human_label", "")
        if human_label:
            verified_label = QLabel(f"✓ {human_label.replace('_', ' ').title()}")
            verified_label.setStyleSheet("color: #a78bfa; font-size: 10px;")
            layout.addWidget(verified_label)
        else:
            # Verification buttons
            tp_btn = QPushButton("✓ TP")
            tp_btn.setToolTip("Mark as True Positive (real anomaly)")
            tp_btn.setStyleSheet("""
                QPushButton {
                    background: rgba(52, 211, 153, 0.2);
                    border: 1px solid rgba(52, 211, 153, 0.4);
                    border-radius: 4px;
                    padding: 3px 6px;
                    color: #34d399;
                    font-size: 10px;
                }
                QPushButton:hover { background: rgba(52, 211, 153, 0.3); }
            """)
            tp_btn.clicked.connect(lambda: self.verify_clicked.emit(self.image_id, "true_positive"))
            layout.addWidget(tp_btn)
            
            fp_btn = QPushButton("✗ FP")
            fp_btn.setToolTip("Mark as False Positive (not actually anomaly)")
            fp_btn.setStyleSheet("""
                QPushButton {
                    background: rgba(248, 113, 113, 0.2);
                    border: 1px solid rgba(248, 113, 113, 0.4);
                    border-radius: 4px;
                    padding: 3px 6px;
                    color: #f87171;
                    font-size: 10px;
                }
                QPushButton:hover { background: rgba(248, 113, 113, 0.3); }
            """)
            fp_btn.clicked.connect(lambda: self.verify_clicked.emit(self.image_id, "false_positive"))
            layout.addWidget(fp_btn)
    
    def _open_image(self):
        """Open the image file in the default viewer."""
        import subprocess
        import platform
        
        if not self.filepath:
            QMessageBox.warning(None, "No File", "No file path available for this image")
            return
            
        filepath = Path(self.filepath)
        if not filepath.exists():
            QMessageBox.warning(None, "File Not Found", f"File not found:\n{self.filepath}")
            return
        
        try:
            if platform.system() == "Darwin":  # macOS
                subprocess.run(["open", str(filepath)], check=True)
            elif platform.system() == "Windows":
                subprocess.run(["start", "", str(filepath)], shell=True, check=True)
            else:  # Linux
                subprocess.run(["xdg-open", str(filepath)], check=True)
        except Exception as e:
            QMessageBox.warning(None, "Error", f"Could not open image:\n{e}")


class VerificationPanel(QWidget):
    """
    Premium verification panel for cross-referencing anomalies.
    """
    
    def __init__(self, api_base: str = "http://localhost:8000"):
        super().__init__()
        self.api_base = api_base
        self.worker = None
        self.calibrate_worker = None
        self.results: List[dict] = []
        self._auto_started = False
        
        self._setup_ui()
        self._load_existing_results()
        self._setup_refresh_timer()
        
        # Schedule auto-configuration check
        QTimer.singleShot(2000, self._auto_configure)
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        
        # Header
        header = self._create_header()
        layout.addLayout(header)
        
        # Stats cards
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(16)
        
        self.stat_total = StatCard("Total Checked", "0", "#7a8599")
        self.stat_known = StatCard("Known Objects", "0", "#5b8def")
        self.stat_unknown = StatCard("Unknown", "0", "#34d399")
        self.stat_published = StatCard("Published", "0", "#fbbf24")
        self.stat_verified = StatCard("Verified", "0", "#a78bfa")
        
        stats_layout.addWidget(self.stat_total)
        stats_layout.addWidget(self.stat_known)
        stats_layout.addWidget(self.stat_unknown)
        stats_layout.addWidget(self.stat_published)
        stats_layout.addWidget(self.stat_verified)
        
        layout.addLayout(stats_layout)
        
        # Main content splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background: rgba(40, 50, 70, 0.3);
                width: 2px;
            }
        """)
        
        # Left side - Results list
        left_panel = self._create_results_panel()
        splitter.addWidget(left_panel)
        
        # Right side - Log
        right_panel = self._create_log_panel()
        splitter.addWidget(right_panel)
        
        splitter.setSizes([600, 300])
        layout.addWidget(splitter, 1)
    
    def _create_header(self) -> QHBoxLayout:
        header = QHBoxLayout()
        
        # Title section
        title_section = QVBoxLayout()
        title_section.setSpacing(4)
        
        title = QLabel("Catalog Verification")
        title.setStyleSheet("""
            font-size: 24px;
            font-weight: 300;
            color: #c8d0e0;
        """)
        title_section.addWidget(title)
        
        subtitle = QLabel("Verify galaxy anomalies: Morphology analysis & catalog cross-reference")
        subtitle.setStyleSheet("font-size: 12px; color: #4a5568;")
        title_section.addWidget(subtitle)
        
        header.addLayout(title_section)
        header.addStretch()
        
        # Controls
        controls = QHBoxLayout()
        controls.setSpacing(12)
        
        # Search radius selector
        radius_label = QLabel("Radius:")
        radius_label.setStyleSheet("color: #7a8599; font-size: 12px;")
        radius_label.setToolTip("Search radius in arcseconds (larger = more matches)")
        controls.addWidget(radius_label)
        
        self.radius_spin = QSpinBox()
        self.radius_spin.setRange(10, 300)
        self.radius_spin.setValue(120)  # 120 arcsec = 2 arcmin, good coverage for galaxies
        self.radius_spin.setSuffix("″")  # arcsecond symbol
        self.radius_spin.setToolTip("120″ = 2 arcmin (good for galaxies)\n60″ = 1 arcmin (for compact objects)")
        self.radius_spin.setStyleSheet("""
            QSpinBox {
                background: rgba(30, 40, 55, 0.8);
                border: 1px solid rgba(60, 70, 90, 0.5);
                border-radius: 6px;
                padding: 6px 8px;
                color: #c8d0e0;
                font-size: 12px;
            }
        """)
        controls.addWidget(self.radius_spin)
        
        # Limit selector
        limit_label = QLabel("Limit:")
        limit_label.setStyleSheet("color: #7a8599; font-size: 12px;")
        controls.addWidget(limit_label)
        
        self.limit_spin = QSpinBox()
        self.limit_spin.setRange(10, 5000)
        self.limit_spin.setValue(500)  # Higher default to test more candidates
        self.limit_spin.setStyleSheet("""
            QSpinBox {
                background: rgba(30, 40, 55, 0.8);
                border: 1px solid rgba(60, 70, 90, 0.5);
                border-radius: 6px;
                padding: 6px 12px;
                color: #c8d0e0;
                font-size: 12px;
            }
        """)
        controls.addWidget(self.limit_spin)
        
        # Cross-reference button
        self.crossref_btn = QPushButton("🔍 Cross-Reference All")
        self.crossref_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(91, 141, 239, 0.9),
                    stop:1 rgba(139, 92, 246, 0.9));
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                color: white;
                font-weight: 500;
                font-size: 13px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(91, 141, 239, 1.0),
                    stop:1 rgba(139, 92, 246, 1.0));
            }
            QPushButton:disabled {
                background: rgba(60, 70, 90, 0.5);
                color: #4a5568;
            }
        """)
        self.crossref_btn.clicked.connect(self._start_crossref)
        controls.addWidget(self.crossref_btn)
        
        # Stop button
        self.stop_btn = QPushButton("⬛ Stop")
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background: rgba(248, 113, 113, 0.2);
                border: 1px solid rgba(248, 113, 113, 0.4);
                border-radius: 8px;
                padding: 10px 16px;
                color: #f87171;
                font-weight: 500;
            }
            QPushButton:hover {
                background: rgba(248, 113, 113, 0.3);
            }
        """)
        self.stop_btn.clicked.connect(self._stop_crossref)
        self.stop_btn.setEnabled(False)
        controls.addWidget(self.stop_btn)
        
        # Calibrate button
        self.calibrate_btn = QPushButton("🔧 Calibrate OOD")
        self.calibrate_btn.setToolTip("Calibrate OOD detector using existing images")
        self.calibrate_btn.setStyleSheet("""
            QPushButton {
                background: rgba(251, 191, 36, 0.2);
                border: 1px solid rgba(251, 191, 36, 0.4);
                border-radius: 8px;
                padding: 10px 16px;
                color: #fbbf24;
                font-weight: 500;
            }
            QPushButton:hover {
                background: rgba(251, 191, 36, 0.3);
            }
        """)
        self.calibrate_btn.clicked.connect(self._start_calibration)
        controls.addWidget(self.calibrate_btn)
        
        # YOLO Verify button - Hidden for Galaxy mode (enable for Transient mode)
        # self.yolo_btn = QPushButton("🎯 YOLO Verify")
        # self.yolo_btn.setToolTip("Run YOLO transient detector on anomaly candidates")
        # self.yolo_btn.clicked.connect(self._run_yolo_verification)
        # controls.addWidget(self.yolo_btn)
        
        # Morphology Analysis button (Galaxy mode)
        self.morph_btn = QPushButton("🌀 Analyze Morphology")
        self.morph_btn.setToolTip("Calculate galaxy morphology features (Asymmetry, Concentration, Clumpiness)")
        self.morph_btn.setCursor(Qt.PointingHandCursor)
        self.morph_btn.setStyleSheet("""
            QPushButton {
                background: rgba(52, 211, 153, 0.2);
                border: 1px solid rgba(52, 211, 153, 0.4);
                border-radius: 8px;
                padding: 10px 16px;
                color: #34d399;
                font-weight: 500;
            }
            QPushButton:hover {
                background: rgba(52, 211, 153, 0.3);
            }
        """)
        self.morph_btn.clicked.connect(self._run_morphology_analysis)
        controls.addWidget(self.morph_btn)
        
        header.addLayout(controls)
        
        return header
    
    def _create_results_panel(self) -> QFrame:
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame {
                background: rgba(15, 20, 28, 0.5);
                border: 1px solid rgba(30, 40, 55, 0.4);
                border-radius: 12px;
            }
        """)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)
        
        # Filter row
        filter_row = QHBoxLayout()
        
        filter_label = QLabel("Show:")
        filter_label.setStyleSheet("color: #7a8599; font-size: 12px;")
        filter_row.addWidget(filter_label)
        
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All", "Unknown (New Discoveries)", "Known Objects", "Needs Review"])
        self.filter_combo.setStyleSheet("""
            QComboBox {
                background: rgba(30, 40, 55, 0.8);
                border: 1px solid rgba(60, 70, 90, 0.5);
                border-radius: 6px;
                padding: 6px 12px;
                color: #c8d0e0;
                font-size: 12px;
            }
            QComboBox::drop-down {
                border: none;
            }
        """)
        self.filter_combo.currentTextChanged.connect(self._apply_filter)
        filter_row.addWidget(self.filter_combo)
        
        filter_row.addStretch()
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background: rgba(30, 40, 55, 0.8);
                border-radius: 4px;
                height: 6px;
                text-align: center;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #5b8def, stop:1 #a78bfa);
                border-radius: 4px;
            }
        """)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedWidth(200)
        filter_row.addWidget(self.progress_bar)
        
        layout.addLayout(filter_row)
        
        # Results scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                background: rgba(20, 27, 38, 0.5);
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: rgba(60, 70, 90, 0.8);
                border-radius: 4px;
                min-height: 30px;
            }
        """)
        
        self.results_container = QWidget()
        self.results_layout = QVBoxLayout(self.results_container)
        self.results_layout.setContentsMargins(0, 0, 0, 0)
        self.results_layout.setSpacing(4)
        self.results_layout.addStretch()
        
        scroll.setWidget(self.results_container)
        layout.addWidget(scroll, 1)
        
        return panel
    
    def _create_log_panel(self) -> QFrame:
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame {
                background: rgba(10, 13, 18, 0.6);
                border: 1px solid rgba(30, 40, 55, 0.4);
                border-radius: 12px;
            }
        """)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)
        
        log_header = QLabel("Activity Log")
        log_header.setStyleSheet("color: #7a8599; font-size: 12px; font-weight: 500;")
        layout.addWidget(log_header)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background: rgba(5, 8, 12, 0.5);
                border: 1px solid rgba(30, 40, 55, 0.3);
                border-radius: 8px;
                padding: 8px;
                color: #7a8599;
                font-family: 'SF Mono', 'Monaco', monospace;
                font-size: 11px;
            }
        """)
        layout.addWidget(self.log_text, 1)
        
        # Clear button
        clear_btn = QPushButton("Clear Log")
        clear_btn.setStyleSheet("""
            QPushButton {
                background: rgba(60, 70, 90, 0.3);
                border: 1px solid rgba(60, 70, 90, 0.5);
                border-radius: 6px;
                padding: 6px 12px;
                color: #7a8599;
                font-size: 11px;
            }
            QPushButton:hover {
                background: rgba(60, 70, 90, 0.5);
            }
        """)
        clear_btn.clicked.connect(self.log_text.clear)
        layout.addWidget(clear_btn)
        
        return panel
    
    def _setup_refresh_timer(self):
        """Periodically refresh stats from results file."""
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self._load_existing_results)
        self.refresh_timer.start(10000)  # Every 10 seconds
    
    def _auto_configure(self):
        """Auto-configure optimal settings and start cross-reference if needed."""
        if self._auto_started:
            return
        
        try:
            # Check if we have anomaly candidates but no results
            response = httpx.get(
                f"{self.api_base}/candidates",
                params={"limit": 5},
                timeout=5.0,
            )
            
            if response.status_code != 200:
                return
            
            candidates = response.json()
            
            # If we have candidates but no results, auto-start
            if candidates and not self.results:
                self._log("Auto-starting cross-reference for discovered anomalies...")
                self._auto_started = True
                # Use optimal settings
                self.radius_spin.setValue(60)  # 1 arcmin - good for galaxies
                self.limit_spin.setValue(min(len(candidates) + 50, 200))
                # Start after short delay
                QTimer.singleShot(500, self._start_crossref)
            
            # Check if OOD needs calibration
            self._check_ood_calibration()
            
        except Exception as e:
            # API might not be running yet
            pass
    
    def _check_ood_calibration(self):
        """Check if OOD detector needs calibration and do it automatically."""
        try:
            state_file = ARTIFACTS_DIR / "data" / "discovery_state.json"
            if not state_file.exists():
                return
            
            with open(state_file, "r") as f:
                state = json.load(f)
            
            threshold = state.get("current_threshold", 0)
            highest_ood = state.get("highest_ood_score", 0)
            
            # If threshold >> highest OOD, calibration is needed
            if highest_ood > 0 and threshold > highest_ood * 3:
                self._log(f"OOD threshold miscalibrated ({threshold:.2f} >> {highest_ood:.2f})")
                self._log("Auto-calibrating OOD detector...")
                QTimer.singleShot(1000, self._start_calibration)
                
        except Exception:
            pass
    
    def _load_existing_results(self):
        """Load existing cross-reference results from file."""
        if CROSS_REF_RESULTS.exists():
            try:
                with open(CROSS_REF_RESULTS, "r") as f:
                    data = json.load(f)
                
                self.results = data.get("results", [])
                self._update_stats()
                self._update_results_display()
                
            except Exception as e:
                self._log(f"Error loading results: {e}")
    
    def _update_stats(self):
        """Update stat cards from current results."""
        total = len(self.results)
        known = sum(1 for r in self.results if r.get("is_known"))
        unknown = sum(1 for r in self.results if not r.get("is_known"))
        published = sum(1 for r in self.results if r.get("is_published"))
        verified = sum(1 for r in self.results if r.get("human_verified"))
        
        self.stat_total.set_value(str(total))
        self.stat_known.set_value(str(known))
        self.stat_unknown.set_value(str(unknown))
        self.stat_published.set_value(str(published))
        self.stat_verified.set_value(str(verified))
    
    def _update_results_display(self):
        """Update the results list display."""
        # Clear existing
        while self.results_layout.count() > 1:  # Keep the stretch
            item = self.results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Apply filter
        filter_text = self.filter_combo.currentText()
        filtered = self.results
        
        if filter_text == "Unknown (New Discoveries)":
            filtered = [r for r in self.results if not r.get("is_known")]
        elif filter_text == "Known Objects":
            filtered = [r for r in self.results if r.get("is_known")]
        elif filter_text == "Needs Review":
            filtered = [r for r in self.results if not r.get("human_verified")]
        
        # Add rows (limit to 100 for performance)
        for result in filtered[:100]:
            row = ResultRow(result)
            row.verify_clicked.connect(self._verify_result)
            row.view_clicked.connect(self._open_url)
            self.results_layout.insertWidget(self.results_layout.count() - 1, row)
    
    def _apply_filter(self, _):
        """Apply filter to results display."""
        self._update_results_display()
    
    def _log(self, message: str):
        """Add message to log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        # Scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def _start_crossref(self):
        """Start cross-reference operation."""
        self.crossref_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        radius = self.radius_spin.value()
        self._log(f"Starting catalog cross-reference (radius={radius}″)...")
        
        self.worker = CrossRefWorker(self.api_base, self.limit_spin.value(), radius)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.log.connect(self._log)
        self.worker.start()
    
    def _stop_crossref(self):
        """Stop cross-reference operation."""
        if self.worker:
            self.worker.stop()
            self._log("Stopping...")
    
    def _on_progress(self, current: int, total: int, result: dict):
        """Handle progress update."""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
    
    def _on_finished(self, stats: dict):
        """Handle completion."""
        self.crossref_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        
        self._log(f"Completed! Total: {stats.get('total', 0)}, "
                  f"Known: {stats.get('known', 0)}, "
                  f"Unknown: {stats.get('unknown', 0)}")
        
        # Reload results
        self._load_existing_results()
    
    def _on_error(self, message: str):
        """Handle error."""
        self.crossref_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        
        self._log(f"ERROR: {message}")
    
    def _verify_result(self, image_id: int, label: str):
        """Verify a result via API."""
        try:
            response = httpx.post(
                f"{self.api_base}/crossref/{image_id}/verify",
                json={"label": label, "verified_by": "user"},
                timeout=10.0,
            )
            response.raise_for_status()
            
            self._log(f"Marked image {image_id} as {label}")
            
            # Reload to update display
            self._load_existing_results()
            
        except Exception as e:
            self._log(f"Error verifying: {e}")
    
    def _open_url(self, url: str):
        """Open URL in browser."""
        QDesktopServices.openUrl(QUrl(url))
    
    def _start_calibration(self):
        """Start OOD calibration."""
        self.calibrate_btn.setEnabled(False)
        self._log("Starting OOD threshold calibration...")
        
        self.calibrate_worker = CalibrateWorker(self.api_base)
        self.calibrate_worker.log.connect(self._log)
        self.calibrate_worker.finished.connect(self._on_calibration_finished)
        self.calibrate_worker.start()
    
    def _on_calibration_finished(self, success: bool, message: str):
        """Handle calibration completion."""
        self.calibrate_btn.setEnabled(True)
        
        if success:
            self._log(f"✓ {message}")
            QMessageBox.information(self, "Calibration Complete", message)
        else:
            self._log(f"⚠ {message}")
            QMessageBox.warning(self, "Calibration Issue", message)
    
    def _run_yolo_verification(self):
        """Run YOLO transient detector on anomaly candidates."""
        self.yolo_btn.setEnabled(False)
        self._log("Starting YOLO transient verification...")
        
        try:
            from inference.yolo_detector import YOLOTransientDetector
            
            detector = YOLOTransientDetector()
            
            if not detector.is_available():
                self._log("⚠ YOLO model not available - run Transient pipeline first")
                QMessageBox.warning(
                    self, 
                    "YOLO Not Available",
                    "YOLO model not found. Please run the Transient pipeline first to train the model."
                )
                self.yolo_btn.setEnabled(True)
                return
            
            self._log("✓ YOLO model loaded")
            
            # Get total count first
            try:
                count_resp = httpx.get(f"{self.api_base}/candidates", params={"limit": 1}, timeout=10.0)
                # Get actual total from a larger query
                all_resp = httpx.get(f"{self.api_base}/candidates", params={"limit": 5000}, timeout=30.0)
                total_anomalies = len(all_resp.json()) if all_resp.status_code == 200 else 0
                self._log(f"Total anomalies in database: {total_anomalies}")
            except:
                total_anomalies = 0
            
            # Get candidates from API
            limit = self.limit_spin.value()
            try:
                response = httpx.get(
                    f"{self.api_base}/candidates",
                    params={"limit": limit},
                    timeout=30.0,
                )
                response.raise_for_status()
                candidates = response.json()
            except Exception as e:
                self._log(f"⚠ Could not fetch candidates: {e}")
                self.yolo_btn.setEnabled(True)
                return
            
            if not candidates:
                self._log("No candidates to verify")
                self.yolo_btn.setEnabled(True)
                return
            
            self._log(f"Testing {len(candidates)}/{total_anomalies} candidates with YOLO...")
            
            confirmed = 0
            rejected = 0
            
            for i, candidate in enumerate(candidates):
                filepath = candidate.get("filepath") or candidate.get("image_path", "")
                if not filepath or not Path(filepath).exists():
                    continue
                
                result = detector.detect(filepath)
                
                if result.is_transient:
                    confirmed += 1
                    self._log(f"  ✓ #{candidate.get('id', i)}: TRANSIENT ({result.confidence_pct})")
                else:
                    rejected += 1
                    self._log(f"  ○ #{candidate.get('id', i)}: Not detected")
            
            self._log(f"\n=== YOLO Results ===")
            self._log(f"  Confirmed transients: {confirmed}")
            self._log(f"  Rejected: {rejected}")
            self._log(f"  Confirmation rate: {confirmed/(confirmed+rejected)*100:.1f}%" if (confirmed+rejected) > 0 else "  No candidates tested")
            
            QMessageBox.information(
                self,
                "YOLO Verification Complete",
                f"Confirmed: {confirmed} transients\nRejected: {rejected} artifacts"
            )
            
        except Exception as e:
            self._log(f"⚠ YOLO verification failed: {e}")
            import traceback
            traceback.print_exc()
        
        # self.yolo_btn.setEnabled(True)  # Disabled for Galaxy mode
    
    def _run_morphology_analysis(self):
        """Run galaxy morphology analysis on anomaly candidates."""
        self.morph_btn.setEnabled(False)
        self._log("Starting galaxy morphology analysis...")
        
        try:
            from features.morphology import GalaxyMorphology
            
            analyzer = GalaxyMorphology()
            self._log("✓ Morphology analyzer loaded")
            
            # Get candidates from API
            try:
                response = httpx.get(
                    f"{self.api_base}/candidates",
                    params={"limit": self.limit_spin.value()},
                    timeout=30.0,
                )
                response.raise_for_status()
                candidates = response.json()
            except Exception as e:
                self._log(f"⚠ Could not fetch candidates: {e}")
                self.morph_btn.setEnabled(True)
                return
            
            if not candidates:
                self._log("No candidates to analyze")
                self.morph_btn.setEnabled(True)
                return
            
            self._log(f"Analyzing {len(candidates)} galaxies...")
            
            irregular = 0
            merger = 0
            compact = 0
            analyzed = 0
            high_score = []
            
            for i, candidate in enumerate(candidates):
                filepath = candidate.get("filepath") or candidate.get("image_path", "")
                if not filepath or not Path(filepath).exists():
                    continue
                
                result = analyzer.analyze(filepath)
                
                if result:
                    analyzed += 1
                    
                    if result.is_irregular:
                        irregular += 1
                    if result.is_merger:
                        merger += 1
                    if result.is_compact:
                        compact += 1
                    
                    # Track high morphology scores
                    if result.morph_score > 0.5:
                        high_score.append({
                            "id": candidate.get('id'),
                            "filename": Path(filepath).name,
                            "score": result.morph_score,
                            "A": result.asymmetry,
                            "C": result.concentration,
                            "S": result.smoothness,
                        })
                
                # Progress every 50
                if (i + 1) % 50 == 0:
                    self._log(f"  Progress: {i+1}/{len(candidates)}")
            
            self._log(f"\n=== Morphology Analysis Complete ===")
            self._log(f"  Total analyzed: {analyzed}")
            self._log(f"  Irregular galaxies: {irregular} ({irregular/max(1,analyzed)*100:.1f}%)")
            self._log(f"  Merger candidates: {merger} ({merger/max(1,analyzed)*100:.1f}%)")
            self._log(f"  Compact objects: {compact} ({compact/max(1,analyzed)*100:.1f}%)")
            
            if high_score:
                self._log(f"\n=== Top Unusual Morphologies ===")
                for hs in sorted(high_score, key=lambda x: x['score'], reverse=True)[:10]:
                    self._log(f"  #{hs['id']}: score={hs['score']:.2f} A={hs['A']:.2f} C={hs['C']:.2f}")
            
            QMessageBox.information(
                self,
                "Morphology Analysis Complete",
                f"Analyzed: {analyzed} galaxies\n\n"
                f"Irregular: {irregular}\n"
                f"Merger candidates: {merger}\n"
                f"Compact: {compact}\n\n"
                f"High morphology score: {len(high_score)}"
            )
            
        except Exception as e:
            self._log(f"⚠ Morphology analysis failed: {e}")
            import traceback
            traceback.print_exc()
        
        self.morph_btn.setEnabled(True)
