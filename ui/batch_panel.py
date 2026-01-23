"""
Batch Analysis Panel

Clean, modern batch analysis interface.
"""

from __future__ import annotations

from typing import List

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QProgressBar, QScrollArea, QSpacerItem, QSizePolicy,
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread

import httpx


class BatchAnalysisWorker(QThread):
    """Background worker for batch analysis."""
    
    progress = pyqtSignal(int, int, str, str)
    anomaly_found = pyqtSignal(dict)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, api: httpx.Client, image_ids: List[int], rebuild_embeddings: bool = False):
        super().__init__()
        self.api = api
        self.image_ids = image_ids
        self.rebuild_embeddings = rebuild_embeddings
        self._cancelled = False
    
    def cancel(self):
        self._cancelled = True
    
    def run(self):
        results = {
            "total": len(self.image_ids),
            "analyzed": 0,
            "anomalies": 0,
            "errors": 0,
            "anomaly_list": [],
        }
        
        if self.rebuild_embeddings:
            # Special mode: just rebuild embeddings
            for i, image_id in enumerate(self.image_ids):
                if self._cancelled:
                    break
                try:
                    self.progress.emit(i + 1, len(self.image_ids), f"Image {image_id}", "Rebuilding...")
                    self.api.post(f"/analysis/full/{image_id}")
                    results["analyzed"] += 1
                except Exception:
                    results["errors"] += 1
            self.finished.emit(results)
            return
        
        for i, image_id in enumerate(self.image_ids):
            if self._cancelled:
                break
            
            try:
                info_response = self.api.get(f"/images/{image_id}")
                filename = "Unknown"
                if info_response.status_code == 200:
                    filename = info_response.json().get("filename", "Unknown")[:30]
                
                self.progress.emit(i + 1, len(self.image_ids), filename, "Analyzing...")
                
                response = self.api.post(f"/analysis/full/{image_id}", timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    results["analyzed"] += 1
                    
                    class_label = data.get("classification", {}).get("class_label", "Unknown")
                    confidence = data.get("classification", {}).get("confidence", 0)
                    is_anomaly = data.get("anomaly", {}).get("is_anomaly", False)
                    ood_score = data.get("anomaly", {}).get("ood_score", 0)
                    
                    result_text = f"{class_label} ({confidence:.0%})"
                    if is_anomaly:
                        result_text = f"🔥 Anomaly: {result_text}"
                        results["anomalies"] += 1
                        
                        anomaly_info = {
                            "id": image_id,
                            "filename": filename,
                            "class_label": class_label,
                            "confidence": confidence,
                            "ood_score": ood_score,
                        }
                        results["anomaly_list"].append(anomaly_info)
                        self.anomaly_found.emit(anomaly_info)
                    
                    self.progress.emit(i + 1, len(self.image_ids), filename, result_text)
                else:
                    results["errors"] += 1
                    
            except Exception:
                results["errors"] += 1
        
        self.finished.emit(results)


class AnomalyCard(QFrame):
    """Compact anomaly discovery card."""
    
    clicked = pyqtSignal(int)
    
    def __init__(self, info: dict):
        super().__init__()
        self.image_id = info["id"]
        
        self.setStyleSheet("""
            QFrame {
                background: rgba(155, 124, 239, 0.08);
                border: 1px solid rgba(155, 124, 239, 0.2);
                border-radius: 10px;
            }
            QFrame:hover {
                background: rgba(155, 124, 239, 0.12);
            }
        """)
        self.setCursor(Qt.PointingHandCursor)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(12)
        
        # Icon
        icon = QLabel("⚡")
        icon.setStyleSheet("font-size: 18px;")
        layout.addWidget(icon)
        
        # Info
        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)
        
        name = QLabel(info.get("filename", "Unknown")[:35])
        name.setStyleSheet("color: #c8d0e0; font-size: 13px; font-weight: 500;")
        text_layout.addWidget(name)
        
        details = QLabel(f"{info.get('class_label', '?')} · Score: {info.get('ood_score', 0):.1f}")
        details.setStyleSheet("color: #7a8599; font-size: 11px;")
        text_layout.addWidget(details)
        
        layout.addLayout(text_layout)
        layout.addStretch()
        
        # View button
        view_btn = QPushButton("View")
        view_btn.setStyleSheet("""
            QPushButton {
                background: rgba(155, 124, 239, 0.15);
                border: none;
                color: #a78bfa;
                padding: 6px 12px;
                border-radius: 6px;
                font-size: 12px;
            }
            QPushButton:hover { background: rgba(155, 124, 239, 0.25); }
        """)
        view_btn.clicked.connect(lambda: self.clicked.emit(self.image_id))
        layout.addWidget(view_btn)


class BatchPanel(QWidget):
    """Clean batch analysis panel."""
    
    view_image = pyqtSignal(int)
    refresh_gallery = pyqtSignal()
    
    def __init__(self, api: httpx.Client):
        super().__init__()
        self.api = api
        self.worker = None
        self.anomaly_cards = []
        
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(24)
        
        # Header
        header_layout = QVBoxLayout()
        header_layout.setSpacing(4)
        
        title = QLabel("Batch Analysis")
        title.setStyleSheet("font-size: 22px; font-weight: 500; color: #c8d0e0;")
        header_layout.addWidget(title)
        
        subtitle = QLabel("Analyze your entire collection and discover anomalies")
        subtitle.setStyleSheet("color: #7a8599; font-size: 13px;")
        header_layout.addWidget(subtitle)
        
        layout.addLayout(header_layout)
        
        # Quick Actions (cards side by side)
        actions_layout = QHBoxLayout()
        actions_layout.setSpacing(16)
        
        # Analyze New card
        new_card = self._create_action_card(
            "🔍", "Analyze New",
            "Process images that haven't been analyzed yet",
            "#34d399", self._analyze_unanalyzed
        )
        actions_layout.addWidget(new_card)
        
        # Re-analyze All card
        all_card = self._create_action_card(
            "🔄", "Re-analyze All", 
            "Run fresh analysis on entire collection",
            "#60a5fa", self._analyze_all
        )
        actions_layout.addWidget(all_card)
        
        # Rebuild Similarity card
        rebuild_card = self._create_action_card(
            "🔗", "Rebuild Similarity",
            "Update 'Find Similar' index after model changes",
            "#a78bfa", self._rebuild_embeddings
        )
        actions_layout.addWidget(rebuild_card)
        
        layout.addLayout(actions_layout)
        
        # Progress section
        progress_frame = QFrame()
        progress_frame.setStyleSheet("""
            QFrame {
                background: rgba(14, 18, 24, 0.4);
                border: 1px solid rgba(30, 40, 55, 0.3);
                border-radius: 12px;
            }
        """)
        progress_layout = QVBoxLayout(progress_frame)
        progress_layout.setContentsMargins(18, 16, 18, 16)
        progress_layout.setSpacing(10)
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #c8d0e0; font-size: 13px; font-weight: 500;")
        progress_layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background: rgba(30, 40, 55, 0.4);
                border: none;
                border-radius: 3px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #6366f1, stop:1 #a78bfa);
                border-radius: 3px;
            }
        """)
        progress_layout.addWidget(self.progress_bar)
        
        self.detail_label = QLabel("")
        self.detail_label.setStyleSheet("color: #4a5568; font-size: 11px;")
        progress_layout.addWidget(self.detail_label)
        
        # Control buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background: rgba(248, 113, 113, 0.15);
                border: none;
                color: #f87171;
                padding: 8px 16px;
                border-radius: 8px;
                font-size: 12px;
            }
            QPushButton:hover { background: rgba(248, 113, 113, 0.25); }
            QPushButton:disabled { opacity: 0.4; color: #4a5568; }
        """)
        self.stop_btn.clicked.connect(self._stop_analysis)
        btn_row.addWidget(self.stop_btn)
        
        btn_row.addStretch()
        progress_layout.addLayout(btn_row)
        
        layout.addWidget(progress_frame)
        
        # Anomalies section
        anomalies_header = QLabel("Discoveries")
        anomalies_header.setStyleSheet("font-size: 16px; font-weight: 500; color: #c8d0e0;")
        layout.addWidget(anomalies_header)
        
        self.no_anomalies_label = QLabel("No anomalies found yet. Run an analysis to discover unusual images.")
        self.no_anomalies_label.setStyleSheet("color: #4a5568; font-size: 12px;")
        layout.addWidget(self.no_anomalies_label)
        
        # Anomalies scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        
        self.anomalies_container = QWidget()
        self.anomalies_container.setStyleSheet("background: transparent;")
        self.anomalies_layout = QVBoxLayout(self.anomalies_container)
        self.anomalies_layout.setContentsMargins(0, 0, 0, 0)
        self.anomalies_layout.setSpacing(8)
        self.anomalies_layout.addStretch()
        
        scroll.setWidget(self.anomalies_container)
        layout.addWidget(scroll, 1)
    
    def _create_action_card(self, icon: str, title: str, desc: str, color: str, callback) -> QFrame:
        """Create a clickable action card."""
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background: rgba(14, 18, 24, 0.5);
                border: 1px solid rgba(30, 40, 55, 0.4);
                border-radius: 12px;
            }}
            QFrame:hover {{
                border-color: {color};
                background: rgba(14, 18, 24, 0.7);
            }}
        """)
        card.setCursor(Qt.PointingHandCursor)
        
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(6)
        
        # Icon + Title row
        header = QHBoxLayout()
        icon_label = QLabel(icon)
        icon_label.setStyleSheet("font-size: 20px;")
        header.addWidget(icon_label)
        
        title_label = QLabel(title)
        title_label.setStyleSheet(f"color: {color}; font-size: 14px; font-weight: 500;")
        header.addWidget(title_label)
        header.addStretch()
        layout.addLayout(header)
        
        desc_label = QLabel(desc)
        desc_label.setStyleSheet("color: #7a8599; font-size: 11px;")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        # Make clickable
        card.mousePressEvent = lambda e: callback()
        
        return card
    
    def _analyze_unanalyzed(self):
        """Analyze only unanalyzed images."""
        try:
            response = self.api.get("/images", params={"limit": 2000})
            if response.status_code != 200:
                self.status_label.setText(f"Error: API returned {response.status_code}")
                return
            
            images = response.json()
            unanalyzed = [img["id"] for img in images if not img.get("class_label")]
            
            if not unanalyzed:
                self.status_label.setText("✓ All images already analyzed")
                self.detail_label.setText(f"{len(images)} total images in collection")
                return
            
            self._start_analysis(unanalyzed)
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)[:50]}")
    
    def _analyze_all(self):
        """Re-analyze all images."""
        try:
            response = self.api.get("/images", params={"limit": 2000})
            if response.status_code != 200:
                self.status_label.setText(f"Error: API returned {response.status_code}")
                return
            
            images = response.json()
            if not images:
                self.status_label.setText("No images to analyze")
                return
            
            self._start_analysis([img["id"] for img in images])
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)[:50]}")
    
    def _rebuild_embeddings(self):
        """Rebuild similarity embeddings."""
        try:
            response = self.api.get("/images", params={"limit": 2000})
            if response.status_code != 200:
                self.status_label.setText(f"Error: API returned {response.status_code}")
                return
            
            images = response.json()
            if not images:
                self.status_label.setText("No images to process")
                return
            
            # Note: Embeddings will be rebuilt during full analysis
            # The /analysis/embeddings/clear endpoint doesn't exist, skip it
            
            self._start_analysis([img["id"] for img in images], rebuild_mode=True)
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)[:50]}")
    
    def _start_analysis(self, image_ids: List[int], rebuild_mode: bool = False):
        """Start batch analysis."""
        if self.worker and self.worker.isRunning():
            return
        
        self.worker = BatchAnalysisWorker(self.api, image_ids, rebuild_mode)
        self.worker.progress.connect(self._on_progress)
        self.worker.anomaly_found.connect(self._on_anomaly)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()
        
        self.stop_btn.setEnabled(True)
        mode = "Rebuilding" if rebuild_mode else "Analyzing"
        self.status_label.setText(f"{mode}...")
    
    def _on_progress(self, current: int, total: int, filename: str, result: str):
        percent = int(current / total * 100) if total > 0 else 0
        self.progress_bar.setValue(percent)
        self.status_label.setText(f"{current}/{total}: {filename}")
        self.detail_label.setText(result)
    
    def _on_anomaly(self, info: dict):
        self.no_anomalies_label.hide()
        
        card = AnomalyCard(info)
        card.clicked.connect(self.view_image.emit)
        self.anomaly_cards.append(card)
        
        # Insert before stretch
        self.anomalies_layout.insertWidget(len(self.anomaly_cards) - 1, card)
    
    def _on_finished(self, results: dict):
        self._cleanup_worker()
        
        analyzed = results.get("analyzed", 0)
        anomalies = results.get("anomalies", 0)
        errors = results.get("errors", 0)
        
        self.status_label.setText(f"Complete: {analyzed} analyzed, {anomalies} anomalies")
        if errors:
            self.detail_label.setText(f"{errors} errors")
        else:
            self.detail_label.setText("Success!")
        
        self.refresh_gallery.emit()
    
    def _stop_analysis(self):
        if self.worker:
            self.worker.cancel()
            self.status_label.setText("Stopping...")
    
    def _cleanup_worker(self):
        self.stop_btn.setEnabled(False)
        if self.worker:
            self.worker.wait(2000)
            self.worker.deleteLater()
            self.worker = None
