"""
Viewer Panel

Detailed image view with analysis results and similar image thumbnails.
"""

from __future__ import annotations

from typing import Optional, List

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QGridLayout, QProgressBar, QTextEdit,
    QSizePolicy,
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, pyqtSlot
from PyQt5.QtGui import QPixmap, QImage

import httpx


class AnalysisWorker(QThread):
    """Background worker for running analysis."""
    
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, api: httpx.Client, image_id: int, analysis_type: str):
        super().__init__()
        self.api = api
        self.image_id = image_id
        self.analysis_type = analysis_type
    
    def run(self):
        try:
            if self.analysis_type == "full":
                response = self.api.post(f"/analysis/full/{self.image_id}")
            elif self.analysis_type == "annotate":
                response = self.api.post(f"/annotate/{self.image_id}")
            else:
                self.error.emit(f"Unknown analysis type: {self.analysis_type}")
                return
            
            if response.status_code == 200:
                self.finished.emit(response.json())
            else:
                self.error.emit(f"API error: {response.status_code}")
        except Exception as e:
            self.error.emit(str(e))


class SimilarImageThumbnail(QFrame):
    """Clickable thumbnail for similar image."""
    
    clicked = pyqtSignal(int)
    
    def __init__(self, image_id: int, similarity: float, api: httpx.Client):
        super().__init__()
        self.image_id = image_id
        self.similarity = similarity
        self.api = api
        
        self.setFixedSize(90, 110)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("""
            QFrame {
                background: rgba(14, 18, 24, 0.6);
                border: 1px solid rgba(40, 50, 70, 0.4);
                border-radius: 8px;
            }
            QFrame:hover {
                border-color: rgba(91, 141, 239, 0.5);
                background: rgba(20, 24, 32, 0.8);
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        
        # Thumbnail
        self.thumb = QLabel()
        self.thumb.setFixedSize(78, 70)
        self.thumb.setAlignment(Qt.AlignCenter)
        self.thumb.setStyleSheet("background: transparent; border-radius: 4px;")
        layout.addWidget(self.thumb)
        
        # Similarity
        sim_label = QLabel(f"{similarity:.0%}")
        sim_label.setAlignment(Qt.AlignCenter)
        sim_label.setStyleSheet("font-size: 11px; color: #6b9fff; font-weight: 500;")
        layout.addWidget(sim_label)
        
        self._load_thumbnail()
    
    def _load_thumbnail(self):
        try:
            response = self.api.get(f"/images/{self.image_id}/file")
            if response.status_code == 200:
                image = QImage.fromData(response.content)
                if not image.isNull():
                    pixmap = QPixmap.fromImage(image)
                    scaled = pixmap.scaled(78, 70, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.thumb.setPixmap(scaled)
                    return
        except Exception:
            pass
        self.thumb.setText("—")
        self.thumb.setStyleSheet("color: #4a5568;")
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.image_id)


class ViewerPanel(QWidget):
    """Panel for viewing and analyzing a single image."""
    
    back_clicked = pyqtSignal()
    image_clicked = pyqtSignal(int)  # For navigating to similar images
    
    def __init__(self, api: httpx.Client):
        super().__init__()
        self.api = api
        self.current_id: Optional[int] = None
        self.current_data: Optional[dict] = None
        self.worker: Optional[AnalysisWorker] = None
        self.similar_thumbnails: List[SimilarImageThumbnail] = []
        
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(20)
        
        # Header
        header = QHBoxLayout()
        
        back_btn = QPushButton("← Back")
        back_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                color: #7a8599;
                font-size: 13px;
                padding: 8px 0;
            }
            QPushButton:hover { color: #a0aec0; }
        """)
        back_btn.clicked.connect(self.back_clicked.emit)
        header.addWidget(back_btn)
        
        header.addStretch()
        
        self.title_label = QLabel("Image Viewer")
        self.title_label.setStyleSheet("font-size: 18px; font-weight: 400; color: #c8d0e0;")
        header.addWidget(self.title_label)
        
        header.addStretch()
        layout.addLayout(header)
        
        # Main content - scrollable
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        
        content_widget = QWidget()
        content_widget.setStyleSheet("background: transparent;")
        content = QHBoxLayout(content_widget)
        content.setSpacing(28)
        content.setContentsMargins(0, 0, 0, 0)
        
        # Left: Image + actions
        left = QVBoxLayout()
        left.setSpacing(16)
        
        self.image_label = QLabel()
        self.image_label.setFixedSize(480, 400)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            background: rgba(6, 8, 12, 0.8);
            border: 1px solid rgba(30, 40, 55, 0.4);
            border-radius: 12px;
        """)
        left.addWidget(self.image_label)
        
        # Action buttons
        actions = QHBoxLayout()
        actions.setSpacing(10)
        
        btn_style = """
            QPushButton {
                background: rgba(20, 24, 32, 0.6);
                border: 1px solid rgba(40, 50, 70, 0.4);
                padding: 10px 16px;
                border-radius: 8px;
                color: #a0aec0;
                font-size: 12px;
            }
            QPushButton:hover {
                background: rgba(30, 40, 55, 0.6);
                border-color: rgba(91, 141, 239, 0.4);
                color: #c8d0e0;
            }
            QPushButton:disabled {
                color: #4a5568;
                border-color: rgba(30, 40, 55, 0.3);
            }
        """
        
        primary_btn = """
            QPushButton {
                background: rgba(91, 141, 239, 0.15);
                border: 1px solid rgba(91, 141, 239, 0.3);
                padding: 10px 16px;
                border-radius: 8px;
                color: #6b9fff;
                font-size: 12px;
                font-weight: 500;
            }
            QPushButton:hover { background: rgba(91, 141, 239, 0.25); }
            QPushButton:disabled { color: #4a5568; background: transparent; }
        """
        
        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.setStyleSheet(primary_btn)
        self.analyze_btn.clicked.connect(self._run_analysis)
        actions.addWidget(self.analyze_btn)
        
        # Store styles for later use
        self._primary_btn_style = primary_btn
        self._btn_style = btn_style
        
        self.annotate_btn = QPushButton("Annotate")
        self.annotate_btn.setStyleSheet(btn_style)
        self.annotate_btn.clicked.connect(self._run_annotation)
        actions.addWidget(self.annotate_btn)
        
        self.similar_btn = QPushButton("Find Similar")
        self.similar_btn.setStyleSheet(btn_style)
        self.similar_btn.clicked.connect(lambda: self._find_similar(5))
        actions.addWidget(self.similar_btn)
        
        actions.addStretch()
        
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.setStyleSheet("""
            QPushButton {
                background: rgba(248, 113, 113, 0.1);
                border: 1px solid rgba(248, 113, 113, 0.2);
                padding: 10px 16px;
                border-radius: 8px;
                color: rgba(248, 113, 113, 0.8);
                font-size: 12px;
            }
            QPushButton:hover { background: rgba(248, 113, 113, 0.2); }
        """)
        self.delete_btn.clicked.connect(self._delete_image)
        actions.addWidget(self.delete_btn)
        
        left.addLayout(actions)
        
        # Progress
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setFixedHeight(4)
        self.progress.setStyleSheet("""
            QProgressBar {
                background: rgba(10, 13, 18, 0.6);
                border: none;
                border-radius: 2px;
            }
            QProgressBar::chunk {
                background: rgba(91, 141, 239, 0.8);
                border-radius: 2px;
            }
        """)
        self.progress.hide()
        left.addWidget(self.progress)
        
        left.addStretch()
        content.addLayout(left)
        
        # Right: Results
        right = QVBoxLayout()
        right.setSpacing(16)
        
        # Classification card
        class_card = QFrame()
        class_card.setStyleSheet("""
            QFrame {
                background: rgba(14, 18, 24, 0.5);
                border: 1px solid rgba(30, 40, 55, 0.4);
                border-radius: 12px;
            }
        """)
        class_layout = QVBoxLayout(class_card)
        class_layout.setContentsMargins(20, 18, 20, 18)
        class_layout.setSpacing(12)
        
        class_title = QLabel("Classification")
        class_title.setStyleSheet("font-size: 13px; font-weight: 500; color: #c8d0e0;")
        class_layout.addWidget(class_title)
        
        self.class_label = QLabel("—")
        self.class_label.setStyleSheet("font-size: 20px; font-weight: 300; color: #6b9fff;")
        class_layout.addWidget(self.class_label)
        
        self.conf_label = QLabel("Confidence: —")
        self.conf_label.setStyleSheet("font-size: 12px; color: #7a8599;")
        class_layout.addWidget(self.conf_label)
        
        self.anomaly_label = QLabel("Anomaly: —")
        self.anomaly_label.setStyleSheet("font-size: 12px; color: #7a8599;")
        class_layout.addWidget(self.anomaly_label)
        
        right.addWidget(class_card)
        
        # Annotation card
        ann_card = QFrame()
        ann_card.setStyleSheet("""
            QFrame {
                background: rgba(14, 18, 24, 0.5);
                border: 1px solid rgba(30, 40, 55, 0.4);
                border-radius: 12px;
            }
        """)
        ann_layout = QVBoxLayout(ann_card)
        ann_layout.setContentsMargins(20, 18, 20, 18)
        ann_layout.setSpacing(10)
        
        ann_title = QLabel("AI Annotation")
        ann_title.setStyleSheet("font-size: 13px; font-weight: 500; color: #c8d0e0;")
        ann_layout.addWidget(ann_title)
        
        self.annotation_text = QTextEdit()
        self.annotation_text.setReadOnly(True)
        self.annotation_text.setMinimumHeight(150)
        self.annotation_text.setStyleSheet("""
            QTextEdit {
                background: transparent;
                border: none;
                color: #a0aec0;
                font-size: 13px;
                line-height: 1.5;
            }
        """)
        self.annotation_text.setPlaceholderText("Click 'Annotate' to generate AI description")
        ann_layout.addWidget(self.annotation_text)
        
        right.addWidget(ann_card)
        
        # Similar images card
        similar_card = QFrame()
        similar_card.setStyleSheet("""
            QFrame {
                background: rgba(14, 18, 24, 0.5);
                border: 1px solid rgba(30, 40, 55, 0.4);
                border-radius: 12px;
            }
        """)
        similar_layout = QVBoxLayout(similar_card)
        similar_layout.setContentsMargins(20, 18, 20, 18)
        similar_layout.setSpacing(12)
        
        similar_header = QHBoxLayout()
        similar_title = QLabel("Similar Images")
        similar_title.setStyleSheet("font-size: 13px; font-weight: 500; color: #c8d0e0;")
        similar_header.addWidget(similar_title)
        similar_header.addStretch()
        
        self.load_more_btn = QPushButton("Load more")
        self.load_more_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                color: #6b9fff;
                font-size: 11px;
            }
            QPushButton:hover { color: #8bb4ff; }
        """)
        self.load_more_btn.hide()
        self.load_more_btn.clicked.connect(self._load_more_similar)
        similar_header.addWidget(self.load_more_btn)
        
        similar_layout.addLayout(similar_header)
        
        # Thumbnails container
        self.similar_container = QWidget()
        self.similar_container.setStyleSheet("background: transparent;")
        self.similar_grid = QHBoxLayout(self.similar_container)
        self.similar_grid.setSpacing(8)
        self.similar_grid.setContentsMargins(0, 0, 0, 0)
        
        self.no_similar_label = QLabel("Click 'Find Similar' to discover related images")
        self.no_similar_label.setStyleSheet("color: #4a5568; font-size: 12px; padding: 20px 0;")
        self.similar_grid.addWidget(self.no_similar_label)
        self.similar_grid.addStretch()
        
        similar_layout.addWidget(self.similar_container)
        
        right.addWidget(similar_card)
        right.addStretch()
        
        content.addLayout(right, 1)
        
        scroll.setWidget(content_widget)
        layout.addWidget(scroll, 1)
    
    def load_image(self, image_id: int):
        """Load image data and display."""
        self.current_id = image_id
        self._clear_similar()
        self.progress.show()
        
        try:
            response = self.api.get(f"/images/{image_id}")
            if response.status_code == 200:
                self.current_data = response.json()
                self._display_data()
            
            img_response = self.api.get(f"/images/{image_id}/file")
            if img_response.status_code == 200:
                image = QImage.fromData(img_response.content)
                if not image.isNull():
                    pixmap = QPixmap.fromImage(image)
                    scaled = pixmap.scaled(460, 380, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.image_label.setPixmap(scaled)
                else:
                    self.image_label.setText("Could not load image")
        except Exception as e:
            self.image_label.setText(f"Error: {e}")
        
        self.progress.hide()
    
    def _display_data(self):
        """Display current data in UI."""
        if not self.current_data:
            return
        
        filename = self.current_data.get("filename", "Image")
        if len(filename) > 40:
            filename = filename[:37] + "..."
        self.title_label.setText(filename)
        
        # Update analyze button based on state
        is_analyzed = bool(self.current_data.get("class_label"))
        if is_analyzed:
            self.analyze_btn.setText("Re-analyze")
            self.analyze_btn.setStyleSheet(self._btn_style)
        else:
            self.analyze_btn.setText("Analyze")
            self.analyze_btn.setStyleSheet(self._primary_btn_style)
        
        # Classification
        class_label = self.current_data.get("class_label")
        if class_label:
            display_name = class_label.replace("_", " ").title()
            self.class_label.setText(display_name)
            self.class_label.setStyleSheet("font-size: 20px; font-weight: 300; color: #6b9fff;")
            conf = self.current_data.get("class_confidence", 0)
            self.conf_label.setText(f"Confidence: {conf:.0%}")
        else:
            self.class_label.setText("Not analyzed")
            self.class_label.setStyleSheet("font-size: 20px; font-weight: 300; color: #4a5568;")
            self.conf_label.setText("Confidence: —")
        
        # Anomaly
        ood = self.current_data.get("ood_score")
        if ood is not None:
            is_anomaly = self.current_data.get("is_anomaly", False)
            if is_anomaly:
                self.anomaly_label.setText(f"Anomaly Score: {ood:.2f} — Flagged")
                self.anomaly_label.setStyleSheet("font-size: 12px; color: #9b7cef;")
            else:
                self.anomaly_label.setText(f"Anomaly Score: {ood:.2f} — Normal")
                self.anomaly_label.setStyleSheet("font-size: 12px; color: #7a8599;")
        else:
            self.anomaly_label.setText("Anomaly: —")
        
        # Annotation
        desc = self.current_data.get("llm_description", "")
        hypo = self.current_data.get("llm_hypothesis", "")
        follow = self.current_data.get("llm_follow_up", "")
        
        if desc or hypo:
            text = ""
            if desc:
                text += f"Description:\n{desc}\n\n"
            if hypo:
                text += f"Hypothesis:\n{hypo}\n\n"
            if follow:
                text += f"Follow-up:\n{follow}"
            self.annotation_text.setPlainText(text)
        else:
            self.annotation_text.clear()
    
    def _set_buttons_enabled(self, enabled: bool):
        """Enable/disable all action buttons to prevent conflicts."""
        self.analyze_btn.setEnabled(enabled)
        self.annotate_btn.setEnabled(enabled)
        self.similar_btn.setEnabled(enabled)
        self.delete_btn.setEnabled(enabled)
    
    def _run_analysis(self):
        if not self.current_id:
            return
        
        self._set_buttons_enabled(False)
        self.progress.show()
        
        self.worker = AnalysisWorker(self.api, self.current_id, "full")
        self.worker.finished.connect(self._on_analysis_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()
    
    def _run_annotation(self):
        if not self.current_id:
            return
        
        self._set_buttons_enabled(False)
        self.progress.show()
        
        self.worker = AnalysisWorker(self.api, self.current_id, "annotate")
        self.worker.finished.connect(self._on_annotation_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()
    
    @pyqtSlot(dict)
    def _on_analysis_done(self, result: dict):
        self.progress.hide()
        self._set_buttons_enabled(True)
        
        if "classification" in result:
            self.current_data["class_label"] = result["classification"]["class_label"]
            self.current_data["class_confidence"] = result["classification"]["confidence"]
        if "anomaly" in result:
            self.current_data["ood_score"] = result["anomaly"]["ood_score"]
            self.current_data["is_anomaly"] = result["anomaly"]["is_anomaly"]
        
        self.class_label.setStyleSheet("font-size: 20px; font-weight: 300; color: #6b9fff;")
        self._display_data()
    
    @pyqtSlot(dict)
    def _on_annotation_done(self, result: dict):
        self.progress.hide()
        self._set_buttons_enabled(True)
        
        self.current_data["llm_description"] = result.get("description", "")
        self.current_data["llm_hypothesis"] = result.get("hypothesis", "")
        self.current_data["llm_follow_up"] = result.get("follow_up", "")
        
        self._display_data()
    
    @pyqtSlot(str)
    def _on_error(self, error: str):
        self.progress.hide()
        self._set_buttons_enabled(True)
        self.annotation_text.setPlainText(f"Error: {error}")
    
    def _load_more_similar(self):
        """Load more similar images."""
        self._find_similar(10)
    
    def _find_similar(self, k: int = 5):
        if not self.current_id:
            return
        
        self._clear_similar()
        self.similar_btn.setEnabled(False)
        self.no_similar_label.setText("Searching...")
        self.no_similar_label.setStyleSheet("color: #6b9fff; padding: 20px 0; font-size: 12px;")
        
        try:
            response = self.api.post(f"/analysis/similar/{self.current_id}?k={k}")
            
            if response.status_code == 200:
                result = response.json()
                similar = result.get("similar", [])
                
                if similar:
                    self.no_similar_label.hide()
                    for item in similar:
                        thumb = SimilarImageThumbnail(
                            item["image_id"],
                            item["similarity"],
                            self.api
                        )
                        thumb.clicked.connect(self._on_similar_clicked)
                        self.similar_thumbnails.append(thumb)
                        self.similar_grid.insertWidget(len(self.similar_thumbnails) - 1, thumb)
                    
                    if k <= 5 and len(similar) >= k:
                        self.load_more_btn.show()
                else:
                    self.no_similar_label.setText("No similar images found. Analyze more images first.")
                    self.no_similar_label.setStyleSheet("color: #4a5568; padding: 20px 0; font-size: 12px;")
                    self.no_similar_label.show()
            elif response.status_code == 404:
                self.no_similar_label.setText("Image not found")
                self.no_similar_label.setStyleSheet("color: #f87171; padding: 20px 0; font-size: 12px;")
                self.no_similar_label.show()
            else:
                error_detail = ""
                try:
                    error_detail = response.json().get("detail", "")
                except:
                    pass
                self.no_similar_label.setText(f"Error {response.status_code}: {error_detail or 'API error'}")
                self.no_similar_label.setStyleSheet("color: #f87171; padding: 20px 0; font-size: 12px;")
                self.no_similar_label.show()
                
        except Exception as e:
            self.no_similar_label.setText(f"Connection error: {e}")
            self.no_similar_label.setStyleSheet("color: #f87171; padding: 20px 0; font-size: 12px;")
            self.no_similar_label.show()
        
        self.similar_btn.setEnabled(True)
    
    def _clear_similar(self):
        for thumb in self.similar_thumbnails:
            thumb.deleteLater()
        self.similar_thumbnails.clear()
        self.no_similar_label.show()
        self.load_more_btn.hide()
    
    def _on_similar_clicked(self, image_id: int):
        """Navigate to clicked similar image."""
        self.load_image(image_id)
        self.image_clicked.emit(image_id)
    
    def _delete_image(self):
        if not self.current_id:
            return
        
        try:
            response = self.api.delete(f"/images/{self.current_id}")
            if response.status_code == 200:
                self.back_clicked.emit()
        except Exception as e:
            self.annotation_text.setPlainText(f"Delete failed: {e}")
