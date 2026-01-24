"""
Gallery Panel

Premium grid view with refined typography and subtle interactions.
"""

from __future__ import annotations

from typing import List, Set

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QScrollArea,
    QLabel, QPushButton, QFrame, QCheckBox, QComboBox, QMessageBox,
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QThread
from PyQt5.QtGui import QPixmap, QImage, QColor

import httpx


class GalleryAnalysisWorker(QThread):
    """Background worker for analyzing selected images."""
    
    progress = pyqtSignal(int, int)  # current, total
    finished = pyqtSignal(int, int)  # success, errors
    
    def __init__(self, api: httpx.Client, image_ids: list):
        super().__init__()
        self.api = api
        self.image_ids = image_ids
    
    def run(self):
        success = 0
        errors = 0
        total = len(self.image_ids)
        
        for i, img_id in enumerate(self.image_ids):
            try:
                r = self.api.post(f"/analysis/full/{img_id}", timeout=30)
                if r.status_code == 200:
                    success += 1
                else:
                    errors += 1
            except Exception:
                errors += 1
            
            self.progress.emit(i + 1, total)
        
        self.finished.emit(success, errors)


class ImageCard(QFrame):
    """
    Premium image card with refined aesthetics.
    Clean, minimal, sophisticated.
    """
    
    clicked = pyqtSignal(int)
    selection_changed = pyqtSignal(int, bool)
    
    def __init__(self, image_data: dict, api: httpx.Client):
        super().__init__()
        self.image_id = image_data["id"]
        self.image_data = image_data
        self.api = api
        self._selected = False
        self.is_anomaly = image_data.get("is_anomaly", False)
        
        self.setFixedSize(200, 270)
        self.setCursor(Qt.PointingHandCursor)
        self._apply_style()
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        
        # Top row: checkbox + status
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        
        self.checkbox = QCheckBox()
        self.checkbox.setStyleSheet("""
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 4px;
                border: 1px solid rgba(60, 70, 90, 0.6);
                background: rgba(10, 13, 18, 0.8);
            }
            QCheckBox::indicator:checked {
                background: rgba(91, 141, 239, 0.8);
                border-color: rgba(91, 141, 239, 0.6);
            }
            QCheckBox::indicator:hover {
                border-color: rgba(91, 141, 239, 0.5);
            }
        """)
        self.checkbox.stateChanged.connect(self._on_checkbox_changed)
        top_row.addWidget(self.checkbox)
        
        top_row.addStretch()
        
        if self.is_anomaly:
            badge = QLabel("Anomaly")
            badge.setStyleSheet("""
                background: rgba(155, 124, 239, 0.2);
                color: rgba(167, 139, 250, 0.9);
                font-size: 10px;
                font-weight: 500;
                padding: 3px 8px;
                border-radius: 10px;
                border: 1px solid rgba(155, 124, 239, 0.3);
            """)
            top_row.addWidget(badge)
        
        layout.addLayout(top_row)
        
        # Thumbnail container
        thumb_container = QFrame()
        thumb_container.setFixedSize(176, 140)
        thumb_container.setStyleSheet("""
            QFrame {
                background: rgba(6, 8, 12, 0.8);
                border: 1px solid rgba(30, 40, 55, 0.4);
                border-radius: 8px;
            }
        """)
        
        thumb_layout = QVBoxLayout(thumb_container)
        thumb_layout.setContentsMargins(2, 2, 2, 2)
        
        self.thumbnail = QLabel()
        self.thumbnail.setFixedSize(172, 136)
        self.thumbnail.setAlignment(Qt.AlignCenter)
        self.thumbnail.setStyleSheet("background: transparent;")
        thumb_layout.addWidget(self.thumbnail)
        
        layout.addWidget(thumb_container)
        
        self._load_thumbnail()
        
        # Filename - subtle
        filename = image_data.get("filename", "Unknown")
        if len(filename) > 24:
            filename = filename[:21] + "..."
        name_label = QLabel(filename)
        name_label.setStyleSheet("""
            font-size: 12px;
            color: #a0aec0;
            font-weight: 400;
        """)
        layout.addWidget(name_label)
        
        # Classification - refined badge
        class_label = image_data.get("class_label")
        result = self._create_result_badge(class_label, image_data.get("class_confidence", 0))
        layout.addWidget(result)

    def _create_result_badge(self, class_label: str, confidence: float) -> QLabel:
        """Create visible, color-coded classification badge."""
        if not class_label:
            badge = QLabel("⏳ Awaiting analysis")
            badge.setStyleSheet("""
                background: rgba(251, 191, 36, 0.1);
                border: 1px solid rgba(251, 191, 36, 0.2);
                color: rgba(251, 191, 36, 0.9);
                font-size: 10px;
                font-weight: 500;
                padding: 5px 8px;
                border-radius: 6px;
            """)
            return badge
        
        class_display = class_label.replace("_", " ").title()
        if len(class_display) > 18:
            class_display = class_display[:15] + "..."
        
        # Visible colors based on confidence
        if confidence >= 0.7:
            color = "rgba(52, 211, 153, 0.95)"  # Emerald - high confidence
            bg = "rgba(52, 211, 153, 0.12)"
            border = "rgba(52, 211, 153, 0.25)"
        elif confidence >= 0.5:
            color = "rgba(78, 205, 196, 0.95)"  # Cyan - medium confidence
            bg = "rgba(78, 205, 196, 0.12)"
            border = "rgba(78, 205, 196, 0.25)"
        elif confidence >= 0.3:
            color = "rgba(107, 159, 255, 0.95)"  # Blue - low confidence
            bg = "rgba(107, 159, 255, 0.12)"
            border = "rgba(107, 159, 255, 0.25)"
        else:
            color = "rgba(156, 163, 175, 0.9)"  # Gray - very low
            bg = "rgba(156, 163, 175, 0.08)"
            border = "rgba(156, 163, 175, 0.2)"
        
        badge = QLabel(f"✓ {class_display} · {confidence:.0%}")
        badge.setStyleSheet(f"""
            background: {bg};
            border: 1px solid {border};
            color: {color};
            font-size: 10px;
            font-weight: 500;
            padding: 5px 8px;
            border-radius: 6px;
        """)
        return badge

    def _apply_style(self):
        """Apply refined card style."""
        if self._selected:
            self.setStyleSheet("""
                ImageCard {
                    background: rgba(91, 141, 239, 0.08);
                    border: 1px solid rgba(91, 141, 239, 0.3);
                    border-radius: 12px;
                }
            """)
        elif self.is_anomaly:
            self.setStyleSheet("""
                ImageCard {
                    background: rgba(14, 18, 24, 0.6);
                    border: 1px solid rgba(155, 124, 239, 0.2);
                    border-radius: 12px;
                }
                ImageCard:hover {
                    background: rgba(20, 24, 32, 0.7);
                    border-color: rgba(155, 124, 239, 0.4);
                }
            """)
        else:
            self.setStyleSheet("""
                ImageCard {
                    background: rgba(14, 18, 24, 0.4);
                    border: 1px solid rgba(30, 40, 55, 0.4);
                    border-radius: 12px;
                }
                ImageCard:hover {
                    background: rgba(20, 24, 32, 0.6);
                    border-color: rgba(60, 70, 90, 0.5);
                }
            """)

    def _on_checkbox_changed(self, state):
        self._selected = state == Qt.Checked
        self._apply_style()
        self.selection_changed.emit(self.image_id, self._selected)

    def set_selected(self, selected: bool):
        self.checkbox.setChecked(selected)

    def _load_thumbnail(self):
        try:
            response = self.api.get(f"/images/{self.image_id}/file")
            if response.status_code == 200:
                image = QImage.fromData(response.content)
                if not image.isNull():
                    pixmap = QPixmap.fromImage(image)
                    scaled = pixmap.scaled(172, 136, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.thumbnail.setPixmap(scaled)
                    return
        except Exception:
            pass
        
        self.thumbnail.setText("No preview")
        self.thumbnail.setStyleSheet("font-size: 12px; color: #4a5568;")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if not self.checkbox.geometry().contains(event.pos()):
                self.clicked.emit(self.image_id)


class DiscoveryBanner(QFrame):
    """Refined discovery notification banner."""
    
    view_clicked = pyqtSignal()
    dismiss_clicked = pyqtSignal()
    
    def __init__(self, count: int = 0):
        super().__init__()
        self.count = count
        
        self.setStyleSheet("""
            QFrame {
                background: rgba(155, 124, 239, 0.08);
                border: 1px solid rgba(155, 124, 239, 0.2);
                border-radius: 12px;
            }
        """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 14, 20, 14)
        
        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)
        
        self.title = QLabel(f"{count} potential discoveries")
        self.title.setStyleSheet("""
            color: #c8d0e0;
            font-size: 14px;
            font-weight: 500;
        """)
        text_layout.addWidget(self.title)
        
        subtitle = QLabel("Unusual patterns detected that may be worth investigating")
        subtitle.setStyleSheet("color: #7a8599; font-size: 12px; font-weight: 400;")
        text_layout.addWidget(subtitle)
        
        layout.addLayout(text_layout)
        layout.addStretch()
        
        view_btn = QPushButton("View")
        view_btn.setStyleSheet("""
            QPushButton {
                background: rgba(155, 124, 239, 0.2);
                border: 1px solid rgba(155, 124, 239, 0.3);
                color: rgba(167, 139, 250, 0.95);
                font-weight: 500;
                padding: 8px 16px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background: rgba(155, 124, 239, 0.3);
            }
        """)
        view_btn.clicked.connect(self.view_clicked.emit)
        layout.addWidget(view_btn)
        
        dismiss = QPushButton("×")
        dismiss.setFixedSize(24, 24)
        dismiss.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #7a8599;
                font-size: 18px;
                border: none;
            }
            QPushButton:hover {
                color: #a0aec0;
            }
        """)
        dismiss.clicked.connect(self.dismiss_clicked.emit)
        layout.addWidget(dismiss)
    
    def set_count(self, count: int):
        self.count = count
        self.title.setText(f"{count} potential discoveries")


class GalleryPanel(QWidget):
    """Premium gallery with pagination for large collections."""
    
    image_selected = pyqtSignal(int)
    upload_clicked = pyqtSignal()
    
    PAGE_SIZE = 30  # Images per page
    
    def __init__(self, api: httpx.Client):
        super().__init__()
        self.api = api
        self.images: List[dict] = []
        self.selected_ids: Set[int] = set()
        self.cards: List[ImageCard] = []
        self.current_page = 0
        self.total_count = 0
        self._analysis_worker = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(24)
        
        # Header - clean, minimal
        header = QHBoxLayout()
        
        title_layout = QVBoxLayout()
        title_layout.setSpacing(4)
        
        title = QLabel("Image Gallery")
        title.setStyleSheet("""
            font-size: 28px;
            font-weight: 300;
            color: #c8d0e0;
            letter-spacing: -0.5px;
        """)
        title_layout.addWidget(title)
        
        subtitle = QLabel("Browse and analyze your astronomical images")
        subtitle.setStyleSheet("font-size: 13px; color: #4a5568; font-weight: 400;")
        title_layout.addWidget(subtitle)
        
        header.addLayout(title_layout)
        header.addStretch()
        
        # Selection info
        self.selection_label = QLabel("")
        self.selection_label.setStyleSheet("color: #6b9fff; font-weight: 500; font-size: 13px;")
        header.addWidget(self.selection_label)
        
        # Control buttons - subtle
        btn_style = """
            QPushButton {
                background: rgba(20, 24, 32, 0.6);
                border: 1px solid rgba(40, 50, 70, 0.4);
                padding: 8px 14px;
                border-radius: 8px;
                color: #7a8599;
                font-size: 12px;
            }
            QPushButton:hover {
                background: rgba(30, 40, 55, 0.6);
                color: #a0aec0;
            }
        """
        
        self.select_all_btn = QPushButton("Select all")
        self.select_all_btn.clicked.connect(self._select_all)
        self.select_all_btn.setStyleSheet(btn_style)
        header.addWidget(self.select_all_btn)
        
        # Analyze Selected button
        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.clicked.connect(self._analyze_selected)
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                background: rgba(52, 211, 153, 0.1);
                border: 1px solid rgba(52, 211, 153, 0.2);
                padding: 8px 14px;
                border-radius: 8px;
                color: rgba(52, 211, 153, 0.8);
                font-size: 12px;
            }
            QPushButton:hover {
                background: rgba(52, 211, 153, 0.2);
            }
            QPushButton:disabled {
                background: transparent;
                border-color: rgba(50, 60, 80, 0.3);
                color: #4a5568;
            }
        """)
        header.addWidget(self.analyze_btn)
        
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(self._delete_selected)
        self.delete_btn.setEnabled(False)
        self.delete_btn.setStyleSheet("""
            QPushButton {
                background: rgba(248, 113, 113, 0.1);
                border: 1px solid rgba(248, 113, 113, 0.2);
                padding: 8px 14px;
                border-radius: 8px;
                color: rgba(248, 113, 113, 0.8);
                font-size: 12px;
            }
            QPushButton:hover {
                background: rgba(248, 113, 113, 0.2);
            }
            QPushButton:disabled {
                background: transparent;
                border-color: rgba(50, 60, 80, 0.3);
                color: #4a5568;
            }
        """)
        header.addWidget(self.delete_btn)
        
        # Filter
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All images", "Anomalies only", "Unanalyzed"])
        self.filter_combo.currentIndexChanged.connect(self.refresh)
        self.filter_combo.setFixedWidth(140)
        header.addWidget(self.filter_combo)
        
        # Refresh
        refresh_btn = QPushButton("↻")
        refresh_btn.setFixedSize(36, 36)
        refresh_btn.setStyleSheet(btn_style + "font-size: 16px;")
        refresh_btn.clicked.connect(self.refresh)
        header.addWidget(refresh_btn)
        
        # Upload
        upload_btn = QPushButton("Upload")
        upload_btn.clicked.connect(self.upload_clicked.emit)
        upload_btn.setStyleSheet("""
            QPushButton {
                background: rgba(91, 141, 239, 0.15);
                border: 1px solid rgba(91, 141, 239, 0.3);
                padding: 10px 18px;
                border-radius: 8px;
                color: #6b9fff;
                font-size: 13px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: rgba(91, 141, 239, 0.25);
            }
        """)
        header.addWidget(upload_btn)
        
        layout.addLayout(header)
        
        # Discovery banner
        self.discovery_banner = DiscoveryBanner()
        self.discovery_banner.view_clicked.connect(self._filter_discoveries)
        self.discovery_banner.dismiss_clicked.connect(self._dismiss_banner)
        self.discovery_banner.hide()
        layout.addWidget(self.discovery_banner)
        
        # Grid scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        
        self.grid_widget = QWidget()
        self.grid_widget.setStyleSheet("background: transparent;")
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(16)
        self.grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        
        self.scroll_area.setWidget(self.grid_widget)
        layout.addWidget(self.scroll_area, 1)
        
        # Pagination controls
        pagination = QHBoxLayout()
        pagination.setSpacing(12)
        
        self.prev_btn = QPushButton("← Previous")
        self.prev_btn.setStyleSheet(btn_style)
        self.prev_btn.clicked.connect(self._prev_page)
        self.prev_btn.setEnabled(False)
        pagination.addWidget(self.prev_btn)
        
        pagination.addStretch()
        
        self.page_label = QLabel("Page 1")
        self.page_label.setStyleSheet("color: #7a8599; font-size: 12px;")
        pagination.addWidget(self.page_label)
        
        pagination.addStretch()
        
        self.next_btn = QPushButton("Next →")
        self.next_btn.setStyleSheet(btn_style)
        self.next_btn.clicked.connect(self._next_page)
        pagination.addWidget(self.next_btn)
        
        layout.addLayout(pagination)
        
        # Empty state
        self.empty_label = QLabel("No images yet\n\nDrag and drop images here or click Upload")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setStyleSheet("""
            color: #4a5568;
            font-size: 14px;
            font-weight: 400;
            background: rgba(14, 18, 24, 0.4);
            border: 1px dashed rgba(50, 60, 80, 0.4);
            border-radius: 16px;
            padding: 60px;
        """)
        self.empty_label.hide()
        layout.addWidget(self.empty_label)
    
    def _update_selection_ui(self):
        count = len(self.selected_ids)
        if count > 0:
            self.selection_label.setText(f"{count} selected")
            self.delete_btn.setEnabled(True)
            self.analyze_btn.setEnabled(True)
            self.select_all_btn.setText("Deselect all")
        else:
            self.selection_label.setText("")
            self.delete_btn.setEnabled(False)
            self.analyze_btn.setEnabled(False)
            self.select_all_btn.setText("Select all")
    
    def _on_card_selection_changed(self, image_id: int, is_selected: bool):
        if is_selected:
            self.selected_ids.add(image_id)
        else:
            self.selected_ids.discard(image_id)
        self._update_selection_ui()
    
    def _select_all(self):
        if len(self.selected_ids) > 0:
            self.selected_ids.clear()
            for card in self.cards:
                card.set_selected(False)
        else:
            for card in self.cards:
                card.set_selected(True)
                self.selected_ids.add(card.image_id)
        self._update_selection_ui()
    
    def _analyze_selected(self):
        """Analyze selected images directly from gallery."""
        if not self.selected_ids:
            return
        
        count = len(self.selected_ids)
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setText(f"Analyzing 0/{count}...")
        
        # Run analysis in background
        self._analysis_worker = GalleryAnalysisWorker(self.api, list(self.selected_ids))
        self._analysis_worker.progress.connect(self._on_analysis_progress)
        self._analysis_worker.finished.connect(self._on_analysis_finished)
        self._analysis_worker.start()
    
    def _on_analysis_progress(self, current: int, total: int):
        self.analyze_btn.setText(f"Analyzing {current}/{total}...")
    
    def _on_analysis_finished(self, success: int, errors: int):
        self.analyze_btn.setText("Analyze")
        self.analyze_btn.setEnabled(len(self.selected_ids) > 0)
        
        if self._analysis_worker:
            self._analysis_worker.wait(1000)
            self._analysis_worker.deleteLater()
            self._analysis_worker = None
        
        # Refresh to show updated analysis results
        self.refresh(reset_page=False)
    
    def _delete_selected(self):
        if not self.selected_ids:
            return
        
        count = len(self.selected_ids)
        reply = QMessageBox.question(
            self, "Delete Images",
            f"Remove {count} image(s)?\n\nThis cannot be undone.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        for image_id in list(self.selected_ids):
            try:
                self.api.delete(f"/images/{image_id}")
            except Exception:
                pass
        
        self.selected_ids.clear()
        self._update_selection_ui()
        self.refresh()
    
    def refresh(self, reset_page: bool = True, preserve_selection: bool = False):
        if reset_page:
            self.current_page = 0
        
        self.cards.clear()
        
        # Only clear selection if not preserving
        if not preserve_selection:
            self.selected_ids.clear()
        
        self._update_selection_ui()
        
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Build query with pagination
        skip = self.current_page * self.PAGE_SIZE
        params = {"limit": self.PAGE_SIZE, "skip": skip}
        filter_idx = self.filter_combo.currentIndex()
        if filter_idx == 1:
            params["anomaly_only"] = True
        
        try:
            response = self.api.get("/images", params=params)
            if response.status_code == 200:
                self.images = response.json()
            else:
                self.images = []
            
            # Get total count for pagination based on current filter
            count_response = self.api.get("/stats")
            if count_response.status_code == 200:
                stats = count_response.json()
                if filter_idx == 1:  # Anomalies filter
                    # Get actual anomaly count
                    self.total_count = stats.get("anomalies", 0)
                elif filter_idx == 2:  # Unanalyzed filter
                    # For unanalyzed, we need to count them
                    total = stats.get("total_images", 0)
                    analyzed = stats.get("analyzed", 0)
                    self.total_count = max(0, total - analyzed)
                else:  # All images
                    self.total_count = stats.get("total_images", 0)
        except Exception:
            self.images = []
            self.total_count = 0
        
        # Client-side filter for unanalyzed
        if filter_idx == 2:
            self.images = [img for img in self.images if not img.get("class_label")]
        
        # Update pagination controls based on filtered count
        total_pages = max(1, (self.total_count + self.PAGE_SIZE - 1) // self.PAGE_SIZE)
        self.page_label.setText(f"Page {self.current_page + 1} of {total_pages}")
        self.prev_btn.setEnabled(self.current_page > 0)
        self.next_btn.setEnabled((self.current_page + 1) * self.PAGE_SIZE < self.total_count)
        
        if not self.images:
            self.empty_label.show()
            self.grid_widget.hide()
            self.discovery_banner.hide()
            return
        
        self.empty_label.hide()
        self.grid_widget.show()
        
        cols = 5
        anomaly_count = 0
        for i, img in enumerate(self.images):
            card = ImageCard(img, self.api)
            card.clicked.connect(self.image_selected.emit)
            card.selection_changed.connect(self._on_card_selection_changed)
            
            # Restore selection state if this image was previously selected
            if img["id"] in self.selected_ids:
                card.set_selected(True)
            
            self.cards.append(card)
            self.grid_layout.addWidget(card, i // cols, i % cols)
            
            if img.get("is_anomaly"):
                anomaly_count += 1
        
        if anomaly_count > 0 and filter_idx != 1:
            self.discovery_banner.set_count(anomaly_count)
            self.discovery_banner.show()
        else:
            self.discovery_banner.hide()
    
    def _prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.refresh(reset_page=False, preserve_selection=True)
            self._scroll_to_top()
    
    def _next_page(self):
        if (self.current_page + 1) * self.PAGE_SIZE < self.total_count:
            self.current_page += 1
            self.refresh(reset_page=False, preserve_selection=True)
            self._scroll_to_top()
    
    def _scroll_to_top(self):
        """Scroll the grid view to the top."""
        self.scroll_area.verticalScrollBar().setValue(0)

    def _filter_discoveries(self):
        self.filter_combo.setCurrentIndex(1)
        self.discovery_banner.hide()

    def _dismiss_banner(self):
        self.discovery_banner.hide()
