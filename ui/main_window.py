"""
AstroLens Main Window

Premium, cutting-edge interface with refined aesthetics.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QStackedWidget, QListWidget, QListWidgetItem, QLabel, QPushButton,
    QStatusBar, QFileDialog, QMessageBox, QProgressBar, QFrame,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QIcon, QDragEnterEvent, QDropEvent

import httpx

from .gallery import GalleryPanel
from .viewer import ViewerPanel
from .chat_panel import ChatPanel
from .control_center import ControlCenter


API_URL = os.environ.get("API_URL", "http://localhost:8000")


class AlertBadge(QFrame):
    """Subtle alert badge with refined styling."""
    
    clicked = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.count = 0
        
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(22, 22)
        self.hide()
        
        self._apply_style()
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.label = QLabel("0")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("""
            color: white;
            font-size: 10px;
            font-weight: 600;
            background: transparent;
        """)
        layout.addWidget(self.label)
    
    def _apply_style(self):
        self.setStyleSheet("""
            QFrame {
                background: rgba(155, 124, 239, 0.9);
                border-radius: 11px;
            }
            QFrame:hover {
                background: rgba(167, 139, 250, 1);
            }
        """)
    
    def set_count(self, count: int):
        self.count = count
        if count > 0:
            self.label.setText(str(min(count, 99)))
            self.show()
        else:
            self.hide()
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()


class MainWindow(QMainWindow):
    """Main application window - Premium design."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AstroLens")
        self.setMinimumSize(1200, 800)
        self.resize(1440, 900)
        
        self.setAcceptDrops(True)
        self.api = httpx.Client(base_url=API_URL, timeout=60.0)
        self.current_image_id: Optional[int] = None
        self.anomaly_count = 0
        
        # Track gallery state for navigation
        self._saved_gallery_page = 0
        self._saved_gallery_scroll = 0
        
        self._setup_ui()
        self._setup_statusbar()
        
        QTimer.singleShot(500, self._check_api_connection)

    def _setup_ui(self):
        """Setup the main UI layout."""
        central = QWidget()
        self.setCentralWidget(central)
        
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        sidebar = self._create_sidebar()
        layout.addWidget(sidebar)
        
        self.content_stack = QStackedWidget()
        self.content_stack.setStyleSheet("background: #0a0d12;")
        layout.addWidget(self.content_stack, 1)
        
        self.gallery_panel = GalleryPanel(self.api)
        self.gallery_panel.image_selected.connect(self._on_image_selected)
        self.gallery_panel.upload_clicked.connect(self._upload_images)
        
        self.viewer_panel = ViewerPanel(self.api)
        self.viewer_panel.back_clicked.connect(self._show_gallery)
        
        self.chat_panel = ChatPanel(self.api)
        self.chat_panel.data_changed.connect(self._on_data_changed)
        
        self.control_center = ControlCenter(self.api)
        self.control_center.view_image.connect(self._on_image_selected)
        self.control_center.refresh_gallery.connect(self._on_data_changed)
        self.control_center.anomaly_count_changed.connect(self._on_anomaly_count_changed)
        
        self.content_stack.addWidget(self.gallery_panel)
        self.content_stack.addWidget(self.viewer_panel)
        self.content_stack.addWidget(self.chat_panel)
        self.content_stack.addWidget(self.control_center)

    def _create_sidebar(self) -> QWidget:
        """Create refined navigation sidebar."""
        sidebar = QWidget()
        sidebar.setFixedWidth(200)
        sidebar.setObjectName("sidebar")
        sidebar.setStyleSheet("""
            #sidebar {
                background: #06080c;
                border-right: 1px solid rgba(30, 40, 55, 0.5);
            }
        """)
        
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(16, 24, 16, 24)
        layout.setSpacing(8)
        
        # Logo - minimal, elegant
        title_row = QHBoxLayout()
        
        title = QLabel("AstroLens")
        title.setStyleSheet("""
            font-size: 18px;
            font-weight: 500;
            color: #c8d0e0;
            letter-spacing: 0.5px;
        """)
        title_row.addWidget(title)
        title_row.addStretch()
        
        self.alert_badge = AlertBadge()
        self.alert_badge.clicked.connect(self._show_anomalies)
        title_row.addWidget(self.alert_badge)
        
        layout.addLayout(title_row)
        layout.addSpacing(32)
        
        # Navigation - refined, minimal
        nav_label = QLabel("NAVIGATION")
        nav_label.setStyleSheet("""
            font-size: 10px;
            font-weight: 600;
            color: #4a5568;
            letter-spacing: 1.5px;
            margin-bottom: 8px;
        """)
        layout.addWidget(nav_label)
        
        self.nav_list = QListWidget()
        self.nav_list.setStyleSheet("""
            QListWidget {
                background: transparent;
                border: none;
                outline: none;
            }
            QListWidget::item {
                padding: 12px 14px;
                border-radius: 8px;
                margin: 1px 0;
                color: #7a8599;
                font-size: 13px;
                font-weight: 400;
            }
            QListWidget::item:selected {
                background: rgba(91, 141, 239, 0.12);
                color: #c8d0e0;
            }
            QListWidget::item:hover:!selected {
                background: rgba(30, 40, 55, 0.4);
                color: #a0aec0;
            }
        """)
        
        nav_items = [
            ("Gallery", 0),
            ("Viewer", 1),
            ("AI Chat", 2),
            ("Settings", 3),
        ]
        
        for label, idx in nav_items:
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, idx)
            self.nav_list.addItem(item)
        
        self.nav_list.setCurrentRow(0)
        self.nav_list.itemClicked.connect(self._on_nav_clicked)
        layout.addWidget(self.nav_list)
        
        layout.addStretch()
        
        # Quick actions - subtle
        actions_label = QLabel("QUICK ACTIONS")
        actions_label.setStyleSheet("""
            font-size: 10px;
            font-weight: 600;
            color: #4a5568;
            letter-spacing: 1.5px;
            margin-bottom: 8px;
        """)
        layout.addWidget(actions_label)
        
        btn_style = """
            QPushButton {
                background: rgba(20, 24, 32, 0.6);
                border: 1px solid rgba(40, 50, 70, 0.4);
                padding: 10px 12px;
                border-radius: 8px;
                text-align: left;
                color: #7a8599;
                font-size: 12px;
                font-weight: 400;
            }
            QPushButton:hover {
                background: rgba(30, 40, 55, 0.6);
                border-color: rgba(91, 141, 239, 0.4);
                color: #a0aec0;
            }
        """
        
        # Discovery Loop - Primary action
        discovery_btn = QPushButton("🔭 Discovery Loop")
        discovery_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(52, 211, 153, 0.15), stop:1 rgba(78, 205, 196, 0.1));
                border: 1px solid rgba(52, 211, 153, 0.3);
                padding: 10px 12px;
                border-radius: 8px;
                text-align: left;
                color: #34d399;
                font-size: 12px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: rgba(52, 211, 153, 0.2);
                border-color: rgba(52, 211, 153, 0.5);
            }
        """)
        discovery_btn.clicked.connect(self._show_discovery)
        layout.addWidget(discovery_btn)
        
        streaming_btn = QPushButton("📡 Streaming Discovery")
        streaming_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(88, 166, 255, 0.15), stop:1 rgba(163, 113, 247, 0.1));
                border: 1px solid rgba(88, 166, 255, 0.3);
                padding: 10px 12px;
                border-radius: 8px;
                text-align: left;
                color: #58a6ff;
                font-size: 12px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: rgba(88, 166, 255, 0.2);
                border-color: rgba(88, 166, 255, 0.5);
            }
        """)
        streaming_btn.clicked.connect(self._show_streaming)
        layout.addWidget(streaming_btn)
        
        download_btn = QPushButton("Download")
        download_btn.setStyleSheet(btn_style)
        download_btn.clicked.connect(self._show_downloads)
        layout.addWidget(download_btn)
        
        analyze_btn = QPushButton("Batch Analyze")
        analyze_btn.setStyleSheet(btn_style)
        analyze_btn.clicked.connect(self._show_batch)
        layout.addWidget(analyze_btn)
        
        layout.addSpacing(12)
        
        upload_btn = QPushButton("Upload Images")
        upload_btn.setStyleSheet("""
            QPushButton {
                background: rgba(91, 141, 239, 0.15);
                border: 1px solid rgba(91, 141, 239, 0.3);
                padding: 12px;
                border-radius: 8px;
                color: #6b9fff;
                font-size: 13px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: rgba(91, 141, 239, 0.25);
                border-color: rgba(91, 141, 239, 0.5);
            }
        """)
        upload_btn.clicked.connect(self._upload_images)
        layout.addWidget(upload_btn)
        
        layout.addSpacing(16)
        
        # Stats - very subtle
        self.stats_label = QLabel("Loading...")
        self.stats_label.setStyleSheet("""
            color: #4a5568;
            font-size: 11px;
            font-weight: 400;
            padding-top: 12px;
            border-top: 1px solid rgba(30, 40, 55, 0.4);
        """)
        layout.addWidget(self.stats_label)
        
        return sidebar

    def _setup_statusbar(self):
        """Setup refined status bar."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #4a5568; font-size: 12px;")
        self.statusbar.addWidget(self.status_label, 1)
        
        self.api_status = QLabel("Connecting...")
        self.api_status.setStyleSheet("color: #4a5568; font-size: 12px;")
        self.statusbar.addPermanentWidget(self.api_status)

    def _check_api_connection(self):
        """Check if API is available."""
        try:
            response = self.api.get("/health")
            if response.status_code == 200:
                self.api_status.setText("Connected")
                self.api_status.setStyleSheet("color: #34d399; font-size: 12px;")
                self._update_stats()
                self.gallery_panel.refresh()
        except Exception as e:
            self.api_status.setText("Offline")
            self.api_status.setStyleSheet("color: #f87171; font-size: 12px;")
            self.status_label.setText(f"API unavailable")

    def _update_stats(self):
        """Update stats label."""
        try:
            response = self.api.get("/stats")
            if response.status_code == 200:
                stats = response.json()
                self.stats_label.setText(
                    f"{stats['total_images']} images · {stats['anomalies']} anomalies"
                )
                self.anomaly_count = stats.get('anomalies', 0)
                self.alert_badge.set_count(self.anomaly_count)
        except Exception:
            pass

    def _on_nav_clicked(self, item: QListWidgetItem):
        idx = item.data(Qt.UserRole)
        self.content_stack.setCurrentIndex(idx)

    def _on_image_selected(self, image_id: int):
        # Save gallery state before navigating to viewer
        self._saved_gallery_page = self.gallery_panel.current_page
        self._saved_gallery_scroll = self.gallery_panel.scroll_area.verticalScrollBar().value()
        
        self.current_image_id = image_id
        self.viewer_panel.load_image(image_id)
        self.content_stack.setCurrentIndex(1)
        self.nav_list.setCurrentRow(1)

    def _show_gallery(self):
        self.content_stack.setCurrentIndex(0)
        self.nav_list.setCurrentRow(0)
        
        # Restore gallery state (page and scroll position)
        self.gallery_panel.current_page = self._saved_gallery_page
        self.gallery_panel.refresh(reset_page=False, preserve_selection=True)
        
        # Restore scroll position after a short delay to allow layout
        QTimer.singleShot(100, lambda: self.gallery_panel.scroll_area.verticalScrollBar().setValue(self._saved_gallery_scroll))
        
        self._update_stats()

    def _show_downloads(self):
        self.content_stack.setCurrentIndex(3)
        self.nav_list.setCurrentRow(3)
        self.control_center.show_sources()

    def _show_batch(self):
        self.content_stack.setCurrentIndex(3)
        self.nav_list.setCurrentRow(3)
        self.control_center.show_batch()

    def _show_discovery(self):
        self.content_stack.setCurrentIndex(3)
        self.nav_list.setCurrentRow(3)
        self.control_center.show_discovery()

    def _show_streaming(self):
        self.content_stack.setCurrentIndex(3)
        self.nav_list.setCurrentRow(3)
        self.control_center.show_streaming()

    def _show_anomalies(self):
        self.content_stack.setCurrentIndex(0)
        self.nav_list.setCurrentRow(0)
        self.gallery_panel.filter_combo.setCurrentIndex(1)
        self.gallery_panel.refresh()

    def _on_data_changed(self):
        self.gallery_panel.refresh()
        self._update_stats()
        self.status_label.setText("Updated")

    def _on_anomaly_count_changed(self, count: int):
        self.anomaly_count += count
        self.alert_badge.set_count(self.anomaly_count)
        self._update_stats()

    def _upload_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", "",
            "Images (*.png *.jpg *.jpeg *.fits);;All Files (*)"
        )
        if files:
            self._do_upload(files)

    def _do_upload(self, files: list):
        self.status_label.setText(f"Uploading {len(files)} files...")
        
        success = 0
        for filepath in files:
            try:
                with open(filepath, "rb") as f:
                    response = self.api.post(
                        "/images",
                        files={"file": (Path(filepath).name, f)},
                    )
                    if response.status_code == 200:
                        success += 1
            except Exception as e:
                self.status_label.setText(f"Error: {e}")
        
        self.status_label.setText(f"Uploaded {success}/{len(files)}")
        self.gallery_panel.refresh()
        self._update_stats()

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        files = []
        for url in event.mimeData().urls():
            if url.isLocalFile():
                path = url.toLocalFile()
                if path.lower().endswith(('.png', '.jpg', '.jpeg', '.fits')):
                    files.append(path)
        if files:
            self._do_upload(files)

    def closeEvent(self, event):
        self.api.close()
        super().closeEvent(event)
