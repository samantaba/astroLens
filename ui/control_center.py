"""
Control Center

Premium settings interface with refined aesthetics.
"""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTabWidget, QFrame,
)
from PyQt5.QtCore import Qt, pyqtSignal

import httpx

from .sources_panel import SourcesPanel
from .download_panel import DownloadPanel
from .batch_panel import BatchPanel
from .training_panel import TrainingPanel
from .discovery_panel import DiscoveryPanel


class ControlCenter(QWidget):
    """
    Premium control center with refined tab navigation.
    """
    
    view_image = pyqtSignal(int)
    refresh_gallery = pyqtSignal()
    anomaly_count_changed = pyqtSignal(int)
    
    def __init__(self, api: httpx.Client):
        super().__init__()
        self.api = api
        
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(24)
        
        # Header
        header_layout = QVBoxLayout()
        header_layout.setSpacing(4)
        
        title = QLabel("Settings")
        title.setStyleSheet("""
            font-size: 28px;
            font-weight: 300;
            color: #c8d0e0;
            letter-spacing: -0.5px;
        """)
        header_layout.addWidget(title)
        
        subtitle = QLabel("Configure data sources, downloads, and model training")
        subtitle.setStyleSheet("font-size: 13px; color: #4a5568; font-weight: 400;")
        header_layout.addWidget(subtitle)
        
        layout.addLayout(header_layout)
        
        # Tab widget - refined
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                background: rgba(10, 13, 18, 0.4);
                border: 1px solid rgba(30, 40, 55, 0.4);
                border-radius: 14px;
                padding: 24px;
                margin-top: -1px;
            }
            QTabBar::tab {
                background: transparent;
                border: none;
                padding: 12px 24px;
                margin-right: 8px;
                color: #4a5568;
                font-size: 13px;
                font-weight: 500;
            }
            QTabBar::tab:selected {
                color: #c8d0e0;
                border-bottom: 2px solid rgba(91, 141, 239, 0.8);
            }
            QTabBar::tab:hover:!selected {
                color: #7a8599;
            }
        """)
        
        self.sources_panel = SourcesPanel()
        self.download_panel = DownloadPanel()
        self.batch_panel = BatchPanel(self.api)
        self.training_panel = TrainingPanel()
        self.discovery_panel = DiscoveryPanel()
        
        self.tabs.addTab(self.discovery_panel, "🔭 Discovery")
        self.tabs.addTab(self.sources_panel, "Data Sources")
        self.tabs.addTab(self.download_panel, "Downloads")
        self.tabs.addTab(self.batch_panel, "Batch Analysis")
        self.tabs.addTab(self.training_panel, "Model Training")
        
        layout.addWidget(self.tabs, 1)
    
    def _connect_signals(self):
        self.sources_panel.download_requested.connect(self._start_download)
        self.download_panel.download_complete.connect(self._on_download_complete)
        self.batch_panel.view_image.connect(self.view_image.emit)
        self.batch_panel.refresh_gallery.connect(self.refresh_gallery.emit)
        self.discovery_panel.anomaly_found.connect(self._on_discovery_anomaly)
    
    def _start_download(self, sources: dict):
        self.tabs.setCurrentWidget(self.download_panel)
        auto_analyze = self.sources_panel.should_auto_analyze()
        self.download_panel.start_download(sources, upload=True, analyze=auto_analyze)
    
    def _on_download_complete(self, results: dict):
        anomalies = results.get("anomalies", 0)
        if anomalies > 0:
            self.anomaly_count_changed.emit(anomalies)
        self.refresh_gallery.emit()
    
    def show_sources(self):
        self.tabs.setCurrentWidget(self.sources_panel)
    
    def show_downloads(self):
        self.tabs.setCurrentWidget(self.download_panel)
    
    def show_batch(self):
        self.tabs.setCurrentWidget(self.batch_panel)
    
    def show_training(self):
        self.tabs.setCurrentWidget(self.training_panel)
    
    def show_discovery(self):
        self.tabs.setCurrentWidget(self.discovery_panel)
    
    def _on_discovery_anomaly(self, candidate: dict):
        """Handle anomaly found by discovery loop."""
        self.anomaly_count_changed.emit(1)
        self.refresh_gallery.emit()
