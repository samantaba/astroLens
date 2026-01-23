"""
Data Sources Panel

Configure astronomical image data sources with refined UI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QCheckBox, QSpinBox, QComboBox, QGroupBox,
    QScrollArea, QTimeEdit, QMessageBox,
)
from PyQt5.QtCore import Qt, pyqtSignal, QTime

from paths import SOURCES_CONFIG_PATH


DEFAULT_SOURCES = {
    "sdss": {
        "name": "SDSS Galaxies",
        "description": "Sloan Digital Sky Survey — Random galaxy images",
        "best_for": "Galaxy classification & training",
        "enabled": True,
        "daily_count": 50,
        "max_count": 500,
    },
    "ztf": {
        "name": "Transient Regions",
        "description": "Sky regions near known transients (supernovae, variables)",
        "best_for": "Transient & anomaly detection",
        "enabled": True,
        "daily_count": 20,
        "max_count": 200,
    },
    "apod": {
        "name": "NASA APOD",
        "description": "Astronomy Picture of the Day (needs NASA_API_KEY env var)",
        "best_for": "Diverse astronomical images",
        "enabled": False,
        "daily_count": 7,
        "max_count": 365,
    },
}


class SourceCard(QFrame):
    """Refined source configuration card."""
    
    config_changed = pyqtSignal()
    
    def __init__(self, source_id: str, config: dict):
        super().__init__()
        self.source_id = source_id
        self.config = config.copy()
        
        self.setStyleSheet("""
            QFrame {
                background: rgba(14, 18, 24, 0.5);
                border: 1px solid rgba(30, 40, 55, 0.4);
                border-radius: 12px;
            }
            QFrame:hover {
                border-color: rgba(60, 70, 90, 0.5);
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)
        
        # Header
        header = QHBoxLayout()
        
        self.enabled_cb = QCheckBox()
        self.enabled_cb.setChecked(config.get("enabled", False))
        self.enabled_cb.stateChanged.connect(self._on_config_changed)
        self.enabled_cb.setStyleSheet("""
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border-radius: 5px;
                border: 1px solid rgba(60, 70, 90, 0.6);
                background: rgba(10, 13, 18, 0.8);
            }
            QCheckBox::indicator:checked {
                background: rgba(91, 141, 239, 0.7);
                border-color: rgba(91, 141, 239, 0.5);
                image: none;
            }
            QCheckBox::indicator:hover {
                border-color: rgba(91, 141, 239, 0.5);
            }
        """)
        header.addWidget(self.enabled_cb)
        
        name_label = QLabel(config.get("name", source_id))
        name_label.setStyleSheet("font-size: 14px; font-weight: 500; color: #c8d0e0;")
        header.addWidget(name_label)
        
        header.addStretch()
        layout.addLayout(header)
        
        # Description
        desc_label = QLabel(config.get("description", ""))
        desc_label.setStyleSheet("color: #7a8599; font-size: 12px; font-weight: 400;")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        # Best for
        best_for = QLabel(f"Best for: {config.get('best_for', 'General use')}")
        best_for.setStyleSheet("color: #6b9fff; font-size: 11px; font-weight: 400;")
        layout.addWidget(best_for)
        
        # Settings
        settings = QHBoxLayout()
        
        count_label = QLabel("Daily images:")
        count_label.setStyleSheet("color: #7a8599; font-size: 12px;")
        settings.addWidget(count_label)
        
        self.count_spin = QSpinBox()
        max_count = config.get("max_count", 500)
        self.count_spin.setRange(1, max_count)
        self.count_spin.setValue(min(config.get("daily_count", 10), max_count))
        self.count_spin.valueChanged.connect(self._on_config_changed)
        self.count_spin.setFixedWidth(80)
        self.count_spin.setStyleSheet("""
            QSpinBox {
                background: rgba(10, 13, 18, 0.8);
                border: 1px solid rgba(40, 50, 70, 0.5);
                border-radius: 6px;
                padding: 6px 10px;
                color: #c8d0e0;
            }
        """)
        settings.addWidget(self.count_spin)
        
        settings.addStretch()
        layout.addLayout(settings)
    
    def _on_config_changed(self):
        self.config["enabled"] = self.enabled_cb.isChecked()
        self.config["daily_count"] = self.count_spin.value()
        self.config_changed.emit()
    
    def get_config(self) -> dict:
        return self.config.copy()
    
    def is_enabled(self) -> bool:
        return self.enabled_cb.isChecked()


class SourcesPanel(QWidget):
    """Refined data sources panel."""
    
    download_requested = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.source_cards = {}
        self.config_path = SOURCES_CONFIG_PATH
        
        self._load_config()
        self._setup_ui()
    
    def _load_config(self):
        self.sources = DEFAULT_SOURCES.copy()
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    saved = json.load(f)
                    for key, val in saved.items():
                        if key in self.sources:
                            self.sources[key].update(val)
            except Exception:
                pass
    
    def _save_config(self):
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            config_to_save = {sid: card.get_config() for sid, card in self.source_cards.items()}
            with open(self.config_path, "w") as f:
                json.dump(config_to_save, f, indent=2)
        except Exception:
            pass
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)
        
        # Header
        header = QLabel("Data Sources")
        header.setStyleSheet("font-size: 18px; font-weight: 500; color: #c8d0e0;")
        layout.addWidget(header)
        
        info = QLabel("Configure which astronomical surveys to download from")
        info.setStyleSheet("color: #7a8599; font-size: 13px; font-weight: 400;")
        layout.addWidget(info)
        
        # Source cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        
        cards_widget = QWidget()
        cards_widget.setStyleSheet("background: transparent;")
        cards_layout = QVBoxLayout(cards_widget)
        cards_layout.setSpacing(12)
        
        for source_id, config in self.sources.items():
            card = SourceCard(source_id, config)
            card.config_changed.connect(self._save_config)
            self.source_cards[source_id] = card
            cards_layout.addWidget(card)
        
        cards_layout.addStretch()
        scroll.setWidget(cards_widget)
        layout.addWidget(scroll, 1)
        
        # Options row
        opts = QHBoxLayout()
        self.auto_analyze_cb = QCheckBox("Auto-analyze after download")
        self.auto_analyze_cb.setChecked(True)
        self.auto_analyze_cb.setToolTip("Automatically run ML analysis on downloaded images")
        self.auto_analyze_cb.setStyleSheet("""
            QCheckBox { color: #a0aec0; font-size: 12px; spacing: 6px; }
            QCheckBox::indicator { width: 14px; height: 14px; border-radius: 3px; 
                border: 1px solid rgba(91, 141, 239, 0.4); background: rgba(10, 13, 18, 0.6); }
            QCheckBox::indicator:checked { background: rgba(91, 141, 239, 0.3); border-color: rgba(91, 141, 239, 0.6); }
        """)
        opts.addWidget(self.auto_analyze_cb)
        opts.addStretch()
        layout.addLayout(opts)
        
        # Actions
        buttons = QHBoxLayout()
        
        self.download_btn = QPushButton("Start Download")
        self.download_btn.setStyleSheet("""
            QPushButton {
                background: rgba(91, 141, 239, 0.15);
                border: 1px solid rgba(91, 141, 239, 0.3);
                padding: 12px 24px;
                border-radius: 10px;
                color: #6b9fff;
                font-size: 14px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: rgba(91, 141, 239, 0.25);
            }
        """)
        self.download_btn.clicked.connect(self._request_download)
        buttons.addWidget(self.download_btn)
        
        buttons.addStretch()
        layout.addLayout(buttons)
    
    def _request_download(self):
        enabled = {sid: card.get_config() for sid, card in self.source_cards.items() if card.is_enabled()}
        if not enabled:
            QMessageBox.warning(self, "No Sources", "Enable at least one data source.")
            return
        self.download_requested.emit(enabled)
    
    def get_enabled_sources(self) -> dict:
        return {sid: card.get_config() for sid, card in self.source_cards.items() if card.is_enabled()}
    
    def should_auto_analyze(self) -> bool:
        return getattr(self, 'auto_analyze_cb', None) and self.auto_analyze_cb.isChecked()
