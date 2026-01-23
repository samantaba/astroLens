"""
Model Training Panel

Clean, modern training interface with educational guidance.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QProgressBar, QComboBox, QSpinBox,
    QFileDialog, QMessageBox, QTextEdit, QScrollArea,
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread

from paths import DATASETS_DIR, WEIGHTS_DIR


class TrainingWorker(QThread):
    """Background worker for model training."""
    
    output = pyqtSignal(str)
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, dataset_path: str, epochs: int, batch_size: int):
        super().__init__()
        self.dataset_path = dataset_path
        self.epochs = epochs
        self.batch_size = batch_size
        self._cancelled = False
    
    def cancel(self):
        self._cancelled = True
    
    def run(self):
        try:
            script_path = Path(__file__).parent.parent / "finetuning" / "train.py"
            
            cmd = [
                sys.executable,
                str(script_path),
                "--data_dir", self.dataset_path,
                "--epochs", str(self.epochs),
                "--batch_size", str(self.batch_size),
                "--output_dir", str(WEIGHTS_DIR / "vit_astrolens"),
            ]
            
            self.output.emit(f"Starting training with dataset: {self.dataset_path}\n")
            self.output.emit(f"Command: {' '.join(cmd)}\n\n")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            
            for line in iter(process.stdout.readline, ''):
                if self._cancelled:
                    process.terminate()
                    self.finished.emit(False, "Cancelled")
                    return
                
                self.output.emit(line)
                
                if "Epoch" in line:
                    try:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == "Epoch":
                                epoch_str = parts[i + 1].replace(":", "").split("/")[0]
                                current_epoch = int(epoch_str)
                                self.progress.emit(current_epoch, self.epochs)
                                break
                    except:
                        pass
            
            process.wait()
            
            if process.returncode == 0:
                self.finished.emit(True, "Training completed successfully!")
            else:
                self.finished.emit(False, f"Training failed (code {process.returncode})")
                
        except Exception as e:
            self.finished.emit(False, str(e))


class TrainingPanel(QWidget):
    """Clean model training panel with educational guidance."""
    
    def __init__(self):
        super().__init__()
        self.worker = None
        self.datasets_dir = DATASETS_DIR
        
        self._setup_ui()
        self._refresh_datasets()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(24)
        
        # Header
        header_layout = QVBoxLayout()
        header_layout.setSpacing(4)
        
        title = QLabel("Model Training")
        title.setStyleSheet("font-size: 22px; font-weight: 500; color: #c8d0e0;")
        header_layout.addWidget(title)
        
        subtitle = QLabel("Fine-tune the AI model on astronomical datasets for better accuracy")
        subtitle.setStyleSheet("color: #7a8599; font-size: 13px;")
        header_layout.addWidget(subtitle)
        
        layout.addLayout(header_layout)
        
        # Main content scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(20)
        
        # Current Model Status
        status_card = self._create_status_card()
        content_layout.addWidget(status_card)
        
        # Dataset Selection
        dataset_card = self._create_dataset_card()
        content_layout.addWidget(dataset_card)
        
        # Training Settings
        settings_card = self._create_settings_card()
        content_layout.addWidget(settings_card)
        
        # Training Progress
        progress_card = self._create_progress_card()
        content_layout.addWidget(progress_card)
        
        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)
    
    def _create_status_card(self) -> QFrame:
        """Model status card."""
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background: rgba(14, 18, 24, 0.5);
                border: 1px solid rgba(30, 40, 55, 0.4);
                border-radius: 12px;
            }
        """)
        
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(8)
        
        header = QHBoxLayout()
        icon = QLabel("🧠")
        icon.setStyleSheet("font-size: 18px;")
        header.addWidget(icon)
        
        title = QLabel("Current Model")
        title.setStyleSheet("color: #c8d0e0; font-size: 14px; font-weight: 500;")
        header.addWidget(title)
        header.addStretch()
        layout.addLayout(header)
        
        # Check model status
        weights_path = WEIGHTS_DIR / "vit_astrolens"
        if weights_path.exists() and (weights_path / "config.json").exists():
            import json
            try:
                with open(weights_path / "config.json") as f:
                    config = json.load(f)
                num_classes = len(config.get("id2label", {}))
                status_text = f"Fine-tuned · {num_classes} classes"
                status_color = "#34d399"
                status_icon = "✓"
            except:
                status_text = "Fine-tuned model available"
                status_color = "#34d399"
                status_icon = "✓"
        else:
            status_text = "Pre-trained only (not fine-tuned yet)"
            status_color = "#fbbf24"
            status_icon = "○"
        
        status_row = QHBoxLayout()
        status_indicator = QLabel(status_icon)
        status_indicator.setStyleSheet(f"color: {status_color}; font-size: 14px;")
        status_row.addWidget(status_indicator)
        
        self.model_status = QLabel(status_text)
        self.model_status.setStyleSheet(f"color: {status_color}; font-size: 13px;")
        status_row.addWidget(self.model_status)
        status_row.addStretch()
        layout.addLayout(status_row)
        
        return card
    
    def _create_dataset_card(self) -> QFrame:
        """Dataset selection card."""
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background: rgba(14, 18, 24, 0.5);
                border: 1px solid rgba(30, 40, 55, 0.4);
                border-radius: 12px;
            }
        """)
        
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)
        
        header = QHBoxLayout()
        icon = QLabel("📁")
        icon.setStyleSheet("font-size: 18px;")
        header.addWidget(icon)
        
        title = QLabel("Training Dataset")
        title.setStyleSheet("color: #c8d0e0; font-size: 14px; font-weight: 500;")
        header.addWidget(title)
        header.addStretch()
        layout.addLayout(header)
        
        # Dataset combo with buttons
        dataset_row = QHBoxLayout()
        dataset_row.setSpacing(10)
        
        self.dataset_combo = QComboBox()
        self.dataset_combo.setMinimumWidth(180)
        self.dataset_combo.setStyleSheet("""
            QComboBox {
                background: rgba(10, 13, 18, 0.7);
                border: 1px solid rgba(40, 50, 70, 0.5);
                border-radius: 8px;
                padding: 8px 12px;
                color: #c8d0e0;
                font-size: 13px;
            }
            QComboBox::drop-down {
                border: none;
                padding-right: 10px;
            }
            QComboBox::down-arrow {
                image: none;
                border: none;
            }
        """)
        dataset_row.addWidget(self.dataset_combo)
        
        btn_style = """
            QPushButton {
                background: rgba(20, 24, 32, 0.6);
                border: 1px solid rgba(40, 50, 70, 0.4);
                padding: 8px 12px;
                border-radius: 8px;
                color: #7a8599;
                font-size: 12px;
            }
            QPushButton:hover {
                background: rgba(30, 40, 55, 0.6);
                color: #a0aec0;
            }
        """
        
        refresh_btn = QPushButton("↻ Refresh")
        refresh_btn.setStyleSheet(btn_style)
        refresh_btn.clicked.connect(self._refresh_datasets)
        dataset_row.addWidget(refresh_btn)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.setStyleSheet(btn_style)
        browse_btn.clicked.connect(self._browse_dataset)
        dataset_row.addWidget(browse_btn)
        
        dataset_row.addStretch()
        layout.addLayout(dataset_row)
        
        # Description
        desc = QLabel(
            "💡 Galaxy10 teaches 10 common galaxy shapes. "
            "Add your own datasets with labeled class folders."
        )
        desc.setStyleSheet("color: #4a5568; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        return card
    
    def _create_settings_card(self) -> QFrame:
        """Training settings card."""
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background: rgba(14, 18, 24, 0.5);
                border: 1px solid rgba(30, 40, 55, 0.4);
                border-radius: 12px;
            }
        """)
        
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)
        
        header = QHBoxLayout()
        icon = QLabel("⚙️")
        icon.setStyleSheet("font-size: 18px;")
        header.addWidget(icon)
        
        title = QLabel("Training Settings")
        title.setStyleSheet("color: #c8d0e0; font-size: 14px; font-weight: 500;")
        header.addWidget(title)
        header.addStretch()
        layout.addLayout(header)
        
        # Settings in a clean row
        settings_row = QHBoxLayout()
        settings_row.setSpacing(24)
        
        # Epochs
        epochs_group = QVBoxLayout()
        epochs_group.setSpacing(4)
        epochs_label = QLabel("Epochs")
        epochs_label.setStyleSheet("color: #7a8599; font-size: 12px;")
        epochs_group.addWidget(epochs_label)
        
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 50)
        self.epochs_spin.setValue(5)
        self.epochs_spin.setFixedWidth(70)
        self.epochs_spin.setStyleSheet("""
            QSpinBox {
                background: rgba(10, 13, 18, 0.7);
                border: 1px solid rgba(40, 50, 70, 0.5);
                border-radius: 6px;
                padding: 6px;
                color: #c8d0e0;
            }
        """)
        epochs_group.addWidget(self.epochs_spin)
        settings_row.addLayout(epochs_group)
        
        # Batch size
        batch_group = QVBoxLayout()
        batch_group.setSpacing(4)
        batch_label = QLabel("Batch Size")
        batch_label.setStyleSheet("color: #7a8599; font-size: 12px;")
        batch_group.addWidget(batch_label)
        
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(4, 64)
        self.batch_spin.setValue(16)
        self.batch_spin.setFixedWidth(70)
        self.batch_spin.setStyleSheet("""
            QSpinBox {
                background: rgba(10, 13, 18, 0.7);
                border: 1px solid rgba(40, 50, 70, 0.5);
                border-radius: 6px;
                padding: 6px;
                color: #c8d0e0;
            }
        """)
        batch_group.addWidget(self.batch_spin)
        settings_row.addLayout(batch_group)
        
        settings_row.addStretch()
        
        # Start button
        self.start_btn = QPushButton("Start Training")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(99, 102, 241, 0.8), stop:1 rgba(139, 92, 246, 0.8));
                border: none;
                color: white;
                font-weight: 500;
                font-size: 13px;
                padding: 10px 20px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(99, 102, 241, 1), stop:1 rgba(139, 92, 246, 1));
            }
            QPushButton:disabled {
                background: rgba(60, 70, 90, 0.5);
                color: #4a5568;
            }
        """)
        self.start_btn.clicked.connect(self._start_training)
        settings_row.addWidget(self.start_btn)
        
        layout.addLayout(settings_row)
        
        # Tips
        tips = QLabel(
            "💡 Epochs = training passes. More epochs = better accuracy but slower. "
            "Batch size affects memory usage."
        )
        tips.setStyleSheet("color: #4a5568; font-size: 11px;")
        tips.setWordWrap(True)
        layout.addWidget(tips)
        
        return card
    
    def _create_progress_card(self) -> QFrame:
        """Training progress card."""
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background: rgba(14, 18, 24, 0.5);
                border: 1px solid rgba(30, 40, 55, 0.4);
                border-radius: 12px;
            }
        """)
        
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)
        
        header = QHBoxLayout()
        icon = QLabel("📊")
        icon.setStyleSheet("font-size: 18px;")
        header.addWidget(icon)
        
        title = QLabel("Training Progress")
        title.setStyleSheet("color: #c8d0e0; font-size: 14px; font-weight: 500;")
        header.addWidget(title)
        header.addStretch()
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background: rgba(248, 113, 113, 0.15);
                border: none;
                color: #f87171;
                padding: 6px 12px;
                border-radius: 6px;
                font-size: 11px;
            }
            QPushButton:hover { background: rgba(248, 113, 113, 0.25); }
            QPushButton:disabled { opacity: 0.4; color: #4a5568; }
        """)
        self.cancel_btn.clicked.connect(self._cancel_training)
        header.addWidget(self.cancel_btn)
        layout.addLayout(header)
        
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
        layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("Ready to train")
        self.progress_label.setStyleSheet("color: #7a8599; font-size: 12px;")
        layout.addWidget(self.progress_label)
        
        # Output log
        self.output_log = QTextEdit()
        self.output_log.setReadOnly(True)
        self.output_log.setMaximumHeight(150)
        self.output_log.setStyleSheet("""
            QTextEdit {
                background: rgba(10, 13, 18, 0.7);
                border: 1px solid rgba(30, 40, 55, 0.3);
                border-radius: 8px;
                color: #7a8599;
                font-family: 'Menlo', 'Monaco', monospace;
                font-size: 11px;
                padding: 8px;
            }
        """)
        self.output_log.setPlaceholderText("Training output will appear here...")
        layout.addWidget(self.output_log)
        
        return card
    
    def _refresh_datasets(self):
        """Refresh available datasets."""
        self.dataset_combo.clear()
        
        if not self.datasets_dir.exists():
            self.datasets_dir.mkdir(parents=True, exist_ok=True)
        
        # Add available datasets
        for item in sorted(self.datasets_dir.iterdir()):
            if item.is_dir():
                # Count classes
                classes = [d for d in item.iterdir() if d.is_dir()]
                if classes:
                    label = f"{item.name} ({len(classes)} classes)"
                else:
                    label = item.name
                self.dataset_combo.addItem(label, str(item))
        
        if self.dataset_combo.count() == 0:
            self.dataset_combo.addItem("No datasets found", "")
    
    def _browse_dataset(self):
        """Browse for a custom dataset folder."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Dataset Folder",
            str(self.datasets_dir),
        )
        
        if folder:
            self.dataset_combo.addItem(Path(folder).name, folder)
            self.dataset_combo.setCurrentIndex(self.dataset_combo.count() - 1)
    
    def _start_training(self):
        """Start the training process."""
        dataset_path = self.dataset_combo.currentData()
        
        if not dataset_path or not Path(dataset_path).exists():
            QMessageBox.warning(self, "No Dataset", "Please select a valid dataset first.")
            return
        
        self.output_log.clear()
        self.progress_bar.setValue(0)
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_label.setText("Starting...")
        
        self.worker = TrainingWorker(
            dataset_path,
            self.epochs_spin.value(),
            self.batch_spin.value(),
        )
        self.worker.output.connect(self._on_output)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()
    
    def _cancel_training(self):
        """Cancel training."""
        if self.worker:
            self.worker.cancel()
            self.progress_label.setText("Cancelling...")
    
    def _on_output(self, text: str):
        self.output_log.append(text.strip())
        scrollbar = self.output_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def _on_progress(self, current: int, total: int):
        percent = int(current / total * 100) if total > 0 else 0
        self.progress_bar.setValue(percent)
        self.progress_label.setText(f"Epoch {current}/{total}")
    
    def _on_finished(self, success: bool, message: str):
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        
        if success:
            self.progress_bar.setValue(100)
            self.progress_label.setText("✓ " + message)
            self.model_status.setText("Fine-tuned model ready")
            self.model_status.setStyleSheet("color: #34d399; font-size: 13px;")
        else:
            self.progress_label.setText("✗ " + message)
        
        if self.worker:
            self.worker.wait(2000)
            self.worker.deleteLater()
            self.worker = None
