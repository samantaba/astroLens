"""
Download Manager Panel

Track and manage image downloads with progress and history.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QProgressBar, QScrollArea, QMessageBox,
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer


class DownloadWorker(QThread):
    """Background worker for downloading images."""
    
    progress = pyqtSignal(int, int, str)
    source_complete = pyqtSignal(str, int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    cancelled = pyqtSignal()
    
    def __init__(self, sources: dict, output_dir: Path, upload: bool = True, analyze: bool = True):
        super().__init__()
        self.sources = sources
        self.output_dir = output_dir
        self.upload = upload
        self.analyze = analyze
        self._cancelled = False
    
    def cancel(self):
        self._cancelled = True
    
    def run(self):
        try:
            scripts_path = Path(__file__).parent.parent
            if str(scripts_path) not in sys.path:
                sys.path.insert(0, str(scripts_path))
            
            all_downloads = []
            total_expected = sum(s.get("daily_count", 10) for s in self.sources.values())
            current = 0
            
            for source_id, config in self.sources.items():
                if self._cancelled:
                    self.cancelled.emit()
                    return
                
                count = config.get("daily_count", 10)
                source_dir = self.output_dir / source_id
                source_name = config.get("name", source_id)
                
                self.progress.emit(current, total_expected, f"Downloading {source_name}...")
                
                try:
                    # Download with progress callback
                    downloads = self._download_source(source_id, count, source_dir, source_name, current, total_expected)
                    
                    if self._cancelled:
                        self.cancelled.emit()
                        return
                    
                    all_downloads.extend(downloads)
                    current += len(downloads)
                    self.source_complete.emit(source_id, len(downloads))
                    self.progress.emit(current, total_expected, f"{source_name}: {len(downloads)} done")
                    
                except Exception as e:
                    self.progress.emit(current, total_expected, f"{source_name} error: {str(e)[:50]}")
                
                if self._cancelled:
                    self.cancelled.emit()
                    return
            
            results = {
                "downloaded": len(all_downloads),
                "files": [str(f) for f in all_downloads],
                "uploaded": 0,
                "analyzed": 0,
                "anomalies": 0,
                "output_dir": str(self.output_dir),
            }
            
            if self.upload and all_downloads and not self._cancelled:
                self.progress.emit(current, total_expected, "Uploading...")
                try:
                    from scripts.nightly_ingest import upload_to_astrolens
                    upload_results = upload_to_astrolens(all_downloads, analyze=self.analyze, alert=True)
                    results["uploaded"] = upload_results.get("uploaded", 0)
                    results["analyzed"] = upload_results.get("analyzed", 0)
                    results["anomalies"] = upload_results.get("anomalies", 0)
                except Exception:
                    pass
            
            if self._cancelled:
                self.cancelled.emit()
            else:
                self.finished.emit(results)
            
        except Exception as e:
            self.error.emit(str(e))
    
    def _download_source(self, source_id: str, count: int, source_dir: Path, source_name: str, base_current: int, total: int) -> list:
        """Download from a source with real-time progress updates."""
        from scripts.nightly_ingest import (
            download_sdss_galaxies,
            download_ztf_alerts,
            download_nasa_apod,
        )
        
        source_dir.mkdir(parents=True, exist_ok=True)
        
        # Progress callback that emits UI updates
        def on_progress(current, source_total):
            if self._cancelled:
                return
            self.progress.emit(
                base_current + current, 
                total, 
                f"{source_name}: {current}/{source_total}"
            )
        
        if source_id == "sdss":
            return download_sdss_galaxies(count, source_dir, on_progress=on_progress)
        elif source_id == "ztf":
            return download_ztf_alerts(count, source_dir, on_progress=on_progress)
        elif source_id == "apod":
            return download_nasa_apod(count, source_dir, on_progress=on_progress)
        
        return []


class DownloadHistoryItem(QFrame):
    """Download history entry."""
    
    open_clicked = pyqtSignal(str)
    delete_clicked = pyqtSignal(str)
    
    def __init__(self, batch_id: str, sources: dict, timestamp: datetime, folder: str):
        super().__init__()
        self.folder = folder
        self.batch_id = batch_id
        
        self.setStyleSheet("""
            QFrame {
                background: rgba(14, 18, 24, 0.5);
                border: 1px solid rgba(30, 40, 55, 0.4);
                border-radius: 10px;
            }
            QFrame:hover { border-color: rgba(60, 70, 90, 0.5); }
        """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        
        info = QVBoxLayout()
        info.setSpacing(4)
        
        title = QLabel(timestamp.strftime("%b %d, %Y at %H:%M"))
        title.setStyleSheet("font-weight: 500; color: #c8d0e0; font-size: 13px;")
        info.addWidget(title)
        
        source_texts = []
        total_count = 0
        for source, count in sources.items():
            icon = "🌌" if source == "sdss" else ("⚡" if source == "ztf" else "🌠")
            source_texts.append(f"{icon} {source.upper()}: {count}")
            total_count += count
        
        details = QLabel(" · ".join(source_texts) if source_texts else "Empty")
        details.setStyleSheet("font-size: 11px; color: #7a8599;")
        info.addWidget(details)
        
        layout.addLayout(info)
        layout.addStretch()
        
        total_label = QLabel(f"{total_count}")
        total_label.setStyleSheet("""
            background: rgba(91, 141, 239, 0.15);
            color: #6b9fff;
            padding: 4px 10px;
            border-radius: 10px;
            font-size: 12px;
            font-weight: 500;
        """)
        layout.addWidget(total_label)
        
        open_btn = QPushButton("Open")
        open_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                color: #7a8599;
                padding: 6px 10px;
                font-size: 12px;
            }
            QPushButton:hover { color: #a0aec0; }
        """)
        open_btn.setCursor(Qt.PointingHandCursor)
        open_btn.clicked.connect(lambda: self.open_clicked.emit(self.folder))
        layout.addWidget(open_btn)
        
        delete_btn = QPushButton("×")
        delete_btn.setFixedWidth(24)
        delete_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                color: #7a8599;
                font-size: 16px;
            }
            QPushButton:hover { color: #f87171; }
        """)
        delete_btn.setCursor(Qt.PointingHandCursor)
        delete_btn.clicked.connect(lambda: self.delete_clicked.emit(self.folder))
        layout.addWidget(delete_btn)


class DownloadPanel(QWidget):
    """Download manager panel."""
    
    download_complete = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.worker: Optional[DownloadWorker] = None
        self.downloads_dir = Path("downloads")
        self.history_items: List[DownloadHistoryItem] = []
        self._cancel_timer: Optional[QTimer] = None
        
        self._setup_ui()
        self._load_history()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)
        
        header = QLabel("Downloads")
        header.setStyleSheet("font-size: 18px; font-weight: 500; color: #c8d0e0;")
        layout.addWidget(header)
        
        # Active download card
        active_frame = QFrame()
        active_frame.setStyleSheet("""
            QFrame {
                background: rgba(14, 18, 24, 0.5);
                border: 1px solid rgba(30, 40, 55, 0.4);
                border-radius: 12px;
            }
        """)
        active_layout = QVBoxLayout(active_frame)
        active_layout.setContentsMargins(20, 18, 20, 18)
        active_layout.setSpacing(12)
        
        active_header = QHBoxLayout()
        active_label = QLabel("Active Download")
        active_label.setStyleSheet("font-size: 13px; font-weight: 500; color: #c8d0e0;")
        active_header.addWidget(active_label)
        active_header.addStretch()
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background: rgba(248, 113, 113, 0.1);
                border: 1px solid rgba(248, 113, 113, 0.2);
                padding: 6px 14px;
                border-radius: 6px;
                color: rgba(248, 113, 113, 0.8);
                font-size: 11px;
            }
            QPushButton:hover { background: rgba(248, 113, 113, 0.2); }
            QPushButton:disabled { color: #4a5568; border-color: rgba(50, 60, 80, 0.3); }
        """)
        self.cancel_btn.clicked.connect(self._cancel_download)
        active_header.addWidget(self.cancel_btn)
        active_layout.addLayout(active_header)
        
        self.status_label = QLabel("No download in progress")
        self.status_label.setStyleSheet("color: #7a8599; font-size: 13px;")
        active_layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background: rgba(10, 13, 18, 0.6);
                border: none;
                border-radius: 3px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(91, 141, 239, 0.9), stop:1 rgba(78, 205, 196, 0.9));
                border-radius: 3px;
            }
        """)
        active_layout.addWidget(self.progress_bar)
        
        self.detail_label = QLabel("")
        self.detail_label.setStyleSheet("color: #4a5568; font-size: 11px;")
        active_layout.addWidget(self.detail_label)
        
        layout.addWidget(active_frame)
        
        # History header
        history_header = QHBoxLayout()
        history_label = QLabel("Download History")
        history_label.setStyleSheet("font-size: 14px; font-weight: 500; color: #c8d0e0;")
        history_header.addWidget(history_label)
        history_header.addStretch()
        
        clear_all_btn = QPushButton("Clear All")
        clear_all_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                color: #7a8599;
                font-size: 12px;
            }
            QPushButton:hover { color: #f87171; }
        """)
        clear_all_btn.clicked.connect(self._clear_all_history)
        history_header.addWidget(clear_all_btn)
        
        open_folder_btn = QPushButton("Open Folder")
        open_folder_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                color: #6b9fff;
                font-size: 12px;
            }
            QPushButton:hover { color: #8bb4ff; }
        """)
        open_folder_btn.clicked.connect(self._open_downloads_folder)
        history_header.addWidget(open_folder_btn)
        
        layout.addLayout(history_header)
        
        # History scroll
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        
        self.history_widget = QWidget()
        self.history_widget.setStyleSheet("background: transparent;")
        self.history_layout = QVBoxLayout(self.history_widget)
        self.history_layout.setSpacing(8)
        self.history_layout.setContentsMargins(0, 0, 0, 0)
        
        self.no_history_label = QLabel("No downloads yet")
        self.no_history_label.setAlignment(Qt.AlignCenter)
        self.no_history_label.setStyleSheet("color: #4a5568; padding: 40px;")
        self.history_layout.addWidget(self.no_history_label)
        self.history_layout.addStretch()
        
        scroll.setWidget(self.history_widget)
        layout.addWidget(scroll, 1)
    
    def _load_history(self):
        for item in self.history_items:
            item.deleteLater()
        self.history_items.clear()
        
        if not self.downloads_dir.exists():
            self.no_history_label.show()
            return
        
        batches = []
        
        for batch_dir in self.downloads_dir.iterdir():
            if batch_dir.is_dir() and len(batch_dir.name) >= 15 and batch_dir.name[:8].isdigit():
                try:
                    timestamp = datetime.strptime(batch_dir.name[:15], "%Y%m%d_%H%M%S")
                except ValueError:
                    continue
                
                sources = {}
                for source_dir in batch_dir.iterdir():
                    if source_dir.is_dir():
                        images = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png"))
                        if images:
                            sources[source_dir.name] = len(images)
                
                if sources:
                    batches.append({
                        "id": batch_dir.name,
                        "folder": str(batch_dir),
                        "timestamp": timestamp,
                        "sources": sources,
                    })
        
        for source_name in ["sdss", "ztf", "apod", "eso"]:
            source_dir = self.downloads_dir / source_name
            if source_dir.is_dir():
                images = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png"))
                if images:
                    latest = max(img.stat().st_mtime for img in images)
                    batches.append({
                        "id": f"legacy_{source_name}",
                        "folder": str(source_dir),
                        "timestamp": datetime.fromtimestamp(latest),
                        "sources": {source_name: len(images)},
                    })
        
        if not batches:
            self.no_history_label.show()
            return
        
        self.no_history_label.hide()
        batches.sort(key=lambda x: x["timestamp"], reverse=True)
        
        for batch in batches[:20]:
            item = DownloadHistoryItem(
                batch["id"],
                batch["sources"],
                batch["timestamp"],
                batch["folder"],
            )
            item.open_clicked.connect(self._open_folder)
            item.delete_clicked.connect(self._delete_batch)
            self.history_items.append(item)
            self.history_layout.insertWidget(len(self.history_items) - 1, item)
    
    def start_download(self, sources: dict, upload: bool = True, analyze: bool = True):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "In Progress", "A download is already running.")
            return
        
        batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.downloads_dir / batch_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.worker = DownloadWorker(sources, output_dir, upload, analyze)
        self.worker.progress.connect(self._on_progress)
        self.worker.source_complete.connect(self._on_source_complete)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.cancelled.connect(self._on_cancelled)
        self.worker.start()
        
        self.cancel_btn.setEnabled(True)
        self.status_label.setText("Starting...")
        self.progress_bar.setValue(0)
        self.detail_label.setText(f"Sources: {', '.join(sources.keys())}")
    
    def _on_progress(self, current: int, total: int, message: str):
        if total > 0:
            self.progress_bar.setValue(int(current / total * 100))
        self.status_label.setText(message)
        self.detail_label.setText(f"{current}/{total} images")
    
    def _on_source_complete(self, source: str, count: int):
        pass
    
    def _on_finished(self, results: dict):
        self._cleanup_worker()
        self.progress_bar.setValue(100)
        
        parts = [f"Downloaded: {results['downloaded']}"]
        if results.get('uploaded'):
            parts.append(f"Uploaded: {results['uploaded']}")
        if results.get('anomalies'):
            parts.append(f"Anomalies: {results['anomalies']}")
        
        self.status_label.setText(" · ".join(parts))
        self.detail_label.setText("Complete")
        
        self._load_history()
        self.download_complete.emit(results)
    
    def _on_error(self, error: str):
        self._cleanup_worker()
        self.status_label.setText(f"Error: {error}")
        self.progress_bar.setValue(0)
    
    def _on_cancelled(self):
        self._cleanup_worker()
        self.status_label.setText("Cancelled")
        self.detail_label.setText("")
        self._load_history()
    
    def _cleanup_worker(self):
        self.cancel_btn.setEnabled(False)
        if self._cancel_timer:
            self._cancel_timer.stop()
            self._cancel_timer = None
        
        # Properly cleanup the worker thread
        if self.worker:
            if self.worker.isRunning():
                self.worker.wait(2000)  # Wait up to 2 seconds
            self.worker.deleteLater()
            self.worker = None
    
    def _cancel_download(self):
        if not self.worker:
            return
        
        self.status_label.setText("Cancelling...")
        self.cancel_btn.setEnabled(False)
        self.worker.cancel()
        
        # Force-terminate after 5 seconds if still running
        self._cancel_timer = QTimer()
        self._cancel_timer.setSingleShot(True)
        self._cancel_timer.timeout.connect(self._force_cancel)
        self._cancel_timer.start(5000)
    
    def _force_cancel(self):
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait(1000)
        self._cleanup_worker()
        self.status_label.setText("Cancelled (forced)")
        self.detail_label.setText("")
        self._load_history()
    
    def _delete_batch(self, folder: str):
        reply = QMessageBox.question(
            self, "Delete Download",
            "Delete this download batch?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                shutil.rmtree(folder)
                self._load_history()
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not delete: {e}")
    
    def _clear_all_history(self):
        if not self.history_items:
            return
        
        reply = QMessageBox.question(
            self, "Clear All",
            f"Delete all {len(self.history_items)} download batches?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                if self.downloads_dir.exists():
                    shutil.rmtree(self.downloads_dir)
                    self.downloads_dir.mkdir(parents=True, exist_ok=True)
                self._load_history()
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not clear: {e}")
    
    def _open_folder(self, folder_path: str):
        import platform
        if platform.system() == "Darwin":
            subprocess.run(["open", folder_path])
        elif platform.system() == "Windows":
            subprocess.run(["explorer", folder_path])
        else:
            subprocess.run(["xdg-open", folder_path])
    
    def _open_downloads_folder(self):
        self.downloads_dir.mkdir(parents=True, exist_ok=True)
        self._open_folder(str(self.downloads_dir))
