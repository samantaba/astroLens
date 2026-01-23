"""
Chat Panel

Premium conversational interface with refined typography.
"""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QLineEdit, QTextEdit, QSizePolicy,
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, pyqtSlot

import httpx


class ChatWorker(QThread):
    """Background worker for chat requests."""
    
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, api: httpx.Client, message: str):
        super().__init__()
        self.api = api
        self.message = message
    
    def run(self):
        try:
            response = self.api.post("/chat", json={"message": self.message})
            if response.status_code == 200:
                data = response.json()
                self.finished.emit(data.get("reply", ""))
            else:
                self.error.emit(f"API error: {response.status_code}")
        except Exception as e:
            self.error.emit(str(e))


class MessageBubble(QFrame):
    """
    Premium message bubble with refined typography.
    Clean, minimal, sophisticated.
    """
    
    def __init__(self, text: str, is_user: bool):
        super().__init__()
        
        # Subtle, refined styling
        if is_user:
            self.setStyleSheet("""
                QFrame {
                    background: rgba(91, 141, 239, 0.12);
                    border: 1px solid rgba(91, 141, 239, 0.2);
                    border-radius: 16px;
                    border-bottom-right-radius: 4px;
                    margin-left: 100px;
                }
            """)
        else:
            self.setStyleSheet("""
                QFrame {
                    background: rgba(20, 24, 32, 0.6);
                    border: 1px solid rgba(40, 50, 70, 0.3);
                    border-radius: 16px;
                    border-bottom-left-radius: 4px;
                    margin-right: 100px;
                }
            """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(6)
        
        # Sender - subtle, minimal
        sender = "You" if is_user else "AstroLens"
        sender_label = QLabel(sender)
        color = "rgba(107, 159, 255, 0.8)" if is_user else "rgba(155, 124, 239, 0.8)"
        sender_label.setStyleSheet(f"""
            font-size: 11px;
            color: {color};
            font-weight: 500;
            letter-spacing: 0.3px;
            background: transparent;
        """)
        layout.addWidget(sender_label)
        
        # Message text - refined typography
        text_label = QLabel(text)
        text_label.setWordWrap(True)
        text_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        text_label.setStyleSheet("""
            font-size: 14px;
            color: #c8d0e0;
            line-height: 1.5;
            background: transparent;
            font-weight: 400;
        """)
        layout.addWidget(text_label)


class ChatPanel(QWidget):
    """
    Premium chat interface.
    Clean, minimal, cutting-edge.
    """
    
    data_changed = pyqtSignal()
    
    def __init__(self, api: httpx.Client):
        super().__init__()
        self.api = api
        self.worker = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(24)
        
        # Header - minimal, refined
        header_layout = QVBoxLayout()
        header_layout.setSpacing(6)
        
        title = QLabel("AI Assistant")
        title.setStyleSheet("""
            font-size: 28px;
            font-weight: 300;
            color: #c8d0e0;
            letter-spacing: -0.5px;
        """)
        header_layout.addWidget(title)
        
        subtitle = QLabel("Natural language control for your observatory")
        subtitle.setStyleSheet("""
            font-size: 13px;
            color: #4a5568;
            font-weight: 400;
        """)
        header_layout.addWidget(subtitle)
        
        layout.addLayout(header_layout)
        
        # Chat area - clean container
        chat_container = QFrame()
        chat_container.setStyleSheet("""
            QFrame {
                background: rgba(10, 13, 18, 0.4);
                border: 1px solid rgba(30, 40, 55, 0.3);
                border-radius: 16px;
            }
        """)
        chat_layout = QVBoxLayout(chat_container)
        chat_layout.setContentsMargins(0, 0, 0, 0)
        
        # Scroll area for messages
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
        """)
        
        self.messages_widget = QWidget()
        self.messages_widget.setStyleSheet("background: transparent;")
        self.messages_layout = QVBoxLayout(self.messages_widget)
        self.messages_layout.setContentsMargins(20, 20, 20, 20)
        self.messages_layout.setSpacing(16)
        self.messages_layout.addStretch()
        
        scroll.setWidget(self.messages_widget)
        chat_layout.addWidget(scroll, 1)
        
        layout.addWidget(chat_container, 1)
        
        # Input area - sleek, modern
        input_frame = QFrame()
        input_frame.setStyleSheet("""
            QFrame {
                background: rgba(14, 18, 24, 0.6);
                border: 1px solid rgba(40, 50, 70, 0.4);
                border-radius: 14px;
            }
        """)
        input_layout = QHBoxLayout(input_frame)
        input_layout.setContentsMargins(16, 12, 12, 12)
        input_layout.setSpacing(12)
        
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Ask anything... (e.g., 'analyze all images', 'find anomalies')")
        self.input_field.setStyleSheet("""
            QLineEdit {
                background: transparent;
                border: none;
                padding: 8px 4px;
                color: #c8d0e0;
                font-size: 14px;
                font-weight: 400;
            }
            QLineEdit::placeholder {
                color: #4a5568;
            }
        """)
        self.input_field.returnPressed.connect(self._send_message)
        input_layout.addWidget(self.input_field)
        
        self.send_btn = QPushButton("→")
        self.send_btn.setFixedSize(40, 40)
        self.send_btn.setCursor(Qt.PointingHandCursor)
        self.send_btn.setStyleSheet("""
            QPushButton {
                background: rgba(91, 141, 239, 0.9);
                border: none;
                border-radius: 10px;
                color: white;
                font-size: 18px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: rgba(107, 159, 255, 1);
            }
            QPushButton:disabled {
                background: rgba(50, 60, 80, 0.4);
                color: #4a5568;
            }
        """)
        self.send_btn.clicked.connect(self._send_message)
        input_layout.addWidget(self.send_btn)
        
        layout.addWidget(input_frame)
        
        # Suggestions - subtle chips
        suggestions = QHBoxLayout()
        suggestions.setSpacing(8)
        
        suggestion_style = """
            QPushButton {
                background: rgba(30, 40, 55, 0.4);
                border: 1px solid rgba(50, 60, 80, 0.3);
                border-radius: 16px;
                padding: 8px 14px;
                color: #7a8599;
                font-size: 12px;
                font-weight: 400;
            }
            QPushButton:hover {
                background: rgba(40, 50, 70, 0.5);
                color: #a0aec0;
                border-color: rgba(91, 141, 239, 0.3);
            }
        """
        
        for text in ["Analyze all images", "Find anomalies", "Show statistics"]:
            btn = QPushButton(text)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet(suggestion_style)
            btn.clicked.connect(lambda _, t=text: self._send_suggestion(t))
            suggestions.addWidget(btn)
        
        suggestions.addStretch()
        layout.addLayout(suggestions)
    
    def _send_suggestion(self, text: str):
        """Send a suggestion as message."""
        self.input_field.setText(text)
        self._send_message()
    
    def _send_message(self):
        """Send user message to API."""
        text = self.input_field.text().strip()
        if not text:
            return
        
        # Add user message
        self._add_message(text, is_user=True)
        self.input_field.clear()
        
        # Disable input while processing
        self.input_field.setEnabled(False)
        self.send_btn.setEnabled(False)
        
        # Start worker
        self.worker = ChatWorker(self.api, text)
        self.worker.finished.connect(self._on_reply)
        self.worker.error.connect(self._on_error)
        self.worker.start()
    
    @pyqtSlot(str)
    def _on_reply(self, reply: str):
        """Handle API reply."""
        self._add_message(reply, is_user=False)
        self.input_field.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.input_field.setFocus()
        self.data_changed.emit()
    
    @pyqtSlot(str)
    def _on_error(self, error: str):
        """Handle API error."""
        self._add_message(f"Error: {error}", is_user=False)
        self.input_field.setEnabled(True)
        self.send_btn.setEnabled(True)
    
    def _add_message(self, text: str, is_user: bool):
        """Add message bubble to chat."""
        bubble = MessageBubble(text, is_user)
        # Insert before the stretch
        self.messages_layout.insertWidget(self.messages_layout.count() - 1, bubble)
        
        # Scroll to bottom
        scroll = self.messages_widget.parent()
        if isinstance(scroll, QScrollArea):
            scroll.verticalScrollBar().setValue(
                scroll.verticalScrollBar().maximum()
            )
