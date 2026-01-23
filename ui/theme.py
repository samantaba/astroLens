"""
AstroLens Theme - Cutting Edge Discovery Experience

Premium, sophisticated UI with refined typography and subtle elegance.
Designed to feel like sitting at the edge of technology.
"""

# Color Palette - Refined & Sophisticated
COLORS = {
    # Backgrounds - Ultra dark with subtle depth
    "void": "#06080c",
    "deep": "#0a0d12",
    "surface": "#0e1218",
    "elevated": "#141820",
    "hover": "#1a2030",
    
    # Borders - Barely visible, ultra subtle
    "border_subtle": "#1e2535",
    "border_dim": "#252d40",
    "border_active": "#3a4560",
    
    # Text - Refined hierarchy (NOT bold/harsh)
    "text_primary": "#c8d0e0",      # Soft white, easy on eyes
    "text_secondary": "#7a8599",    # Muted
    "text_tertiary": "#4a5568",     # Very subtle
    "text_accent": "#6b9fff",       # Soft blue accent
    
    # Accent - Subtle, sophisticated
    "accent_blue": "#5b8def",
    "accent_purple": "#9b7cef",
    "accent_cyan": "#4ecdc4",
    "accent_emerald": "#34d399",
    "accent_amber": "#fbbf24",
    "accent_rose": "#f472b6",
}

# Main application stylesheet - Premium feel
MAIN_STYLESHEET = """
/* ============================================
   GLOBAL - Refined Typography & Smooth Surfaces
   ============================================ */

* {
    font-family: 'SF Pro Display', 'Inter', 'Segoe UI', -apple-system, sans-serif;
}

QWidget {
    background-color: #0a0d12;
    color: #c8d0e0;
    font-size: 13px;
    font-weight: 400;
}

QMainWindow {
    background-color: #06080c;
}

QLabel {
    background: transparent;
    color: #c8d0e0;
    font-weight: 400;
}

/* ============================================
   SCROLLBARS - Nearly invisible, smooth
   ============================================ */

QScrollBar:vertical {
    background: transparent;
    width: 6px;
    margin: 4px 2px;
}
QScrollBar::handle:vertical {
    background: rgba(90, 100, 120, 0.3);
    border-radius: 3px;
    min-height: 40px;
}
QScrollBar::handle:vertical:hover {
    background: rgba(107, 159, 255, 0.5);
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: transparent;
    height: 0;
}

QScrollBar:horizontal {
    background: transparent;
    height: 6px;
    margin: 2px 4px;
}
QScrollBar::handle:horizontal {
    background: rgba(90, 100, 120, 0.3);
    border-radius: 3px;
}

/* ============================================
   BUTTONS - Subtle, elegant interactions
   ============================================ */

QPushButton {
    background: rgba(30, 37, 50, 0.6);
    border: 1px solid rgba(50, 60, 80, 0.4);
    border-radius: 10px;
    padding: 10px 18px;
    color: #a0aec0;
    font-weight: 500;
    font-size: 13px;
}
QPushButton:hover {
    background: rgba(40, 50, 70, 0.7);
    border-color: rgba(107, 159, 255, 0.4);
    color: #e2e8f0;
}
QPushButton:pressed {
    background: rgba(50, 60, 80, 0.5);
}
QPushButton:disabled {
    background: rgba(20, 25, 35, 0.4);
    color: #4a5568;
    border-color: transparent;
}

/* Primary Button - Soft glow, not harsh */
QPushButton#primary {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 rgba(91, 141, 239, 0.85), stop:1 rgba(107, 159, 255, 0.9));
    border: 1px solid rgba(107, 159, 255, 0.3);
    color: #ffffff;
    font-weight: 600;
}
QPushButton#primary:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 rgba(107, 159, 255, 0.95), stop:1 rgba(130, 180, 255, 1));
}

/* ============================================
   INPUTS - Clean, minimal
   ============================================ */

QLineEdit, QTextEdit, QSpinBox {
    background: rgba(10, 13, 18, 0.8);
    border: 1px solid rgba(40, 50, 70, 0.5);
    border-radius: 10px;
    padding: 12px 16px;
    color: #c8d0e0;
    font-size: 13px;
    selection-background-color: rgba(91, 141, 239, 0.4);
}
QLineEdit:focus, QTextEdit:focus, QSpinBox:focus {
    border-color: rgba(91, 141, 239, 0.6);
    background: rgba(14, 18, 24, 0.9);
}
QLineEdit::placeholder {
    color: #4a5568;
}

QComboBox {
    background: rgba(10, 13, 18, 0.8);
    border: 1px solid rgba(40, 50, 70, 0.5);
    border-radius: 10px;
    padding: 10px 14px;
    color: #a0aec0;
    min-width: 120px;
}
QComboBox:hover {
    border-color: rgba(91, 141, 239, 0.5);
}
QComboBox::drop-down {
    border: none;
    width: 24px;
}
QComboBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #4a5568;
    margin-right: 8px;
}
QComboBox QAbstractItemView {
    background: #0e1218;
    border: 1px solid rgba(40, 50, 70, 0.6);
    border-radius: 10px;
    padding: 4px;
    selection-background-color: rgba(91, 141, 239, 0.3);
}

/* ============================================
   GROUP BOX - Subtle containers
   ============================================ */

QGroupBox {
    background: rgba(14, 18, 24, 0.5);
    border: 1px solid rgba(30, 40, 55, 0.5);
    border-radius: 14px;
    margin-top: 20px;
    padding: 20px 16px 16px 16px;
    font-weight: 500;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 18px;
    top: 6px;
    padding: 0 10px;
    color: #7a8599;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* ============================================
   PROGRESS BAR - Smooth, elegant
   ============================================ */

QProgressBar {
    background: rgba(10, 13, 18, 0.6);
    border: none;
    border-radius: 6px;
    height: 8px;
    text-align: center;
    font-size: 11px;
    color: transparent;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 rgba(91, 141, 239, 0.9), stop:0.5 rgba(155, 124, 239, 0.9), stop:1 rgba(78, 205, 196, 0.9));
    border-radius: 6px;
}

/* ============================================
   CHECKBOX & RADIO - Minimal, refined
   ============================================ */

QCheckBox {
    spacing: 10px;
    color: #a0aec0;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 5px;
    border: 1px solid rgba(60, 70, 90, 0.6);
    background: rgba(10, 13, 18, 0.8);
}
QCheckBox::indicator:checked {
    background: rgba(91, 141, 239, 0.7);
    border-color: rgba(91, 141, 239, 0.5);
}
QCheckBox::indicator:hover {
    border-color: rgba(91, 141, 239, 0.6);
}

QRadioButton::indicator {
    width: 16px;
    height: 16px;
    border-radius: 8px;
    border: 1px solid rgba(60, 70, 90, 0.6);
    background: rgba(10, 13, 18, 0.8);
}
QRadioButton::indicator:checked {
    background: #5b8def;
    border: 4px solid rgba(10, 13, 18, 0.9);
}

/* ============================================
   TAB WIDGET - Clean, modern tabs
   ============================================ */

QTabWidget::pane {
    background: rgba(14, 18, 24, 0.4);
    border: 1px solid rgba(30, 40, 55, 0.4);
    border-radius: 14px;
    padding: 20px;
    margin-top: -1px;
}
QTabBar::tab {
    background: transparent;
    border: none;
    padding: 14px 28px;
    margin-right: 4px;
    color: #4a5568;
    font-weight: 500;
    font-size: 13px;
}
QTabBar::tab:selected {
    color: #c8d0e0;
    border-bottom: 2px solid #5b8def;
}
QTabBar::tab:hover:!selected {
    color: #7a8599;
}

/* ============================================
   LIST WIDGET - Minimal selection
   ============================================ */

QListWidget {
    background: transparent;
    border: none;
    outline: none;
}
QListWidget::item {
    padding: 14px 18px;
    border-radius: 10px;
    margin: 2px 4px;
    color: #7a8599;
}
QListWidget::item:selected {
    background: rgba(91, 141, 239, 0.15);
    color: #c8d0e0;
    border-left: 2px solid #5b8def;
}
QListWidget::item:hover:!selected {
    background: rgba(30, 40, 55, 0.4);
    color: #a0aec0;
}

/* ============================================
   STATUS BAR - Subtle
   ============================================ */

QStatusBar {
    background: rgba(6, 8, 12, 0.9);
    border-top: 1px solid rgba(30, 40, 55, 0.3);
    padding: 6px 16px;
    color: #4a5568;
    font-size: 12px;
}

/* ============================================
   TOOLTIPS - Refined
   ============================================ */

QToolTip {
    background: #141820;
    border: 1px solid rgba(50, 60, 80, 0.5);
    border-radius: 8px;
    padding: 8px 12px;
    color: #c8d0e0;
    font-size: 12px;
}
"""
