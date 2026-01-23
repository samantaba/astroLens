"""
AstroLens Desktop Application Entry Point

Run with: python -m ui.main

Deep space discovery experience.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPalette, QColor

from ui.main_window import MainWindow
from ui.theme import MAIN_STYLESHEET


def main():
    """Launch the AstroLens desktop application."""
    # High DPI support
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    app.setApplicationName("AstroLens")
    app.setOrganizationName("AstroLens")
    
    # Try to use a more elegant font
    font_candidates = [
        ("JetBrains Mono", 11),
        ("Fira Code", 11),
        ("SF Pro Display", 11),
        ("Inter", 11),
        ("Segoe UI", 10),
    ]
    
    for font_name, size in font_candidates:
        font = QFont(font_name, size)
        if font.exactMatch():
            app.setFont(font)
            break
    else:
        app.setFont(QFont("Segoe UI", 10))
    
    # Apply cosmic theme
    app.setStyleSheet(MAIN_STYLESHEET)
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
