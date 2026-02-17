#!/usr/bin/env python3
"""
UI Component Import Tests for AstroLens.

Verifies that all UI modules can be imported and their key classes are accessible.
These tests do NOT require a display server -- they only check Python imports.

Expected: 15/15 passed.
"""

import importlib
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Ensure Qt can operate without a display (headless CI environments)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PASSED = 0
FAILED = 0
ERRORS: list[str] = []


def check(label: str, import_path: str, class_names: list[str]) -> None:
    """Import a module and verify expected attributes exist."""
    global PASSED, FAILED
    try:
        mod = importlib.import_module(import_path)
        for name in class_names:
            if not hasattr(mod, name):
                raise AttributeError(f"{import_path} has no attribute '{name}'")
        PASSED += 1
        print(f"  PASS  {label}")
    except Exception as exc:
        FAILED += 1
        msg = f"  FAIL  {label}: {exc}"
        print(msg)
        ERRORS.append(msg)


def main() -> int:
    print("=" * 60)
    print("AstroLens UI Component Tests")
    print("=" * 60)

    # 1. Theme constants
    check("Theme constants", "ui.theme", ["COLORS", "MAIN_STYLESHEET"])

    # 2. Main window
    check("MainWindow", "ui.main_window", ["MainWindow"])

    # 3. Gallery panel
    check("GalleryPanel", "ui.gallery", ["GalleryPanel"])

    # 4. Viewer panel
    check("ViewerPanel", "ui.viewer", ["ViewerPanel"])

    # 5. Batch analysis panel
    check("BatchPanel", "ui.batch_panel", ["BatchPanel"])

    # 6. Chat panel
    check("ChatPanel", "ui.chat_panel", ["ChatPanel"])

    # 7. Control center
    check("ControlCenter", "ui.control_center", ["ControlCenter"])

    # 8. Discovery panel
    check("DiscoveryPanel", "ui.discovery_panel", ["DiscoveryPanel"])

    # 9. Download panel
    check("DownloadPanel", "ui.download_panel", ["DownloadPanel"])

    # 10. Sources panel
    check("SourcesPanel", "ui.sources_panel", ["SourcesPanel"])

    # 11. Training panel
    check("TrainingPanel", "ui.training_panel", ["TrainingPanel"])

    # 12. Transient panel
    check("TransientPanel", "ui.transient_panel", ["TransientPanel"])

    # 13. Verification panel
    check("VerificationPanel", "ui.verification_panel", ["VerificationPanel"])

    # 14. Streaming panel
    check("StreamingPanel", "ui.streaming_panel", ["StreamingPanel"])

    # 15. Package-level import (ui.__init__)
    check("ui package (MainWindow re-export)", "ui", ["MainWindow"])

    # Summary
    total = PASSED + FAILED
    print("=" * 60)
    print(f"Results: {PASSED}/{total} passed, {FAILED} failed")
    print("=" * 60)

    if ERRORS:
        print("\nFailures:")
        for err in ERRORS:
            print(err)

    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
