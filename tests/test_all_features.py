#!/usr/bin/env python3
"""
AstroLens Feature Tests (by section).

Usage:
    python tests/test_all_features.py                    # Run all sections
    python tests/test_all_features.py --section ood      # OOD detection only
    python tests/test_all_features.py --section morphology
    python tests/test_all_features.py --section reconstruction
    python tests/test_all_features.py --section gpu
    python tests/test_all_features.py --section web
    python tests/test_all_features.py --section build

Each section tests imports and basic functionality without external services.
"""

import argparse
import importlib
import json
import os
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Log directory -- respects ASTROLENS_ARTIFACTS_DIR if set
_artifacts = os.environ.get(
    "ASTROLENS_ARTIFACTS_DIR",
    str(Path(__file__).parent.parent.parent / "astrolens_artifacts"),
)
LOG_DIR = Path(_artifacts) / "test_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

PASSED = 0
FAILED = 0
RESULTS: list[dict] = []


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _record(section: str, name: str, ok: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if ok:
        PASSED += 1
        print(f"  PASS  [{section}] {name}")
    else:
        FAILED += 1
        print(f"  FAIL  [{section}] {name}: {detail}")
    RESULTS.append({"section": section, "test": name, "passed": ok, "detail": detail})


def _try(section: str, name: str, fn) -> None:
    """Run *fn*; record pass/fail."""
    try:
        fn()
        _record(section, name, True)
    except Exception as exc:
        _record(section, name, False, f"{type(exc).__name__}: {exc}")


# ─────────────────────────────────────────────────────────────
# OOD Detection
# ─────────────────────────────────────────────────────────────

def test_ood() -> None:
    sec = "ood"
    print(f"\n--- {sec.upper()} ---")

    _try(sec, "import inference.ood", lambda: importlib.import_module("inference.ood"))

    def check_class():
        from inference.ood import OODDetector, OODOutput  # noqa: F401
        assert callable(OODDetector), "OODDetector is not callable"
    _try(sec, "OODDetector class exists", check_class)

    def check_output_fields():
        from inference.ood import OODOutput
        fields = {f.name for f in OODOutput.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        for expected in ("ood_score", "is_anomaly", "threshold"):
            assert expected in fields, f"Missing field: {expected}"
    _try(sec, "OODOutput dataclass fields", check_output_fields)


# ─────────────────────────────────────────────────────────────
# Morphology
# ─────────────────────────────────────────────────────────────

def test_morphology() -> None:
    sec = "morphology"
    print(f"\n--- {sec.upper()} ---")

    _try(sec, "import features.morphology", lambda: importlib.import_module("features.morphology"))

    def check_classes():
        from features.morphology import GalaxyMorphology, MorphologyResult, analyze_morphology  # noqa: F401
    _try(sec, "key classes importable", check_classes)

    def quick_analyze():
        import numpy as np
        from PIL import Image
        from features.morphology import GalaxyMorphology

        analyzer = GalaxyMorphology()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            size = 64
            img = np.zeros((size, size), dtype=np.uint8)
            c = size // 2
            for y in range(size):
                for x in range(size):
                    r = ((x - c) ** 2 + (y - c) ** 2) ** 0.5
                    if r < c:
                        img[y, x] = int(255 * (1 - r / c))
            Image.fromarray(img).save(f.name)
            tmp = f.name

        try:
            result = analyzer.analyze(tmp)
            assert result is not None, "analyze returned None"
            assert 0 <= result.morph_score <= 1
        finally:
            Path(tmp).unlink(missing_ok=True)
    _try(sec, "quick morphology analysis", quick_analyze)


# ─────────────────────────────────────────────────────────────
# Reconstruction
# ─────────────────────────────────────────────────────────────

def test_reconstruction() -> None:
    sec = "reconstruction"
    print(f"\n--- {sec.upper()} ---")

    _try(sec, "import features.reconstruction", lambda: importlib.import_module("features.reconstruction"))

    def check_classes():
        from features.reconstruction import (  # noqa: F401
            ReconstructionResult,
            PCAReconstructor,
            FeatureReconstructor,
            create_pca_detector,
            fit_and_detect,
        )
    _try(sec, "key classes importable", check_classes)


# ─────────────────────────────────────────────────────────────
# GPU Detection
# ─────────────────────────────────────────────────────────────

def test_gpu() -> None:
    sec = "gpu"
    print(f"\n--- {sec.upper()} ---")

    _try(sec, "import inference.gpu_utils", lambda: importlib.import_module("inference.gpu_utils"))

    def check_device():
        from inference.gpu_utils import get_device, get_device_summary
        dev = get_device()
        assert dev is not None, "get_device returned None"
        summary = get_device_summary()
        assert isinstance(summary, str) and len(summary) > 0
        print(f"        device summary: {summary}")
    _try(sec, "get_device / get_device_summary", check_device)


# ─────────────────────────────────────────────────────────────
# Web Interface
# ─────────────────────────────────────────────────────────────

def test_web() -> None:
    sec = "web"
    print(f"\n--- {sec.upper()} ---")

    _try(sec, "import web.app", lambda: importlib.import_module("web.app"))

    def check_app():
        from web.app import app  # noqa: F401
        assert app is not None
    _try(sec, "FastAPI app object exists", check_app)


# ─────────────────────────────────────────────────────────────
# Build
# ─────────────────────────────────────────────────────────────

def test_build() -> None:
    sec = "build"
    print(f"\n--- {sec.upper()} ---")

    _try(sec, "import build.build", lambda: importlib.import_module("build.build"))

    def check_funcs():
        from build.build import get_platform, generate_spec, build  # noqa: F401
    _try(sec, "key functions importable", check_funcs)

    def check_platform():
        from build.build import get_platform
        plat = get_platform()
        assert plat in ("macos", "linux", "windows"), f"unexpected platform: {plat}"
    _try(sec, "get_platform returns valid value", check_platform)


# ─────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────

SECTIONS = {
    "ood": test_ood,
    "morphology": test_morphology,
    "reconstruction": test_reconstruction,
    "gpu": test_gpu,
    "web": test_web,
    "build": test_build,
}


def main() -> int:
    global PASSED, FAILED, RESULTS

    parser = argparse.ArgumentParser(description="AstroLens Feature Tests")
    parser.add_argument(
        "--section",
        choices=list(SECTIONS.keys()),
        help="Run a specific section only",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("AstroLens Feature Tests")
    print("=" * 60)

    sections = [args.section] if args.section else list(SECTIONS.keys())

    for sec in sections:
        try:
            SECTIONS[sec]()
        except Exception:
            traceback.print_exc()
            _record(sec, "UNEXPECTED ERROR", False, traceback.format_exc())

    total = PASSED + FAILED
    print("\n" + "=" * 60)
    print(f"Results: {PASSED}/{total} passed, {FAILED} failed")
    print("=" * 60)

    # Write JSON log
    try:
        log_file = LOG_DIR / f"test_results_{datetime.now():%Y%m%d_%H%M%S}.json"
        log_file.write_text(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "sections": sections,
            "passed": PASSED,
            "failed": FAILED,
            "total": total,
            "results": RESULTS,
        }, indent=2))
        print(f"Log saved to: {log_file}")
    except Exception as exc:
        print(f"Warning: could not write log file: {exc}")

    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
