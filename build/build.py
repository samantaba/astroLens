#!/usr/bin/env python3
"""
AstroLens Build Pipeline

Builds standalone executables for Mac, Linux, and Windows using PyInstaller.

Usage:
    python build/build.py              # Build for current platform
    python build/build.py --platform all    # Build for all platforms (cross-compile)
    python build/build.py --clean      # Clean build artifacts
    
Requirements:
    pip install pyinstaller
"""

import argparse
import logging
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
BUILD_DIR = PROJECT_ROOT / "build" / "dist"
SPEC_DIR = PROJECT_ROOT / "build" / "specs"

# Ensure dirs exist
BUILD_DIR.mkdir(parents=True, exist_ok=True)
SPEC_DIR.mkdir(parents=True, exist_ok=True)


def get_platform() -> str:
    """Get current platform identifier."""
    system = platform.system().lower()
    if system == "darwin":
        return "macos"
    elif system == "linux":
        return "linux"
    elif system == "windows":
        return "windows"
    return system


def get_icon_path() -> str:
    """Get icon path for the current platform."""
    assets = PROJECT_ROOT / "assets"
    if get_platform() == "macos":
        icns = assets / "logo.icns"
        if icns.exists():
            return str(icns)
    elif get_platform() == "windows":
        ico = assets / "logo.ico"
        if ico.exists():
            return str(ico)
    png = assets / "logo.png"
    return str(png) if png.exists() else ""


def generate_spec() -> str:
    """Generate PyInstaller .spec file."""
    icon = get_icon_path()
    plat = get_platform()
    
    # Collect all data files
    datas = [
        (str(PROJECT_ROOT / "assets"), "assets"),
        (str(PROJECT_ROOT / "web" / "templates"), "web/templates"),
    ]
    
    # Static files if they exist
    static_dir = PROJECT_ROOT / "web" / "static"
    if static_dir.exists():
        datas.append((str(static_dir), "web/static"))
    
    datas_str = ",\n        ".join(f"('{src}', '{dst}')" for src, dst in datas)
    
    # Hidden imports that PyInstaller might miss
    hidden_imports = [
        "uvicorn.logging",
        "uvicorn.protocols",
        "uvicorn.lifespan",
        "uvicorn.lifespan.on",
        "sqlalchemy.dialects.sqlite",
        "sklearn.utils._cython_blas",
        "sklearn.neighbors._typedefs",
        "faiss",
        "PIL",
        "httpx",
        "PyQt5.sip",
    ]
    
    hidden_str = ",\n        ".join(f"'{h}'" for h in hidden_imports)
    
    # Platform-specific options
    if plat == "macos":
        bundle_type = "BUNDLE"
        extra = f"""
app = BUNDLE(
    exe,
    name='AstroLens.app',
    icon='{icon}',
    bundle_identifier='com.astrolens.app',
    info_plist={{
        'NSHighResolutionCapable': True,
        'CFBundleShortVersionString': '1.0.0',
        'CFBundleDisplayName': 'AstroLens',
    }},
)"""
    else:
        extra = ""
    
    spec_content = f"""# -*- mode: python ; coding: utf-8 -*-
# AstroLens PyInstaller Spec - {plat}
# Generated automatically by build/build.py

import sys
import os

block_cipher = None

a = Analysis(
    ['{str(PROJECT_ROOT / "ui" / "main.py")}'],
    pathex=['{str(PROJECT_ROOT)}'],
    binaries=[],
    datas=[
        {datas_str}
    ],
    hiddenimports=[
        {hidden_str}
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[
        'matplotlib',  # Large, not needed for core UI
        'notebook',
        'jupyter',
        'IPython',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AstroLens',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No console window
    disable_windowed_traceback=False,
    argv_emulation=True,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='{icon}' if '{icon}' else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AstroLens',
)
{extra}
"""
    
    spec_path = SPEC_DIR / f"astrolens_{plat}.spec"
    spec_path.write_text(spec_content)
    logger.info(f"Generated spec: {spec_path}")
    return str(spec_path)


def build(spec_path: str = None, clean: bool = False):
    """Run PyInstaller build."""
    
    if clean:
        logger.info("Cleaning build artifacts...")
        for d in [BUILD_DIR, SPEC_DIR, PROJECT_ROOT / "build" / "work"]:
            if d.exists():
                shutil.rmtree(d)
                logger.info(f"  Removed {d}")
        return
    
    # Check PyInstaller
    try:
        import PyInstaller
        logger.info(f"PyInstaller version: {PyInstaller.__version__}")
    except ImportError:
        logger.error("PyInstaller not installed. Run: pip install pyinstaller")
        sys.exit(1)
    
    if spec_path is None:
        spec_path = generate_spec()
    
    plat = get_platform()
    work_dir = str(PROJECT_ROOT / "build" / "work")
    dist_dir = str(BUILD_DIR / plat)
    
    cmd = [
        sys.executable, "-m", "PyInstaller",
        spec_path,
        "--workpath", work_dir,
        "--distpath", dist_dir,
        "--noconfirm",
        "--clean",
    ]
    
    logger.info(f"Building for {plat}...")
    logger.info(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    
    if result.returncode == 0:
        logger.info(f"\n✓ Build successful!")
        logger.info(f"  Output: {dist_dir}/AstroLens")
        
        if plat == "macos":
            logger.info(f"  App bundle: {dist_dir}/AstroLens.app")
            logger.info(f"\n  To create DMG:")
            logger.info(f"    hdiutil create -volname AstroLens -srcfolder {dist_dir}/AstroLens.app -ov AstroLens.dmg")
        elif plat == "linux":
            logger.info(f"\n  To create AppImage: use linuxdeploy or appimagetool")
        elif plat == "windows":
            logger.info(f"\n  To create installer: use Inno Setup or NSIS")
    else:
        logger.error(f"Build failed with code {result.returncode}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="AstroLens Build Pipeline")
    parser.add_argument("--clean", action="store_true", help="Clean build artifacts")
    parser.add_argument("--spec-only", action="store_true", help="Generate spec file only")
    parser.add_argument("--platform", type=str, default=None, help="Target platform")
    args = parser.parse_args()
    
    if args.clean:
        build(clean=True)
        return
    
    if args.spec_only:
        generate_spec()
        return
    
    build()


if __name__ == "__main__":
    main()
