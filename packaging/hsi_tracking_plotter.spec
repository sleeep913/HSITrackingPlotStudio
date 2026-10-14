# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

project_root = Path(SPEC).resolve().parent.parent

analysis = Analysis(
    [str(project_root / "src" / "hsi_plotter" / "__main__.py")],
    pathex=[str(project_root / "src")],
    binaries=[],
    datas=[
        (str(project_root / "resources" / "Groundtruth"), "Groundtruth"),
    ],
    hiddenimports=[
        "matplotlib.backends.backend_agg",
        "matplotlib.backends.backend_pdf",
        "matplotlib.backends.backend_ps",
        "matplotlib.backends.backend_svg",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

# Qt 6 on supported Windows versions uses the operating-system ICU shim.
# Codex's tool PATH also contains Poppler ICU 78 DLLs; PyInstaller can mistake
# those unrelated DLLs for Qt dependencies.  Bundling them shadows the Windows
# ICU shim and makes QtCore fail with WinError 127 at startup.
excluded_toolchain_dlls = {"icuuc.dll", "icudt78.dll"}
analysis.binaries = [
    entry
    for entry in analysis.binaries
    if Path(entry[0]).name.casefold() not in excluded_toolchain_dlls
]

pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="HSITrackingPlotStudio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=str(project_root / "resources" / "icon.ico"),
)

collection = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="HSITrackingPlotStudio-v1.4.0",
)
