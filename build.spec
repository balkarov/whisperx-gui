# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for WhisperX UI.
Build: pyinstaller build.spec
"""

import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

# Collect all needed data/submodules
datas = []
hiddenimports = []

# WhisperX and its dependencies
hiddenimports += collect_submodules('whisperx')
hiddenimports += collect_submodules('faster_whisper')
hiddenimports += collect_submodules('pyannote')
hiddenimports += collect_submodules('lightning_fabric')
hiddenimports += collect_submodules('speechbrain')
hiddenimports += collect_submodules('transformers')
hiddenimports += [
    'torch', 'torchaudio', 'ctranslate2',
    'huggingface_hub', 'tokenizers',
    'sklearn', 'sklearn.cluster',
    'customtkinter',
]

# Data files
datas += collect_data_files('whisperx')
datas += collect_data_files('faster_whisper')
datas += collect_data_files('customtkinter')
datas += collect_data_files('pyannote')
datas += collect_data_files('speechbrain')
datas += collect_data_files('lightning_fabric')
datas += collect_data_files('transformers')

# Package metadata needed at runtime
for pkg in ['torch', 'torchaudio', 'torchcodec', 'transformers',
            'huggingface_hub', 'tokenizers', 'safetensors',
            'ctranslate2', 'faster_whisper', 'whisperx',
            'pyannote.audio', 'pyannote.core', 'pyannote.pipeline',
            'speechbrain', 'lightning_fabric']:
    try:
        datas += copy_metadata(pkg)
    except Exception:
        print(f">>> Skipping metadata for {pkg} (not installed)")

# Bundle FFmpeg
ffmpeg_path = r'C:\Users\Ruslan\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin'
datas += [
    (os.path.join(ffmpeg_path, 'ffmpeg.exe'), '.'),
    (os.path.join(ffmpeg_path, 'ffprobe.exe'), '.'),
]

# Models are downloaded at runtime via HuggingFace token - not bundled

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib', 'notebook', 'jupyter',
        'IPython', 'PIL', 'cv2',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='WhisperX-UI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No console window
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='WhisperX-UI',
)
