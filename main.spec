# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

import os
import importlib

block_cipher = None

hidden_imports = collect_submodules('tensorflow_core')
hidden_imports_2 = collect_submodules('matplotlib')
hidden_imports_3 = collect_submodules('sklearn')
hidden_imports_4 = collect_submodules('tensorflow')
hidden_imports_5 = collect_submodules('ctypes')

all_hidden_imports = hidden_imports + hidden_imports_2 + hidden_imports_3 + hidden_imports_4 + hidden_imports_5

datas_1 = [(os.path.join(os.path.dirname(importlib.import_module('tensorflow').__file__), "lite/experimental/microfrontend/python/ops/_audio_microfrontend_op.so"),"tensorflow/lite/experimental/microfrontend/python/ops/")]
datas_2 = [(os.path.join(os.path.dirname(importlib.import_module('tensorflow').__file__), "python/ops/numpy_ops/python/ops/"),"tensorflow/python/ops/")]

all_datas = datas_1 + datas_2

a = Analysis(['main.py'],
             pathex=['C:\\Users\\laura\\PycharmProjects\\StockPriceGrabber'],
             binaries=[('C:\Windows\System32\msvcp140_1.dll', '.')],
             datas=datas_1,
             hiddenimports=all_hidden_imports,
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='main',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='main')
