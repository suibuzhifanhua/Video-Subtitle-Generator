# -*- mode: python ; coding: utf-8 -*-
# PyInstaller 打包配置 - 视频字幕生成器

import os
import sys
import shutil
from pathlib import Path
import site

block_cipher = None

try:
    BASE_DIR = Path(__file__).parent.resolve()
except NameError:
    BASE_DIR = Path('.').resolve()

SITE_PACKAGES = Path(site.getsitepackages()[1])

TEMPLATE_DIR = BASE_DIR / "templates"

# Tree 是 PyInstaller spec 内置类，无需 import

a = Analysis(
    [BASE_DIR / "app.py"],
    pathex=[str(BASE_DIR)],
    binaries=[],
    datas=[
        (str(TEMPLATE_DIR), "templates"),
        (str(BASE_DIR / "requirements.txt"), "."),
        (str(BASE_DIR / "translate_helper.py"), "."),
    ],
    hiddenimports=[
        "flask", "flask_cors", "flask.app",
        "transformers", "transformers.models.marian", "transformers.models.marian_mt",
        "transformers.generation", "transformers.generation.utils",
        "transformers.generation.configuration_utils", "transformers.generation.candidate_generator",
        "transformers.generation.logits_process", "transformers.generation.stopping_criteria",
        "transformers.generation.streamers", "transformers.generation.watermarking",
        "transformers.masking_utils", "transformers.modeling_utils",
        "faster_whisper", "ctranslate2",
        "jinja2", "markupsafe",
        "werkzeug", "itsdangerous",
        "huggingface_hub",
        "requests", "urllib3", "charset_normalizer",
        "idna", "certifi",
        "torch", "torch.nn", "torch.nn.functional", "torch.nn.modules",
        "torch.jit", "torch._dynamo", "torch._inductor",
        "torchvision",
    ],
    hookspath=[],
    hooksconfig={},
    keys=[],
    name="subtitle_generator",
    debug=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    **({"icon": str(BASE_DIR / "icon.ico")} if (BASE_DIR / "icon.ico").exists() else {}),
)

# 用 Tree 把完整的 transformers/torch 目录注册为可导入模块
# Tree 会把目录里的 .py 文件都当作模块处理，而不是普通数据文件
a.datas += Tree(str(SITE_PACKAGES / "transformers"), "transformers", excludes=[])
a.datas += Tree(str(SITE_PACKAGES / "torch"), "torch", excludes=[])
a.datas += Tree(str(SITE_PACKAGES / "torchvision"), "torchvision", excludes=[])

pyz = PYZ(a.pure, a.zipped_data, check_internet_platform_windows=False, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="subtitle_generator",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    **({"icon": str(BASE_DIR / "icon.ico")} if (BASE_DIR / "icon.ico").exists() else {}),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="subtitle_generator",
)
