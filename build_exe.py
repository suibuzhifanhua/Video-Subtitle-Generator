#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频字幕生成器 - 打包脚本
自动安装依赖并生成 .exe 可执行文件

使用方法:
  python build_exe.py

打包完成后，exe 文件位于 dist/subtitle_generator/
"""
import subprocess
import sys
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()

def run(cmd, desc=""):
    print(f"\n{'='*50}")
    print(f"  {desc or cmd}")
    print('='*50)
    result = subprocess.run(cmd, shell=True, cwd=str(BASE_DIR))
    if result.returncode != 0:
        print(f"❌ 命令执行失败: {cmd}")
        sys.exit(1)
    print(f"✅ 完成")

def main():
    print("""
╔═══════════════════════════════════════════╗
║   视频字幕生成器 - 打包工具                ║
║   自动安装依赖并生成 .exe 文件              ║
╚═══════════════════════════════════════════╝
""")

    # 1. 安装依赖
    print("📦 正在安装 Python 依赖...")
    deps = [
        "flask>=3.0",
        "flask-cors>=4.0",
        "faster-whisper>=1.0",
        "transformers>=4.30.0",
        "sentencepiece>=0.1.99",
        "pyinstaller>=6.0",
    ]
    for dep in deps:
        run(f'pip install "{dep}"', f"安装 {dep}")

    # 2. 清理旧构建
    dist = BASE_DIR / "dist"
    build = BASE_DIR / "build"
    if dist.exists():
        print("🗑️  清理旧构建文件...")
        import shutil
        shutil.rmtree(dist, ignore_errors=True)
        shutil.rmtree(build, ignore_errors=True)

    # 3. PyInstaller 打包
    print("\n🔨 正在使用 PyInstaller 打包...")
    spec_file = BASE_DIR / "subtitle_generator.spec"
    if not spec_file.exists():
        print(f"❌ 找不到 spec 文件: {spec_file}")
        sys.exit(1)

    run(
        f'pyinstaller "{spec_file}" --noconfirm --clean',
        "PyInstaller 打包"
    )

    # 4. 复制 whisper_models 目录（存放模型的目录）
    model_src = BASE_DIR / "whisper_models"
    dist_dir  = dist / "subtitle_generator"
    model_dst = dist_dir / "whisper_models"

    if model_src.exists():
        print("📁 复制模型目录...")
        import shutil
        shutil.copytree(model_src, model_dst, dirs_exist_ok=True)
    else:
        model_dst.mkdir(parents=True, exist_ok=True)
        print(f"⚠️  模型目录不存在，已创建空目录: {model_dst}")
        print("   首次运行时会自动下载模型到该目录")

    # 5. 复制 config.json
    cfg_src = BASE_DIR / "config.json"
    cfg_dst = dist_dir / "config.json"
    if cfg_src.exists():
        import shutil
        shutil.copy2(cfg_src, cfg_dst)

    # 6. 完成
    exe_path = dist_dir / "subtitle_generator.exe"
    print(f"""
╔═══════════════════════════════════════════╗
║   ✅ 打包完成！                           ║
╚═══════════════════════════════════════════╝

📂 exe 文件位置:
   {exe_path}

📂 模型存放目录:
   {model_dst}

📂 字幕输出目录（程序内）:
   {dist_dir / "outputs"}

🚀 运行方式:
   双击 {exe_path}
   浏览器访问 http://localhost:54124

💡 首次运行会自动下载 Whisper 模型到:
   {model_dst}
   （Base 模型约 140 MB，Large 模型约 3 GB）
""")

if __name__ == "__main__":
    main()
