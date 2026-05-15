#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""下载翻译模型到本地"""
import os
import sys

# 设置环境变量
os.environ["USERNAME"] = "lfish"
os.environ["HOME"] = "C:/Users/lfish"
os.environ["USERPROFILE"] = "C:/Users/lfish"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_JIT"] = "0"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-en-zh"
local_dir = "D:/shipmodels/opus-mt-en-zh"

print(f"下载模型到: {local_dir}", flush=True)
print(f"模型名称: {model_name}", flush=True)

try:
    tokenizer = MarianTokenizer.from_pretrained(model_name, cache_dir="D:/shipmodels")
    print("Tokenizer 下载完成", flush=True)

    model = MarianMTModel.from_pretrained(model_name, cache_dir="D:/shipmodels")
    print("Model 下载完成", flush=True)

    # 列出模型文件
    files = os.listdir(local_dir)
    model_files = [f for f in files if f.endswith(".bin") or f.endswith(".safetensors") or f.endswith(".pt")]
    print(f"模型文件列表: {model_files}", flush=True)
    print(f"总计 {len(files)} 个文件", flush=True)
    print("下载成功！", flush=True)
except Exception as e:
    print(f"下载失败: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
