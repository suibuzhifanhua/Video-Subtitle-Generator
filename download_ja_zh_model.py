#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""下载日语到中文翻译模型"""
import os
import sys

os.environ["USERNAME"] = "lfish"
os.environ["HOME"] = "C:/Users/lfish"
os.environ["USERPROFILE"] = "C:/Users/lfish"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_JIT"] = "0"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-ja-zh"
local_dir = "D:/shipmodels/opus-mt-ja-zh"

print(f"下载模型: {model_name}", flush=True)
print(f"目标目录: {local_dir}", flush=True)

try:
    tokenizer = MarianTokenizer.from_pretrained(model_name, cache_dir="D:/shipmodels")
    print("Tokenizer 下载完成", flush=True)

    model = MarianMTModel.from_pretrained(model_name, cache_dir="D:/shipmodels")
    print("Model 下载完成", flush=True)

    # 复制到目标目录
    import shutil
    cache_path = os.path.join("D:/shipmodels/models--Helsinki-NLP--opus-mt-ja-zh/snapshots")
    snapshots = os.listdir(cache_path)
    latest = os.path.join(cache_path, snapshots[-1])

    files = os.listdir(latest)
    print(f"模型文件: {files}", flush=True)

    for f in files:
        src = os.path.join(latest, f)
        dst = os.path.join(local_dir, f)
        shutil.copy2(src, dst)

    print(f"已复制到 {local_dir}", flush=True)

    # 测试
    test_text = "こんにちは、元気ですか？"
    src = ">>zho<< " + test_text
    encoded = tokenizer([src], return_tensors="pt", padding=True)
    generated = model.generate(**encoded)
    result = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    print(f"测试翻译: '{test_text}' -> '{result}'", flush=True)

except Exception as e:
    print(f"下载失败: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
