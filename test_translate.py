#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试翻译模型"""
import os
import sys

os.environ["USERNAME"] = "lfish"
os.environ["HOME"] = "C:/Users/lfish"
os.environ["USERPROFILE"] = "C:/Users/lfish"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import MarianMTModel, MarianTokenizer

local_dir = "D:/shipmodels/opus-mt-en-zh"
print(f"从本地加载: {local_dir}", flush=True)

try:
    tokenizer = MarianTokenizer.from_pretrained(local_dir)
    print("Tokenizer 加载成功", flush=True)

    model = MarianMTModel.from_pretrained(local_dir)
    print("Model 加载成功", flush=True)

    # 测试翻译
    test_text = "Hello, how are you?"
    src = ">>zho<< " + test_text
    encoded = tokenizer([src], return_tensors="pt", padding=True)
    generated = model.generate(**encoded)
    result = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    print(f"测试翻译: '{test_text}' -> '{result}'", flush=True)

except Exception as e:
    print(f"错误: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
