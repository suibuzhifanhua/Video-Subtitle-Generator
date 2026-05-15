#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""下载日语到中文翻译模型 (M2M100)"""
import os
import sys

os.environ["USERNAME"] = "lfish"
os.environ["HOME"] = "C:/Users/lfish"
os.environ["USERPROFILE"] = "C:/Users/lfish"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_JIT"] = "0"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

model_name = "Nekofox/M2M100-ja-zh"
local_dir = "D:/shipmodels/M2M100-ja-zh"

print(f"下载模型: {model_name}", flush=True)
print(f"目标目录: {local_dir}", flush=True)

try:
    tokenizer = M2M100Tokenizer.from_pretrained(model_name, cache_dir="D:/shipmodels")
    print("Tokenizer 下载完成", flush=True)

    model = M2M100ForConditionalGeneration.from_pretrained(model_name, cache_dir="D:/shipmodels")
    print("Model 下载完成", flush=True)

    # 测试
    test_text = "こんにちは、元気ですか？"
    tokenizer.src_lang = "ja"
    encoded = tokenizer(test_text, return_tensors="pt", padding=True)
    generated = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id("zh"))
    result = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    print(f"测试翻译: '{test_text}' -> '{result}'", flush=True)

    print("\n下载成功！", flush=True)
except Exception as e:
    print(f"下载失败: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
