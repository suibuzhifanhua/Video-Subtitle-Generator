#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试翻译少量字幕"""
import os
import sys

os.environ["USERNAME"] = "lfish"
os.environ["HOME"] = "C:/Users/lfish"
os.environ["USERPROFILE"] = "C:/Users/lfish"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_JIT"] = "0"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

local_dir = "D:/shipmodels/M2M100-ja-zh"
print(f"从本地加载: {local_dir}", flush=True)

tokenizer = M2M100Tokenizer.from_pretrained(local_dir)
model = M2M100ForConditionalGeneration.from_pretrained(local_dir)
print("模型加载成功", flush=True)

# 测试几条日语字幕
test_lines = [
    "僕たち夫婦の関係はずっと変わらないと思っていた",
    "でも心の底から信じ合っている二人でも",
    "やばいやばいやばい",
    "ネクタイの締め方わかんなくて遅刻しちゃう",
    "ありがとう",
]

for ja in test_lines:
    tokenizer.src_lang = "ja"
    encoded = tokenizer(ja.strip(), return_tensors="pt", padding=True, truncation=True, max_length=512)
    generated = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id("zh"))
    zh = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    print(f"日: {ja}")
    print(f"中: {zh}")
    print()
