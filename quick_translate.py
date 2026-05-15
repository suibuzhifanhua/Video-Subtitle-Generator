#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""快速测试翻译脚本"""
import os
import sys

os.environ["USERNAME"] = "lfish"
os.environ["HOME"] = "C:/Users/lfish"
os.environ["USERPROFILE"] = "C:/Users/lfish"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_JIT"] = "0"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 强制使用 CPU

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

local_dir = "D:/shipmodels/M2M100-ja-zh"
print(f"加载模型: {local_dir}")

tokenizer = M2M100Tokenizer.from_pretrained(local_dir)
model = M2M100ForConditionalGeneration.from_pretrained(local_dir)
print("模型加载成功")

# 读取字幕
input_file = "C:/Users/lfish/Downloads/JUQ-553.restored (2).srt"
output_file = "C:/Users/lfish/Downloads/test_translated.srt"

with open(input_file, "r", encoding="utf-8") as f:
    content = f.read()

blocks = content.strip().split("\n\n")
print(f"共 {len(blocks)} 条字幕")

def do_translate(text):
    tokenizer.src_lang = "ja"
    encoded = tokenizer(text.strip(), return_tensors="pt", padding=True, truncation=True, max_length=512)
    generated = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id("zh"))
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

out_blocks = []
errors = 0
for i, block in enumerate(blocks):
    lines = block.strip().split("\n")
    if len(lines) < 3:
        out_blocks.append(block)
        continue
    try:
        translated = do_translate("\n".join(lines[2:]))
        out_blocks.append(f"{lines[0]}\n{lines[1]}\n{translated}")
    except Exception as e:
        print(f"错误 {i}: {e}", flush=True)
        out_blocks.append(block)
        errors += 1
    if i % 50 == 0:
        print(f"进度: {i+1}/{len(blocks)}", flush=True)
        sys.stdout.flush()

with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n\n".join(out_blocks) + "\n")

print(f"\n完成！翻译 {len(blocks) - errors}/{len(blocks)} 条，错误 {errors} 条")
print(f"输出文件: {output_file}")
