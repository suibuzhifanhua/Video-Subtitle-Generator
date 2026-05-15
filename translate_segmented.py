#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分段翻译字幕，每 500 条保存一次"""
import os
import sys

# 设置环境变量
os.environ["USERNAME"] = "lfish"
os.environ["HOME"] = "C:/Users/lfish"
os.environ["USERPROFILE"] = "C:/Users/lfish"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_JIT"] = "0"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch

local_dir = "D:/shipmodels/M2M100-ja-zh"
print(f"加载模型: {local_dir}", flush=True)

tokenizer = M2M100Tokenizer.from_pretrained(local_dir)
model = M2M100ForConditionalGeneration.from_pretrained(local_dir)
model.eval()
print("模型加载成功", flush=True)

# 读取字幕
input_file = "C:/Users/lfish/Downloads/JUQ-553.restored (2).srt"
output_file = "C:/Users/lfish/Downloads/test_translated.srt"

with open(input_file, "r", encoding="utf-8") as f:
    content = f.read()

blocks = content.strip().split("\n\n")
total = len(blocks)
print(f"共 {total} 条字幕", flush=True)

def do_translate(text):
    tokenizer.src_lang = "ja"
    with torch.no_grad():
        encoded = tokenizer(text.strip(), return_tensors="pt", padding=True, truncation=True, max_length=512)
        generated = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id("zh"))
        result = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    del encoded, generated
    return result

out_blocks = []
errors = 0

try:
    for i, block in enumerate(blocks):
        lines = block.strip().split("\n")
        if len(lines) < 3:
            out_blocks.append(block)
            continue
        try:
            translated = do_translate("\n".join(lines[2:]))
            out_blocks.append(f"{lines[0]}\n{lines[1]}\n{translated}")
        except Exception as e:
            print(f"错误 [{i}]: {lines[2][:30]}... -> {e}", flush=True)
            out_blocks.append(block)
            errors += 1
        if (i + 1) % 100 == 0:
            print(f"进度: {i+1}/{total}", flush=True)
            # 每 100 条强制刷新
            sys.stdout.flush()
except KeyboardInterrupt:
    print("\n用户中断", flush=True)
except Exception as e:
    print(f"致命错误: {e}", flush=True)
    import traceback
    traceback.print_exc()
finally:
    print(f"写入文件（{len(out_blocks)} 条）...", flush=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(out_blocks) + "\n")
    print(f"完成！翻译 {len(out_blocks) - errors}/{total} 条，错误 {errors} 条", flush=True)
    print(f"输出文件: {output_file}", flush=True)
