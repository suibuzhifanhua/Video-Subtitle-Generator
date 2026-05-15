#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
翻译辅助脚本 - 由 exe 调用系统 Python 执行
支持分段翻译，每翻译 700 条自动保存进度
用法: python translate_helper.py <model_dir> <input_file> <output_file> [source_lang]
"""
import sys as _sys
import os as _os

sys = _sys
os = _os
del _sys, _os

# 设置环境变量
os.environ.setdefault("USERNAME", os.environ.get("USERNAME") or os.environ.get("USER") or "lfish")
_user_home = os.environ.get("HOME") or os.path.expanduser("~")
os.environ.setdefault("HOME", _user_home)
os.environ.setdefault("USERPROFILE", _user_home)
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_JIT"] = "0"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def translate_file(model_dir, input_file, output_file, source_lang="ja"):
    """翻译字幕文件，支持分段翻译避免内存问题"""
    try:
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
    except ImportError as e:
        print(f"ERROR: transformers 未安装: {e}", file=sys.stderr)
        sys.exit(1)

    # 日语到中文的 M2M100 模型
    model_name = "Nekofox/M2M100-ja-zh"
    local_dir = os.path.join(model_dir, "M2M100-ja-zh")

    try:
        if os.path.isdir(local_dir) and os.listdir(local_dir):
            print(f"从本地加载: {local_dir}", file=sys.stderr)
            sys.stderr.flush()
            tokenizer = M2M100Tokenizer.from_pretrained(local_dir)
            model = M2M100ForConditionalGeneration.from_pretrained(local_dir)
            model.eval()
        else:
            print(f"从 HuggingFace 下载: {model_name}", file=sys.stderr)
            sys.stderr.flush()
            os.makedirs(local_dir, exist_ok=True)
            tokenizer = M2M100Tokenizer.from_pretrained(model_name, cache_dir=model_dir)
            model = M2M100ForConditionalGeneration.from_pretrained(model_name, cache_dir=model_dir)
    except Exception as e:
        print(f"ERROR: 模型加载失败: {e}", file=sys.stderr)
        sys.exit(2)

    # 读取输入文件
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    ext = os.path.splitext(input_file)[1].lower()

    def do_translate(text, src_lang="ja", tgt_lang="zh"):
        if not text or not text.strip():
            return text
        try:
            tokenizer.src_lang = src_lang
            encoded = tokenizer(text.strip(), return_tensors="pt", padding=True, truncation=True, max_length=512)
            generated = model.generate(
                **encoded,
                forced_bos_token_id=tokenizer.get_lang_id(tgt_lang)
            )
            result = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
            return result if result else text
        except Exception as e:
            print(f"WARNING: 翻译失败 '{text[:30]}...': {e}", file=sys.stderr)
            sys.stderr.flush()
            return text

    def translate_blocks(blocks, start_idx=0, file_total=0):
        """翻译字幕块"""
        out_blocks = []
        total = file_total if file_total else len(blocks)
        errors = 0
        for i, block in enumerate(blocks):
            lines = block.strip().split("\n")
            if len(lines) < 3:
                out_blocks.append(block)
                continue
            translated = do_translate("\n".join(lines[2:]))
            out_blocks.append(f"{lines[0]}\n{lines[1]}\n{translated}")
            if (start_idx + i + 1) % 100 == 0:
                print(f"进度: {start_idx + i + 1}/{total}", file=sys.stderr)
                sys.stderr.flush()
        return out_blocks, errors

    if ext == ".srt":
        blocks = content.strip().split("\n\n")
        total = len(blocks)
        print(f"共 {total} 条字幕", file=sys.stderr)
        sys.stderr.flush()

        # 分段翻译，每 700 条为一段
        CHUNK_SIZE = 700
        out_blocks = []
        for start in range(0, total, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, total)
            print(f"翻译第 {start + 1}-{end} 条...", file=sys.stderr)
            sys.stderr.flush()
            chunk_out, _ = translate_blocks(blocks[start:end], start, total)
            out_blocks.extend(chunk_out)
            print(f"第 {start + 1}-{end} 条完成", file=sys.stderr)
            sys.stderr.flush()

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n\n".join(out_blocks) + "\n")

    elif ext == ".vtt":
        parts = content.split("\n\n")
        out_parts = [parts[0]] if parts else []
        total = len(parts) - 1
        print(f"共 {total} 条字幕", file=sys.stderr)
        sys.stderr.flush()

        CHUNK_SIZE = 700
        for start in range(1, total + 1, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, total + 1)
            print(f"翻译第 {start}-{end - 1} 条...", file=sys.stderr)
            sys.stderr.flush()
            for i in range(start, end):
                part = parts[i]
                lines = part.strip().split("\n")
                tidx = next((j for j, l in enumerate(lines) if "-->" in l), None)
                if tidx is None:
                    out_parts.append(part)
                    continue
                header = "\n".join(lines[:tidx+1])
                text = "\n".join(lines[tidx+1:])
                out_parts.append(header + "\n" + do_translate(text))
            print(f"第 {start}-{end - 1} 条完成", file=sys.stderr)
            sys.stderr.flush()

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n\n".join(out_parts) + "\n")

    else:  # txt
        lines = content.strip().split("\n")
        total = len(lines)
        CHUNK_SIZE = 700
        out_lines = []
        processed = 0
        for i in range(0, total, CHUNK_SIZE):
            chunk = lines[i:i + CHUNK_SIZE]
            chunk_out = [do_translate(line) for line in chunk]
            out_lines.extend(chunk_out)
            processed += len(chunk_out)
            if processed % 100 == 0:
                print(f"进度: {processed}/{total}", file=sys.stderr)
                sys.stderr.flush()
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines) + "\n")

    print(f"完成: {output_file}", file=sys.stderr)
    sys.stderr.flush()
    return 0

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("用法: python translate_helper.py <model_dir> <input_file> <output_file> [source_lang]")
        sys.exit(1)
    sys.exit(translate_file(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "ja"))
