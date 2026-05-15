import os
import sys

# ── 修复 exe 环境变量（PyTorch 需要 HOME/USERNAME）─────────────
if not os.environ.get("HOME"):
    try:
        import pathlib
        os.environ["HOME"] = str(pathlib.Path.home())
    except Exception:
        pass
if not os.environ.get("USERNAME"):
    try:
        import getpass
        os.environ["USERNAME"] = getpass.getuser()
    except Exception:
        os.environ["USERNAME"] = "lfish"

import json
import uuid
import threading
import time
import shutil
import subprocess
from pathlib import Path
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS

# ── PyInstaller 兼容处理 ─────────────────────────────────────────
def get_resource_path(relative_path):
    """获取资源文件的绝对路径，兼容开发环境和 PyInstaller 打包后环境"""
    if getattr(sys, 'frozen', False):
        # PyInstaller 打包后，资源文件在临时解压目录 _MEIPASS 下
        # 使用 exe 目录 + "_internal" 作为基准，确保返回绝对路径
        exe_dir = Path(sys.executable).parent.resolve()
        return exe_dir / "_internal" / relative_path
    # 开发环境：相对于 app.py 所在目录
    return Path(__file__).parent / relative_path

app = Flask(__name__, template_folder=str(get_resource_path('templates')))
CORS(app)

# ── 目录设置 ─────────────────────────────────────────────────────
# exe 所在目录（PyInstaller 环境下是 exe 所在目录）
# 关键：使用 sys.executable 的父目录，而不是 __file__ 的父目录
# 因为 __file__ 在打包后指向 _internal 目录
EXE_DIR = Path(sys.executable).parent.resolve() if getattr(sys, 'frozen', False) else Path(__file__).parent.resolve()
BASE_DIR = EXE_DIR  # 保持向后兼容

UPLOAD_FOLDER = BASE_DIR / "uploads"
OUTPUT_FOLDER = BASE_DIR / "outputs"
_SUBTITLE_TRANSLATE_FOLDER = BASE_DIR / "subtitle_translate"
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)
_SUBTITLE_TRANSLATE_FOLDER.mkdir(exist_ok=True)

# 默认模型存放目录（可让用户改）
CONFIG_FILE = BASE_DIR / "config.json"

def load_config():
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"model_dir": str(BASE_DIR / "whisper_models")}

def save_config(cfg):
    CONFIG_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

config = load_config()
MODEL_DIR = Path(config.get("model_dir", str(BASE_DIR / "whisper_models")))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"mp4", "mkv", "avi", "mov", "wmv", "flv", "webm", "m4v", "mpeg", "mpg"}

# 所有可用模型及大小（近似）
MODEL_INFO = {
    "tiny":   {"size_mb": 75,   "desc": "最快，精度低，适合测试"},
    "base":   {"size_mb": 140,  "desc": "推荐，速度与精度平衡"},
    "small":  {"size_mb": 460,  "desc": "较准确，速度适中"},
    "medium": {"size_mb": 1500, "desc": "高精度，较慢"},
    "large-v3": {"size_mb": 3100, "desc": "最高精度，需要大量内存"},
}
for k, v in MODEL_INFO.items():
    v["downloaded"] = False  # 运行时动态更新

# 任务状态
tasks = {}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def format_timestamp_srt(seconds):
    hours   = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs    = int(seconds % 60)
    millis  = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def format_timestamp_vtt(seconds):
    hours   = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs    = int(seconds % 60)
    millis  = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

# 模型名称到 HF repo ID 的映射
HF_REPO_MAP = {
    "tiny":      "Systran/faster-whisper-tiny",
    "base":      "Systran/faster-whisper-base",
    "small":     "Systran/faster-whisper-small",
    "medium":    "Systran/faster-whisper-medium",
    "large-v3":  "Systran/faster-whisper-large-v3",
}

def _find_model_in_hf_cache(hf_cache_root, model_name):
    """在 HF Hub 缓存目录中递归查找指定模型的 snapshot 路径"""
    if not hf_cache_root.exists():
        return None
    try:
        for entry in hf_cache_root.iterdir():
            if not entry.is_dir():
                continue
            # 检查是否匹配模型名
            model_key = model_name.replace("-", "")
            entry_key = entry.name.lower().replace("-", "")
            if model_key not in entry_key:
                continue
            # 检查 snapshots 子目录
            snapshots = entry / "snapshots"
            if snapshots.exists():
                for sub in sorted(snapshots.iterdir()):
                    if sub.is_dir() and (sub / "model.bin").exists():
                        return str(sub)
    except PermissionError:
        pass
    return None

def get_model_local_path(model_size):
    """返回指定模型的 snapshot 绝对路径，找不到返回 None"""
    # 1. 先检查 MODEL_DIR（用户自定义目录）
    model_path = _find_model_in_hf_cache(MODEL_DIR, model_size)
    if model_path:
        return model_path
    # 2. 检查 HF 默认缓存
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    return _find_model_in_hf_cache(hf_cache, model_size)

def scan_models():
    """扫描所有已知位置，返回 {model_size: snapshot_path}"""
    all_found = {}
    for model_size in MODEL_INFO:
        path = get_model_local_path(model_size)
        if path:
            all_found[model_size] = path
    # 同步状态
    for k in MODEL_INFO:
        MODEL_INFO[k]["downloaded"] = k in all_found
        MODEL_INFO[k]["path"] = all_found.get(k, None)
    return all_found

# ── 字幕生成 ───────────────────────────────────────────────────
def ensure_vad_model(model_dir):
    """确保 VAD 模型已下载到指定目录"""
    try:
        import huggingface_hub as hf_hub
        from pathlib import Path
        # VAD 模型会下载到 HF_HOME/hub/models--silero-vad--*/snapshots/
        vad_pattern = "models--silero-vad--"
        hf_cache = Path(model_dir)
        for entry in hf_cache.iterdir():
            if vad_pattern in entry.name.lower():
                snapshots = entry / "snapshots"
                if snapshots.exists():
                    for snap in snapshots.iterdir():
                        if any(f.suffix in [".onnx", ".bin"] for f in snap.iterdir() if f.is_file()):
                            return str(entry)
        # 没找到，触发下载
        print(f"[DEBUG] 正在下载 VAD 模型到 {model_dir}...")
        hf_hub.snapshot_download(
            repo_id="snakers4/silero-vad",
            cache_dir=str(model_dir),
            local_files_only=False
        )
        # 重新检查
        for entry in hf_cache.iterdir():
            if vad_pattern in entry.name.lower():
                return str(entry)
        return None
    except Exception as e:
        print(f"[WARN] VAD 模型下载失败: {e}")
        return None

def generate_subtitles(task_id, video_path, language, model_size):
    try:
        tasks[task_id]["status"]  = "processing"
        tasks[task_id]["progress"] = 10
        tasks[task_id]["message"]  = "正在加载 Whisper 模型..."

        try:
            from faster_whisper import WhisperModel
        except ImportError:
            tasks[task_id]["status"]  = "error"
            tasks[task_id]["message"] = "faster-whisper 未安装，请运行: pip install faster-whisper"
            return

        # 检查可用设备：优先 GPU (CUDA)，其次 CPU
        import torch
        _use_gpu = torch.cuda.is_available()
        # 关键：HF_HOME 必须指向包含 models--xxx 目录的父目录
        os.environ["HF_HOME"] = str(MODEL_DIR)

        if _use_gpu:
            tasks[task_id]["message"] = f"正在加载模型 ({model_size}，使用 GPU）..."
            print(f"[DEBUG] 检测到 GPU: {torch.cuda.get_device_name(0)}，使用 CUDA + float16")
            model_kwargs = {"device": "cuda", "compute_type": "float16"}
        else:
            tasks[task_id]["message"] = f"正在加载模型 ({model_size}，使用 CPU）..."
            print(f"[DEBUG] 未检测到 GPU，使用 CPU + int8")
            model_kwargs = {"device": "cpu", "compute_type": "int8"}
        local_path = get_model_local_path(model_size)
        if local_path:
            # 已有本地模型，直接加载本地 snapshot 路径
            print(f"[DEBUG] 使用本地模型: {local_path}")
            model = WhisperModel(local_path, **model_kwargs)
        else:
            # 没有本地模型，需要下载
            print(f"[DEBUG] 本地未找到模型，设置 HF_HOME={MODEL_DIR}，尝试从 HF 下载...")
            model = WhisperModel(HF_REPO_MAP.get(model_size, model_size), **model_kwargs)

        tasks[task_id]["progress"] = 40
        tasks[task_id]["message"]  = "正在识别语音..."

        # 确保 VAD 模型已下载到用户目录
        tasks[task_id]["message"]  = "正在检查 VAD 模型..."
        ensure_vad_model(MODEL_DIR)

        lang_param = None if language == "auto" else language
        # VAD 加载失败 → 尝试下载（3次） → 仍失败则降级
        _vad_ok = False
        _segments_gen = None
        _detected_lang = None

        for _attempt in range(3):
            try:
                tasks[task_id]["message"] = f"正在识别语音（VAD 检测，第 {_attempt + 1}/3 次）..."
                # faster-whisper transcribe 返回 (segments_generator, info)
                _segments_gen, _detected_lang = model.transcribe(
                    str(video_path),
                    language=lang_param,
                    beam_size=5,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                _vad_ok = True
                break
            except Exception as vad_error:
                print(f"[WARN] VAD 加载失败（第 {_attempt + 1}/3 次）: {vad_error}")
                if _attempt < 2:
                    tasks[task_id]["message"] = f"VAD 模型缺失，尝试下载中（第 {_attempt + 1}/3 次）..."
                    ensure_vad_model(MODEL_DIR)
                else:
                    print(f"[WARN] VAD 模型3次加载均失败，降级处理")
                    tasks[task_id]["message"] = "VAD 模型加载失败，跳过静音过滤..."
                    _segments_gen, _detected_lang = model.transcribe(
                        str(video_path),
                        language=lang_param,
                        beam_size=5,
                        vad_filter=False
                    )

        # 迭代生成器收集结果，同时更新进度
        tasks[task_id]["progress"] = 45
        tasks[task_id]["message"]  = "正在识别语音..."
        segments_list = []
        _total_segs = getattr(_segments_gen, '_model_info', None)
        # segments 是生成器，逐个消费并更新进度
        try:
            for idx, seg in enumerate(_segments_gen):
                segments_list.append(seg)
                # 根据已处理片段数粗略估算进度（45% ~ 75%）
                pct = min(74, 45 + (idx + 1) * 2)
                tasks[task_id]["progress"] = pct
                if (idx + 1) % 10 == 0:
                    tasks[task_id]["message"] = f"正在识别语音... 已处理 {idx + 1} 段"
        except Exception as seg_error:
            print(f"[ERROR] 片段处理异常: {seg_error}")
            raise

        detected_lang = _detected_lang.language if _detected_lang else "auto"

        tasks[task_id]["progress"] = 80
        tasks[task_id]["message"]  = "正在生成字幕文件..."

        # SRT
        srt_path = OUTPUT_FOLDER / f"{task_id}.srt"
        with open(srt_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments_list, 1):
                f.write(f"{i}\n")
                f.write(f"{format_timestamp_srt(seg.start)} --> {format_timestamp_srt(seg.end)}\n")
                f.write(f"{seg.text.strip()}\n\n")

        # VTT
        vtt_path = OUTPUT_FOLDER / f"{task_id}.vtt"
        with open(vtt_path, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for i, seg in enumerate(segments_list, 1):
                f.write(f"{i}\n")
                f.write(f"{format_timestamp_vtt(seg.start)} --> {format_timestamp_vtt(seg.end)}\n")
                f.write(f"{seg.text.strip()}\n\n")

        # TXT
        txt_path = OUTPUT_FOLDER / f"{task_id}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            for seg in segments_list:
                f.write(seg.text.strip() + "\n")

        # 预览数据
        preview = []
        for seg in segments_list[:50]:
            preview.append({
                "start": round(seg.start, 2),
                "end":   round(seg.end, 2),
                "text":  seg.text.strip()
            })

        tasks[task_id]["status"]           = "done"
        tasks[task_id]["progress"]         = 100
        tasks[task_id]["message"]          = "字幕生成完成！"
        tasks[task_id]["detected_language"] = detected_lang
        tasks[task_id]["segment_count"]    = len(segments_list)
        tasks[task_id]["preview"]          = preview
        tasks[task_id]["files"] = {
            "srt": f"{task_id}.srt",
            "vtt": f"{task_id}.vtt",
            "txt": f"{task_id}.txt"
        }

    except Exception as e:
        tasks[task_id]["status"]  = "error"
        tasks[task_id]["message"] = f"处理失败: {str(e)}"
    finally:
        try:
            video_path.unlink(missing_ok=True)
        except Exception:
            pass

# ── 翻译 ───────────────────────────────────────────────────────
def _find_system_python():
    """查找系统可用的 Python（exe 环境 fallback）
    优先用系统 Python 3.14（完整的 Windows Python 安装）
    """
    # 系统 Python 3.14（优先，比 managed 3.13 更完整）
    for ver in ["314", "313"]:
        for candidate in [
            Path(f"C:/Users/lfish/AppData/Local/Programs/Python/Python{ver}/python.exe"),
            Path(f"C:/Python{ver}/python.exe"),
        ]:
            if candidate.exists():
                return str(candidate)
    # Managed Python 3.13.12（备用）
    managed_py = Path("C:/Users/lfish/.workbuddy/binaries/python/versions/3.13.12/python.exe")
    if managed_py.exists():
        return str(managed_py)
    # PATH 中的 python
    import shutil
    return shutil.which("python") or shutil.which("python3")

def _init_marian_translator():
    """初始化 MarianMT 翻译器（本地离线翻译，使用 transformers）"""
    import traceback
    # 确保环境变量存在
    if not os.environ.get("HOME"):
        os.environ["HOME"] = str(Path.home())
    if not os.environ.get("USERNAME"):
        try:
            import getpass
            os.environ["USERNAME"] = getpass.getuser()
        except Exception:
            os.environ["USERNAME"] = "lfish"

    # 禁用 torch dynamo 编译
    os.environ["TORCHDUMP"] = "0"
    os.environ["TORCHINDUCTOR_DISABLE"] = "1"
    os.environ["PYTORCH_JIT"] = "0"
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 国内访问镜像
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    try:
        import torch
        torch._dynamo.config.disable = True
    except Exception:
        pass

    try:
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
    except Exception as e:
        traceback.print_exc()
        return None, None, f"transformers 未安装，请运行: pip install transformers (详细: {e})"

    # 日语到中文的 M2M100 模型
    model_name = "Nekofox/M2M100-ja-zh"
    try:
        local_dir = MODEL_DIR / "M2M100-ja-zh"
        if local_dir.exists() and any(local_dir.iterdir()):
            # 目录存在且非空，尝试加载
            tokenizer = M2M100Tokenizer.from_pretrained(str(local_dir))
            model = M2M100ForConditionalGeneration.from_pretrained(str(local_dir))
        elif local_dir.exists():
            # 目录存在但为空（下载中途失败），清理后提示用 subprocess 下载
            import shutil
            shutil.rmtree(str(local_dir))
            return None, None, f"翻译模型未下载完成，请先进行一次字幕翻译（系统会自动从 HuggingFace 下载模型）"
        else:
            # 目录不存在，也提示用 subprocess 下载
            return None, None, f"翻译模型未下载，请先进行一次字幕翻译（系统会自动从 HuggingFace 下载模型）"
        return model, tokenizer, None
    except Exception as e:
        traceback.print_exc()
        return None, None, f"M2M100 模型加载失败: {e}"

def do_translate(task_id, target_lang="zh"):
    try:
        tasks[task_id]["translate_status"]  = "processing"
        tasks[task_id]["translate_message"] = "正在加载翻译引擎..."

        task = tasks[task_id]
        if task.get("status") != "done":
            tasks[task_id]["translate_status"]  = "error"
            tasks[task_id]["translate_message"] = "原始字幕尚未生成完成"
            return

        srt_path = OUTPUT_FOLDER / f"{task_id}.srt"
        if not srt_path.exists():
            tasks[task_id]["translate_status"]  = "error"
            tasks[task_id]["translate_message"] = "找不到原始 SRT 文件"
            return

        tasks[task_id]["translate_message"] = "正在初始化翻译模型..."

        # 两层策略：先尝试内置，再尝试系统 Python
        model, tokenizer, err = _init_marian_translator()
        use_system_python = bool(err)

        if use_system_python:
            sys_py = _find_system_python()
            if not sys_py:
                tasks[task_id]["translate_status"] = "error"
                tasks[task_id]["translate_message"] = err
                return
            tasks[task_id]["translate_message"] = "使用系统 Python 翻译..."
            helper = get_resource_path("translate_helper.py")
            zh_srt = OUTPUT_FOLDER / f"{task_id}_zh.srt"
            cmd = [sys_py, str(helper), str(MODEL_DIR), str(srt_path), str(zh_srt)]
            try:
                import subprocess
                exe_dir = str(Path(sys.executable).parent.resolve())
                clean_env = {
                    **os.environ,
                    "PYTHONIOENCODING": "utf-8",
                    "HF_ENDPOINT": "https://hf-mirror.com",
                    "PYTORCH_JIT": "0",
                    "TORCHINDUCTOR_DISABLE": "1",
                    "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
                    "HOME": os.environ.get("HOME") or os.path.expanduser("~"),
                    "USERNAME": os.environ.get("USERNAME") or os.environ.get("USER") or "lfish",
                }
                if "PATH" in clean_env:
                    clean_env["PATH"] = ";".join(
                        p for p in clean_env["PATH"].split(";") if Path(p).resolve() != Path(exe_dir)
                    )
                # stderr 重定向到临时文件（避免管道死锁）
                _err_file = OUTPUT_FOLDER / f"{task_id}_translate_stderr.log"
                with open(_err_file, "w", encoding="utf-8") as ef:
                    r = subprocess.run(
                        cmd, stdout=subprocess.PIPE, stderr=ef,
                        encoding="utf-8", errors="replace", timeout=7200, env=clean_env
                    )
                # 读取错误日志
                _err_text = _err_file.read_text(encoding="utf-8", errors="replace") if _err_file.exists() else ""
                _err_file.unlink(missing_ok=True)
                if r.returncode == 0:
                    tasks[task_id]["translate_status"]   = "done"
                    tasks[task_id]["translate_progress"] = 100
                    tasks[task_id]["translate_message"]  = "翻译完成！"
                    tasks[task_id]["translate_files"] = {
                        "srt": f"{task_id}_zh.srt",
                        "vtt": f"{task_id}_zh.vtt",
                        "txt": f"{task_id}_zh.txt"
                    }
                else:
                    tasks[task_id]["translate_status"]  = "error"
                    tasks[task_id]["translate_message"] = f"翻译失败: {_err_text[:500] or r.stderr}"
            except subprocess.TimeoutExpired:
                tasks[task_id]["translate_status"]  = "error"
                tasks[task_id]["translate_message"] = "翻译超时"
            return

        def safe_translate(text, src_lang="ja", tgt_lang="zh"):
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
                print(f"[WARN] 翻译失败: {e}")
                return text

        tasks[task_id]["translate_message"] = "正在翻译字幕..."

        with open(srt_path, "r", encoding="utf-8") as f:
            content = f.read()

        blocks = content.strip().split("\n\n")
        translated_blocks = []
        preview_zh = []
        total_blocks = len(blocks)

        # 分段翻译，每 700 条为一段，避免内存问题
        CHUNK_SIZE = 700
        for chunk_start in range(0, total_blocks, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, total_blocks)
            tasks[task_id]["translate_message"] = f"正在翻译字幕（第 {chunk_start + 1}-{chunk_end} 条）..."

            for i in range(chunk_start, chunk_end):
                block = blocks[i]
                lines = block.strip().split("\n")
                if len(lines) < 3:
                    translated_blocks.append(block)
                    continue
                index_line = lines[0]
                time_line  = lines[1]
                text_lines = "\n".join(lines[2:])
                translated_text = safe_translate(text_lines)
                translated_blocks.append(f"{index_line}\n{time_line}\n{translated_text}")
                if len(translated_blocks) - 1 < 50:
                    preview_zh.append({"index": index_line, "time": time_line, "text": translated_text})

            # 更新进度
            pct = min(90, int(chunk_end / max(total_blocks, 1) * 90))
            tasks[task_id]["translate_progress"] = pct

        tasks[task_id]["translate_message"] = "正在写入翻译文件..."

        # 写入翻译 SRT
        zh_srt = OUTPUT_FOLDER / f"{task_id}_zh.srt"
        with open(zh_srt, "w", encoding="utf-8") as f:
            f.write("\n\n".join(translated_blocks) + "\n")

        # VTT
        vtt_path = OUTPUT_FOLDER / f"{task_id}.vtt"
        if vtt_path.exists():
            with open(vtt_path, "r", encoding="utf-8") as f:
                vtt_content = f.read()
            vtt_parts = vtt_content.split("\n\n")
            translated_vtt_parts = [vtt_parts[0]] if vtt_parts else []
            for part in vtt_parts[1:]:
                lines = part.strip().split("\n")
                if len(lines) < 2:
                    translated_vtt_parts.append(part)
                    continue
                time_idx = next((j for j, l in enumerate(lines) if "-->" in l), None)
                if time_idx is None:
                    translated_vtt_parts.append(part)
                    continue
                header_lines = lines[:time_idx + 1]
                text_lines   = "\n".join(lines[time_idx + 1:])
                t = safe_translate(text_lines) if text_lines.strip() else text_lines
                translated_vtt_parts.append("\n".join(header_lines) + "\n" + t)
            zh_vtt = OUTPUT_FOLDER / f"{task_id}_zh.vtt"
            with open(zh_vtt, "w", encoding="utf-8") as f:
                f.write("\n\n".join(translated_vtt_parts) + "\n")

        # TXT
        zh_txt = OUTPUT_FOLDER / f"{task_id}_zh.txt"
        with open(zh_txt, "w", encoding="utf-8") as f:
            for b in translated_blocks:
                lines = b.strip().split("\n")
                if len(lines) >= 3:
                    f.write("\n".join(lines[2:]) + "\n")

        tasks[task_id]["translate_status"]   = "done"
        tasks[task_id]["translate_progress"] = 100
        tasks[task_id]["translate_message"]  = "翻译完成！"
        tasks[task_id]["translate_preview"] = preview_zh
        tasks[task_id]["translate_files"] = {
            "srt": f"{task_id}_zh.srt",
            "vtt": f"{task_id}_zh.vtt",
            "txt": f"{task_id}_zh.txt"
        }

    except Exception as e:
        tasks[task_id]["translate_status"]  = "error"
        tasks[task_id]["translate_message"] = f"翻译失败: {str(e)}"

# ── 路由 ─────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/check_deps")
def check_deps():
    return jsonify({
        "faster_whisper": _check_import("faster_whisper"),
        "flask":          _check_import("flask"),
        "transformers":   _check_import("transformers"),
    })

@app.route("/api/config", methods=["GET", "POST"])
def api_config():
    global MODEL_DIR, config
    if request.method == "GET":
        found = scan_models()
        return jsonify({
            "model_dir": str(MODEL_DIR),
            "models": MODEL_INFO,
            "model_paths": {k: v for k, v in found.items()},
            "deps": {
                "faster_whisper": _check_import("faster_whisper"),
                "flask":          _check_import("flask"),
                "transformers":   _check_import("transformers"),
            }
        })
    else:
        data = request.get_json(force=True, silent=True) or {}
        new_dir = data.get("model_dir", "").strip()
        if new_dir:
            p = Path(new_dir)
            try:
                p.mkdir(parents=True, exist_ok=True)
                MODEL_DIR = p
                config["model_dir"] = str(p)
                save_config(config)
                found = scan_models()
                return jsonify({
                    "ok": True,
                    "model_dir": str(MODEL_DIR),
                    "models": MODEL_INFO,
                    "model_paths": {k: v for k, v in found.items()}
                })
            except Exception as e:
                return jsonify({"error": f"无法设置目录: {e}"}), 400
        return jsonify({"error": "未提供 model_dir"}), 400

def _check_import(name):
    try:
        __import__(name)
        return True
    except ImportError:
        return False

@app.route("/api/models/download", methods=["POST"])
def api_download_model():
    """后台下载指定模型"""
    data = request.get_json(force=True, silent=True) or {}
    model_size = data.get("model", "base")
    if model_size not in MODEL_INFO:
        return jsonify({"error": "不支持的模型大小"}), 400

    task_id = f"model_{model_size}"
    if task_id in tasks and tasks[task_id].get("translate_status") == "processing":
        return jsonify({"error": "正在下载中，请稍候"}), 400

    tasks[task_id] = {"status": "processing", "progress": 0, "message": f"正在下载 {model_size} 模型..."}

    def _download():
        try:
            from faster_whisper import WhisperModel
            # 关键：HF_HOME 必须指向包含 models--xxx 目录的父目录
            os.environ["HF_HOME"] = str(MODEL_DIR)
            # 触发下载（不实际转录，只下载模型）
            # 通过创建模型对象来触发下载
            import huggingface_hub as hf_hub
            # 直接通过 huggingface_hub 下载
            repo_map = {
                "tiny":      "Systran/faster-whisper-tiny",
                "base":      "Systran/faster-whisper-base",
                "small":      "Systran/faster-whisper-small",
                "medium":     "Systran/faster-whisper-medium",
                "large-v3":   "Systran/faster-whisper-large-v3",
            }
            repo = repo_map.get(model_size, f"Systran/faster-whisper-{model_size}")
            tasks[task_id]["message"]  = f"正在从 HuggingFace 下载 {repo}..."
            tasks[task_id]["progress"] = 10
            local_dir = hf_hub.snapshot_download(repo_id=repo, cache_dir=str(MODEL_DIR))
            tasks[task_id]["progress"] = 100
            tasks[task_id]["status"]   = "done"
            tasks[task_id]["message"]  = f"模型 {model_size} 下载完成！"
            scan_models()
        except Exception as e:
            tasks[task_id]["status"]  = "error"
            tasks[task_id]["message"] = f"下载失败: {str(e)}"

    t = threading.Thread(target=_download, daemon=True)
    t.start()
    return jsonify({"ok": True, "message": "开始下载"})

@app.route("/api/models/status/<model_size>")
def api_model_status(model_size):
    task_id = f"model_{model_size}"
    if task_id not in tasks:
        scan_models()
        return jsonify({"status": "none", "downloaded": MODEL_INFO.get(model_size, {}).get("downloaded", False)})
    t = tasks[task_id]
    return jsonify({
        "status":    t.get("status", "none"),
        "progress":   t.get("progress", 0),
        "message":    t.get("message", ""),
        "downloaded": MODEL_INFO.get(model_size, {}).get("downloaded", False)
    })

@app.route("/api/models/delete", methods=["POST"])
def api_delete_model():
    data = request.get_json(force=True, silent=True) or {}
    model_size = data.get("model", "")
    if model_size not in MODEL_INFO:
        return jsonify({"error": "不支持的模型"}), 400
    # 删除模型目录
    deleted = False
    for d in [MODEL_DIR] + ([Path.home() / ".cache" / "huggingface" / "hub"] if True else []):
        if not d.exists():
            continue
        for sub in d.iterdir():
            if model_size.replace("-", "") in sub.name.lower().replace("-", ""):
                try:
                    shutil.rmtree(sub)
                    deleted = True
                except Exception as e:
                    return jsonify({"error": f"删除失败: {e}"}), 500
    scan_models()
    return jsonify({"ok": True, "deleted": deleted})

# ── 翻译模块管理 ───────────────────────────────────────────────
TRANSLATE_MODEL_NAME = "Nekofox/M2M100-ja-zh"
TRANSLATE_MODEL_LOCAL = MODEL_DIR / "M2M100-ja-zh"

def check_translate_module():
    """检查翻译模块是否已下载"""
    return TRANSLATE_MODEL_LOCAL.exists() and any(TRANSLATE_MODEL_LOCAL.iterdir())

@app.route("/api/translate_module/status")
def api_translate_module_status():
    """查询翻译模块状态"""
    downloaded = check_translate_module()
    path = str(TRANSLATE_MODEL_LOCAL) if downloaded else None
    task_id = "translate_module"
    if task_id in tasks:
        t = tasks[task_id]
        return jsonify({
            "status": t.get("status", "none"),
            "progress": t.get("progress", 0),
            "message": t.get("message", ""),
            "downloaded": downloaded,
            "path": path
        })
    return jsonify({
        "status": "none",
        "progress": 100 if downloaded else 0,
        "message": "已下载" if downloaded else "未下载",
        "downloaded": downloaded,
        "path": path
    })

@app.route("/api/translate_module/download", methods=["POST"])
def api_download_translate_module():
    """下载翻译模块（M2M100-ja-zh）"""
    if check_translate_module():
        return jsonify({"ok": True, "message": "模块已存在"})

    task_id = "translate_module"
    if task_id in tasks and tasks[task_id].get("status") == "processing":
        return jsonify({"error": "正在下载中"}), 400

    tasks[task_id] = {"status": "processing", "progress": 0, "message": "准备下载..."}

    def _download():
        try:
            from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
            tasks[task_id]["message"] = f"正在下载翻译模型 {TRANSLATE_MODEL_NAME}..."
            tasks[task_id]["progress"] = 10
            # 下载 tokenizer
            tokenizer = M2M100Tokenizer.from_pretrained(TRANSLATE_MODEL_NAME, cache_dir=str(MODEL_DIR))
            tasks[task_id]["progress"] = 40
            tasks[task_id]["message"] = "正在下载模型权重..."
            # 下载 model
            model = M2M100ForConditionalGeneration.from_pretrained(TRANSLATE_MODEL_NAME, cache_dir=str(MODEL_DIR))
            tasks[task_id]["progress"] = 100
            tasks[task_id]["status"] = "done"
            tasks[task_id]["message"] = "翻译模块下载完成！"
        except Exception as e:
            tasks[task_id]["status"] = "error"
            tasks[task_id]["message"] = f"下载失败: {str(e)}"

    t = threading.Thread(target=_download, daemon=True)
    t.start()
    return jsonify({"ok": True, "message": "开始下载"})

@app.route("/api/translate_module/delete", methods=["POST"])
def api_delete_translate_module():
    """删除翻译模块"""
    if TRANSLATE_MODEL_LOCAL.exists():
        try:
            shutil.rmtree(TRANSLATE_MODEL_LOCAL)
            return jsonify({"ok": True, "message": "已删除"})
        except Exception as e:
            return jsonify({"error": f"删除失败: {e}"}), 500
    return jsonify({"ok": True, "message": "模块不存在"})

@app.route("/api/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "未找到视频文件"}), 400
    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "未选择文件"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": f"不支持的文件格式，支持: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    language   = request.form.get("language", "auto")
    model_size = request.form.get("model", "base")

    task_id = str(uuid.uuid4())
    ext      = file.filename.rsplit(".", 1)[1].lower()
    video_path = UPLOAD_FOLDER / f"{task_id}.{ext}"
    file.save(str(video_path))

    tasks[task_id] = {
        "status":  "pending",
        "progress": 0,
        "message":  "任务已创建，等待处理...",
        "filename": file.filename
    }

    t = threading.Thread(
        target=generate_subtitles,
        args=(task_id, video_path, language, model_size),
        daemon=True
    )
    t.start()
    return jsonify({"task_id": task_id, "message": "上传成功，开始处理"})

@app.route("/api/status/<task_id>")
def task_status(task_id):
    if task_id not in tasks:
        return jsonify({"error": "任务不存在"}), 404
    return jsonify(tasks[task_id])

@app.route("/api/download/<task_id>/<fmt>")
def download(task_id, fmt):
    if task_id not in tasks:
        return jsonify({"error": "任务不存在"}), 404
    task = tasks[task_id]
    if task["status"] != "done":
        return jsonify({"error": "任务未完成"}), 400
    if fmt not in ("srt", "vtt", "txt"):
        return jsonify({"error": "不支持的格式"}), 400
    file_path = OUTPUT_FOLDER / f"{task_id}.{fmt}"
    if not file_path.exists():
        return jsonify({"error": "文件不存在"}), 404
    original_name = Path(task.get("filename", "subtitle")).stem
    download_name  = f"{original_name}.{fmt}"
    return send_file(str(file_path), as_attachment=True, download_name=download_name, mimetype="text/plain")

@app.route("/api/translate/<task_id>", methods=["POST"])
def translate_subtitles(task_id):
    if task_id not in tasks:
        return jsonify({"error": "任务不存在"}), 404
    task = tasks[task_id]
    if task.get("status") != "done":
        return jsonify({"error": "字幕尚未生成完成"}), 400
    ts = task.get("translate_status")
    if ts == "processing":
        return jsonify({"message": "翻译已在进行中", "translate_status": ts})
    tasks[task_id]["translate_status"]   = "pending"
    tasks[task_id]["translate_progress"]  = 0
    tasks[task_id]["translate_message"]  = "任务已创建，等待翻译..."
    t = threading.Thread(target=do_translate, args=(task_id,), daemon=True)
    t.start()
    return jsonify({"message": "翻译任务已启动", "task_id": task_id})

@app.route("/api/translate_status/<task_id>")
def translate_status(task_id):
    if task_id not in tasks:
        return jsonify({"error": "任务不存在"}), 404
    task = tasks[task_id]
    return jsonify({
        "translate_status":   task.get("translate_status", "none"),
        "translate_progress":  task.get("translate_progress", 0),
        "translate_message":   task.get("translate_message", ""),
        "translate_preview":   task.get("translate_preview", []),
        "translate_files":     task.get("translate_files", {})
    })

@app.route("/api/download_zh/<task_id>/<fmt>")
def download_zh(task_id, fmt):
    if task_id not in tasks:
        return jsonify({"error": "任务不存在"}), 404
    task = tasks[task_id]
    if task.get("translate_status") != "done":
        return jsonify({"error": "翻译未完成"}), 400
    if fmt not in ("srt", "vtt", "txt"):
        return jsonify({"error": "不支持的格式"}), 400
    file_path = OUTPUT_FOLDER / f"{task_id}_zh.{fmt}"
    if not file_path.exists():
        return jsonify({"error": "文件不存在"}), 404
    original_name = Path(task.get("filename", "subtitle")).stem
    download_name  = f"{original_name}_中文.{fmt}"
    return send_file(str(file_path), as_attachment=True, download_name=download_name, mimetype="text/plain")

# ── 独立翻译 API（不依赖视频任务）─────────────────────────────
_translator_cache = {}  # 缓存已加载的翻译器
_subtitle_file_tasks: dict = {}  # task_id -> {status, progress, message, output_path, preview}

@app.route("/api/translate/text", methods=["POST"])
def translate_text():
    """直接翻译输入的文本，不依赖视频任务"""
    import tempfile, traceback
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "请输入要翻译的文本"}), 400

    # 优先尝试内置翻译器
    global _translator_cache
    if "model" in _translator_cache:
        model = _translator_cache["model"]
        tokenizer = _translator_cache["tokenizer"]
        try:
            src_text = ">>zho<< " + text
            encoded = tokenizer([src_text], return_tensors="pt", padding=True, truncation=True, max_length=512)
            generated = model.generate(**encoded)
            result = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
            return jsonify({"original": text, "translated": result})
        except Exception as e:
            # 翻译器出问题了，清除缓存，用系统 Python 重试
            traceback.print_exc()
            _translator_cache.clear()

    # Fallback：调用系统 Python（必须清理 PATH 避免 DLL 冲突）
    sys_py = _find_system_python()
    if not sys_py:
        return jsonify({"error": "无可用 Python 环境（内置翻译失败且找不到系统 Python）"}), 500

    # 写临时文件调用翻译
    tmp_in = _SUBTITLE_TRANSLATE_FOLDER / f"_t_{uuid.uuid4().hex}_in.txt"
    tmp_out = _SUBTITLE_TRANSLATE_FOLDER / f"_t_{uuid.uuid4().hex}_out.txt"
    tmp_in.write_text(text, encoding="utf-8")

    # 清理 PATH：移除 exe 自身目录，避免 DLL 冲突（Windows 用 ; 分隔）
    # 同时设置 HF 镜像（国内必须）
    clean_env = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "HF_ENDPOINT": "https://hf-mirror.com",
        "PYTORCH_JIT": "0",
        "TORCHINDUCTOR_DISABLE": "1",
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
        "HOME": os.environ.get("HOME") or os.path.expanduser("~"),
        "USERNAME": os.environ.get("USERNAME") or os.environ.get("USER") or "lfish",
    }
    exe_dir = str(Path(sys.executable).parent.resolve())
    if "PATH" in clean_env:
        clean_env["PATH"] = ";".join(
            p for p in clean_env["PATH"].split(";") if Path(p).resolve() != Path(exe_dir)
        )

    helper = Path(__file__).parent / "translate_helper.py"
    cmd = [sys_py, str(helper), str(MODEL_DIR), str(tmp_in), str(tmp_out)]
    _err_file_tx = _SUBTITLE_TRANSLATE_FOLDER / f"_t_{uuid.uuid4().hex}_stderr.log"
    try:
        with open(_err_file_tx, "w", encoding="utf-8") as ef:
            r = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=ef,
                encoding="utf-8", errors="replace", timeout=120, env=clean_env
            )
        _err_text_tx = _err_file_tx.read_text(encoding="utf-8", errors="replace") if _err_file_tx.exists() else ""
        _err_file_tx.unlink(missing_ok=True)
        if r.returncode == 0 and tmp_out.exists():
            translated = tmp_out.read_text(encoding="utf-8").strip()
            return jsonify({"original": text, "translated": translated, "via": "system_python"})
        else:
            return jsonify({"error": f"系统 Python 翻译失败: {_err_text_tx[:500] or r.stderr}"}), 500
    except subprocess.TimeoutExpired:
        return jsonify({"error": "翻译超时（120秒）"}), 500
    except Exception as e:
        return jsonify({"error": f"翻译失败: {str(e)}"}), 500
    finally:
        try:
            tmp_in.unlink(missing_ok=True)
            tmp_out.unlink(missing_ok=True)
        except Exception:
            pass

# ── 字幕文件翻译 API（上传字幕文件，直接翻译导出）────────────────
SUBTITLE_ALLOWED = {"srt", "vtt", "txt"}

@app.route("/api/subtitle/translate", methods=["POST"])
def translate_subtitle_file():
    """上传字幕文件，立即返回 task_id，后台异步翻译"""
    if "subtitle" not in request.files:
        return jsonify({"error": "未找到字幕文件"}), 400
    file = request.files["subtitle"]
    if file.filename == "":
        return jsonify({"error": "未选择文件"}), 400
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in SUBTITLE_ALLOWED:
        return jsonify({"error": f"不支持的文件格式，支持: {', '.join(SUBTITLE_ALLOWED)}"}), 400

    task_id = str(uuid.uuid4())
    input_path = _SUBTITLE_TRANSLATE_FOLDER / f"{task_id}_input.{ext}"
    file.save(str(input_path))

    _subtitle_file_tasks[task_id] = {
        "status": "processing",
        "progress": 0,
        "message": "正在启动翻译...",
        "output_path": None,
        "preview": [],
        "ext": ext,
        "orig_name": file.filename,
    }

    import threading
    t = threading.Thread(target=_run_subtitle_translation,
                         args=(task_id, input_path, ext), daemon=True)
    t.start()

    return jsonify({"ok": True, "task_id": task_id})


def _run_subtitle_translation(task_id, input_path, ext):
    """后台线程：执行字幕翻译并更新 _subtitle_file_tasks"""
    def set_progress(pct, msg):
        _subtitle_file_tasks[task_id]["progress"] = pct
        _subtitle_file_tasks[task_id]["message"] = msg

    try:
        out_file = f"{task_id}_translated.{ext}"
        out_path = _SUBTITLE_TRANSLATE_FOLDER / out_file

        sys_py = _find_system_python()
        if not sys_py:
            _subtitle_file_tasks[task_id].update({"status": "error", "message": "找不到可用的 Python 环境"})
            return

        helper = get_resource_path("translate_helper.py")
        cmd = [sys_py, str(helper), str(MODEL_DIR), str(input_path), str(out_path)]
        exe_dir = str(Path(sys.executable).parent.resolve())
        clean_env = {
            **os.environ,
            "PYTHONIOENCODING": "utf-8",
            "HF_ENDPOINT": "https://hf-mirror.com",
            "PYTORCH_JIT": "0",
            "TORCHINDUCTOR_DISABLE": "1",
            "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
            "HOME": os.environ.get("HOME") or os.path.expanduser("~"),
            "USERNAME": os.environ.get("USERNAME") or os.environ.get("USER") or "lfish",
        }
        if "PATH" in clean_env:
            clean_env["PATH"] = ";".join(
                p for p in clean_env["PATH"].split(";") if Path(p).resolve() != Path(exe_dir)
            )

        set_progress(5, "正在启动翻译引擎...")

        import re as _re
        import subprocess as _sp
        proc = _sp.Popen(cmd, stdout=_sp.PIPE, stderr=_sp.PIPE, env=clean_env,
                         encoding="utf-8", errors="replace")
        total = None
        for line in proc.stderr:
            line = line.strip()
            m = _re.match(r"进度:\s*(\d+)/(\d+)", line)
            if m:
                cur, tot = int(m.group(1)), int(m.group(2))
                if total is None:
                    total = tot
                pct = int(5 + cur / tot * 90)
                set_progress(pct, f"翻译中... {cur}/{tot}")
            elif line:
                set_progress(_subtitle_file_tasks[task_id]["progress"], line[:80])

        proc.wait()
        if proc.returncode != 0:
            stderr_tail = "".join(proc.stderr.readlines())
            _subtitle_file_tasks[task_id].update({
                "status": "error",
                "message": f"翻译失败: {stderr_tail[:300]}"
            })
            return

        # 生成预览
        preview = []
        try:
            for line in out_path.read_text(encoding="utf-8").splitlines()[:60]:
                if line.strip():
                    preview.append({"text": line.strip()})
        except Exception:
            pass

        _subtitle_file_tasks[task_id].update({
            "status": "done",
            "progress": 100,
            "message": "翻译完成！",
            "output_path": str(out_path),
            "preview": preview[:50],
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        _subtitle_file_tasks[task_id].update({"status": "error", "message": f"翻译异常: {e}"})
    finally:
        try:
            input_path.unlink(missing_ok=True)
        except Exception:
            pass


@app.route("/api/subtitle/status/<task_id>", methods=["GET"])
def subtitle_translate_status(task_id):
    """查询字幕翻译进度"""
    t = _subtitle_file_tasks.get(task_id)
    if not t:
        return jsonify({"error": "任务不存在"}), 404
    return jsonify(t)


def _translate_srt(content, safe_translate):
    """翻译 SRT 字幕"""
    blocks = content.strip().split("\n\n")
    translated_blocks = []
    preview = []
    for i, block in enumerate(blocks):
        lines = block.strip().split("\n")
        if len(lines) < 3:
            translated_blocks.append(block)
            continue
        index_line = lines[0]
        time_line = lines[1]
        text_lines = "\n".join(lines[2:])
        translated_text = safe_translate(text_lines)
        translated_blocks.append(f"{index_line}\n{time_line}\n{translated_text}")
        if i < 50:
            preview.append({"index": index_line, "time": time_line, "text": translated_text})
    return "\n\n".join(translated_blocks) + "\n", preview

def _translate_vtt(content, safe_translate):
    """翻译 VTT 字幕"""
    parts = content.split("\n\n")
    translated_parts = [parts[0]] if parts else []
    preview = []
    for i, part in enumerate(parts[1:]):
        lines = part.strip().split("\n")
        if len(lines) < 2:
            translated_parts.append(part)
            continue
        time_idx = next((j for j, l in enumerate(lines) if "-->" in l), None)
        if time_idx is None:
            translated_parts.append(part)
            continue
        header_lines = lines[:time_idx + 1]
        text_lines = "\n".join(lines[time_idx + 1:])
        t = safe_translate(text_lines) if text_lines.strip() else text_lines
        translated_parts.append("\n".join(header_lines) + "\n" + t)
        if i < 50:
            preview.append({"time": lines[time_idx], "text": t})
    return "\n\n".join(translated_parts) + "\n", preview

def _translate_txt(content, safe_translate):
    """翻译 TXT 纯文本"""
    lines = content.strip().split("\n")
    translated_lines = []
    preview = []
    for i, line in enumerate(lines):
        if line.strip():
            t = safe_translate(line.strip())
            translated_lines.append(t)
            if i < 50:
                preview.append({"text": t})
        else:
            translated_lines.append("")
    return "\n".join(translated_lines) + "\n", preview

@app.route("/api/subtitle/download/<task_id>/<ext>")
def download_translated_subtitle(task_id, ext):
    """下载翻译后的字幕文件"""
    if ext not in SUBTITLE_ALLOWED:
        return jsonify({"error": "不支持的格式"}), 400
    filename = f"{task_id}_translated.{ext}"
    file_path = _SUBTITLE_TRANSLATE_FOLDER / filename
    if not file_path.exists():
        return jsonify({"error": "文件不存在"}), 404
    return send_file(str(file_path), as_attachment=True, download_name=filename, mimetype="text/plain")

# ── 静态文件服务（打包后用） ───────────────────────────────────
@app.route("/api/open_output", methods=["POST"])
def open_output_folder():
    """打开输出文件夹（仅本地生效）"""
    try:
        os.startfile(str(OUTPUT_FOLDER))
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/open_model_folder", methods=["POST"])
def open_model_folder():
    """打开模型文件夹（仅本地生效）"""
    try:
        md = Path(MODEL_DIR)
        if not md.exists():
            md.mkdir(parents=True, exist_ok=True)
        os.startfile(str(md))
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── 启动 ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import webbrowser
    import threading

    def open_browser():
        webbrowser.open('http://127.0.0.1:54124')

    # 2秒后自动打开浏览器
    threading.Timer(2.0, open_browser).start()

    print("=" * 50)
    print("  视频字幕生成器")
    print("=" * 50)
    print(f"  模型存放目录: {MODEL_DIR}")
    print(f"  字幕输出目录: {OUTPUT_FOLDER}")
    print("=" * 50)
    scan_models()
    app.run(debug=False, host="127.0.0.1", port=54124, threaded=True)
