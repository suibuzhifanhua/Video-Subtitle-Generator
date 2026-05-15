# 视频字幕生成器

上传视频，自动识别语音，生成 SRT / VTT / TXT 字幕文件。

## 功能特性

- 支持拖拽上传视频（MP4、MKV、AVI、MOV、WMV、FLV、WebM 等）
- 自动检测语言（中文、英文、日语等 99+ 种语言）
- 生成三种格式：`.srt`（通用）、`.vtt`（网页）、`.txt`（纯文本）
- 实时进度显示 + 字幕预览
- **🤖 模型管理**：右上角 ⚙️ 设置中可查看已下载的模型、自主下载/删除指定模型
- **📁 自定义模型存放路径**：支持将 Whisper 模型存放到指定目录
- **🌐 一键翻译成中文**：字幕生成后可一键翻译成中文（免费无需 API Key）

## 快速启动

### Windows（命令行）

```bash
pip install -r requirements.txt
python app.py
```

### 打包成 exe（无需安装 Python）

```bash
python build_exe.py
```

打包完成后，双击 `dist/subtitle_generator/subtitle_generator.exe` 即可运行。

## ⚙️ 模型存放目录

| 模型 | 大小 | 推荐用途 |
|------|------|---------|
| Tiny | ~75 MB | 快速测试 |
| Base | ~140 MB | **推荐日常使用** |
| Small | ~460 MB | 高精度 |
| Medium | ~1.5 GB | 专业级 |
| Large v3 | ~3.1 GB | 最高精度 |

> 首次使用会自动从 HuggingFace 下载模型。
> 右上角 ⚙️ 按钮 → 可修改存放目录、查看已下载模型、自主下载或删除指定模型。

## 翻译功能说明

字幕生成完成后，在结果区域点击「🌐 一键翻译成中文」按钮，系统将自动：
1. 调用翻译引擎将字幕原文翻译成中文
2. 生成中文版的 `.srt`、`.vtt`、`.txt` 三种格式
3. 展示翻译预览，支持单独下载中文字幕

> ⚠️ 翻译功能需要网络连接，翻译速度取决于字幕段数。

## 依赖

- Python 3.8+
- Flask + Flask-CORS
- faster-whisper（基于 CTranslate2 的 Whisper 高速实现）
- translators（多翻译引擎聚合）
- huggingface-hub（模型下载）
- PyInstaller（仅打包时需要）

如果你用这个项目看片看得更得劲了，不妨请我喝杯咖啡。

<img width="300" height="327" alt="image" src="https://github.com/user-attachments/assets/f29f5a9d-b93e-49b2-8ac3-5c2dfcf1224f" />

<img width="300" height="327" alt="image" src="https://github.com/user-attachments/assets/9e1f04f8-9b7e-42c6-a3f5-7dc23933fdb0" />

