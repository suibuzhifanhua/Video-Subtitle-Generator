# 视频字幕生成器

> 上传视频，自动识别语音生成字幕，支持一键翻译成中文。

## 下载使用（Windows）

下载 Release 中的两个文件，放在**同一目录下**即可直接运行，无需安装 Python：

| 文件 | 说明 |
|------|------|
| `subtitle_generator.exe` | 主程序 |
| `subtitle_generator_v1.0.0_internal.zip` | 解压后得到 `_internal` 文件夹 |

### 使用步骤

1. 下载并解压 `subtitle_generator_v1.0.0_internal.zip`
2. 将 `subtitle_generator.exe` 放入解压后的目录
3. 双击 `subtitle_generator.exe` 启动
4. 打开浏览器访问显示的地址（默认 `http://127.0.0.1:5000`）

## 功能特性

- **支持多种格式**：MP4、MKV、AVI、MOV、WMV、FLV、WebM 等
- **多语言识别**：自动检测中、英、日、韩等 99+ 种语言
- **三种输出格式**：`.srt`（通用）、`.vtt`（网页）、`.txt`（纯文本）
- **实时进度**：字幕生成过程实时显示进度和预览
- **一键翻译**：生成字幕后可一键翻译成中文（免费，无需 API Key）
- **模型管理**：右上角 ⚙️ 设置中可管理 Whisper 模型

## 模型说明

首次使用会自动从 HuggingFace 下载 Whisper 模型。

| 模型 | 大小 | 推荐用途 |
|------|------|---------|
| Tiny | ~75 MB | 快速测试 |
| Base | ~140 MB | **推荐日常使用** |
| Small | ~460 MB | 高精度 |
| Medium | ~1.5 GB | 专业级 |
| Large v3 | ~3.1 GB | 最高精度 |

> 右上角 ⚙️ 按钮 → 可修改模型存放路径、查看/下载/删除指定模型。

## 翻译功能

字幕生成完成后，点击「🌐 一键翻译成中文」，系统自动：
1. 将字幕原文翻译成中文
2. 生成中文版 `.srt`、`.vtt`、`.txt` 三种格式
3. 支持单独下载中文字幕文件

> ⚠️ 翻译需要网络连接，速度取决于字幕长度。

## 从源码运行

```bash
# 安装依赖
pip install -r requirements.txt

# 启动
python app.py
```

## 打包 exe

```bash
python build_exe.py
```

打包完成后双击 `dist/subtitle_generator/subtitle_generator.exe` 运行。

## 技术栈

- **语音识别**：faster-whisper（CTranslate2 加速版 Whisper）
- **翻译**：Nekofox/M2M100-ja-zh（日译中）/ opus-mt（多语言）
- **前端**：Flask + 原生 HTML/JS
- **打包**：PyInstaller

## 系统要求

- Windows 10 / 11
- 内存建议 8GB+（使用 Large 模型建议 16GB+）

---

如果你用这个项目看片看得更得劲了，不妨请我喝杯咖啡 ☕

<img width="300" height="327" alt="支付宝" src="https://github.com/user-attachments/assets/f29f5a9d-b93e-49b2-8ac3-5c2dfcf1224f" />
<img width="300" height="327" alt="微信" src="https://github.com/user-attachments/assets/9e1f04f8-9b7e-42c6-a3f5-7dc23933fdb0" />
