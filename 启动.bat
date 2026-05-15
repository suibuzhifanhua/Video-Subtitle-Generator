@echo off
chcp 65001 > nul
echo ==================================================
echo   视频字幕生成器 - 启动中...
echo ==================================================
echo.

REM 检查 Python 是否可用
python --version > nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.8+
    pause
    exit /b 1
)

REM 启动 Flask 服务器（后台）
echo [1/2] 正在启动服务器...
start /b python "%~dp0app.py"

REM 等待服务器启动
echo [2/2] 正在打开浏览器...
timeout /t 3 > nul

REM 打开浏览器
start http://127.0.0.1:5000

echo.
echo ==================================================
echo   服务器已启动，浏览器已打开
echo   关闭此窗口将停止服务器
echo ==================================================
echo.

REM 保持窗口打开，等待用户按 Ctrl+C
python -c "import time; time.sleep(999999)"
