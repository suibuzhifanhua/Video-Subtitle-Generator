@echo off
chcp 65001 > nul
echo ================================================
echo        视频字幕生成器 - 启动脚本
echo ================================================
echo.

:: 检查 Python
python --version > nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.8+
    pause
    exit /b 1
)

:: 切换到脚本所在目录
cd /d "%~dp0"

:: 安装依赖
echo [1/2] 正在检查/安装依赖...
pip install -r requirements.txt -q
if errorlevel 1 (
    echo [错误] 依赖安装失败，请检查网络连接
    pause
    exit /b 1
)

echo [2/2] 启动服务器...
echo.
echo  访问地址: http://localhost:5000
echo  按 Ctrl+C 停止服务
echo.
python app.py

pause
