#!/bin/bash
echo "================================================"
echo "       视频字幕生成器 - 启动脚本"
echo "================================================"
echo ""

# 切换到脚本目录
cd "$(dirname "$0")"

# 安装依赖
echo "[1/2] 正在检查/安装依赖..."
pip install -r requirements.txt -q

echo "[2/2] 启动服务器..."
echo ""
echo " 访问地址: http://localhost:5000"
echo " 按 Ctrl+C 停止服务"
echo ""

python app.py
