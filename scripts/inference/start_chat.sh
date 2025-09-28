#!/bin/bash
# 启动交互式高中教学AI助手

echo "🚀 启动高中教学AI助手"

# 检查虚拟环境
if [ ! -f "venv/bin/activate" ]; then
    echo "❌ 未找到虚拟环境 venv"
    echo "请先运行: python -m venv venv"
    echo "然后运行: source venv/bin/activate"
    echo "安装依赖: pip install -r config/requirements.txt"
    exit 1
fi

# 激活虚拟环境
source venv/bin/activate

# 启动交互式助手
python scripts/inference/interactive_chat.py

echo "👋 感谢使用！"

