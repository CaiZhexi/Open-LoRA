@echo off
REM 启动交互式高中教学AI助手 (Windows)

echo 🚀 启动高中教学AI助手

REM 检查虚拟环境
if not exist "venv\Scripts\activate.bat" (
    echo ❌ 未找到虚拟环境 venv
    echo 请先运行: python -m venv venv
    echo 然后运行: venv\Scripts\activate.bat
    echo 安装依赖: pip install -r config\requirements.txt
    pause
    exit /b 1
)

REM 激活虚拟环境
call venv\Scripts\activate.bat

REM 启动交互式助手
python scripts\inference\interactive_chat.py

echo 👋 感谢使用！
pause

