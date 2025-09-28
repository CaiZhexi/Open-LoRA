#!/bin/bash
# Qwen3-8B 高中各学科数据集微调启动脚本 (Linux)

echo "===== Qwen3-8B 高中各学科数据集 QLoRA 微调 ====="

# 检查虚拟环境
if [ ! -f "venv/bin/activate" ]; then
    echo "错误: 未找到虚拟环境 venv"
    echo "请先运行: python -m venv venv"
    echo "然后运行: source venv/bin/activate"
    echo "安装依赖: pip install -r config/requirements.txt"
    exit 1
fi

# 激活虚拟环境
source venv/bin/activate

# 检查CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 检查训练脚本是否存在
if [ ! -f "scripts/training/simple_train_ultimate.py" ]; then
    echo "错误: 未找到训练脚本"
    exit 1
fi

echo ""
echo "环境检查完成，开始训练..."
echo "按 Ctrl+C 取消训练，或按回车键继续..."
read -p ""

# 开始训练
python scripts/training/simple_train_ultimate.py

echo ""
echo "训练完成！"
