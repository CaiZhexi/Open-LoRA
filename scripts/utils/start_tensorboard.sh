#!/bin/bash

echo "========================================"
echo "   启动 TensorBoard 监控界面"
echo "========================================"
echo

echo "正在检查日志目录..."
if [ ! -d "./logs" ]; then
    echo "❌ 日志目录不存在！请先运行训练生成日志"
    echo "训练会自动创建 logs 目录"
    exit 1
fi

echo "✅ 日志目录已找到"
echo

echo "正在启动 TensorBoard..."
echo "请在浏览器中访问: http://localhost:6006"
echo "按 Ctrl+C 停止 TensorBoard"
echo "========================================"
tensorboard --logdir=logs --port=6006 --host=0.0.0.0 