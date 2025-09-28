#!/usr/bin/env python3
"""
模型加载测试脚本 - 路径修复版
使用项目根目录相对路径，兼容跨平台
"""

import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def get_project_root():
    """获取项目根目录"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..', '..')
    return os.path.abspath(project_root)

def main():
    # 使用项目根目录相对路径
    project_root = get_project_root()
    model_path = os.path.join(project_root, "models", "qwen2-7b-instruct")
    
    print(f"项目根目录: {project_root}")
    print(f"模型路径: {model_path}")
    
    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        print("请确保模型文件位于正确位置")
        return
    
    print("正在加载模型...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",  # 自动分配到 GPU
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        print("✅ 模型加载成功")
        
        # 测试对话
        messages = [
            {"role": "system", "content": "你是一个有帮助的客服机器人。"},
            {"role": "user", "content": "你好，我下的订单怎么还没到？"}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        print("正在生成回复...")
        outputs = model.generate(**inputs, max_new_tokens=100)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print("="*50)
        print("🤖 模型回复:")
        print(response)
        print("="*50)
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
    
    finally:
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 GPU缓存已清理")

if __name__ == "__main__":
    main()
