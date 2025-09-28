#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将JDDC数据集的test.txt文件转换为微调可用的jsonl格式
"""
import json
import os
import re


def process_test_data(input_file, output_file):
    """
    处理test.txt数据并转换为jsonl格式
    
    Args:
        input_file: 输入的test.txt文件路径
        output_file: 输出的jsonl文件路径
    """
    print(f"正在读取数据文件: {input_file}")
    
    output_data = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"成功读取 {len(lines)} 行数据")
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # 每行是一段对话，用逗号分隔
            sentences = [s.strip() for s in line.split(',') if s.strip()]
            
            if len(sentences) < 2:  # 至少需要两句话
                continue
            
            # 假设交替模式：用户->客服->用户->客服...
            # 第一句通常是用户的问题或陈述
            messages = []
            
            for j, sentence in enumerate(sentences):
                # 简单的启发式规则来判断角色
                # 用户消息特征：询问、抱怨、请求等
                # 客服消息特征：解答、道歉、服务用语等
                
                if j == 0:
                    # 第一句假设是用户
                    role = "user"
                elif j % 2 == 0:
                    # 偶数索引假设是用户
                    role = "user"
                else:
                    # 奇数索引假设是客服
                    role = "assistant"
                
                # 根据内容特征进行调整
                service_keywords = ['请您稍等', '正在为您', '有什么问题', '谢谢您', '感谢您', '请问还有', '亲', '您好', '抱歉', '不好意思']
                user_keywords = ['什么时候', '怎么', '为什么', '能不能', '可以', '我的', '我想', '急用', '着急']
                
                is_service_msg = any(keyword in sentence for keyword in service_keywords)
                is_user_msg = any(keyword in sentence for keyword in user_keywords)
                
                if is_service_msg and not is_user_msg:
                    role = "assistant"
                elif is_user_msg and not is_service_msg:
                    role = "user"
                
                messages.append({
                    'role': role,
                    'content': sentence
                })
            
            # 确保对话符合要求：第一个是用户，用户和助手交替
            if not messages:
                continue
                
            # 确保第一个是用户消息
            if messages[0]['role'] != 'user':
                # 寻找第一个用户消息
                user_start = -1
                for k, msg in enumerate(messages):
                    if msg['role'] == 'user':
                        user_start = k
                        break
                
                if user_start == -1:
                    continue
                    
                messages = messages[user_start:]
            
            # 优化对话，确保用户和助手交替出现
            optimized_messages = []
            last_role = None
            
            for msg in messages:
                if msg['role'] != last_role or last_role is None:
                    optimized_messages.append(msg)
                    last_role = msg['role']
            
            # 确保至少有一对对话
            if len(optimized_messages) < 2:
                continue
            
            # 确保最后一条是助手消息
            if optimized_messages[-1]['role'] == 'user':
                if len(optimized_messages) > 1:
                    optimized_messages = optimized_messages[:-1]
                else:
                    continue
            
            # 检查是否有用户和助手的对话
            user_count = sum(1 for msg in optimized_messages if msg['role'] == 'user')
            assistant_count = sum(1 for msg in optimized_messages if msg['role'] == 'assistant')
            
            if user_count == 0 or assistant_count == 0:
                continue
            
            # 创建jsonl条目
            conversation_entry = {
                "messages": optimized_messages
            }
            
            output_data.append(conversation_entry)
    
    except Exception as e:
        print(f"处理文件出错: {e}")
        return
    
    print(f"生成了 {len(output_data)} 个有效对话")
    
    # 写入jsonl文件
    print(f"正在写入输出文件: {output_file}")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in output_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"成功写入 {len(output_data)} 条记录到 {output_file}")
    
    # 显示几个样例
    print("\n生成的数据样例:")
    for i, entry in enumerate(output_data[:3]):
        print(f"\n样例 {i+1}:")
        print(json.dumps(entry, ensure_ascii=False, indent=2))


def main():
    # 文件路径
    input_file = "datasets/JDDC-Baseline-Seq2Seq-master/data/test/test.txt"  
    output_file = "data/jddc_test_train.jsonl"
    
    if not os.path.exists(input_file):
        print(f"输入文件不存在: {input_file}")
        return
    
    # 处理数据
    process_test_data(input_file, output_file)


if __name__ == "__main__":
    main()