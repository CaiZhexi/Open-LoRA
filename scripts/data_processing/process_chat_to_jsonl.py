#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将JDDC数据集的chat.txt文件转换为微调可用的jsonl格式
"""
import json
import pandas as pd
from collections import defaultdict
import os


def process_chat_data(input_file, output_file):
    """
    处理chat.txt数据并转换为jsonl格式
    
    Args:
        input_file: 输入的chat.txt文件路径
        output_file: 输出的jsonl文件路径
    """
    print(f"正在读取数据文件: {input_file}")
    
    # 读取数据文件 - 逐行解析以处理格式不一致的问题
    conversations_data = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            header_line = f.readline().strip()
            print(f"文件头: {header_line}")
            
            # 解析头部字段
            headers = header_line.split('\t')
            print(f"字段数量: {len(headers)}")
            print(f"字段名称: {headers}")
            
            line_num = 1
            for line in f:
                line_num += 1
                line = line.strip()
                if not line:
                    continue
                    
                # 分割字段
                fields = line.split('\t')
                
                # 如果字段数量不匹配，尝试处理
                if len(fields) >= 6:  # 至少需要基本字段
                    try:
                        session_id = fields[0]
                        user_id = fields[1] 
                        waiter_send = int(fields[2])
                        is_transfer = int(fields[3])
                        is_repeat = int(fields[4])
                        # content字段可能包含制表符，所以合并剩余字段
                        content = '\t'.join(fields[5:])
                        
                        conversations_data.append({
                            'session_id': session_id,
                            'user_id': user_id,
                            'waiter_send': waiter_send,
                            'is_transfer': is_transfer,
                            'is_repeat': is_repeat,
                            'content': content
                        })
                    except (ValueError, IndexError) as e:
                        print(f"第{line_num}行解析错误: {e}, 跳过该行")
                        continue
                else:
                    print(f"第{line_num}行字段不足({len(fields)}个)，跳过")
                    continue
                    
        print(f"成功读取 {len(conversations_data)} 条记录")
        
        # 显示前几行数据结构
        print("\n前5行数据样例:")
        for i, row in enumerate(conversations_data[:5]):
            print(f"第{i+1}行: {row}")
        
    except Exception as e:
        print(f"读取文件出错: {e}")
        return
    
    # 按session_id分组对话
    conversations = defaultdict(list)
    
    for row in conversations_data:
        session_id = row['session_id']
        content = str(row['content']).strip()
        waiter_send = row['waiter_send']  # 0=用户, 1=客服
        
        # 跳过空内容
        if not content or content == 'nan' or content == '':
            continue
            
        # 确定角色
        role = "assistant" if waiter_send == 1 else "user"
        
        conversations[session_id].append({
            'role': role,
            'content': content
        })
    
    print(f"\n找到 {len(conversations)} 个对话会话")
    
    # 转换为jsonl格式
    valid_conversations = 0
    output_data = []
    
    for session_id, messages in conversations.items():
        # 检查对话是否符合要求
        if len(messages) < 2:  # 至少需要一对对话
            continue
            
        # 确保第一条消息是用户消息
        if messages[0]['role'] != 'user':
            # 如果第一条不是用户消息，尝试找到第一条用户消息
            user_start = -1
            for i, msg in enumerate(messages):
                if msg['role'] == 'user':
                    user_start = i
                    break
            
            if user_start == -1:  # 没有用户消息，跳过
                continue
                
            messages = messages[user_start:]  # 从第一条用户消息开始
        
        # 检查是否有用户和助手的交替对话
        user_count = sum(1 for msg in messages if msg['role'] == 'user')
        assistant_count = sum(1 for msg in messages if msg['role'] == 'assistant')
        
        if user_count == 0 or assistant_count == 0:
            continue
            
        # 优化对话，确保用户和助手交替出现
        optimized_messages = []
        last_role = None
        
        for msg in messages:
            # 如果当前角色与上一个角色相同，则跳过（但保留第一条消息）
            if msg['role'] != last_role or last_role is None:
                optimized_messages.append(msg)
                last_role = msg['role']
        
        # 确保至少有一对对话
        if len(optimized_messages) < 2:
            continue
            
        # 确保第一条是用户消息，最后一条是助手消息
        if optimized_messages[0]['role'] != 'user':
            continue
            
        # 如果最后一条不是助手消息，添加一个简单的助手回复或删除最后的用户消息
        if optimized_messages[-1]['role'] == 'user':
            if len(optimized_messages) > 1:
                optimized_messages = optimized_messages[:-1]  # 删除最后的用户消息
            else:
                continue  # 如果只有一条用户消息，跳过
        
        # 最终检查：确保有完整的用户-助手对话
        final_user_count = sum(1 for msg in optimized_messages if msg['role'] == 'user')
        final_assistant_count = sum(1 for msg in optimized_messages if msg['role'] == 'assistant')
        
        if final_user_count == 0 or final_assistant_count == 0:
            continue
            
        # 创建jsonl条目
        conversation_entry = {
            "messages": optimized_messages
        }
        
        output_data.append(conversation_entry)
        valid_conversations += 1
    
    print(f"生成了 {valid_conversations} 个有效对话")
    
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


def validate_jsonl_format(file_path, num_samples=5):
    """
    验证生成的jsonl文件是否符合微调要求
    
    Args:
        file_path: jsonl文件路径
        num_samples: 验证的样本数量
    """
    print(f"\n验证文件格式: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"总共 {len(lines)} 条记录")
    
    valid_count = 0
    issues = []
    
    for i, line in enumerate(lines[:num_samples]):
        try:
            # 1. 每行是一个独立的JSON对象
            data = json.loads(line.strip())
            
            # 2. 每个对象须包含一个键名为messages的数组，数组不能为空
            if 'messages' not in data or not isinstance(data['messages'], list) or len(data['messages']) == 0:
                issues.append(f"第{i+1}行: messages字段缺失或为空")
                continue
            
            messages = data['messages']
            
            # 3. messages中每个元素必须包含role和content两个字段
            for j, msg in enumerate(messages):
                if 'role' not in msg or 'content' not in msg:
                    issues.append(f"第{i+1}行消息{j+1}: 缺少role或content字段")
                    continue
                
                # 4. role只能是system、user或assistant
                if msg['role'] not in ['system', 'user', 'assistant']:
                    issues.append(f"第{i+1}行消息{j+1}: role字段值无效 ({msg['role']})")
                    continue
            
            # 5. 如果有system角色消息，应在数组首位
            system_indices = [j for j, msg in enumerate(messages) if msg['role'] == 'system']
            if system_indices and system_indices[0] != 0:
                issues.append(f"第{i+1}行: system消息不在首位")
                continue
            
            # 6. 第一条非system消息必须是user角色
            first_non_system_idx = 0
            while first_non_system_idx < len(messages) and messages[first_non_system_idx]['role'] == 'system':
                first_non_system_idx += 1
            
            if first_non_system_idx < len(messages) and messages[first_non_system_idx]['role'] != 'user':
                issues.append(f"第{i+1}行: 第一条非system消息不是user角色")
                continue
            
            # 7. user和assistant角色的消息应当交替、成对出现，不少于1对
            non_system_messages = [msg for msg in messages if msg['role'] != 'system']
            if len(non_system_messages) < 2:
                issues.append(f"第{i+1}行: 用户-助手对话少于1对")
                continue
            
            # 检查交替模式
            user_count = sum(1 for msg in non_system_messages if msg['role'] == 'user')
            assistant_count = sum(1 for msg in non_system_messages if msg['role'] == 'assistant')
            
            if user_count == 0 or assistant_count == 0:
                issues.append(f"第{i+1}行: 缺少用户或助手消息")
                continue
            
            valid_count += 1
            
        except json.JSONDecodeError as e:
            issues.append(f"第{i+1}行: JSON格式错误 - {e}")
            continue
    
    print(f"验证了 {min(num_samples, len(lines))} 条记录")
    print(f"有效记录: {valid_count}")
    print(f"问题记录: {len(issues)}")
    
    if issues:
        print("\n发现的问题:")
        for issue in issues[:10]:  # 只显示前10个问题
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... 还有 {len(issues) - 10} 个问题")
    else:
        print("✅ 所有验证的记录都符合格式要求！")


def main():
    # 文件路径
    input_file = "datasets/JDDC-Baseline-Seq2Seq-master/data/chat.txt"
    output_file = "data/jddc_chat_train.jsonl"
    
    if not os.path.exists(input_file):
        print(f"输入文件不存在: {input_file}")
        return
    
    # 处理数据
    process_chat_data(input_file, output_file)
    
    # 验证生成的文件
    if os.path.exists(output_file):
        validate_jsonl_format(output_file, num_samples=10)


if __name__ == "__main__":
    main()