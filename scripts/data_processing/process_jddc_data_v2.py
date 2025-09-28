#!/usr/bin/env python3
"""
JDDC数据处理脚本 V2 - 路径修复版
使用项目根目录相对路径，兼容跨平台
"""

import pandas as pd
import json
import os
from collections import defaultdict
import re

# 添加项目根目录路径计算
def get_project_root():
    """获取项目根目录"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..', '..')
    return os.path.abspath(project_root)

def clean_content(content):
    """深度清理content内容"""
    if pd.isna(content) or content == '':
        return ''
    
    content = str(content).strip()
    
    # 1. 删除开头的纯数字（如 "1186710", "4383667"）
    content = re.sub(r'^\d+\s*', '', content)
    
    # 2. 删除URL链接
    content = re.sub(r'https?://[^\s]+', '', content)
    
    # 3. 删除#E-s[数字x]等噪声标签
    content = re.sub(r'#E-s\[[^\]]*\]', '', content)
    
    # 4. 清理其他噪声标签
    content = re.sub(r'#[A-Za-z0-9-]+\[[^\]]*\]', '', content)
    
    # 5. 替换真实订单号为占位符
    content = re.sub(r'\[ORDERID_\d+\]', '<ORDER_ID>', content)
    content = re.sub(r'订单号[:：]\s*\d+', '订单号: <ORDER_ID>', content)
    
    # 6. 替换其他敏感信息占位符
    content = re.sub(r'\[订单编号:[^\]]+\]', '<ORDER_INFO>', content)
    content = re.sub(r'\[金额[^\]]*\]', '<AMOUNT>', content)
    content = re.sub(r'\[日期[^\]]*\]', '<DATE>', content)
    content = re.sub(r'\[时间[^\]]*\]', '<TIME>', content)
    
    # 7. 去除多余的空白字符
    content = re.sub(r'\s+', ' ', content)
    content = content.strip()
    
    return content

def is_meaningful_content(content):
    """判断内容是否有意义"""
    if not content or len(content.strip()) == 0:
        return False
    
    # 过滤掉极短的无意义内容
    if len(content.strip()) <= 2 and content.strip() in ['?', '？', '.', '。', '!', '！', 'ok', 'OK']:
        return False
    
    # 过滤掉只包含标点和空格的内容
    if re.match(r'^[\s\.,!?？。！]*$', content):
        return False
        
    return True

def merge_consecutive_messages(messages):
    """合并同一角色的连续消息"""
    if not messages:
        return []
    
    merged = []
    current_role = None
    current_contents = []
    
    for msg in messages:
        role = msg['role']
        content = msg['content']
        
        if role == current_role:
            # 同一角色，累积内容
            if content:  # 只添加非空内容
                current_contents.append(content)
        else:
            # 角色切换，保存之前累积的内容
            if current_role and current_contents:
                merged_content = ' '.join(current_contents)
                if is_meaningful_content(merged_content):
                    merged.append({
                        'role': current_role,
                        'content': merged_content
                    })
            
            # 开始新角色的累积
            current_role = role
            current_contents = [content] if content else []
    
    # 处理最后一组
    if current_role and current_contents:
        merged_content = ' '.join(current_contents)
        if is_meaningful_content(merged_content):
            merged.append({
                'role': current_role,
                'content': merged_content
            })
    
    return merged

def ensure_alternating_pattern(messages):
    """确保user和assistant交替出现"""
    if not messages:
        return []
    
    result = []
    prev_role = None
    
    for msg in messages:
        current_role = msg['role']
        
        # 如果是连续的同一角色，跳过（已经在merge阶段处理过了）
        if current_role == prev_role:
            continue
            
        result.append(msg)
        prev_role = current_role
    
    return result

def load_and_clean_data(file_path):
    """加载并清理数据"""
    print("正在加载数据...")
    
    data_rows = []
    with open(file_path, 'r', encoding='utf-8') as f:
        headers = f.readline().strip().split('\t')
        print(f"数据头部: {headers}")
        
        for line_num, line in enumerate(f, 1):
            if line_num % 10000 == 0:
                print(f"已处理 {line_num} 行...")
                
            parts = line.strip().split('\t')
            if len(parts) >= 6:
                session_id = parts[0]
                user_id = parts[1]
                waiter_send = int(parts[2]) if parts[2].isdigit() else parts[2]
                is_transfer = parts[3]
                is_repeat = parts[4]
                content = '\t'.join(parts[5:]) if len(parts) > 6 else parts[5]
                content = clean_content(content)
                
                # 只保留有意义的内容
                if is_meaningful_content(content):
                    data_rows.append({
                        'session_id': session_id,
                        'user_id': user_id,
                        'waiter_send': waiter_send,
                        'is_transfer': is_transfer,
                        'is_repeat': is_repeat,
                        'content': content
                    })
    
    df = pd.DataFrame(data_rows)
    print(f"成功加载 {len(df)} 行有效数据")
    return df

def process_sessions(df):
    """按session_id分组处理对话"""
    print("正在处理会话数据...")
    
    grouped = df.groupby('session_id')
    print(f"共发现 {len(grouped)} 个会话")
    
    processed_conversations = []
    skipped_sessions = {
        'empty_after_clean': 0,
        'single_side': 0,
        'too_short': 0,
        'too_long': 0,
        'bad_pattern': 0
    }
    
    for session_id, session_data in grouped:
        # 去除重复内容的消息
        session_data = session_data.drop_duplicates(subset=['content'])
        session_data = session_data.reset_index(drop=True)
        
        if len(session_data) == 0:
            skipped_sessions['empty_after_clean'] += 1
            continue
        
        # 检查是否只包含单方发言
        unique_senders = session_data['waiter_send'].unique()
        if len(unique_senders) == 1:
            skipped_sessions['single_side'] += 1
            continue
        
        # 构建原始对话消息列表
        raw_messages = []
        for _, row in session_data.iterrows():
            role = "user" if row['waiter_send'] == 0 else "assistant"
            raw_messages.append({
                "role": role,
                "content": row['content']
            })
        
        # 合并连续的同角色消息
        merged_messages = merge_consecutive_messages(raw_messages)
        
        if len(merged_messages) == 0:
            skipped_sessions['empty_after_clean'] += 1
            continue
        
        # 确保交替模式
        alternating_messages = ensure_alternating_pattern(merged_messages)
        
        # 检查最终对话质量
        if len(alternating_messages) < 2:  # 至少要有2条消息才能构成对话
            skipped_sessions['too_short'] += 1
            continue
        
        # 检查是否有user和assistant两种角色
        roles_in_conversation = set(msg['role'] for msg in alternating_messages)
        if len(roles_in_conversation) < 2:
            skipped_sessions['single_side'] += 1
            continue
        
        # 确保对话以user开始（更自然）
        if alternating_messages[0]['role'] != 'user':
            # 如果第一条是assistant，尝试调整或跳过
            if len(alternating_messages) >= 3:
                # 如果有足够消息，跳过第一条assistant消息
                alternating_messages = alternating_messages[1:]
            else:
                skipped_sessions['bad_pattern'] += 1
                continue
        
        # 控制对话长度（3-6轮）
        if len(alternating_messages) < 3:
            skipped_sessions['too_short'] += 1
            continue
        elif len(alternating_messages) > 6:
            alternating_messages = alternating_messages[:6]
        
        # 构建最终对话
        conversation = []
        
        # 添加system消息
        conversation.append({
            "role": "system", 
            "content": "你是一个京东客服机器人，帮助用户解答问题。"
        })
        
        # 添加处理后的对话内容
        conversation.extend(alternating_messages)
        
        # 检查token长度
        total_chars = sum(len(msg['content']) for msg in conversation)
        if total_chars > 2048:
            skipped_sessions['too_long'] += 1
            continue
        
        processed_conversations.append({
            "conversations": conversation
        })
    
    print(f"处理完成:")
    print(f"  成功处理: {len(processed_conversations)} 个会话")
    print(f"  跳过统计:")
    for reason, count in skipped_sessions.items():
        print(f"    {reason}: {count}")
    
    return processed_conversations

def save_jsonl(data, output_file):
    """保存为JSONL格式"""
    print(f"正在保存到 {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"成功保存 {len(data)} 条训练样本")

def main():
    """主函数"""
    # 使用项目根目录相对路径
    project_root = get_project_root()
    input_file = os.path.join(project_root, "datasets", "JDDC-Baseline-Seq2Seq-master", "data", "chat.txt")
    output_file = os.path.join(project_root, "data", "train_v2.jsonl")
    
    print("=== JDDC数据清洗V2开始 ===")
    print(f"项目根目录: {project_root}")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        print("请确保数据集文件位于正确位置")
        return
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 1. 加载和清理数据
    df = load_and_clean_data(input_file)
    
    # 2. 处理会话
    conversations = process_sessions(df)
    
    # 3. 保存结果
    save_jsonl(conversations, output_file)
    
    # 4. 显示示例
    print("\n=== V2处理结果示例 ===")
    if conversations:
        for i, example in enumerate(conversations[:5], 1):
            print(f"\n示例 {i}:")
            for msg in example['conversations']:
                role_map = {'system': '系统', 'user': '用户', 'assistant': '客服'}
                role = role_map.get(msg['role'], msg['role'])
                content = msg['content'][:100] + '...' if len(msg['content']) > 100 else msg['content']
                print(f"  {role}: {content}")
    
    print(f"\n=== 数据清洗V2完成 ===")

if __name__ == "__main__":
    main() 