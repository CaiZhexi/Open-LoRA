#!/usr/bin/env python3
"""
E-commerce数据集预处理脚本
将E-commerce数据集格式转换为适用于Qwen2-7B LoRA微调的JSONL格式

数据格式转换:
输入: label \t conversation_utterances (splited by \t) \t response
输出: {"conversations": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EcommerceDataProcessor:
    def __init__(self, project_root: str = None):
        """初始化数据处理器"""
        if project_root is None:
            # 获取项目根目录
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.project_root = os.path.join(script_dir, '..', '..')
        else:
            self.project_root = project_root
        
        self.project_root = os.path.abspath(self.project_root)
        
        # 设置输入和输出路径
        self.ecommerce_dir = os.path.join(self.project_root, "datasets", "E-commerce dataset")
        self.output_dir = os.path.join(self.project_root, "data")
        
        # 系统提示语
        self.system_prompt = "你是一个电商客服机器人，帮助用户解答购物相关问题。请以友好、专业的态度回复用户。"
    
    def read_ecommerce_data(self, file_path: str) -> List[Dict[str, Any]]:
        """读取E-commerce数据集文件"""
        logger.info(f"正在读取数据文件: {file_path}")
        
        data_list = []
        total_lines = 0
        error_lines = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    total_lines += 1
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        # 分割数据: label \t conversation_utterances \t response
                        parts = line.split('\t')
                        if len(parts) < 3:
                            error_lines += 1
                            continue
                        
                        label = parts[0]
                        response = parts[-1]  # 最后一个是回复
                        
                        # 中间的所有部分组成对话历史
                        conversation_parts = parts[1:-1]
                        conversation_history = ' '.join(conversation_parts)
                        
                        data_item = {
                            'label': label,
                            'conversation_history': conversation_history,
                            'response': response,
                            'line_num': line_num
                        }
                        
                        data_list.append(data_item)
                        
                    except Exception as e:
                        error_lines += 1
                        logger.warning(f"解析第{line_num}行时出错: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"读取文件失败: {e}")
            return []
        
        logger.info(f"文件读取完成: 总行数={total_lines}, 成功解析={len(data_list)}, 错误行数={error_lines}")
        return data_list
    
    def clean_spaces(self, text: str) -> str:
        """清理文本中的多余空格，保持自然的中文表达"""
        if not text:
            return text
        
        import re
        
        # 首先去除首尾空格
        text = text.strip()
        
        # 去除连续的多个空格，替换为单个空格
        text = re.sub(r'\s+', ' ', text)
        
        # 多次迭代去除中文字符之间的空格，确保完全清理
        prev_text = ""
        while prev_text != text:
            prev_text = text
            # 去除中文字符之间的空格
            text = re.sub(r'([\u4e00-\u9fff])\s+([\u4e00-\u9fff])', r'\1\2', text)
        
        # 去除中文字符和标点符号之间的空格
        text = re.sub(r'([\u4e00-\u9fff])\s+([，。！？：；、（）""''【】《》])', r'\1\2', text)
        text = re.sub(r'([，。！？：；、（）""''【】《》])\s+([\u4e00-\u9fff])', r'\1\2', text)
        
        # 去除数字和中文之间的空格
        text = re.sub(r'(\d)\s+([\u4e00-\u9fff])', r'\1\2', text)
        text = re.sub(r'([\u4e00-\u9fff])\s+(\d)', r'\1\2', text)
        
        # 去除英文字母和中文之间的空格
        text = re.sub(r'([a-zA-Z])\s+([\u4e00-\u9fff])', r'\1\2', text)
        text = re.sub(r'([\u4e00-\u9fff])\s+([a-zA-Z])', r'\1\2', text)
        
        # 特殊处理：去除中文和数字、字母、标点符号组合中的空格
        text = re.sub(r'([\u4e00-\u9fff])\s+([a-zA-Z0-9，。！？：；、（）""''【】《》])', r'\1\2', text)
        text = re.sub(r'([a-zA-Z0-9，。！？：；、（）""''【】《》])\s+([\u4e00-\u9fff])', r'\1\2', text)
        
        # 最后再次去除多余的空格
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def convert_to_conversation_format(self, data_item: Dict[str, Any]) -> Dict[str, Any]:
        """将单个数据项转换为对话格式"""
        conversation_history = self.clean_spaces(data_item['conversation_history'])
        response = self.clean_spaces(data_item['response'])
        
        # 构建对话格式
        conversations = [
            {
                "role": "system",
                "content": self.system_prompt
            }
        ]
        
        # 处理对话历史
        if conversation_history:
            # 假设对话历史是用户的多轮对话，我们将其作为用户输入
            conversations.append({
                "role": "user", 
                "content": conversation_history
            })
        
        # 添加助手回复
        conversations.append({
            "role": "assistant",
            "content": response
        })
        
        return {"conversations": conversations}
    
    def process_file(self, input_file: str, output_file: str, max_samples: int = None) -> bool:
        """处理单个文件"""
        input_path = os.path.join(self.ecommerce_dir, input_file)
        output_path = os.path.join(self.output_dir, output_file)
        
        if not os.path.exists(input_path):
            logger.error(f"输入文件不存在: {input_path}")
            return False
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 读取原始数据
        raw_data = self.read_ecommerce_data(input_path)
        if not raw_data:
            logger.error(f"读取数据失败: {input_path}")
            return False
        
        # 限制样本数量（用于测试）
        if max_samples:
            raw_data = raw_data[:max_samples]
            logger.info(f"限制样本数量为: {max_samples}")
        
        # 转换格式并写入文件
        logger.info(f"正在转换格式并写入: {output_path}")
        
        converted_count = 0
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for data_item in raw_data:
                    try:
                        converted_item = self.convert_to_conversation_format(data_item)
                        json_line = json.dumps(converted_item, ensure_ascii=False)
                        f.write(json_line + '\n')
                        converted_count += 1
                    except Exception as e:
                        logger.warning(f"转换第{data_item.get('line_num', '?')}行时出错: {e}")
                        continue
            
            logger.info(f"文件处理完成: {output_file}, 成功转换 {converted_count} 个样本")
            return True
            
        except Exception as e:
            logger.error(f"写入文件失败: {e}")
            return False
    
    def process_all_files(self, max_samples: int = None):
        """处理所有数据文件"""
        logger.info("=== E-commerce数据集预处理开始 ===")
        logger.info(f"项目根目录: {self.project_root}")
        logger.info(f"输入目录: {self.ecommerce_dir}")
        logger.info(f"输出目录: {self.output_dir}")
        
        # 文件映射
        file_mapping = {
            'train.txt': 'ecommerce_train.jsonl',
            'dev.txt': 'ecommerce_dev.jsonl', 
            'test.txt': 'ecommerce_test.jsonl'
        }
        
        success_count = 0
        for input_file, output_file in file_mapping.items():
            logger.info(f"\n--- 处理 {input_file} -> {output_file} ---")
            if self.process_file(input_file, output_file, max_samples):
                success_count += 1
            else:
                logger.error(f"处理 {input_file} 失败")
        
        logger.info(f"\n=== 预处理完成: {success_count}/{len(file_mapping)} 个文件处理成功 ===")
        
        # 显示输出文件信息
        self.show_output_info()
    
    def show_output_info(self):
        """显示输出文件信息"""
        logger.info("\n--- 输出文件信息 ---")
        output_files = ['ecommerce_train.jsonl', 'ecommerce_dev.jsonl', 'ecommerce_test.jsonl']
        
        for filename in output_files:
            filepath = os.path.join(self.output_dir, filename)
            if os.path.exists(filepath):
                # 统计行数
                with open(filepath, 'r', encoding='utf-8') as f:
                    line_count = sum(1 for _ in f)
                
                file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
                logger.info(f"{filename}: {line_count} 个样本, {file_size:.2f} MB")
            else:
                logger.info(f"{filename}: 文件不存在")
    
    def preview_converted_data(self, filename: str = 'ecommerce_train.jsonl', num_samples: int = 3):
        """预览转换后的数据"""
        filepath = os.path.join(self.output_dir, filename)
        if not os.path.exists(filepath):
            logger.error(f"文件不存在: {filepath}")
            return
        
        logger.info(f"\n--- 预览 {filename} 前 {num_samples} 个样本 ---")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= num_samples:
                        break
                    
                    data = json.loads(line.strip())
                    logger.info(f"\n样本 {i+1}:")
                    logger.info(json.dumps(data, ensure_ascii=False, indent=2))
                    
        except Exception as e:
            logger.error(f"预览数据失败: {e}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="E-commerce数据集预处理")
    parser.add_argument('--max_samples', type=int, default=None, 
                       help='最大样本数量（用于测试，默认处理全部数据）')
    parser.add_argument('--preview_only', action='store_true',
                       help='仅预览现有转换后的数据，不进行处理')
    parser.add_argument('--project_root', type=str, default=None,
                       help='项目根目录路径')
    
    args = parser.parse_args()
    
    # 创建处理器
    processor = EcommerceDataProcessor(args.project_root)
    
    if args.preview_only:
        # 仅预览数据
        processor.preview_converted_data()
    else:
        # 处理数据
        processor.process_all_files(args.max_samples)
        
        # 预览转换结果
        processor.preview_converted_data()

if __name__ == "__main__":
    main()