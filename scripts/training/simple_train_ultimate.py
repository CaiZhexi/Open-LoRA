#!/usr/bin/env python3
"""
Qwen3-8B QLoRA 微调脚本 - 高中各学科数据集版本
完全绕过accelerate设备分发问题
"""

import os
import sys
import json
import torch
import logging
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# 添加配置文件路径到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'config'))
from train_config import get_config

import warnings
warnings.filterwarnings("ignore")

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(data_path: str, max_samples=None):
    """加载训练数据 - 适配高中各学科数据集格式"""
    logger.info(f"正在加载数据: {data_path}")
    
    conversations_list = []
    
    # 检查是否是目录（包含多个学科数据集）
    if os.path.isdir(data_path):
        # 加载目录中所有jsonl文件
        import glob
        jsonl_files = glob.glob(os.path.join(data_path, "*.jsonl"))
        logger.info(f"找到 {len(jsonl_files)} 个学科数据集")
        
        for jsonl_file in jsonl_files:
            logger.info(f"正在加载: {os.path.basename(jsonl_file)}")
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                file_count = 0
                for line in f:
                    if max_samples and len(conversations_list) >= max_samples:
                        break
                    try:
                        data = json.loads(line.strip())
                        # 适配新格式：messages -> conversations
                        if "messages" in data:
                            # 将messages转换为conversations格式
                            conversations = []
                            for msg in data["messages"]:
                                conversations.append({
                                    "role": msg["role"],
                                    "content": msg["content"]
                                })
                            conversations_list.append(conversations)
                            file_count += 1
                    except Exception as e:
                        logger.warning(f"跳过无效行: {e}")
                        continue
                logger.info(f"从 {os.path.basename(jsonl_file)} 加载了 {file_count} 个对话")
    else:
        # 单个文件
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                try:
                    data = json.loads(line.strip())
                    # 兼容两种格式
                    if "conversations" in data:
                        conversations_list.append(data["conversations"])
                    elif "messages" in data:
                        conversations = []
                        for msg in data["messages"]:
                            conversations.append({
                                "role": msg["role"],
                                "content": msg["content"]
                            })
                        conversations_list.append(conversations)
                except Exception as e:
                    logger.warning(f"跳过无效行: {e}")
                    continue
    
    logger.info(f"成功加载 {len(conversations_list)} 个对话")
    return Dataset.from_dict({"conversations": conversations_list})

def format_conversations(conversations, tokenizer):
    """格式化对话"""
    text = ""
    for conv in conversations:
        role = conv["role"]
        content = conv["content"]
        if role == "system":
            text += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == "user":
            text += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    
    return text

def preprocess_data(dataset, tokenizer, max_length=2048):
    """预处理数据"""
    logger.info("正在预处理数据...")
    
    def tokenize_function(examples):
        texts = []
        for conversations in examples["conversations"]:
            text = format_conversations(conversations, tokenizer)
            texts.append(text)
        
        # 分词
        model_inputs = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
            add_special_tokens=False
        )
        
        # labels与input_ids相同
        model_inputs["labels"] = [ids.copy() for ids in model_inputs["input_ids"]]
        
        return model_inputs
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    logger.info(f"预处理完成，数据集大小: {len(tokenized_dataset)}")
    return tokenized_dataset

def main():
    # 获取当前脚本的根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..', '..')
    
    # 加载配置
    mode = "normal"  # 从 "test" 改为 "normal"
    config = get_config(mode)
    
    # 生成带日期后缀的输出目录
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    original_output_dir = config['training']['output_dir']
    # 使用相对于项目根目录的路径 - 使用高中各学科数据集标识
    output_dir_with_date = os.path.join(project_root, "models", f"qwen3-8b-lora-highschool-{current_date}")
    config['training']['output_dir'] = output_dir_with_date
    
    # 更新其他路径为绝对路径 - 使用高中各学科数据集
    config['model']['model_path'] = os.path.join(project_root, "models", "qwen3_8b")  # 使用本地模型
    config['data']['train_file'] = os.path.join(project_root, "datasets", "高中各学科jsonl数据集")
    
    logger.info("=== Qwen3-8B 高中各学科数据集 QLoRA 微调开始 ===")
    logger.info(f"模式: {mode}")
    logger.info(f"项目根目录: {project_root}")
    logger.info(f"模型路径: {config['model']['model_path']}")
    logger.info(f"高中各学科数据目录: {config['data']['train_file']}")
    logger.info(f"输出目录: {config['training']['output_dir']}")
    logger.info(f"训练时间: {current_date}")
    
    # 检查CUDA
    if not torch.cuda.is_available():
        logger.error("CUDA不可用，请检查GPU环境")
        return
    
    logger.info(f"使用GPU: {torch.cuda.get_device_name()}")
    logger.info(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 清理GPU缓存
    torch.cuda.empty_cache()
    
    # 设置环境变量，禁用accelerate的自动设备映射
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["ACCELERATE_USE_CPU"] = "false"
    
    # 配置量化
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # 加载分词器
    logger.info("正在加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['model_path'],
        trust_remote_code=config['model']['trust_remote_code'],
        padding_side="right",
        use_fast=False
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型 - 终极修复版本
    logger.info("正在加载模型...")
    try:
        # 方法1: 尝试不使用任何accelerate相关参数
        import transformers
        old_dispatch = getattr(transformers.modeling_utils, 'dispatch_model', None)
        
        # 临时禁用dispatch_model函数
        def dummy_dispatch(model, **kwargs):
            logger.info("跳过设备分发，直接返回模型")
            return model
        
        # 替换dispatch_model函数
        if old_dispatch:
            transformers.modeling_utils.dispatch_model = dummy_dispatch
            # 同时也替换accelerate中的
            try:
                import accelerate.big_modeling
                accelerate.big_modeling.dispatch_model = dummy_dispatch
            except:
                pass
        
        model = AutoModelForCausalLM.from_pretrained(
            config['model']['model_path'],
            quantization_config=bnb_config,
            trust_remote_code=config['model']['trust_remote_code'],
            torch_dtype=torch.float16
        )
        
        # 恢复原函数
        if old_dispatch:
            transformers.modeling_utils.dispatch_model = old_dispatch
            try:
                import accelerate.big_modeling
                accelerate.big_modeling.dispatch_model = old_dispatch
            except:
                pass
        
        logger.info("✅ 模型加载成功")
        
        # 检查模型设备
        device = next(model.parameters()).device
        logger.info(f"模型设备: {device}")
        
        # 如果模型在CPU上，手动移动到GPU（量化模型应该已经在正确设备上）
        if device.type == 'cpu' and torch.cuda.is_available():
            logger.info("检测到模型在CPU上，这对于量化模型是不正常的")
            
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 准备LoRA
    logger.info("正在准备LoRA训练...")
    try:
        model = prepare_model_for_kbit_training(model)
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=config['lora']['r'],
            lora_alpha=config['lora']['alpha'],
            lora_dropout=config['lora']['dropout'],
            target_modules=config['lora']['target_modules'],
            bias=config['lora']['bias']
        )
        
        model = get_peft_model(model, lora_config)
        logger.info("✅ LoRA配置完成")
        model.print_trainable_parameters()
        
    except Exception as e:
        logger.error(f"❌ LoRA准备失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 加载数据
    try:
        dataset = load_data(
            config['data']['train_file'], 
            config['data']['max_train_samples']
        )
        
        # 预处理
        train_dataset = preprocess_data(dataset, tokenizer, config['model']['max_length'])
        
    except Exception as e:
        logger.error(f"❌ 数据处理失败: {e}")
        return
    
    # 数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt"
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=config['training']['output_dir'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        num_train_epochs=config['training']['num_train_epochs'],
        max_steps=config['training']['max_steps'],
        learning_rate=config['training']['learning_rate'],
        warmup_steps=config['training']['warmup_steps'],
        logging_steps=1,  # 每步都记录日志，便于tensorboard监控
        save_steps=config['training']['save_steps'],
        save_total_limit=config['training']['save_total_limit'],
        fp16=config['training']['fp16'],
        dataloader_pin_memory=False,
        remove_unused_columns=config['training']['remove_unused_columns'],
        optim=config['training']['optim'],
        group_by_length=False,
        report_to=["tensorboard"],  # 启用tensorboard
        logging_dir=os.path.join(project_root, "logs"),  # tensorboard日志目录
        dataloader_num_workers=0,
        gradient_checkpointing=config['optimization']['use_gradient_checkpointing'],
        save_safetensors=True,
        ddp_find_unused_parameters=False,
        prediction_loss_only=True
    )
    
    # 创建训练器
    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        logger.info("✅ 训练器创建成功")
        
    except Exception as e:
        logger.error(f"❌ 训练器创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 开始训练
    logger.info("🚀 开始训练...")
    
    try:
        trainer.train()
        
        # 保存模型
        trainer.save_model()
        logger.info(f"✅ 训练完成！模型已保存到: {config['training']['output_dir']}")
        
        # 创建最新模型的软链接
        try:
            latest_link = os.path.join(project_root, "models", "qwen3-8b-lora-highschool-latest")
            if os.path.exists(latest_link) or os.path.islink(latest_link):
                os.remove(latest_link)
            
            # 在Windows上创建目录链接
            if os.name == 'nt':  # Windows
                import subprocess
                subprocess.run(['mklink', '/D', latest_link, config['training']['output_dir']], 
                             shell=True, check=False)
            else:  # Linux/Mac
                os.symlink(config['training']['output_dir'], latest_link)
            
            logger.info(f"🔗 已创建最新模型链接: {latest_link}")
        except Exception as e:
            logger.warning(f"⚠️ 创建软链接失败: {e}")
        
        # 显示显存使用情况
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1024**3
            logger.info(f"📊 最大显存使用: {memory_used:.1f} GB")
            
        logger.info("="*50)
        logger.info("🎉 高中各学科数据集训练完成摘要:")
        logger.info(f"📁 模型保存位置: {config['training']['output_dir']}")
        logger.info(f"🔗 最新链接: {os.path.join(project_root, 'models', 'qwen3-8b-lora-highschool-latest')}")
        logger.info(f"⏱️ 训练时间: {current_date}")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # 清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("🧹 GPU缓存已清理")

if __name__ == "__main__":
    main() 