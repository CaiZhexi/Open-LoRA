"""
Qwen3-8B QLoRA 微调配置文件 - 高中各学科数据集版本
使用相对路径，兼容跨平台部署
"""

import os

def get_project_root():
    """获取项目根目录"""
    config_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(config_dir, '..')
    return os.path.abspath(project_root)

# 模型配置
MODEL_CONFIG = {
    "model_name": "qwen3_8b",  # 本地模型目录名称
    "trust_remote_code": True,
    "max_length": 2048
}

# 数据配置  
DATA_CONFIG = {
    "train_file_name": "高中各学科jsonl数据集",  # 高中各学科数据集目录
    "max_train_samples": None,  # None表示使用全部数据，可设置为100等小数值用于测试
}

# LoRA配置
LORA_CONFIG = {
    "r": 32,                    # 增加LoRA rank，提高模型容量
    "alpha": 64,                # LoRA alpha，一般设置为r的2倍
    "dropout": 0.05,            # 降低Dropout率，减少过度正则化
    "target_modules": "all-linear",  # 目标模块
    "bias": "none"              # bias设置
}

# 量化配置
QUANTIZATION_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type": "nf4",
    "compute_dtype": "float16"
}

# 训练配置
TRAINING_CONFIG = {
    "output_dir_name": "qwen3-8b-lora-highschool-output",  # 高中各学科数据集输出目录名
    "per_device_train_batch_size": 1,      # 恢复为1，避免显存溢出
    "gradient_accumulation_steps": 4,       # 恢复为4，保持有效batch size=4
    "num_train_epochs": 3,                  # 增加训练轮数，让模型充分学习
    "max_steps": 1000,                      # 增加最大步数，确保充分收敛
    "learning_rate": 1e-5,                  # 降低学习率，减少震荡
    "warmup_steps": 100,                    # 增加预热步数，更稳定的开始
    "logging_steps": 10,                    # 减少日志频率，避免过多输出
    "save_steps": 200,                      # 调整保存间隔
    "save_total_limit": 3,                  # 最多保存模型数量
    "fp16": True,                           # 使用半精度，节省显存
    "dataloader_pin_memory": False,         # 关闭pin_memory，避免内存问题
    "remove_unused_columns": False,
    "optim": "paged_adamw_32bit",          # 优化器，适合量化训练
    "group_by_length": True,               # 按长度分组，提高训练效率
    "report_to": [],                       # 不使用wandb等记录工具
}

# GPU和内存优化
OPTIMIZATION_CONFIG = {
    "use_gradient_checkpointing": True,     # 梯度检查点，节省显存
    "dataloader_num_workers": 0,           # 数据加载器worker数量，Linux可以设置为2-4
}

# 测试配置（快速验证环境）
TEST_CONFIG = {
    "max_train_samples": 10,               # 仅使用10个样本
    "max_steps": 5,                        # 仅训练5步
    "save_steps": 2,                       # 每2步保存一次
}

def get_config(mode="normal", project_root=None):
    """
    获取配置，自动计算完整路径
    mode: "normal" | "test"
    project_root: 项目根目录，如果不提供则自动计算
    """
    if project_root is None:
        project_root = get_project_root()
    
    # 构建完整路径
    config = {
        "model": {
            **MODEL_CONFIG,
            "model_path": os.path.join(project_root, "models", MODEL_CONFIG["model_name"])
        },
        "data": {
            **DATA_CONFIG,
            "train_file": os.path.join(project_root, "datasets", DATA_CONFIG["train_file_name"])
        },
        "lora": LORA_CONFIG,
        "quantization": QUANTIZATION_CONFIG,
        "training": {
            **TRAINING_CONFIG,
            "output_dir": os.path.join(project_root, "models", TRAINING_CONFIG["output_dir_name"])
        },
        "optimization": OPTIMIZATION_CONFIG
    }
    
    if mode == "test":
        # 测试模式，使用小数据量快速验证
        config["data"].update(TEST_CONFIG)
        config["training"]["max_steps"] = TEST_CONFIG["max_steps"]
        config["training"]["save_steps"] = TEST_CONFIG["save_steps"]
    
    return config 