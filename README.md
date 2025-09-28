# 🎓 Open-LoRA：高中教学AI助手微调项目

> 基于Qwen3-8B的高中各学科智能教学助手，使用QLoRA技术高效微调，支持数学、物理、化学、生物、语文、英语、历史、地理、政治九大学科的专业问答。

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-4.40+-yellow.svg)](https://huggingface.co/transformers)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

## 📋 目录

- [✨ 项目特色](#-项目特色)
- [🚀 快速开始](#-快速开始)
- [📁 项目结构](#-项目结构)
- [⚙️ 环境配置](#️-环境配置)
- [🎯 使用指南](#-使用指南)
  - [模型训练](#模型训练)
  - [模型推理](#模型推理)
  - [数据处理](#数据处理)
- [🔧 高级配置](#-高级配置)
- [📊 监控与调试](#-监控与调试)
- [❓ 常见问题](#-常见问题)
- [🤝 贡献指南](#-贡献指南)

## ✨ 项目特色

### 🧠 智能化教学
- **多学科专业支持**：涵盖高中九大学科的专业数据集和教学策略
- **自动学科识别**：根据问题内容智能识别学科，切换对应专家角色
- **个性化回答**：针对不同学科特点提供专业、准确、易懂的解答

### 🔬 技术优势
- **QLoRA微调**：使用参数高效的QLoRA技术，显存占用低，训练效率高
- **4bit量化**：支持4bit量化推理，在普通GPU上也能流畅运行
- **流式生成**：实时流式输出，提供更好的交互体验
- **跨平台支持**：同时支持Windows和Linux系统

### 📚 丰富数据集
- **高中各学科数据集**：包含9个学科的专业教学数据
- **标准化格式**：支持多种数据格式的自动转换和处理
- **质量保证**：经过精心筛选和格式化的高质量训练数据

## 🚀 快速开始

### 1. 克隆项目
```bash
git clone https://github.com/your-username/Open-LoRA.git
cd Open-LoRA
```

### 2. 创建虚拟环境
```bash
# 创建虚拟环境（必须在项目根目录下）
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 3. 安装依赖
```bash
# CUDA环境（推荐）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r config/requirements.txt

# CPU环境
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r config/requirements.txt
```

### 4. 验证安装
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### 5. 准备模型
将Qwen3-8B模型文件放置在 `models/qwen3_8b/` 目录下，确保包含以下文件：
- `config.json`
- `tokenizer.json`
- `model.safetensors` 或 `pytorch_model.bin`
- 其他必要的模型文件

### 6. 开始使用
```bash
# 启动交互式AI助手（Windows）
scripts\inference\start_chat.bat

# 启动交互式AI助手（Linux）
scripts/inference/start_chat.sh
```

## 📁 项目结构

```
Open-LoRA/
├── 📋 README.md                    # 项目说明文档
├── 📦 config/                      # 配置文件
│   ├── requirements.txt           # Python依赖
│   └── train_config.py           # 训练配置
├── 📊 datasets/                    # 数据集目录
│   ├── 高中各学科jsonl数据集/      # 九大学科训练数据
│   ├── COIG通用数据集/            # 通用对话数据
│   ├── K12数据集/                 # K12教育数据
│   └── 小学数学数据集/            # 数学专项数据
├── 🤖 models/                      # 模型文件
│   ├── qwen3_8b/                  # 基础模型（需要手动下载）
│   └── qwen3-8b-lora-*            # 微调输出模型
├── 🛠️ scripts/                     # 脚本工具
│   ├── training/                  # 训练脚本
│   │   ├── simple_train_ultimate.py    # 主训练脚本
│   │   └── run_qwen3_8b_highschool.sh  # Linux训练启动脚本
│   ├── inference/                 # 推理脚本
│   │   ├── interactive_chat.py    # 交互式对话
│   │   ├── start_chat.bat        # Windows启动脚本
│   │   └── start_chat.sh         # Linux启动脚本
│   ├── data_processing/           # 数据处理工具
│   │   ├── process_chat_to_jsonl.py     # 对话数据转换
│   │   ├── process_jddc_data_v*.py      # JDDC数据处理
│   │   └── process_ecommerce_data.py    # 电商数据处理
│   └── utils/                     # 实用工具
│       ├── start_tensorboard.bat # TensorBoard启动（Windows）
│       ├── start_tensorboard.sh  # TensorBoard启动（Linux）
│       └── install_dependencies.* # 依赖安装脚本
├── 📝 logs/                        # 训练日志和TensorBoard
├── 📂 cache/                       # 缓存目录
└── 💾 data/                        # 临时数据目录
```

## ⚙️ 环境配置

### 系统要求
- **操作系统**：Windows 10+ 或 Linux
- **Python**：3.8+
- **显存**：建议8GB+（支持4bit量化可降低至4GB）
- **内存**：建议16GB+

### GPU支持
- **NVIDIA GPU**：推荐RTX 3060以上
- **CUDA**：12.1+（可根据具体GPU调整）
- **显存优化**：支持4bit量化、梯度检查点等技术

### CPU支持
- 完全支持CPU推理和训练
- 推荐使用多核CPU以获得更好性能

## 🎯 使用指南

### 模型训练

#### 方法一：使用启动脚本（推荐）
```bash
# Linux
bash scripts/training/run_qwen3_8b_highschool.sh

# Windows（可以直接双击运行训练脚本）
python scripts/training/simple_train_ultimate.py
```

#### 方法二：直接运行训练脚本
```bash
python scripts/training/simple_train_ultimate.py
```

#### 训练配置说明
训练参数可在 `config/train_config.py` 中修改：

```python
# 主要配置项
TRAINING_CONFIG = {
    "per_device_train_batch_size": 1,      # 批次大小
    "gradient_accumulation_steps": 4,       # 梯度累积
    "num_train_epochs": 3,                  # 训练轮数
    "max_steps": 1000,                      # 最大步数
    "learning_rate": 1e-5,                  # 学习率
    "save_steps": 200,                      # 保存间隔
}

# LoRA配置
LORA_CONFIG = {
    "r": 32,                    # LoRA rank
    "alpha": 64,                # LoRA alpha
    "dropout": 0.05,            # Dropout率
    "target_modules": "all-linear",  # 目标模块
}
```

### 模型推理

#### 启动交互式助手
```bash
# Windows
scripts\inference\start_chat.bat

# Linux  
bash scripts/inference/start_chat.sh

# 或直接运行
python scripts/inference/interactive_chat.py
```

#### 功能特色
1. **智能学科识别**：自动识别问题所属学科
2. **专家角色切换**：根据学科切换对应专家身份
3. **流式输出**：实时显示生成内容
4. **模型切换**：支持在不同模型间切换
5. **交互命令**：
   - `quit`/`exit`：退出程序
   - `clear`：清屏
   - `switch`：切换模型

#### 使用示例
```
🤔 请输入您的问题: 求解二次方程 x²-5x+6=0

🤖 AI助手: 这是一个二次方程求解问题，我来为您详细解答。

给定方程：x² - 5x + 6 = 0

方法一：因式分解法
我们需要找到两个数，它们的乘积等于常数项6，和等于一次项系数-5。
这两个数是-2和-3，因为：(-2) × (-3) = 6，(-2) + (-3) = -5

所以原方程可以分解为：
(x - 2)(x - 3) = 0

因此：x = 2 或 x = 3

方法二：求根公式
对于一般形式ax² + bx + c = 0，求根公式为：
x = [-b ± √(b² - 4ac)] / (2a)

在本题中，a = 1, b = -5, c = 6
判别式：Δ = b² - 4ac = (-5)² - 4×1×6 = 25 - 24 = 1 > 0
所以方程有两个不等实根：
x = [5 ± √1] / (2×1) = (5 ± 1) / 2
x₁ = (5 + 1) / 2 = 3
x₂ = (5 - 1) / 2 = 2

答案：x = 2 或 x = 3
```

### 数据处理

#### 支持的数据格式
项目支持多种数据格式的处理和转换：

1. **JSONL格式**（推荐）
```json
{"messages": [
  {"role": "system", "content": "你是一位数学专家..."},
  {"role": "user", "content": "请解这道题..."},
  {"role": "assistant", "content": "解题步骤如下..."}
]}
```

2. **CSV格式转换**
```bash
python scripts/data_processing/process_chat_to_jsonl.py
```

3. **自定义数据处理**
```bash
python scripts/data_processing/process_jddc_data_v3.py  # JDDC数据集处理
python scripts/data_processing/process_ecommerce_data.py  # 电商数据处理
```

#### 数据集要求
- **格式**：JSONL格式，每行一个JSON对象
- **字段**：必须包含`messages`字段，内含`role`和`content`
- **角色**：支持`system`、`user`、`assistant`三种角色
- **交替**：用户和助手消息需交替出现

## 🔧 高级配置

### 量化配置
```python
# 在 config/train_config.py 中
QUANTIZATION_CONFIG = {
    "load_in_4bit": True,                    # 启用4bit量化
    "bnb_4bit_use_double_quant": True,       # 双重量化
    "bnb_4bit_quant_type": "nf4",           # 量化类型
    "compute_dtype": "float16"               # 计算精度
}
```

### 显存优化
```python
# 优化配置
OPTIMIZATION_CONFIG = {
    "use_gradient_checkpointing": True,      # 梯度检查点
    "dataloader_num_workers": 0,             # 数据加载器进程数
}
```

### 测试模式
```python
# 快速验证环境
config = get_config(mode="test")  # 使用少量数据快速测试
```

## 📊 监控与调试

### TensorBoard监控
```bash
# Windows
scripts\utils\start_tensorboard.bat

# Linux
bash scripts/utils/start_tensorboard.sh

# 或直接启动
tensorboard --logdir=logs --port=6006
```

访问 http://localhost:6006 查看训练进度：
- 损失曲线
- 学习率变化
- 梯度统计
- 其他训练指标

### 日志文件
- **训练日志**：控制台输出，包含详细的训练过程信息
- **TensorBoard日志**：`logs/` 目录下的事件文件
- **模型检查点**：`models/qwen3-8b-lora-*` 目录下的模型文件

### 调试技巧
1. **内存监控**：程序会自动显示显存使用情况
2. **错误排查**：检查CUDA环境、模型路径、数据格式
3. **性能调优**：调整批次大小、学习率等参数

## ❓ 常见问题

### Q1: CUDA内存不足怎么办？
**A**: 尝试以下解决方案：
- 减少`per_device_train_batch_size`
- 启用`gradient_checkpointing`
- 使用4bit量化
- 减少`max_length`

### Q2: 模型加载失败？
**A**: 检查以下项目：
- 模型文件是否完整下载
- 路径配置是否正确
- Python依赖是否安装完整
- CUDA环境是否正确配置

### Q3: 训练收敛慢或不收敛？
**A**: 调整训练参数：
- 降低学习率
- 增加warmup步数
- 检查数据质量
- 增加训练数据量

### Q4: 如何添加新的学科数据？
**A**: 步骤如下：
1. 准备JSONL格式数据
2. 放入`datasets/`目录
3. 修改`config/train_config.py`中的数据路径
4. 更新专家提示词（如需要）

### Q5: 推理速度慢怎么优化？
**A**: 优化建议：
- 使用4bit量化
- 启用GPU加速
- 减少max_tokens
- 使用更快的硬件

## 🤝 贡献指南

欢迎贡献代码、数据或文档！

### 贡献方式
1. **Fork本仓库**
2. **创建特性分支**：`git checkout -b feature/amazing-feature`
3. **提交更改**：`git commit -m 'Add amazing feature'`
4. **推送分支**：`git push origin feature/amazing-feature`
5. **开启Pull Request**

### 贡献内容
- 🐛 Bug修复
- ✨ 新功能开发
- 📚 数据集贡献
- 📝 文档改进
- 🎨 界面优化

### 开发规范
- 遵循Python PEP8代码规范
- 添加必要的注释和文档
- 编写单元测试
- 确保向后兼容性

---

## 📄 许可证

本项目采用 Apache 2.0 许可证。详情请参阅 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- [Qwen团队](https://github.com/QwenLM/Qwen) 提供优秀的基础模型
- [Hugging Face](https://huggingface.co/) 提供transformers库和模型托管
- [Microsoft LoRA](https://github.com/microsoft/LoRA) 提供参数高效微调技术
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) 提供量化技术支持

## 📞 联系方式

- **项目主页**：[https://github.com/your-username/Open-LoRA](https://github.com/your-username/Open-LoRA)
- **问题反馈**：[GitHub Issues](https://github.com/your-username/Open-LoRA/issues)
- **邮箱**：your-email@example.com

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给我们一个Star！⭐**

Made with ❤️ by Open-LoRA Team

</div>
