#!/usr/bin/env python3
"""
交互式高中教学AI助手
自动检测models目录下的所有模型，提供选择界面和交互式问答
"""

import os
import sys
import glob
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', '..')

class InteractiveChat:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = ""
        self.base_model_path = os.path.join(project_root, "models", "qwen3_8b")
        
    def scan_models(self):
        """扫描models目录下的所有可用模型"""
        models_dir = os.path.join(project_root, "models")
        available_models = []
        seen_paths = set()  # 用于去重
        
        # 添加基础模型
        if os.path.exists(self.base_model_path):
            available_models.append({
                "name": "Qwen3-8B (原始模型)",
                "path": self.base_model_path,
                "type": "base"
            })
            seen_paths.add(self.base_model_path)
        
        # 扫描微调模型 - 只扫描highschool相关的
        pattern_path = os.path.join(models_dir, "qwen3-8b-lora-highschool-*")
        for model_dir in sorted(glob.glob(pattern_path)):
            if os.path.isdir(model_dir) and model_dir not in seen_paths:
                # 检查是否包含adapter文件
                adapter_file = os.path.join(model_dir, "adapter_model.safetensors")
                if os.path.exists(adapter_file):
                    model_name = os.path.basename(model_dir)
                    if "latest" in model_name:
                        display_name = f"{model_name} (最新微调模型)"
                    else:
                        display_name = f"{model_name} (微调模型)"
                    
                    available_models.append({
                        "name": display_name,
                        "path": model_dir,
                        "type": "lora"
                    })
                    seen_paths.add(model_dir)
        
        return available_models
    
    def display_models(self, models):
        """显示可用模型列表"""
        print("\n" + "="*60)
        print("🤖 高中教学AI助手 - 模型选择")
        print("="*60)
        print("\n可用模型:")
        
        for i, model in enumerate(models, 1):
            print(f"{i}. {model['name']}")
        
        print(f"\n0. 退出程序")
        print("="*60)
    
    def load_model(self, model_info):
        """加载指定模型"""
        print(f"\n🚀 正在加载模型: {model_info['name']}")
        print("请稍等...")
        
        try:
            # 清理之前的模型
            if self.model is not None:
                del self.model
                torch.cuda.empty_cache()
            
            # 加载分词器
            print("📝 加载分词器...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_path,
                trust_remote_code=True,
                use_fast=False
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 创建临时offload目录
            import tempfile
            offload_dir = tempfile.mkdtemp(prefix="model_offload_")
            
            print("🧠 加载基础模型...")
            
            # 尝试量化加载（推荐）
            try:
                from transformers import BitsAndBytesConfig
                
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16
                )
                
                print("🔧 使用4bit量化加载（推荐模式）...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_path,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                print("✅ 量化模型加载成功")
                
            except Exception as e1:
                print("⚠️ 量化加载失败，尝试标准GPU模式...")
                try:
                    # 尝试完整GPU加载
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.base_model_path,
                        torch_dtype=torch.float16,
                        device_map={"": 0},  # 强制所有层都在GPU 0上
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
                    print("✅ GPU完整加载成功")
                    
                except Exception as e2:
                    print("⚠️ GPU完整加载失败，尝试CPU+GPU混合模式...")
                    try:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.base_model_path,
                            torch_dtype=torch.float16,
                            device_map="auto",
                            trust_remote_code=True,
                            low_cpu_mem_usage=True,
                            offload_folder=offload_dir,
                            max_memory={0: "14GB", "cpu": "8GB"}
                        )
                    except Exception as e3:
                        print("⚠️ 最后尝试纯CPU模式...")
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.base_model_path,
                            torch_dtype=torch.float32,
                            device_map="cpu",
                            trust_remote_code=True,
                            low_cpu_mem_usage=True
                        )
            
            # 如果是LoRA模型，加载适配器
            if model_info['type'] == 'lora':
                print("🔧 加载LoRA适配器...")
                try:
                    # 检查模型是否是量化模型
                    is_quantized = hasattr(self.model, 'quantization_config') and self.model.quantization_config is not None
                    
                    if is_quantized:
                        print("📦 检测到量化模型，使用兼容模式加载LoRA...")
                        # 量化模型需要特殊处理
                        from peft import prepare_model_for_kbit_training
                        self.model = prepare_model_for_kbit_training(self.model)
                        self.model = PeftModel.from_pretrained(self.model, model_info['path'])
                        print("✅ LoRA适配器加载成功（量化兼容模式）")
                    else:
                        # 非量化模型
                        self.model = PeftModel.from_pretrained(self.model, model_info['path'])
                        print("🔄 合并LoRA权重...")
                        self.model = self.model.merge_and_unload()  # 合并权重
                        print("✅ LoRA权重合并完成")
                        
                except Exception as e:
                    print(f"⚠️ LoRA加载失败，继续使用基础模型: {e}")
            
            self.model_name = model_info['name']
            print(f"✅ 模型加载成功: {self.model_name}")
            
            # 显示模型信息
            try:
                if torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated() / 1024**3
                    if memory_used > 0:
                        print(f"📊 显存使用: {memory_used:.1f} GB")
                    else:
                        print("📊 运行模式: CPU")
                        
                    # 检查设备映射（简化显示）
                    if hasattr(self.model, 'hf_device_map'):
                        device_map = self.model.hf_device_map
                        gpu_layers = sum(1 for v in device_map.values() if str(v).startswith('cuda') or str(v) == '0')
                        cpu_layers = sum(1 for v in device_map.values() if str(v) == 'cpu')
                        print(f"🎯 设备分布: GPU层数={gpu_layers}, CPU层数={cpu_layers}")
                else:
                    print("📊 运行模式: CPU")
            except Exception as e:
                print("📊 运行模式: 混合模式")
            
            # 清理临时目录
            import shutil
            try:
                shutil.rmtree(offload_dir)
            except:
                pass
            
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def get_subject_prompt(self, question):
        """根据问题内容智能选择学科专家身份"""
        # 关键词匹配
        subject_keywords = {
            "数学": ["方程", "函数", "几何", "三角", "代数", "概率", "统计", "微分", "积分", "数列", "计算"],
            "物理": ["力", "电", "磁", "光", "热", "波", "能量", "功", "牛顿", "欧姆", "焦耳"],
            "化学": ["反应", "元素", "分子", "原子", "化合", "酸", "碱", "盐", "氧化", "还原"],
            "生物": ["细胞", "遗传", "进化", "生态", "植物", "动物", "DNA", "蛋白质", "酶"],
            "语文": ["古诗", "文言文", "作文", "诗歌", "散文", "小说", "成语", "修辞", "阅读"],
            "英语": ["grammar", "translate", "english", "vocabulary", "sentence", "语法", "翻译"],
            "历史": ["朝代", "战争", "皇帝", "革命", "文明", "古代", "近代", "现代"],
            "地理": ["气候", "地形", "河流", "山脉", "经纬度", "季风", "板块", "城市"],
            "政治": ["法律", "制度", "权利", "义务", "宪法", "民主", "政府", "公民"]
        }
        
        # 专家身份模板
        expert_prompts = {
            "数学": "你是位精通代数、几何、三角函数、概率统计、数列、导数、解析几何、数学建模及高考数学命题规律等方面的数学教育专家，特别擅长高中数学概念深度解析、公式定理灵活运用、多解法思路拓展、典型例题分类精讲、高考真题考点拆解与应试技巧提炼，能精准指导学生突破重难点、构建数学思维体系。",
            "物理": "你是位精通高考物理的专家，擅长力学、电磁学、热学、光学和近代物理等核心板块，能高效解析物理模型、受力分析、电路设计、实验题及计算题，帮助学生快速掌握解题思路与应试技巧。",
            "化学": "你是位精通高中化学的教育专家，擅长无机化学、有机化学、物理化学等各个方面，能够深入浅出地解析化学概念、反应机理、实验操作等，帮助学生建立扎实的化学知识基础。",
            "生物": "你是位精通高中生物的教育专家，擅长细胞生物学、遗传学、生态学等各个领域，能够生动形象地讲解生命现象、生物过程，帮助学生理解生命科学的奥秘。",
            "语文": "你是位精通高考语文训练、论述类文本阅读分析、文言文解读、现代文理解、选择题解析、作文辅导等方面的语文辅导专家，特别擅长文本深度剖析和高考考点把握。",
            "英语": "你是位精通高中英语的教育专家，擅长语法解析、词汇拓展、阅读理解、写作指导等，能够帮助学生提高英语综合能力。",
            "历史": "你是位精通高中历史的教育专家，对中外历史发展脉络、重大历史事件、历史人物有深入了解，能够帮助学生构建完整的历史知识体系。",
            "地理": "你是位精通高中地理的教育专家，擅长自然地理、人文地理、区域地理等各个方面，能够帮助学生理解地理现象和地理规律。",
            "政治": "你是位精通高中政治的教育专家，熟悉马克思主义哲学、政治经济学、科学社会主义等理论，能够帮助学生理解政治制度和社会现象。"
        }
        
        # 分析问题内容
        question_lower = question.lower()
        for subject, keywords in subject_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                return expert_prompts.get(subject, expert_prompts["数学"])
        
        # 默认返回通用专家身份
        return "你是一位资深的高中教育专家，擅长各个学科的教学，能够针对学生的问题提供专业、准确、易懂的解答。"
    
    def generate_response_stream(self, question):
        """流式生成回答"""
        if not self.model or not self.tokenizer:
            yield "❌ 请先选择并加载模型"
            return
        
        try:
            # 智能选择专家身份
            system_prompt = self.get_subject_prompt(question)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
            
            # 使用chat template格式化
            text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # 编码 - 智能设备处理
            inputs = self.tokenizer(text, return_tensors="pt")
            
            # 智能设备映射处理
            try:
                # 检查是否有设备映射
                if hasattr(self.model, 'hf_device_map') and self.model.hf_device_map:
                    # 对于有设备映射的模型，找到embeddings层的设备
                    device_map = self.model.hf_device_map
                    embed_device = None
                    
                    # 查找embed_tokens的设备
                    for key, device in device_map.items():
                        if 'embed' in key.lower():
                            embed_device = device
                            break
                    
                    # 如果没找到embed设备，使用第一个非CPU设备
                    if embed_device is None:
                        for device in device_map.values():
                            if str(device) != 'cpu':
                                embed_device = device
                                break
                        if embed_device is None:
                            embed_device = 'cpu'
                    
                    # 将输入移动到正确的设备
                    if str(embed_device) != 'cpu':
                        inputs = inputs.to(f"cuda:{embed_device}" if isinstance(embed_device, int) else embed_device)
                    else:
                        inputs = inputs.to('cpu')
                        
                else:
                    # 没有设备映射，使用传统方法
                    device = next(self.model.parameters()).device
                    inputs = inputs.to(device)
                    
            except Exception as e:
                yield f"⚠️ 设备处理警告: {e}"
                return
            
            # 流式生成回答
            from transformers import TextIteratorStreamer
            from threading import Thread
            
            # 创建流式器
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,  # 跳过输入部分
                skip_special_tokens=True
            )
            
            generation_kwargs = {
                **inputs,
                "max_new_tokens": 1024,  # 增加到1024，支持更长回答
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,
                "streamer": streamer
            }
            
            # 在单独线程中运行生成
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # 流式输出
            for new_text in streamer:
                if new_text:
                    yield new_text
            
            thread.join()  # 等待生成完成
            
        except Exception as e:
            yield f"❌ 生成回答时出错: {e}"
    
    def generate_response(self, question):
        """非流式生成回答（备用方法）"""
        if not self.model or not self.tokenizer:
            return "❌ 请先选择并加载模型"
        
        # 收集流式输出为完整响应
        response_parts = []
        try:
            for chunk in self.generate_response_stream(question):
                response_parts.append(chunk)
            return ''.join(response_parts).strip()
        except Exception as e:
            return f"❌ 生成回答时出错: {e}"
    
    def interactive_chat(self):
        """交互式聊天"""
        if not self.model:
            print("❌ 请先加载模型")
            return
        
        print("\n" + "="*60)
        print(f"🎓 高中教学AI助手 - {self.model_name}")
        print("="*60)
        print("💡 提示:")
        print("  - 输入您的问题，AI会自动识别学科并提供专业解答")
        print("  - 支持流式响应，实时显示生成内容")
        print("  - 生成过程中按 Ctrl+C 可中断回答")
        print("  - 输入 'quit' 或 'exit' 退出对话")
        print("  - 输入 'clear' 清屏")
        print("  - 输入 'switch' 切换模型")
        print("="*60)
        
        while True:
            try:
                # 获取用户输入
                question = input("\n🤔 请输入您的问题: ").strip()
                
                if not question:
                    continue
                
                # 处理特殊命令
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 再见！感谢使用高中教学AI助手！")
                    break
                elif question.lower() in ['clear', 'cls']:
                    os.system('cls' if os.name == 'nt' else 'clear')
                    continue
                elif question.lower() == 'switch':
                    return 'switch'
                
                # 流式生成回答
                print("\n🤖 AI助手: ", end="", flush=True)
                
                try:
                    response_parts = []
                    for chunk in self.generate_response_stream(question):
                        print(chunk, end="", flush=True)  # 实时输出
                        response_parts.append(chunk)
                    
                    # 确保换行
                    print()
                    print("\n" + "-"*60)
                    
                except KeyboardInterrupt:
                    print("\n⏸️ 生成被中断")
                except Exception as e:
                    print(f"\n❌ 生成错误: {e}")
                
            except KeyboardInterrupt:
                print("\n👋 再见！感谢使用高中教学AI助手！")
                break
            except Exception as e:
                print(f"❌ 发生错误: {e}")
    
    def run(self):
        """运行主程序"""
        print("🚀 正在启动高中教学AI助手...")
        
        # 检查CUDA
        if torch.cuda.is_available():
            print(f"🎯 检测到GPU: {torch.cuda.get_device_name()}")
            # 启用一些优化
            torch.backends.cudnn.benchmark = True
        else:
            print("💻 使用CPU运行")
        
        while True:
            # 扫描可用模型
            models = self.scan_models()
            
            if not models:
                print("❌ 未找到可用模型！")
                print("请确保以下路径存在模型文件:")
                print(f"  - 基础模型: {self.base_model_path}")
                print(f"  - 微调模型: {os.path.join(project_root, 'models')}/*lora*")
                return
            
            # 显示模型选择界面
            self.display_models(models)
            
            try:
                choice = input("\n🎯 请选择模型编号: ").strip()
                
                if choice == '0':
                    print("👋 再见！")
                    break
                
                try:
                    model_index = int(choice) - 1
                    if 0 <= model_index < len(models):
                        selected_model = models[model_index]
                        
                        # 加载模型
                        if self.load_model(selected_model):
                            # 开始交互式对话
                            result = self.interactive_chat()
                            
                            # 如果用户选择切换模型，继续循环
                            if result == 'switch':
                                print("\n🔄 正在返回模型选择界面...")
                                continue
                            else:
                                break
                        else:
                            input("\n按回车键返回模型选择界面...")
                    else:
                        print("❌ 无效的选择！")
                        input("按回车键继续...")
                        
                except ValueError:
                    print("❌ 请输入有效的数字！")
                    input("按回车键继续...")
                    
            except KeyboardInterrupt:
                print("\n👋 再见！")
                break

def main():
    """主函数"""
    try:
        chat = InteractiveChat()
        chat.run()
    except KeyboardInterrupt:
        print("\n👋 程序已退出")
    except Exception as e:
        print(f"❌ 程序运行出错: {e}")

if __name__ == "__main__":
    main()
