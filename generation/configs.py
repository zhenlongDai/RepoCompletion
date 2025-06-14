import argparse
import yaml
from dataclasses import dataclass, fields
from typing import Any
    
@dataclass
class InferenceConfig:
    model_path: str
    eval_dataset_name: str
    dp_size: int
    tp_size: int
    dtype: str
    dp_master_ip: str
    dp_master_port: int
    data_path_dir: str
    debug_mode: bool
    language: str
    max_input_tokens: int
    save_file_path: str
    system_prompt: str
    temperature: int
    top_p: int
    gpu_memory_utilization: float
    use_lora: bool = False
    lora_path: str = ""
    without_context: bool = False
    eval_mode: str = "test"
    prompt_mode: str = "split"

def load_config(config_path: str) -> InferenceConfig:
    """从 YAML 文件加载配置并返回 InferenceConfig 实例"""
    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file)
    return InferenceConfig(**config_data)