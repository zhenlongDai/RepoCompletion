import argparse
import yaml
from dataclasses import dataclass, fields
from typing import Any


import re

def extract_content(s):
    # 去除外层<answer>标签及周围空白
    stripped = re.sub(r'^\s*<answer>\s*|[\r\n]+\s*</answer>\s*$', '', s, flags=re.DOTALL)
    
    # 提取所有'''包裹的内容
    matches = re.findall(r"'''((?:.|\n)*?)'''", stripped, re.DOTALL)
    
    if matches:
        # 取最后一个'''块并去除前后空白
        return matches[-1].strip()
    else:
        # 没有'''则取标签内容并去除前后空白
        return re.sub(r'^\s+|\s+$', '', stripped, flags=re.DOTALL)
    
@dataclass
class InferenceConfig:
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 10

def load_config(config_path: str) -> InferenceConfig:
    """从 YAML 文件加载配置并返回 InferenceConfig 实例"""
    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file)
    return InferenceConfig(**config_data)

def parse_args() -> InferenceConfig:
    """解析命令行参数并返回配置实例，命令行参数会覆盖配置文件中的值"""
    # 创建 ArgumentParser
    parser = argparse.ArgumentParser(description="训练配置")

    # 添加 --config 参数来指定配置文件路径
    parser.add_argument('--config', type=str, help="YAML 配置文件路径", default='config.yaml')

    # 解析 --config 参数
    args = parser.parse_args()

    # 先从 YAML 文件加载配置
    config = load_config(args.config)

    # 创建 ArgumentParser 来添加其它参数
    for field in fields(config):
        parser.add_argument(f'--{field.name}', type=type(getattr(config, field.name)), default=getattr(config, field.name))

    # 重新解析命令行参数
    args = parser.parse_args()

    # 使用命令行参数覆盖配置文件中的默认值
    for field in fields(config):
        if getattr(args, field.name) is not None:
            setattr(config, field.name, getattr(args, field.name))

    return config

def train(config: InferenceConfig):
    print(f"开始训练：学习率 {config.learning_rate}, 批次大小 {config.batch_size}, 训练轮数 {config.epochs}")

if __name__ == "__main__":
    config = parse_args()  # 解析命令行参数并加载配置
    train(config)  # 使用最终的配置启动训练
