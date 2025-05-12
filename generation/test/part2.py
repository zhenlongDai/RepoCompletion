import argparse
import sys
from dataclasses import fields
import gc

def get_part_eval_dataset(data_list, local_dp_rank, dp_size):
    # 计算每个进程处理的数据量
    data_per_rank = len(data_list) // dp_size
    start = local_dp_rank * data_per_rank
    end = start + data_per_rank
    
    # 最后一个进程处理剩余的数据
    if local_dp_rank == dp_size - 1:
        end = len(data_list)
        
    # 获取当前进程需要处理的数据
    part_data_list = data_list[start:end]
    
    # 打印出当前进程处理的任务量
    print(f"DP rank {local_dp_rank} needs to process {len(part_data_list)} prompts")
    
    # 构造处理的 prompt 列表
    part_prompt_list = [{"prompt": data['prompt']} for data in part_data_list]
    
    # 显式删除部分数据以释放内存
    del part_data_list  # 删除分配给当前进程的数据
    
    # 运行垃圾回收，帮助释放内存
    gc.collect()

    return part_prompt_list

def parse_args() -> InferenceConfig:
    """解析命令行参数并返回配置实例，命令行参数会覆盖配置文件中的值"""
    # 创建 ArgumentParser
    parser = argparse.ArgumentParser(description="训练配置")
    
    # 先解析 --config 参数，确保配置文件路径正确
    # 手动只传递 --config 参数给 argparse
    parser.add_argument('--config', type=str, help="YAML 配置文件路径", default='config.yaml')
    
    # 通过 sys.argv 只传递 --config 参数来解析
    args = parser.parse_args(['--config', sys.argv[1]] if len(sys.argv) > 1 else ['--config', 'config.yaml'])

    # 加载配置文件
    config = load_config(args.config)

    # 动态添加其它参数
    for field in fields(config):
        parser.add_argument(f'--{field.name}', type=type(getattr(config, field.name)), default=getattr(config, field.name))

    # 重新解析命令行参数，包括 --model_path 等
    args = parser.parse_args()

    # 使用命令行参数覆盖配置文件中的默认值
    for field in fields(config):
        if getattr(args, field.name) is not None:
            setattr(config, field.name, getattr(args, field.name))

    return config
