#!/usr/bin/env python3
"""
Model Evaluation Automation Pipeline
自动化模型评估流程：批量运行模型生成和评估，找出最佳模型并生成最终测试结果
"""

import os
import yaml
import glob
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import copy
import sys
import subprocess
import time
import json

from generation.InferencePipeline import InferencePipeline
from generation.configs import InferenceConfig
from evaluation.eval_metric import compute_metric, calculate_results
from utils.json_util import save_list_to_json, load_list_from_json


class ModelEvaluationPipeline:
    """模型评估自动化管道"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.dirs = self.setup_directories()
        
    def setup_directories(self):
        """创建必要的目录结构"""
        base_dir = Path(self.config['output_dir'])
        dirs = {
            'base': base_dir,
            'generations': os.path.join(base_dir,'generations'),
            'evaluations': os.path.join(base_dir, 'evaluations'), 
            'results': os.path.join(base_dir, 'results')
        }
        
        for dir_path in dirs.values():
            # 创建目录
            dir_path = Path(dir_path)
            if not dir_path.exists():
                print(f"Creating directory: {dir_path}")
                dir_path.mkdir(parents=True, exist_ok=True)
            else:
                print(f"Directory already exists: {dir_path}")
        return dirs
    
    def discover_models(self) -> List[str]:
        """发现模型文件夹下的所有模型权重"""
        models_dir = Path(self.config['models_dir'])
        if not models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {models_dir}")
            
        # 查找所有模型目录，和记录id
        model_paths = []
        for item in models_dir.iterdir():
            if item.is_dir():
                # 检查是否包含模型文件
                if (any(item.glob("*.bin")) or 
                    any(item.glob("*.safetensors")) or 
                    any(item.glob("pytorch_model.bin")) or
                    any(item.glob("model.safetensors")) or
                    (item / "config.json").exists()):
                    # 只添加包含模型文件的目录
                    print(f"Discovered model directory: {item}")
                    # 添加模型路径并且记录文件名的id
                    # 这里假设模型目录名格式为 checkpoint-<id>
                    path_name = str(item)
                    id = int(Path(item).name.split('-')[-1]) if '-' in Path(path_name).name else 0
                    model_paths.append({"id":id, "path":path_name})
        # 按照id排序    
        model_paths = sorted(model_paths, key=lambda x: x['id'])
      
        print(f"Discovered {len(model_paths)} models: {[Path(p['path']).name for p in model_paths]}")
        return model_paths
        
    # def create_inference_args(self, model_path: str, output_file: str, eval_mode: str) -> InferenceConfig:
    #     """创建推理参数配置"""
    #     # 加载基础配置
    #     base_config = load_config(self.config.get('generation_config', 'config.yaml'))
        
    #     # 更新配置参数
    #     base_config.model_path = model_path
    #     base_config.save_file_path = output_file
    #     base_config.eval_dataset_name = self.config.get('eval_dataset_name')
    #     base_config.data_path_dir = os.path.join(self.config.get(f'{eval_mode}_data_path_dir')) #错误已经没有这个参数了
    #     base_config.language = self.config.get('language')
    #     base_config.debug_mode = self.config.get('debug_mode')
    #     base_config.prompt_mode = self.config.get('prompt_mode')

    #     # 根据模式设置不同的数据集
    #     base_config.eval_dataset_name = self.config.get('eval_dataset_name')
            
    #     # 设置生成参数
    #     generation_params = self.config.get('generation_params', {})
    #     for key, value in generation_params.items():
    #         if hasattr(base_config, key):
    #             setattr(base_config, key, value)
                
    #     return base_config
        
    def run_generation(self, model_path: str, eval_mode: str) -> str:
        """运行单个模型的生成（使用subprocess确保资源隔离）"""
        generation_params = self.config.get('generation_params', {})
        use_lora = generation_params.get('use_lora')
        if use_lora:
            base_model_path = generation_params.get('model_path')
            lora_path = model_path
            model_name = Path(lora_path).name
            output_file = os.path.join(self.dirs['generations'], f"{lora_path}_generation.json")
        else:
            base_model_path = model_path
            model_name = Path(model_path).name
            output_file = os.path.join(self.dirs['generations'], f"{model_name}_generation.json")
        if eval_mode == "test":
            output_file = os.path.join(self.dirs['results'], f"{model_name}_generation.json")
            print(output_file)


        if Path(output_file).exists():
            print(f"Generation completed for {model_name}: {output_file}")
            return str(output_file)
        else:
            print(f"Running generation for model: {model_name}")
            
        try:
            # 构建命令
            # 从配置中读取其他参数
            cmd = [
                sys.executable, "-m", "generation.InferencePipeline",
                "--config", generation_params.get('inference_config'),
                "--model", base_model_path,
                "--data_path_dir", generation_params.get(f'{eval_mode}_data_path_dir'),
                "--save_file_path", str(output_file),
                "--language",  self.config.get('language'),
                "--eval_mode", eval_mode,
                "--prompt_mode", self.config.get('prompt_mode')
            ]

            # 添加模型相关参数
            cmd.extend([
                "--dtype", generation_params.get('dtype'),
                "--gpu_memory_utilization", str(generation_params.get('gpu_memory_utilization')),
                "--temperature", str(generation_params.get('temperature')),
                "--dp_size", str(generation_params.get('dp_size')),
                "--tp_size", str(generation_params.get('tp_size')),
            ])
            
            # 如果启用LoRA，添加LoRA参数
            if generation_params.get('use_lora', False):
                cmd.extend([
                    "--use_lora", str(generation_params.get('use_lora')),
                    "--lora_path", lora_path,
                ])

        
            # 构建完整的shell命令，包含环境变量
            shell_cmd = []
            
            # 添加CUDA设备设置
            cuda_devices = generation_params.get('cuda_visible_devices')
            if cuda_devices:
                shell_cmd.append(f"export CUDA_VISIBLE_DEVICES={cuda_devices}")
            shell_cmd.append(' '.join(cmd))
            full_cmd = ' && '.join(shell_cmd)

            print(f"Command: {full_cmd}")
            # 运行命令并等待完成
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                timeout=self.config.get('generation_timeout', 3600)
            )
            
            if result.returncode == 0 and Path(output_file).exists():
                print(f"Generation completed for {model_name}: {output_file}")
                return str(output_file)
            else:
                print(f"ERROR: Generation failed for {model_name}")
                if result.stderr:
                    print(f"stderr: {result.stderr}")
                if result.stdout:
                    print(f"stdout: {result.stdout}")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"ERROR: Generation timeout for {model_name}")
            return None
        except Exception as e:
            print(f"ERROR: Generation error for {model_name}: {e}")
            return None
        
    def create_eval_args(self, generation_file: str, eval_mode: str):
        """创建评估参数"""
        config = self.config
        class EvalArgs:
            def __init__(self):
                self.prediction_file = generation_file
                self.data_dir_path = str(config.get(f'generation_params_{eval_mode}_data_path_dir'))
                self.language = config.get('language')
                self.ts_lib = config.get('ts_lib')
                self.eval_dataset_name = config.get('eval_dataset_name')
                self.eval_mode = eval_mode
                self.prompt_mode = config.get('prompt_mode')
        eval_args = EvalArgs()
        print(f"Creating evaluation args for language:{eval_args.language} with generation file: {generation_file}")
        return eval_args
            
    def run_evaluation(self, generation_file: str, model_name: str, eval_mode: str) -> Optional[Dict]:
        """运行单个模型的评估"""
        if not generation_file or not Path(generation_file).exists():
            print(f"Generation file not found: {generation_file}")
            return None
            
        print(f"Running evaluation for model: {model_name}")
        
        try:
            # 创建评估参数
            eval_args = self.create_eval_args(generation_file, eval_mode)
            detailed_results, _ = compute_metric(eval_args) # 运行评估
            
            # 计算结果
            eval_results = calculate_results(detailed_results)
            eval_results['model_name'] = model_name
            if eval_mode == "test":
                eval_results['detailed_results'] = detailed_results
            
            # 保存评估结果
            eval_file = os.path.join(self.dirs['evaluations'], f"{model_name}_evaluation.json")
            with open(eval_file, 'w') as f:
                json.dump(eval_results, f, indent=2)
                
            print(f"Evaluation completed for {model_name}")
            return eval_results
            
        except Exception as e:
            print(f"Evaluation error for {model_name}: {e}")
            return None
            
    def select_best_model(self, all_results: List[Dict]) -> Tuple[str, Dict]:
        """根据评估结果选择最佳模型"""
        if not all_results:
            raise ValueError("No evaluation results available")
            
        # 定义评估指标权重
        weights = self.config.get('metric_weights', {
            'em_ratio': 0.0,
            'edit_sim': 1.0, 
            'id_f1': 0.0,
            'id_precision': 0.0,
            'id_recall': 0.0
        })
        
        best_model_info = None
        best_score = -1
        
        for result in all_results:
            # 计算加权分数
            score = 0
            for metric, weight in weights.items():
                if metric in result:
                    # 将百分比指标转换为0-1范围
                    value = result[metric]
                    if metric in ['em_ratio', 'id_em_ratio', 'id_precision', 'id_recall', 'id_f1']:
                        value = value / 100.0
                    score += value * weight
                    
            result['composite_score'] = score
            
            if score > best_score:
                best_score = score
                best_model_info = result
                
        print(f"Best model: {best_model_info['model_path']} with score: {best_score:.4f}")
        return best_model_info['model_path'], best_model_info
        
    def run_final_test(self, best_model_path: str) -> str:
        """使用最佳模型运行最终测试"""
        model_name = Path(best_model_path).name
        print(f"Running final test with best model: {model_name}")
        
        # 运行生成
        generation_file = self.run_generation(best_model_path, "test")  
        if not generation_file:
            print(f"Skipping model {model_name} due to generation failure")
              
        eval_result = self.run_evaluation(generation_file, model_name, eval_mode='test')
        return eval_result
    
    def save_summary_report(self, all_results: List[Dict], best_model: Dict, final_eval_result: str = None):
        """保存总结报告"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'mode': "test",
            'total_models_tested': len(all_results),
            'final_test_results': final_eval_result,
            'best_model': best_model,
            'all_results': all_results,
            'config': self.config
        }
        
        summary_file = os.path.join(self.dirs['results'],'evaluation_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"Summary report saved: {summary_file}")
        
        # 创建简化的markdown报告
        self.create_markdown_report(summary)
        
    def create_markdown_report(self, summary: Dict):
        """创建markdown格式的报告"""
        md_file = os.path.join(self.dirs['results'],'evaluation_report.md')
        
        with open(md_file, 'w') as f:
            f.write(f"# Model Evaluation Report (test Mode)\n\n")
            f.write(f"**Generated on:** {summary['timestamp']}\n\n")
            f.write(f"**Total Models Tested:** {summary['total_models_tested']}\n\n")
            
            f.write("## final_test_results of Best Model\n\n")
            best = summary['final_test_results']
            f.write(f"**Model:** {best['model_name']}\n\n")
            f.write("**Metrics:**\n")
            for key, value in best.items():
                if key != 'model_name' and key != 'detailed_results' and key != 'id2tag_map' and isinstance(value, (int, float)):
                    f.write(f"- {key}: {value:.4f}\n")
                    
            f.write("\n## All Results\n\n")
            f.write("| Model | EM Ratio | Edit Sim | ID F1 | Composite Score |\n")
            f.write("|-------|----------|----------|-------|----------------|\n")
            
            for result in summary['all_results']:
                f.write(f"| {result['model_name']} | "
                       f"{result.get('em_ratio', 0):.2f} | "
                       f"{result.get('edit_sim', 0):.2f} | "
                       f"{result.get('id_f1', 0):.2f} | "
                       f"{result.get('composite_score', 0):.4f} |\n")
                       
        print(f"Markdown report saved: {md_file}")
        
    def run_pipeline(self):
        """运行完整的评估管道"""
        print(f"Starting model evaluation pipeline in dev mode")
        
        # 1. 发现所有模型
        model_paths_list = self.discover_models()
        if not model_paths_list:
            raise ValueError("No models found in the specified models directory!")
        print(f"Found {len(model_paths_list)} models to evaluate.")
        for item in model_paths_list:
            print(f"id:{item['id']} Model path: {item['path']}")

        # 2. 对每个模型运行生成和评估
        all_results = []
        model_generation_files = {}
        
        for i, item in enumerate(model_paths_list):
            model_path = item['path']
            model_id = item['id']
            model_name = Path(model_path).name
            print(f"\n=== Processing model {model_id}: {model_name} ===")
            
            # 运行生成
            generation_file = self.run_generation(model_path, "dev")  
            if not generation_file:
                print(f"Skipping model {model_name} due to generation failure")
                continue
                
            model_generation_files[model_path] = generation_file
            
            # 运行评估（仅在dev模式下）
            eval_result = self.run_evaluation(generation_file, model_name, eval_mode='dev')
            # print(eval_result)
            # input()
            eval_result['model_path'] = model_path
            if eval_result:
                all_results.append(eval_result)
            else:
                raise ValueError("Evaluation failed for model {model_name}")

            
            # 可选：添加间隔时间让GPU冷却
            if i < len(model_paths_list) - 1:  # 不是最后一个模型
                print("Waiting for GPU cooldown...")
                time.sleep(1)
                    
        # 后续处理逻辑保持不变...
        
        if not all_results:
            raise ValueError("ERROR: No successful evaluations!")
            
        # 选择最佳模型并运行最终测试
        best_model_path, best_model_info = self.select_best_model(all_results)

        if best_model_info is not None:
            eval_result = self.run_final_test(best_model_path)
            self.save_summary_report(all_results, best_model_info, eval_result)
            print("Pipeline completed successfully!")
            print(f"Best model: {best_model_path}")
        else:
            print("ERROR: Best model path not found!")
     


def load_config(config_file: str) -> Dict:
    """加载yaml配置文件"""
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

def override_config_with_args(config: Dict, args: argparse.Namespace) -> Dict:
    """用命令行参数覆盖配置文件中的对应项"""
    args_dict = vars(args)
    for key, value in args_dict.items():
        if key != 'config' and value is not None:
            config[key] = value
            print("key:", key, "value:", value)
    
    # 处理嵌套的generation_params参数
    if 'generation_params' not in config:
        config['generation_params'] = {}
    # print(args)
    # input()
    # 检查所有以generation_params_开头的参数
    for attr_name in dir(args):
        if attr_name.startswith('generation_params_'):
            value = getattr(args, attr_name)
            if value is not None:
                param_name = attr_name.replace('generation_params_', '')
                config['generation_params'][param_name] = value

    return config

def main():
    parser = argparse.ArgumentParser(description="Model Evaluation Automation Pipeline")
    parser.add_argument("--config", type=str, default="./evaluation/config_file/evalConfig.yaml",
                       help="Configuration file path")
    parser.add_argument("--generation_params.inference_config",type=str, default="generation/config_file/ContextConfig.yaml",
                       dest='generation_params_inference_config', help="Configuration file path of infernce pipeline")
    parser.add_argument("--models_dir", type=str, help="Directory containing model weights")
    parser.add_argument("--output_dir", type=str, help="Output directory for results")
    parser.add_argument("--language", type=str, help="language")
    parser.add_argument("--ts_lib", type=str, help="Tree-sitter library path")
    parser.add_argument('--generation_params.base_model_path', type=str, dest='generation_params_base_model_path', default=None,
                        help="Base model path for LoRA, if using LoRA")
    parser.add_argument('--generation_params.cuda_visible_devices', type=str, dest='generation_params_cuda_visible_devices', default="0,1",
                        help="CUDA_VISIBLE_DEVICES for generation")
    parser.add_argument('--generation_params.dev_data_path_dir', type=str, dest='generation_params_dev_data_path_dir')
    parser.add_argument('--generation_params.test_data_path_dir', type=str, dest='generation_params_test_data_path_dir')
    parser.add_argument('--generation_params.use_lora', type=bool, dest='generation_params_use_lora', default=False, help="Whether to use LoRA for generation")
        
    
    args = parser.parse_args()
    # 加载配置
    config = load_config(args.config)
    # 命令行参数覆盖配置文件
    config = override_config_with_args(config, args)
    # 运行管道
    pipeline = ModelEvaluationPipeline(config)
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()