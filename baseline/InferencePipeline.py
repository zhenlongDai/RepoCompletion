import argparse
from dataclasses import dataclass, fields
from baseline.configs import InferenceConfig, load_config
from typing import Any
import sys
import multiprocessing
from multiprocessing import Process, Manager, Lock, Queue
from vllm.utils import get_open_port
import os
from time import sleep
from baseline.dataset_util import load_test_dataset_by_baselineName
from utils.json_util import save_list_to_json
from vllm import LLM, SamplingParams
import copy
import gc
from datasets import load_dataset
from vllm.lora.request import LoRARequest
from multiprocessing import Barrier

    
def parse_args() -> InferenceConfig:
    """解析命令行参数并返回配置实例，命令行参数会覆盖配置文件中的值"""
    # 创建 ArgumentParser
    parser = argparse.ArgumentParser(description="训练配置")
    parser.add_argument('--config', type=str, help="YAML 配置文件路径", default='config.yaml')
    args = parser.parse_args(['--config', sys.argv[2]] if len(sys.argv) > 1 else ['--config', 'config.yaml'])
    
    config = load_config(args.config)
    for field in fields(config):
        parser.add_argument(f'--{field.name}', type=type(getattr(config, field.name)), default=getattr(config, field.name))
    args = parser.parse_args() 

    for field in fields(config):
        if getattr(args, field.name) is not None:
            setattr(config, field.name, getattr(args, field.name))

    return config

def init_os_env(local_dp_rank, dp_size, dp_master_ip, dp_master_port):    
        os.environ["VLLM_DP_RANK"] = str(local_dp_rank)
        os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
        os.environ["VLLM_DP_SIZE"] = str(dp_size)
        os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
        os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

def get_part_eval_dataset(data_list, local_dp_rank, dp_size, lock):
    # 计算每个进程处理的数据量
    data_per_rank = len(data_list) // dp_size
    start = local_dp_rank * data_per_rank
    end = start + data_per_rank
    if local_dp_rank == dp_size - 1:  # 最后一个进程处理剩余的数据
        end = len(data_list)
    # 获取当前进程需要处理的数据
    # 将data_list中start到end的数据分配给当前进程，新建list,避免进程卡死
    new_data_list = []
    with lock:
        for data in data_list[start:end]:
            new_data_list.append(copy.deepcopy(data))
    
    print(f"DP rank {local_dp_rank} needs to process {len(new_data_list)} prompts")

    part_prompt_list = [{"prompt": data['prompt']} for data in new_data_list]
    return new_data_list, part_prompt_list

class InferencePipeline:
    
    def __init__(self, args):
        self.model_name = args.model_name
        self.model_path = args.model_path
        self.dp_size = args.dp_size
        self.tp_size = args.tp_size
        self.dp_master_ip = "127.0.0.1"
        self.dp_master_port = get_open_port() if args.dp_master_port==0 else args.dp_master_port
        self.data_path_dir = args.data_path_dir 
        self.debug_mode = args.debug_mode
        self.language = args.language
        self.max_input_tokens = args.max_input_tokens
        self.max_output_tokens = args.max_output_tokens 
        self.save_file_path = args.save_file_path
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.eval_dataset_name = args.eval_dataset_name
        self.gpu_memory_utilization = args.gpu_memory_utilization
        self.dtype = args.dtype
        self.use_lora = args.use_lora
        self.lora_path = args.lora_path
        self.without_context = args.without_context
        self.eval_mode = args.eval_mode
        self.prompt_mode = args.prompt_mode
        print("self.dp_master_port", self.dp_master_port)
        data_list = load_test_dataset_by_baselineName(self.model_name, args.eval_dataset_name, self.data_path_dir, self.language, 
                                      self.model_path, self.max_input_tokens, self.debug_mode, self.without_context, 
                                    )
        
        manager = Manager()
        self.data_list = manager.list(data_list)
        #self.result_queue =  multiprocessing.Queue()
        self.result_queue = manager.Queue()  # 用于存储每个进程的结果
        self.lock = Lock()  # 创建锁
        self.barrier = Barrier(self.dp_size)

        print("data_list[0]:", self.data_list[0])
        print("------prompt----------\n" , self.data_list[0]['prompt'], "\n---------------------")
        #input()
    
    def generation(self, data_list, model_path, dp_size, tp_size, local_dp_rank, dp_master_ip, dp_master_port,
                temperature, top_p, gpu_memory_utilization, result_queue, lock, dtype, max_input_tokens, max_output_tokens,
                use_lora, lora_path = None, barrier = None, debug_mode = True):
        init_os_env(local_dp_rank, dp_size, dp_master_ip, dp_master_port)
        # 处理进程数据
        part_eval_dataset, part_prompt_list = get_part_eval_dataset(data_list, local_dp_rank, dp_size, lock)
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens = max_output_tokens)
        print(f"DP rank {local_dp_rank} start loading model......")
        llm = LLM(model=model_path,
                tensor_parallel_size=tp_size,
                gpu_memory_utilization=gpu_memory_utilization,
                dtype = dtype,
                enforce_eager=True,
                enable_lora=use_lora
                )
        
        print(f"rank {local_dp_rank} start generation......")
        if use_lora:
            print(f"DP rank {local_dp_rank} using LoRA with path: {lora_path}")
            outputs = llm.generate(part_prompt_list, 
                                   sampling_params,
                                   lora_request=LoRARequest(lora_name=f"lora_adapter_{local_dp_rank}",lora_int_id = local_dp_rank+1, lora_path=lora_path))
        else:
            outputs = llm.generate(part_prompt_list, sampling_params)
        # process the outputs.
        result = []
        for i, output in enumerate(outputs):
            prompt = output.prompt
            if prompt != part_eval_dataset[i]['prompt']:
               #抛出异常
               raise ValueError(f"DP rank {local_dp_rank}, Prompt: {prompt!r} not equal to output prompt: {part_eval_dataset[i]['prompt']!r}") 
            
            generated_text = output.outputs[0].text
            #"prompt": copy.deepcopy(part_eval_dataset[i]['prompt']),
            new_data = { "generated_text": copy.deepcopy(generated_text), "tag": copy.deepcopy(part_eval_dataset[i]['tag'])
                        , "id": copy.deepcopy(part_eval_dataset[i]['id'])}
            result.append(new_data)
        
            #if debug_mode:
            #    print(f"DP rank {local_dp_rank}, Prompt: {prompt!r}, " f"Generated text: {generated_text!r}")
        print(">>> Generation completed")
        result_queue.put(result)
        print(">>> Put in result queue")
        # del part_eval_dataset
        # del part_prompt_list
        # del outputs
        # del data_list
        # Give engines time to pause their processing loops before exiting.
        #sleep(10)
        #gc.collect()
        barrier.wait()  # 等待所有进程到达此点
        print(f"Exit the process {local_dp_rank}")
    
    def run_in_parpallel(self):

        procs = []
        for local_dp_rank in range(0, self.dp_size):
            print("local_dp_rank", local_dp_rank)
            proc = Process(target=self.generation,
                            args=(self.data_list, self.model_path, self.dp_size, self.tp_size,
                                  local_dp_rank, self.dp_master_ip, self.dp_master_port, 
                                  self.temperature, self.top_p, self.gpu_memory_utilization,
                                  self.result_queue, self.lock, self.dtype, self.max_input_tokens, self.max_output_tokens, 
                                  self.use_lora, self.lora_path, self.barrier, self.debug_mode)
                            )
            proc.start()
            procs.append(proc)
        
        # 等待所有子进程结束
        for proc in procs:
            proc.join()
            if proc.exitcode != 0:
                print(f"Process {proc.pid} exited with code {proc.exitcode}")

        # 从 result_queue 获取每个子进程的返回值
        results = []
        while not self.result_queue.empty():
            results.extend(self.result_queue.get())
            
        results = sorted(results, key=lambda x: x["id"])
        save_list_to_json(results, self.save_file_path )
        print("len(results)",len(results))
        print("results[0]:\n",results[0])

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    args = parse_args()  # 解析命令行参数并加载配置
    print(args)
    print("without_context:", args.without_context)
    inference_pipeline = InferencePipeline(args)
    inference_pipeline.run_in_parpallel()  # 启动推理管道
    print("finished\n")
    #inference_pipeline.test_in_parallel()  # 启动推理管道