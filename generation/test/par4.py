import os
import time
import gc
import copy
from multiprocessing import Process, Lock, Queue

def generation(self, data_list, model_path, dp_size, tp_size, local_dp_rank, dp_master_ip, dp_master_port,
               temperature, top_p, gpu_memory_utilization, result_queue, lock, debug_mode=True):
    init_os_env(local_dp_rank, dp_size, dp_master_ip, dp_master_port)

    # 处理进程数据
    part_eval_dataset, part_prompt_list = get_part_eval_dataset(data_list, local_dp_rank, dp_size, lock)
    
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p)
    llm = LLM(model=model_path, tensor_parallel_size=tp_size, gpu_memory_utilization=gpu_memory_utilization, enforce_eager=True)
    
    print(f"rank {local_dp_rank} start generation......")
    outputs = llm.generate(part_prompt_list, sampling_params)
    
    result = []
    for i, output in enumerate(outputs):
        prompt = output.prompt
        if prompt != part_eval_dataset[i]['prompt']:
            raise ValueError(f"DP rank {local_dp_rank}, Prompt: {prompt!r} not equal to output prompt: {part_eval_dataset[i]['prompt']!r}")
        
        generated_text = output.outputs[0].text
        new_data = {"prompt": copy.deepcopy(part_eval_dataset[i]['prompt']), "generated_text": copy.deepcopy(generated_text), 
                    "tag": copy.deepcopy(part_eval_dataset[i]['tag']), "id": copy.deepcopy(part_eval_dataset[i]['id'])}
        result.append(new_data)

    # 使用锁保护队列操作
    with lock:
        result_queue.put(result)

    print("生成完成")
    del part_eval_dataset
    del part_prompt_list
    del outputs

    # 让引擎暂停，避免进程提前退出
    time.sleep(1)  # 如果需要延迟退出
    gc.collect()  # 强制垃圾回收
    print("退出进程")
    os._exit(0)  # 强制退出当前进程

def run_in_parpallel(self):
    procs = []
    for local_dp_rank in range(0, self.tp_size):
        proc = Process(target=self.generation,
                       args=(self.data_list, self.model_path, self.dp_size, self.tp_size,
                             local_dp_rank, self.dp_master_ip, self.dp_master_port, 
                             self.temperature, self.top_p, self.gpu_memory_utilization,
                             self.result_queue, self.lock, self.debug_mode))
        proc.start()
        procs.append(proc)
    
    # 等待所有子进程结束
    for proc in procs:
        proc.join(timeout=30)  # 使用 timeout 防止进程挂起
        if proc.is_alive():
            print(f"Process {proc.pid} is still running, terminating it.")
            proc.terminate()  # 强制终止子进程

    # 从 result_queue 获取每个子进程的返回值
    results = []
    while not self.result_queue.empty():
        results.extend(self.result_queue.get())

    save_list_to_json(results, self.save_file_path)
    print("len(results)", len(results))
    print("results[0]:\n", results[0])
