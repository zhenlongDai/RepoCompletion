from datasets import load_dataset
from utils.json_util import ensure_dir,read_parquet_to_list,save_list_to_json,save_list_to_parquet 
import json
import argparse
from transformers import AutoTokenizer
from tqdm import tqdm

def download_and_filt_2k(dataset_name = "tianyang/repobench_python_v1.1", prefix_path = "./dataprocess/datafile/python_datafile-2k"):

    dataset = load_dataset(dataset_name, verification_mode="no_checks")
    data_list = []
    data_name_list = []
    for split_name, split_data in dataset.items():
        data = []
        for item in split_data:
            data.append(item)
        json_data=[]
        for item in data:
            if item['level'] == '2k':
                json_data.append(item)
        data_list.append(json_data)
        data_name_list.append(split_name)

    ensure_dir(prefix_path)
    for i, data_name in enumerate(data_name_list):
        with open(f"{prefix_path}/{data_name}.json", "w", encoding="utf-8") as f:
            data = data_list[i]
            json.dump(data, f, indent=4)

def download_compeltion_dataset(dataset_name = "tianyang/repobench-c", task_name = "java_cff", split_name="train", prefix_path = "./dataprocess/datafile/repobench-c"):
    dataset = load_dataset(dataset_name,  data_dir = task_name, split=split_name, revision="refs/convert/parquet",verification_mode="no_checks")
    ensure_dir(prefix_path)
    save_path = f"{prefix_path}/{task_name}_{split_name}.parquet"
    dataset.to_parquet(save_path)  
    print(f"Dataset saved to: {save_path}")


def get_code_line(code_str: str):
    """
    获取代码行数
    :param code_str: 代码字符串
    :return: 代码行数
    """
    return len(code_str.split('\n'))

def filt_by_length(data_list, context_max_length=100, code_max_length = 30):
    """
    过滤数据列表，保留符合长度要求的数据
    :param data_list: 数据列表
    :param context_max_length: 上下文最大长度
    :param code_max_length: 代码最大长度
    :return: 过滤后的数据列表
    """
    filtered_data = []
    for data in data_list:
        if get_code_line(data['context']) <= context_max_length and get_code_line(data['code']) <= code_max_length:
            filtered_data.append(data)
    return filtered_data

def get_token_length(tokenizer, text):
    """
    获取文本的token长度
    :param tokenizer: 分词器
    :param text: 文本字符串
    :return: token长度
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)

def add_message_tag(data_list, tokenizer_path, file_name, prompt_max_tokens = 512):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    new_data_list = []
    count_1024 = 0
    count_768 = 0
    count_512 = 0
    for data in tqdm(data_list):
        prompt = data['prompt']
        prompt_tokens = get_token_length(tokenizer, prompt)
        data['prompt_tokens'] = prompt_tokens
        data['tag'] = file_name
        new_data_list.append(data)
        if prompt_tokens <= 1024:
            count_1024 += 1
        if prompt_tokens <= 768:
            count_768 += 1
        if prompt_tokens <= 512:
            count_512 += 1
        #if prompt_tokens <= prompt_max_tokens:
            # print(prompt_tokens)
            # print(prompt)
            # input()
    print(f"file_name: {file_name}")
    print(f"count_1024: {count_1024}")
    print(f"count_768: {count_768}")
    print(f"count_512: {count_512}")

    return new_data_list

def process_parquet(file_name,
                        parquet_path = "./dataprocess/datafile/repobench-c/origin/java_cff_train.parquet", 
                        tokenizer_path = "/data/develop/dzl/cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/Qwen2.5-7B-instruct",
                        new_parquet_path = "./dataprocess/datafile/repobench-c/tag/java_cff_train_token_tag.parquet",
                        json_path = "./dataprocess/datafile/repobench-c/tag/java_cff_train.json",
                        save_mode = "parquet"):
    
    data_list = read_parquet_to_list(parquet_path)
    print(len(data_list))
    data_list = add_message_tag(data_list, tokenizer_path, file_name)
    if save_mode == "json":
        save_list_to_json(data_list, json_path)
    elif save_mode == "parquet":
        save_list_to_parquet(data_list, new_parquet_path)

def main(args):
    if args.process_mode == "download_test_data":
        download_dataset_name_list = ['python','java']
        for dataset_name in download_dataset_name_list:
            dataset_name = f"tianyang/repobench_{dataset_name}_v1.1"
            download_and_filt_2k(dataset_name = dataset_name, prefix_path = f"./dataprocess/datafile/{dataset_name}_datafile-2k")

    elif args.process_mode == "download_compeltion_train_dataset":
        download_task_name_list = ['java_cff','java_cfr','java_if','python_cff','python_cfr','python_if']
        for task_name in download_task_name_list:
            download_compeltion_dataset(task_name=task_name)

    elif args.process_mode == "process_parquet":
        file_name_list = ['java_cff','java_cfr','java_if','python_cff','python_cfr','python_if']
        for file_name in file_name_list:
            process_parquet(file_name, parquet_path = f"./dataprocess/datafile/repobench-c/origin/{file_name}_train.parquet", \
                                new_parquet_path = f"./dataprocess/datafile/repobench-c/tag/{file_name}_tag.parquet",
                                #json_path = f"./dataprocess/datafile/repobench-c/tag/{qarquet}_train.json", \
                            )
    else:
        print("Invalid process mode. Please choose from 'download_2k', 'download_compeltion_dataset', or 'process_parquet'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--process_mode", type=str, default=None, help="the name of processing dataset")
    #parser.add_argument("--language", type=str, default=None, help="the name of processing dataset")
    args = parser.parse_args()
    main(args)

    pass