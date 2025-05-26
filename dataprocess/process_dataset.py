from datasets import load_dataset
from utils.json_util import ensure_dir,read_parquet_to_list,save_list_to_json,save_list_to_parquet, load_list_from_json
import json
import argparse
from tqdm import tqdm
import os
from dataprocess.process_completionBaseR_dataset import produce_completion_Base_retrieval_dataset, add_message_tag, restruct_retrieval_train_data_for_dataset

def download_and_filt_2k(dataset_name = "tianyang/repobench_python_v1.1", prefix_path = "./datasets/datafile/python_datafile-2k/"):

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

    
    for i, data_name in enumerate(data_name_list):
        file_path = os.path.join(prefix_path, f"{data_name}.json")
        ensure_dir(file_path)
        with open(file_path, "w", encoding="utf-8") as f:
            data = data_list[i]
            json.dump(data, f, indent=4)

def download_compeltion_dataset(dataset_name = "tianyang/repobench-c", task_name = "java_cff", split_name="train", prefix_path = "./dataprocess/datafile/repobench-c"):
    dataset = load_dataset(dataset_name,  data_dir = task_name, split=split_name, revision="refs/convert/parquet",verification_mode="no_checks")
    if "/" in task_name:
        task_name = task_name.replace("/", "_")
    save_path = f"{prefix_path}/{task_name}_{split_name}.parquet"
    ensure_dir(save_path)
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




def process_train_dev_data(file_name,
                        language,
                        file_path = "./dataprocess/datafile/repobench-c/java_cff_train.parquet", 
                        tokenizer_path = "/data/dzl/package/CodeLLM/Qwen2.5-Coder-7B-Instruct",
                        save_path = "./dataprocess/datafile/repobench-c/tag/java_cff_train.parquet",
                        read_mode = "parquet",
                        save_mode = "parquet"):
    if read_mode == "parquet":    
        data_list = read_parquet_to_list(file_path)
    elif read_mode == "json":
        data_list = load_list_from_json(file_path)

    print("the number of data list:", len(data_list)) 
    data_list = add_message_tag(data_list, tokenizer_path, file_name, language)
    ensure_dir(save_path)
    if save_mode == "json":
        save_list_to_json(data_list, save_path)
    elif save_mode == "parquet":
        save_list_to_parquet(data_list, save_path)

def get_import_train_data_from_tag_file(language, 
                                        tag_parquet_path = "./dataprocess/datafile/repobench-c/tag/",
                                        save_json_path = "./dataprocess/datafile/codeCompeletion",
                                        dataset_tag="import",
                                        up_train_number = 1300,
                                        up_dev_number = 50,
                                        ):
    file_name_list = ['cff','cfr','if']
    token_length_nums= [512,768,1024]
    import_train_data_list = []
    import_dev_data_list = []
    for file_name in tqdm(file_name_list, desc="processing each file"):
        parquet_path = os.path.join(tag_parquet_path, f"{language}_{file_name}_tag.parquet")
        data_list = read_parquet_to_list(parquet_path)
        data_map ={ f"{token_length}": [] for token_length in token_length_nums}
        
        for data in data_list:
            data_token_length = data['prompt_tokens']
            for up_token_legnth in token_length_nums:
                if data_token_length <= up_token_legnth and len(data_map[str(up_token_legnth)]) < up_train_number + up_dev_number:
                    data_map[str(up_token_legnth)].append(data)
                    break

        for up_token_legnth, import_data_list in data_map.items():
            print(f"{file_name}: up token length is {up_token_legnth}, the number of import_data_list:", len(import_data_list))
            import_train_data_list.extend(import_data_list[:up_train_number])
            import_dev_data_list.extend(import_data_list[up_train_number:])
        
    train_data_file_path = os.path.join(save_json_path, language, dataset_tag, "train.json")
    dev_data_file_path = os.path.join(save_json_path, language, dataset_tag, "dev.json")
    
    ensure_dir(train_data_file_path)
    print(f">>> language: {language}")
    print(">>> The number of train data list:", len(import_train_data_list))
    save_list_to_json(import_train_data_list, train_data_file_path)    
    print(">>> The number of dev data list:", len(import_dev_data_list))
    save_list_to_json(import_dev_data_list, dev_data_file_path) 

def main(args):
    if args.process_mode == "download_test_data":
        download_language_name_list = ['python','java']
        for language_name in download_language_name_list:
            dataset_name = f"tianyang/repobench_{language_name}_v1.1"
            download_and_filt_2k(dataset_name = dataset_name, prefix_path = f"./datasets/repobench/{language_name}/test/")

    elif args.process_mode == "process_import_compeltion_dataset":
        #1. download the dataset
        download_task_name_list = ['java_cff','java_cfr','java_if','python_cff','python_cfr','python_if']
        for task_name in download_task_name_list:
            download_compeltion_dataset(task_name=task_name)
        #2. process_train_dev_data 
        split_names = ['cff','cfr','if']
        language_name_list=['python','java']
        for language in language_name_list:
            for split_name in split_names:
                file_name = f"{language}_{split_name}"
                process_train_dev_data(file_name, language = language,file_path = f"./dataprocess/datafile/repobench-c/{file_name}_train.parquet", 
                                    save_path = f"./dataprocess/datafile/repobench-c/tag/{file_name}_tag.parquet"
                                )
        #3. generate import training data
        languages = ['java','python']
        for language in languages:
            get_import_train_data_from_tag_file(language = language,  dataset_tag="import")

    elif args.process_mode == "process_retrieval_compeltion_dataset":
        #1. download the dataset
        download_split_names = ['cff','cfr','if']
        download_task_name_list=['python','java']
        for task_name in download_task_name_list:
            for split_name in download_split_names:
                    download_compeltion_dataset(dataset_name="tianyang/repobench-p",task_name=f"{task_name}/{split_name}", split_name= 'train', 
        
                                                 prefix_path="./dataprocess/datafile/repobench_retrieval")
        #2. process the dataset
        produce_completion_Base_retrieval_dataset(prefix_path="./dataprocess/datafile/repobench_retrieval",
                                                  retrieval_model_name=args.retrieval_model_name,
                                                  retrieval_model_path=args.retrieval_model_path)
        
        #3. restruct the retrieval data for dataset (adding tag/prompt_tokens/prompt, restructing context etc.)
        language_name_list=['python','java']
        restruct_retrieval_train_data_for_dataset(prefix_path="./dataprocess/datafile/repobench_retrieval/top5",
                                                tokenizer_path = args.tokenizer_path,
                                                code_column_name = "code",
                                                save_prefix_path = "./dataprocess/datafile/repobench_retrieval/top5/tag"
                                                )

        #4. spilt train/dev data
        split_names = ['cff','cfr','if']
        language_name_list=['python','java']
        number_map = {'train':{'cff': 800, 'cfr': 300, 'if': 200},
                        'dev':{'cff': 50, 'cfr': 50, 'if': 50}}   
        for language in language_name_list:
            train_data_list = []
            dev_data_list = []
            for split_name in split_names:
                file_name = f"{language}_{split_name}"
                prefix_path = f"./dataprocess/datafile/repobench_retrieval/top5/tag/"
                file_path = os.path.join(prefix_path, f"{file_name}.json")
                data_list = load_list_from_json(file_path)
                train_number = number_map['train'][split_name]
                dev_number = number_map['dev'][split_name]
                train_data_list.extend(data_list[:train_number])
                dev_data_list.extend(data_list[train_number:train_number+dev_number])
            save_prefix_path = f"./dataprocess/datafile/codeCompeletion/{language}/retrieval"
            train_save_path = os.path.join(save_prefix_path, "train.json")
            dev_save_path = os.path.join(save_prefix_path, "dev.json")
            ensure_dir(save_prefix_path)
            save_list_to_json(train_data_list, train_save_path)
            save_list_to_json(dev_data_list, dev_save_path)

    elif args.process_mode == "merge_import_and_retrieval":
        language_name_list=['python','java']
        file_name_list = ['import','retrieval']
        for language in language_name_list:
            data_list = []
            dev_list = []
            for file_name in file_name_list:
                file_path = f"./dataprocess/datafile/codeCompeletion/{language}/{file_name}/train.json"
                data_list.extend(load_list_from_json(file_path))
                dev_file_path = f"./dataprocess/datafile/codeCompeletion/{language}/{file_name}/dev.json"
                dev_list.extend(load_list_from_json(dev_file_path))
            train_save_path = f"./datasets/repobench/{language}/train.json"
            dev_save_path = f"./datasets/repobench//{language}/dev.json"
            ensure_dir(train_save_path)
            save_list_to_json(data_list, train_save_path)
            save_list_to_json(dev_list, dev_save_path)
        print(f"Merge import and retrieval data for {language} completed.")

    else:
        print("Invalid process mode. Please choose from 'download_test_data', 'process_import_compeltion_dataset', " \
        "'process_retrieval_compeltion_dataset', 'merge_import_and_retrieval'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--process_mode", type=str, default=None, help="the name of processing dataset")
    parser.add_argument("--retrieval_model_name", type=str, default=None, help="the name of retrieval model")
    parser.add_argument("--retrieval_model_path", type=str, default=None, help="the path of retriever model")
    parser.add_argument("--tokenizer_path", type=str, default="/data/dzl/package/CodeLLM/Qwen2.5-Coder-7B-Instruct", help="the path of tokenizer")
    args = parser.parse_args()
    main(args)

    pass