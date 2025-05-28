import re
import os
from utils.json_util import load_list_from_json
from trl.data_utils import maybe_apply_chat_template
from transformers import AutoTokenizer
from utils.code_util import comment_out
import json
from datasets import  Dataset, DatasetDict
from transformers import BertTokenizer

def load_json_as_hf_dataset(json_file_path):
    data_list = load_list_from_json(json_file_path)
    # 转换格式
    processed_data = []
    for row in data_list:
        processed_data.append({
            "prompt": row["prompt"],
            "completion": f"<answer>```\n{row['next_line']}\n```</answer>",
        })
    # 用 HuggingFace Dataset 构建
    return Dataset.from_list(processed_data)
    

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

def make_conversation(example, system_prompt):
    
    prompt = []
    if system_prompt is not None:
        prompt.append({"role": "system", "content": system_prompt})
    prompt.append({"role": "user", "content": example["prompt"]})
    return {"prompt": prompt}

def construct_prompt(
    data: dict, 
    language: str = "java",
    tokenizer= None,
    code_column_name = "cropped_code",
    max_token_nums: int = 15800,
    without_context: bool = False
    ) -> str:
    """
    Construct the prompt for next line prediction.

    :param data: data point from the dataset
    :param language: the language of the code
    :param tokenizer: the tokenizer of the evaluation model
    :param max_token_nums: the maximum number of tokens constraint for the prompt

    :return: the constructed prompt
    """

    # comment symbol for different languages
    comment_symbol = "#" if language == "python" else "//"

    # construct the cross-file prompt and in-file prompt separately
    # cross-file prompt
    cross_file_prompt = f"{comment_symbol} Repo Name: {data['repo_name']}\n"

    if not without_context:
        for snippet in data['context']:
            code_comment = comment_out(snippet['snippet'], language)
            cross_file_prompt += f"{comment_symbol} Path: {snippet['path']}\n{code_comment}" + "\n\n"
    
    # in-file prompt
    in_file_prompt = f"{comment_symbol} Path: {data['file_path']}\n{data['import_statement']}\n{data[code_column_name]}\n"

    # if we assign the tokenizer and the max_token_nums, we will truncate the cross-file prompt to meet the constraint
    if tokenizer is not None and max_token_nums is not None:
        
        cross_file_prompt_token_nums = len(tokenizer.encode(cross_file_prompt))
        in_file_prompt_token_nums = len(tokenizer.encode(in_file_prompt))

        exceed_token_nums = cross_file_prompt_token_nums + in_file_prompt_token_nums - max_token_nums

        if exceed_token_nums > 0:
            # split the cross-file prompt into lines
            cross_file_prompt_lines = cross_file_prompt.split("\n")
            extra_token_num = exceed_token_nums
            # drop lines from end until the extra token number is less than 0
            for i in range(len(cross_file_prompt_lines)-1, -1, -1):
                extra_token_num -= len(tokenizer.encode(cross_file_prompt_lines[i]))
                if extra_token_num < 0:
                    break
            
            # join the lines back
            cross_file_prompt = "\n".join(cross_file_prompt_lines[:i]) + "\n\n"
    
    # combine the cross-file prompt and in-file prompt
    prompt = cross_file_prompt + in_file_prompt

    # normalize some empty lines
    prompt = re.sub(r'\n{4,}', '\n\n', prompt)

    return prompt

def construct_model_prompt(data, language, tokenizer, max_input_tokens, system_prompt, without_context = False):
    """
        ```maybe_apply_chat_template
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
        >>> example = {
        ...     "prompt": [{"role": "user", "content": "What color is the sky?"}],
        ...     "completion": [{"role": "assistant", "content": "It is blue."}]
        ... }
        >>> apply_chat_template(example, tokenizer)
        {'prompt': '<|user|>\nWhat color is the sky?<|end|>\n<|assistant|>\n', 'completion': 'It is blue.<|end|>\n<|endoftext|>'}
        ```
    """
    example = {}
    example['prompt'] = construct_prompt(data, language, tokenizer, code_column_name = "cropped_code", max_token_nums = max_input_tokens, without_context= without_context) #constrcut input of a specific task
    example['prompt'] = f"```\n{ example['prompt']}\n```"
    example = make_conversation(example, system_prompt) #constrcut conversation based on input
    model_prompt = maybe_apply_chat_template(example, tokenizer)["prompt"]  # construct input of model
    return model_prompt

def load_eval_dataset(system_prompt, eval_dataset_name, data_dir_path, language, model_path, max_input_tokens, debug_mode, without_context = False):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    data_list = []

    if eval_dataset_name == "repobench":
        file_name2tag_map = {"cross_file_first":"cff", "cross_file_random":"cfr", "in_file":"if"} 
        for file_name, tag_name in file_name2tag_map.items():
            data_file_path = os.path.join(data_dir_path, f"{file_name}.json")
            print(">>> start load data from data_file_path:", data_file_path)
            ori_data_list = load_list_from_json(data_file_path)
            data_num = len(data_list)
            for idx, data in enumerate(ori_data_list):
                temp_data = {}
                temp_data['prompt'] = construct_model_prompt(data, language, tokenizer, max_input_tokens, system_prompt, without_context = without_context)
                temp_data['tag'] = tag_name
                temp_data['id'] = idx + data_num
                temp_data['ground_truth'] = data['next_line']
                data_list.append(temp_data)
    print("the number of data_list:", len(data_list))
    #input()
    return data_list if not debug_mode else data_list[:10]

def load_ground_truth(eval_dataset_name, data_dir_path, code_column_name="cropped_code"): 
    data_list = []
    if eval_dataset_name == "repobench":
        file_name2tag_map = {"cross_file_first":"cff", "cross_file_random":"cfr", "in_file":"if"} 
        for file_name, tag_name in file_name2tag_map.items():
            data_file_path = os.path.join(data_dir_path, f"{file_name}.json")
            print(">>> start load data from data_file_path:", data_file_path)
            ori_data_list = load_list_from_json(data_file_path)
            data_num = len(data_list)
            for idx, data in enumerate(ori_data_list):
                temp_data = {}
                temp_data['in_file_prompt'] = f"{data['import_statement']}\n{data[code_column_name]}\n"
                temp_data['tag'] = tag_name
                temp_data['id'] = idx + data_num
                temp_data['ground_truth'] = data['next_line']
                data_list.append(temp_data)
    print("the number of data_list:", len(data_list))

    return data_list 


def load_compeltion_dataset(dataset_file_path):
    """
    Load the dataset from the given file path.
    
    Args:
        dataset_file_path (str): Path to the dataset file.
        
    Returns:
        datasets.Dataset: Loaded dataset.
    """
    if not os.path.exists(dataset_file_path):
        raise FileNotFoundError(f"Dataset file not found at {dataset_file_path}")
    train_file_path = os.path.join(dataset_file_path, "train.json")
    dev_file_path = os.path.join(dataset_file_path, "dev.json")
    train_dataset = load_json_as_hf_dataset(train_file_path)
    dev_dataset = load_json_as_hf_dataset(dev_file_path)
    dataset = DatasetDict({
        "train": train_dataset,
        "test": dev_dataset
    })
        
    # Ensure the dataset has the required columns
    if "prompt" not in dataset["train"].column_names or "completion" not in dataset["train"].column_names:
        raise ValueError("Dataset must contain 'prompt' and 'completion' columns.")
    
    return dataset