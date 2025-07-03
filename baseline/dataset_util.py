import re
import os
from utils.json_util import load_list_from_json
from trl.data_utils import maybe_apply_chat_template
from transformers import AutoTokenizer
from utils.code_util import comment_out
import json
from datasets import  Dataset, DatasetDict
from transformers import BertTokenizer
from utils.eval_utils import uncomment_code
from baseline.aixcoder_prompt import construct_aixcoder_prompt

def load_test_dataset_by_baselineName(eval_dataset_name, data_dir_path, language, model_path, max_input_tokens, debug_mode, without_context = False):
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
                temp_data['prompt'] = construct_aixcoder_prompt(data, language, tokenizer, eval_dataset_name, max_input_tokens, without_context)
                temp_data['tag'] = tag_name
                temp_data['id'] = idx + data_num
                temp_data['ground_truth'] = data['next_line']
                data_list.append(temp_data)
    print("the number of data_list:", len(data_list))
    #input()
    return data_list if not debug_mode else data_list[:10]