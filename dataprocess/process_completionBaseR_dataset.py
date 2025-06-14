from utils.json_util import ensure_dir,read_parquet_to_list,save_list_to_json,save_list_to_parquet , load_list_from_json
import os
from utils.model_utils.retriever import Retriever
from tqdm import tqdm
from transformers import AutoTokenizer
from utils.code_util import comment_out
import re

'''
{
    "repo_name": "repository name of the data point",
    "file_path": "path/to/current_file",
    "context": [
        {
            "path": "path/to/cross_file_1",
            "identifier": "identifier of the cross-file module",
            "snippet": "the code snippet of the cross-file module",
        },
        // ...
        {
            "path": "path/to/cross_file_k",
            "identifier": "identifier of the cross-file module",
            "snippet": "the code snippet of the cross-file module",
        },
    ],
    "import_statement": "all import statements in current file",
    "code": "the code for next-line prediction",
    "next_line": "the next line of the code",
    "gold_snippet_index": 2 // NOTE: Only for "cross_file_first" and "cross_file_random" settings, for "in_file" setting, we set it to -1.

}
'''
def produce_completion_Base_retrieval_dataset(prefix_path="./dataprocess/datafile/repobench_retrieval", retrieval_model_name=None, 
                                              retrieval_model_path = None, reserve_num=5, debug_mode=False):
    split_names = ['cff','cfr','if']
    language_name_list = ['python','java']
    
    for language_name in tqdm(language_name_list, desc="Processing languages"):
        for split_name in tqdm(split_names, desc="Processing splits"):
            print(f"Processing {language_name} {split_name} dataset")
            data_file_path = os.path.join(prefix_path, f"{language_name}_{split_name}_train.parquet")
            data_list = read_parquet_to_list(data_file_path)
            if debug_mode:
                data_list = data_list[:10]  # For debugging, only use the first 10 data points
            data_list = add_and_sort_dataset_by_RetrievalSocre(data_list, retrieval_model_name, retrieval_model_path, reserve_num = 5)
            new_data_file_path = os.path.join(prefix_path, f"top{reserve_num}", f"{language_name}_{split_name}_train.json")
            ensure_dir(new_data_file_path)
            save_list_to_json(data_list, new_data_file_path)


def get_token_length(tokenizer, text):
    """
    获取文本的token长度
    :param tokenizer: 分词器
    :param text: 文本字符串
    :return: token长度
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)

def cut_text_to_max_token_length(text, tokenizer, max_token_length, mode='right'):
    """
    截断文本到最大token长度
    :param text: 文本字符串
    :param tokenizer: 分词器
    :param max_token_length: 最大token长度
    :return: 截断后的文本字符串
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) > max_token_length:
        if mode == 'left':
            tokens = tokens[-max_token_length:]  # 保留最后max_token_length个token
        elif mode == 'right':
            tokens = tokens[:max_token_length]
    return tokenizer.decode(tokens, skip_special_tokens=True)

def construct_specific_data(
    data: dict, 
    language: str = "java",
    tokenizer= None,
    code_column_name = "cropped_code",
    max_token_nums: int = 15800,
    max_snippet_tokens: int = 300,
    topk = 3,
    topn = 5
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

    snippet_num = 0
    new_conetxt = []
    flag = False
    data['origin_context'] = data['context']
    for id, snippet in enumerate(data['context'][:topk]):
        #print("id", id)
        if id == data['gold_snippet_index']: 
            flag = True
        code_comment = comment_out(snippet['snippet'], language)
        snippet_token_length = get_token_length(tokenizer, code_comment)
        if snippet_token_length <= max_snippet_tokens:
            snippet_num += 1        
            cross_file_prompt += f"{comment_symbol} Path: {snippet['path']}\n{code_comment}" + "\n\n"
            new_conetxt.append(snippet)
        # if the snippet is too long, we will cut it to the max_snippet_tokens
        elif snippet_token_length <= max_snippet_tokens*2 or id == data['gold_snippet_index']:
            code_comment = cut_text_to_max_token_length(code_comment, tokenizer, max_snippet_tokens)
            cross_file_prompt += f"{comment_symbol} Path: {snippet['path']}\n{code_comment}" + "\n\n"
            new_conetxt.append(snippet)
    
    # add the gold snippet at the end
    if flag == False:
        if len(new_conetxt) != 0:
            new_conetxt.pop()
        gold_snippet = data['context'][data['gold_snippet_index']]
        code_comment = cut_text_to_max_token_length(gold_snippet['snippet'], tokenizer, max_snippet_tokens)
        gold_snippet['snippet'] = code_comment
        code_comment = comment_out(code_comment, language)
        cross_file_prompt += f"{comment_symbol} Path: {gold_snippet['path']}\n{code_comment}" + "\n\n"
        new_conetxt.append(gold_snippet)
        data['gold_snippet_index'] = len(new_conetxt) - 1
    
    data['context'] = new_conetxt
    
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
    data['prompt'] = prompt
    return data

def add_message_tag(data_list, tokenizer_path, file_name, language, code_column_name = "cropped_code", prompt_max_tokens = 1024):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    new_data_list = []
    count_2048 = 0
    count_1024 = 0
    count_512 = 0
    for data in tqdm(data_list):
        prompt = data['prompt']
        prompt_tokens = get_token_length(tokenizer, prompt)
        data['prompt_tokens'] = prompt_tokens
        data['tag'] = file_name
        new_data_list.append(data)
        if prompt_tokens <= 2048:
            count_2048 += 1
        if prompt_tokens <= 1024:
            count_1024 += 1
        if prompt_tokens <= 512:
            count_512 += 1
        # if prompt_tokens <= prompt_max_tokens:
        #     print("prompt_tokens", prompt_tokens)
        #     print(prompt)
        #     input()
    print(f"file_name: {file_name}")
    print(f"count_2048: {count_2048}")
    print(f"count_1024: {count_1024}")
    print(f"count_512: {count_512}")

    return new_data_list


def restruct_retrieval_train_data(data_list, tokenizer_path, file_name, language, code_column_name = "cropped_code", prompt_max_tokens = 2048):
    """
    Restructure the retrieval train data by adding prompt and token length.
    :param data_list: list of data points
    :param tokenizer_path: path to the tokenizer
    :param file_name: name of the file to be processed
    :param language: programming language of the code
    :param code_column_name: name of the code column in the data
    :param prompt_max_tokens: maximum number of tokens for the prompt
    :return: list of restructured data points
    """

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    new_data_list = []
    count_2048 = 0
    for data in tqdm(data_list):
        data = construct_specific_data(data, language=language, tokenizer=tokenizer, code_column_name= code_column_name)
        if len(data['context']) < 2:
            continue
        prompt = data['prompt']
        prompt_tokens = get_token_length(tokenizer, prompt)
        data['prompt_tokens'] = prompt_tokens
        data['tag'] = file_name
        if prompt_tokens <= 650:
            continue
        if prompt_tokens <= 2048:
            count_2048+= 1
            new_data_list.append(data)
        # if prompt_tokens <= prompt_max_tokens:
        #     print("prompt_tokens", prompt_tokens)
        #     print(prompt)
        #     input()
    print(f"file_name: {file_name}")
    print(f"count_2048: {count_2048}")

    return new_data_list

def restruct_retrieval_train_data_for_dataset(prefix_path="./dataprocess/datafile/repobench_retrieval/top5", tokenizer_path=None, 
                                code_column_name="code", 
                                save_prefix_path = "./dataprocess/datafile/repobench_retrieval/top5/tag",
                                debug_mode=False):
    
    split_names = ['cff','cfr','if']
    language_name_list = ['python','java']
    
    for language_name in tqdm(language_name_list, desc="Processing languages"):
        for split_name in tqdm(split_names, desc="Processing splits"):
            print(f"Processing {language_name} {split_name} dataset, adding column 'prompt' and token length")
            data_file_path = os.path.join(prefix_path, f"{language_name}_{split_name}_train.json")
            data_list = load_list_from_json(data_file_path)
            data_list = restruct_retrieval_train_data(data_list, tokenizer_path, f"{language_name}_{split_name}", language_name, code_column_name = code_column_name)
            save_file_path = os.path.join(save_prefix_path, f"{language_name}_{split_name}.json")
            ensure_dir(save_file_path)
            save_list_to_json(data_list, save_file_path)


def add_and_sort_dataset_by_RetrievalSocre(data_list, retrieval_model_name, retrieval_model_path, reserve_num = 5):
    """
    Add and sort dataset by retrieval score.
    """
    retriever = Retriever(retrieval_model_name, retrieval_model_path)
    #result_data_list = []
    count = 0
    for data in tqdm(data_list, desc="Adding and sorting dataset by retrieval score"):
        query = data['code']
        query_embedding = retriever.get_sentence_embeddings(source_texts = [query])[0]
        context_list = data['context']
        gold_snippet_index = data['gold_snippet_index']
        gold_snippet = context_list[gold_snippet_index]
        for context in context_list:
            context_embedding = retriever.get_sentence_embeddings(source_texts = [context['snippet']])[0]
            context['retrieval_score'] = retriever.cosine_similarity(query_embedding, context_embedding)
        # Sort the context list by retrieval score
        context_list = sorted(context_list,key=lambda x: x['retrieval_score'], reverse=True)
        # Select the top k context
        result_context_list = context_list[:reserve_num]    
        # 如果context_list[gold_snippet_index]不在context_list[:reserve_num]，取context_list[:reserve_num-1]
        if gold_snippet_index != -1: # reprocess the gold snippet index
            if gold_snippet in result_context_list:
                for i, context in enumerate(result_context_list):
                    if context['snippet'] == gold_snippet['snippet']:
                        data['gold_snippet_index'] = i
                        break
                data['context'] = result_context_list
            else:
                data['context'] = result_context_list[:-1]
                data['context'].append(gold_snippet)
                data['gold_snippet_index'] = reserve_num - 1
        else:
            data['context'] = result_context_list
            
        if gold_snippet_index == reserve_num - 1:
            count += 1

        # print(data)
        # print(len(data['context']))
        # input("Press Enter to continue...")
    print(f"Total number of data with gold snippet in the lowest sorting number: {count}")
    return data_list

    