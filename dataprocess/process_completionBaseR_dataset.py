from utils.json_util import ensure_dir,read_parquet_to_list,save_list_to_json,save_list_to_parquet 
import os
from utils.model_utils.retriever import Retriever
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
def produce_completion_Base_retrieval_dataset(prefix_path="./dataprocess/datafile/repobench_retrieval", retrieval_model_name=None, reserve_num=5):
    split_names = ['cff','cfr','if']
    language_name_list = ['python','java']
    for language_name in language_name_list:
        for split_name in split_names:
            data_file_path = os.path.join(prefix_path, f"{language_name}_{split_name}_train.parquet")
            data_list = read_parquet_to_list(data_file_path)
            data_list = add_and_sort_dataset_by_RetrievalSocre(data_list, retrieval_model_name, reserve_num = 5)
            


def add_and_sort_dataset_by_RetrievalSocre(data_list, retrieval_model_name, reserve_num = 5):
    """
    Add and sort dataset by retrieval score.
    """
    Retriever = Retriever(retrieval_model_name)
    for data in data_list:
        query = data['code']
        query_embedding = Retriever.get_sentence_embeddings(source_text = query)
        context_list = data['context']
        gold_snippet_index = data['gold_snippet_index']
        gold_snippet = context_list[gold_snippet_index]
        for context in context_list:
            context_embedding = Retriever.get_sentence_embeddings(source_text = context['snippet'])
            context['retrieval_score'] = Retriever.cosine_similarity(query_embedding, context_embedding)
        # Sort the context list by retrieval score
        context_list = sorted(context_list,key=lambda x: x['retrieval_score'], reverse=True)
        # Select the top k context
        result_context_list = context_list[:reserve_num]    
        # 如果context_list[gold_snippet_index]不在context_list[:reserve_num]，取context_list[:reserve_num-1]
        if gold_snippet_index != -1 and gold_snippet not in data['context']:
            data['context'].append(gold_snippet)
            
    