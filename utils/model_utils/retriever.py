import torch.nn as nn
import torch    
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from enum import Enum
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

class Retrieve_ModelName(Enum):
    codebert = "codebert"
    unixcoder = "unixcoder"
    GIST_large = "GIST_large"
    NV_Embed_v2 = "NV_Embed_v2"
    gte_Qwen2_7B_instruct = "gte_Qwen2_7B_instruct"
    gte_Qwen2_2B_instruct = "gte_Qwen2_2B_instruct"
    snow = "snow"
    jina_code = "jina_code"
#from datasets import CodeBlock




def codebert_tokenize(text, tokenizer, max_length, is_query):
    """
    Converts text to a list of token ids.
    :param text: The text to be converted
    :param tokenizer: The tokenizer to use
    :param max_length: The maximum input length
    :param is_query: A flag indicating whether the text is a query
    :return: A list of token ids
    """
    
    tokens = tokenizer.tokenize(text)

    # Apply manual truncation
    if is_query:
        tokens = tokens[-max_length:]  # Keep the last `max_length` tokens (from the right)
    else:
        tokens = tokens[:max_length]   # Keep the first `max_length` tokens (from the left)
    assert len(tokens) <= max_length
    # Convert tokens back to token IDs
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    # Pad if necessary
    padding_length = max_length - len(token_ids)
    if padding_length > 0:
        token_ids += [tokenizer.pad_token_id] * padding_length
    assert len(token_ids) <= max_length
    return token_ids


def add_eos(input_examples, eos_token): # used by NV_Embed_v2
  input_examples = [input_example + eos_token for input_example in input_examples]
  return input_examples

class Retriever:
    """
    Retriever model, used to compute sentence embeddings and retrieve similar code blocks.
    :param retrieval_model_name: The name of the retrieval model
    """
    def __init__(self, retrieval_model_name, retriever_model_path):
        super(Retriever, self).__init__()
        
        self.retrieval_model_name = retrieval_model_name
        self.retriever_model_path = retriever_model_path

        if self.retrieval_model_name == Retrieve_ModelName.codebert.value or self.retrieval_model_name == Retrieve_ModelName.unixcoder.value:
            self.tokenizer = AutoTokenizer.from_pretrained(self.retriever_model_path)
           
            self.model = AutoModel.from_pretrained(self.retriever_model_path)
            self.model = torch.nn.DataParallel(self.model).cuda()
            self.model.eval()
        else:
            if self.retrieval_model_name == Retrieve_ModelName.NV_Embed_v2.value:
                self.model = SentenceTransformer(self.retriever_model_path, trust_remote_code=True)
                self.model.max_seq_length = 512 
                self.model.tokenizer.padding_side="left"
            elif self.retrieval_model_name == Retrieve_ModelName.gte_Qwen2_7B_instruct.value or self.retrieval_model_name == Retrieve_ModelName.gte_Qwen2_2B_instruct.value:
                self.model = SentenceTransformer(self.retriever_model_path, trust_remote_code=True)
                self.model.max_seq_length = 512 
            else:
                self.model = SentenceTransformer(self.retriever_model_path)
                self.model.max_seq_length = 512
            self.tokenizer = None

    def setMaxlength(self, max_length):    
        if self.retrieval_model_name != Retrieve_ModelName.codebert.value and \
            self.retrieval_model_name != Retrieve_ModelName.unixcoder.value and \
            self.retrieval_model_name != Retrieve_ModelName.GIST_large.value:
            self.model.max_seq_length = max_length
        if self.retrieval_model_name == Retrieve_ModelName.NV_Embed_v2.value:
            self.model.tokenizer.padding_side="right"


    def unixcoder_tokenize(text, tokenizer, max_length):
        """
        Converts text to a list of token ids.
        :param text: The text to be converted
        :param tokenizer: The tokenizer to use
        :param max_length: The maximum input length
        :return: A list of token ids
        """
        tokens = tokenizer.tokenize(text)
        tokens = tokens[:max_length - 4]
        tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + tokens + [tokenizer.sep_token]
        tokens_id = tokenizer.convert_tokens_to_ids(tokens)
        padding_length = max_length - len(tokens_id)
        tokens_id += [tokenizer.pad_token_id] * padding_length

        return tokens_id  
    
    def get_sentence_embeddings(self, source_ids = None, source_text = None, query_prefix= None):
        """
        Forward propagation function, used to generate the embedding representation of the input.
        :param input_ids: The sequence of input IDs
        :return: The embedding representation
        """
        if self.retrieval_model_name == Retrieve_ModelName.codebert.value or self.retrieval_model_name == Retrieve_ModelName.unixcoder.value:
            if self.retrieval_model_name == Retrieve_ModelName.codebert.value:
                mask = source_ids.ne(self.tokenizer.pad_token_id)
                sentence_embeddings = self.model(source_ids, attention_mask=mask)[1]
            elif self.retrieval_model_name == Retrieve_ModelName.unixcoder.value:
                mask = source_ids.ne(self.tokenizer.pad_token_id)
                token_embeddings = self.model(source_ids, attention_mask=mask)[0]
                sentence_embeddings = (token_embeddings * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1)
                sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        else:
            if self.retrieval_model_name == Retrieve_ModelName.NV_Embed_v2.value:
                if query_prefix == None:
                    sentence_embeddings = self.model.encode(add_eos(source_text, self.model.tokenizer.eos_token), batch_size=len(source_text), normalize_embeddings=True)
                else:
                    sentence_embeddings = self.model.encode(add_eos(source_text, self.model.tokenizer.eos_token), batch_size=len(source_text), prompt=query_prefix, normalize_embeddings=True)
            elif self.retrieval_model_name == Retrieve_ModelName.gte_Qwen2_7B_instruct.value or self.retrieval_model_name == Retrieve_ModelName.gte_Qwen2_2B_instruct.value:
                if query_prefix == None:
                    sentence_embeddings = self.model.encode(source_text)
                else:
                    sentence_embeddings = self.model.encode(source_text, prompt_name="query")
            elif self.retrieval_model_name == Retrieve_ModelName.snow.value:
                if query_prefix == None:
                    sentence_embeddings = self.model.encode(source_text)
                else:
                    sentence_embeddings = self.model.encode(source_text, prompt_name="query")
                    
            else: 
                sentence_embeddings = self.model.encode(source_text, convert_to_tensor=True)

        return sentence_embeddings

    def cosine_similarity(self, a_vec, b_vec):
        """
        Compute the cosine similarity between two vectors.  
        :param a: The first vector
        :param b: The second vector
        :return: The cosine similarity
        """
        cosine_sim = F.cosine_similarity(a_vec, b_vec, dim=0)
        return cosine_sim.item()