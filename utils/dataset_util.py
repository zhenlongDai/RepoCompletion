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

def load_json_as_hf_dataset(json_file_path, language=None, prompt_mode="split", eval_mode = "train", debug_mode=False):
    data_list = load_list_from_json(json_file_path)
    # 转换格式
    processed_data = []
    for data in data_list:
        input_code = data['import_statement'] + "\n" + data['code']
        processed_data.append({
            "language": language,
            "input_code": input_code,
            "prompt": construct_task_prompt(data = data, language = language, prompt_mode = prompt_mode),
            "solution": data['next_line'],
            "labels": data['next_line']
        })
    # 如果是调试模式，只取前10条数据
    if debug_mode:
        processed_data = processed_data[:10]
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

def construct_test_prompt( data: dict, 
    language: str = "java",
    tokenizer= None,
    code_column_name = "cropped_code",
    max_token_nums: int = 15800,
    without_context: bool = False,
    prompt_mode = "split"
    ) -> str:
    if prompt_mode == "comment":
        return construct_comment_prompt(data, language, tokenizer, code_column_name, max_token_nums, without_context)
    elif prompt_mode == "split":
        return construct_split_prompt(data, language, tokenizer, code_column_name, max_token_nums, without_context)
    
def construct_split_prompt(
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

     # construct the prompt with split snippets
    if isinstance(data['context'], str):
        context_list = split_snippets(data['context'], language)
    else:
        context_list = data['context']
    prompt = f"Repository name: {data['repo_name']}\n"
    if without_context is False:
        # add the context snippets to the prompt
        prompt += f"Code snippets:\n"
        for snippet in context_list:
            prompt += f"Path: {snippet['path']}\n```\n{snippet['snippet']}\n```\n"
    prompt += f"The incomplete code:\n"
    prompt += f"Path: {data['file_path']}\n```\n{data['import_statement']}\n{data[code_column_name]}\n```"

    return prompt
  
def construct_comment_prompt(
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

def split_snippets(snippets_str, language):
    # each snippet is "// Path: xxxx\n{code_snippet}// Path: xxxx\n{code_snippet}..." 的结构
    # 返回 [{"path": path, "snippet": snippet}, ...]
    snippets = []

    snippets_str = uncomment_code(snippets_str, language)
    snippet_list = snippets_str.split("Path: ")
    for snippet in snippet_list:
        snippet = snippet.rstrip()
        if snippet:
            # 取 path（第一行）和 snippet（剩余内容）
            lines = snippet.split('\n', 1)
            path = lines[0].rstrip()
            code = lines[1].rstrip() if len(lines) > 1 else ""
            snippets.append({"path": path, "snippet": code})
    return snippets

def construct_task_prompt(data, language, prompt_mode = "comment"):
    """
    Construct the task prompt for next line prediction.

    :param data: data point from the dataset
    :param prompt_mode: the mode of the prompt, can be "comment" or "split"
    :return: the constructed task prompt
    """
    if prompt_mode == "comment":
        # construct the prompt with comments
        prompt = data['prompt']
        prompt = f"```\n{ data['prompt']}\n```"
    elif prompt_mode == "split":
        
        # construct the prompt with split snippets
        if isinstance(data['context'], str):
            context_list = split_snippets(data['context'], language)
        else:
            context_list = data['context']
        prompt = f"Repository name: {data['repo_name']}\n"
        prompt += f"Code snippets:\n"
        for snippet in context_list:
            prompt += f"Path: {snippet['path']}\n```\n{snippet['snippet']}\n```\n"
        prompt += f"The incomplete code:\n"
        prompt += f"Path: {data['file_path']}\n```\n{data['import_statement']}\n{data['code']}\n```"
    else:
        raise ValueError(f"Unknown prompt mode: {prompt_mode}")
    
    return prompt

def construct_model_prompt(data, language, tokenizer = None, max_input_tokens = None, system_prompt = "", without_context = False, eval_mode=None, prompt_mode = "split"):
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
    if eval_mode == "test":
        example['prompt'] = construct_test_prompt(data, language, tokenizer, "cropped_code", max_input_tokens, without_context, prompt_mode) #constrcut input of a specific task
        example['prompt'] = f"```\n{ example['prompt']}\n```"
    elif eval_mode == "train" or eval_mode == "dev":
        example['prompt'] = construct_task_prompt(data, language, prompt_mode) # pass
   
    example = make_conversation(example, system_prompt) #constrcut conversation based on input
    model_prompt = maybe_apply_chat_template(example, tokenizer)["prompt"]  # construct input of model
    return model_prompt

def load_eval_dataset(system_prompt, eval_dataset_name, data_dir_path, language, model_path, max_input_tokens, debug_mode, without_context = False, eval_mode="test", prompt_mode = "split"):
    if eval_mode == "test":
        return load_test_dataset(system_prompt, eval_dataset_name, data_dir_path, language, model_path, max_input_tokens, prompt_mode, debug_mode, without_context)
    elif eval_mode == "dev":
        return load_dev_dataset(system_prompt, eval_dataset_name, data_dir_path, language, model_path, max_input_tokens,  prompt_mode, debug_mode)
    
def load_test_dataset(system_prompt, eval_dataset_name, data_dir_path, language, model_path, max_input_tokens, prompt_mode, debug_mode, without_context = False):
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
                temp_data['prompt'] = construct_model_prompt(data, language, tokenizer, max_input_tokens, system_prompt, without_context, "test", prompt_mode)
                temp_data['tag'] = tag_name
                temp_data['id'] = idx + data_num
                temp_data['ground_truth'] = data['next_line']
                data_list.append(temp_data)
    print("the number of data_list:", len(data_list))
    #input()
    return data_list if not debug_mode else data_list[:10]

def load_dev_dataset(system_prompt, eval_dataset_name, data_dir_path, language, model_path, max_input_tokens, prompt_mode, debug_mode):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    data_list = []

    data_file_path = os.path.join(data_dir_path, "dev.json")
    print(">>> start load data from data_file_path:", data_file_path)
    ori_data_list = load_list_from_json(data_file_path)
    for idx, data in enumerate(ori_data_list):
        temp_data = {}
        temp_data['prompt'] = construct_model_prompt(data, language, tokenizer, max_input_tokens, system_prompt, False, eval_mode="dev", prompt_mode = prompt_mode)
        temp_data['tag'] = data['tag']
        temp_data['id'] = idx
        temp_data['ground_truth'] = data['next_line']
        data_list.append(temp_data)
    print("the number of data_list:", len(data_list))
    #input()
    return data_list if not debug_mode else data_list[:10]

def load_ground_truth(eval_dataset_name, data_dir_path, eval_mode, code_column_name="cropped_code"): 
    data_list = []
    if eval_mode == "test":
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

    elif eval_mode == "dev":
        data_file_path = os.path.join(data_dir_path, f"{eval_mode}.json")
        print(">>> start load data from data_file_path:", data_file_path)
        ori_data_list = load_list_from_json(data_file_path)
        for idx, data in enumerate(ori_data_list):
            temp_data = {}
            temp_data['in_file_prompt'] = data['prompt']
            temp_data['tag'] = data['tag']
            temp_data['id'] = idx
            temp_data['ground_truth'] = data['next_line']
            data_list.append(temp_data)

    print("the number of data_list:", len(data_list))

    return data_list 


def load_compeltion_dataset(dataset_file_path, language=None, prompt_mode ="split", debug_mode=False):
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
    train_dataset = load_json_as_hf_dataset(train_file_path, language, prompt_mode, "train")
    dev_dataset = load_json_as_hf_dataset(dev_file_path, language, prompt_mode, "dev")
    dataset = DatasetDict({
        "train": train_dataset,
        "test": dev_dataset
    })
        
    # Ensure the dataset has the required columns
    if "prompt" not in dataset["train"].column_names or "solution" not in dataset["train"].column_names:
        raise ValueError("Dataset must contain 'prompt' and 'solution' columns.")
    
    return dataset

def example():
    context = "// Path: query/src/main/java/io/keen/client/java/exceptions/KeenQueryClientException.java\n// public class KeenQueryClientException extends KeenException {\n//     private static final long serialVersionUID = -8714276738565293346L;\n// \n//     public KeenQueryClientException() {\n//         super();\n//     }\n// \n//     public KeenQueryClientException(Throwable cause) {\n//         super(cause);\n//     }\n// \n//     public KeenQueryClientException(String message) {\n//         super(message);\n//     }\n// \n//     public KeenQueryClientException(String message, Throwable cause) {\n//         super(message, cause);\n//     }\n// }\n// \n// Path: core/src/main/java/io/keen/client/java/http/HttpMethods.java\n// public final class HttpMethods {\n//     private HttpMethods() {}\n// \n//     public final static String GET = \"GET\";\n//     public final static String POST = \"POST\";\n//     public final static String PUT = \"PUT\";\n//     public final static String DELETE = \"DELETE\";\n// }\n\n"

    code = split_snippets(context,  "java")
    language="java"
    prompt_mode = "split"
    data =  {
        "repo_name": "keenlabs/KeenClient-Java",
        "file_path": "query/src/main/java/io/keen/client/java/KeenQueryRequest.java",
        "context": "// Path: query/src/main/java/io/keen/client/java/exceptions/KeenQueryClientException.java\n// public class KeenQueryClientException extends KeenException {\n//     private static final long serialVersionUID = -8714276738565293346L;\n// \n//     public KeenQueryClientException() {\n//         super();\n//     }\n// \n//     public KeenQueryClientException(Throwable cause) {\n//         super(cause);\n//     }\n// \n//     public KeenQueryClientException(String message) {\n//         super(message);\n//     }\n// \n//     public KeenQueryClientException(String message, Throwable cause) {\n//         super(message, cause);\n//     }\n// }\n// \n// Path: core/src/main/java/io/keen/client/java/http/HttpMethods.java\n// public final class HttpMethods {\n//     private HttpMethods() {}\n// \n//     public final static String GET = \"GET\";\n//     public final static String POST = \"POST\";\n//     public final static String PUT = \"PUT\";\n//     public final static String DELETE = \"DELETE\";\n// }\n\n",
        "import_statement": "import java.net.URL;\nimport java.util.Collection;\nimport java.util.Map;\nimport io.keen.client.java.exceptions.KeenQueryClientException;\nimport io.keen.client.java.http.HttpMethods;",
        "code": "package io.keen.client.java;\n\n\n\n/**\n * Interface to be implemented by a query request\n *\n * @author baumatron\n */\nabstract class KeenQueryRequest {\n    abstract URL getRequestURL(RequestUrlBuilder urlBuilder, String projectId)\n            throws KeenQueryClientException;\n\n    // By default, we POST to get most of our query results.\n    String getHttpMethod() {",
        "prompt": "// Path: query/src/main/java/io/keen/client/java/exceptions/KeenQueryClientException.java\n// public class KeenQueryClientException extends KeenException {\n//     private static final long serialVersionUID = -8714276738565293346L;\n// \n//     public KeenQueryClientException() {\n//         super();\n//     }\n// \n//     public KeenQueryClientException(Throwable cause) {\n//         super(cause);\n//     }\n// \n//     public KeenQueryClientException(String message) {\n//         super(message);\n//     }\n// \n//     public KeenQueryClientException(String message, Throwable cause) {\n//         super(message, cause);\n//     }\n// }\n// \n// Path: core/src/main/java/io/keen/client/java/http/HttpMethods.java\n// public final class HttpMethods {\n//     private HttpMethods() {}\n// \n//     public final static String GET = \"GET\";\n//     public final static String POST = \"POST\";\n//     public final static String PUT = \"PUT\";\n//     public final static String DELETE = \"DELETE\";\n// }\n\n\n// Path: query/src/main/java/io/keen/client/java/KeenQueryRequest.java\nimport java.net.URL;\nimport java.util.Collection;\nimport java.util.Map;\nimport io.keen.client.java.exceptions.KeenQueryClientException;\nimport io.keen.client.java.http.HttpMethods;\n\npackage io.keen.client.java;\n\n\n\n/**\n * Interface to be implemented by a query request\n *\n * @author baumatron\n */\nabstract class KeenQueryRequest {\n    abstract URL getRequestURL(RequestUrlBuilder urlBuilder, String projectId)\n            throws KeenQueryClientException;\n\n    // By default, we POST to get most of our query results.\n    String getHttpMethod() {",
        "next_line": "        return HttpMethods.POST;",
        "prompt_tokens": 379,
        "tag": "java_cff"
    }
    code2 = construct_task_prompt(data, language, prompt_mode) 
    print(code2)

def example2():
    language="python"
    prompt_mode = "split"
    data = {
        "repo_name": "leapcode/leap_mx",
        "file_path": "src/leap/mx/vendor/pgpy/packet/packets.py",
        "context": [
            {
                "path": "src/leap/mx/vendor/pgpy/packet/fields.py",
                "identifier": "ECDSAPriv",
                "snippet": "class ECDSAPriv(PrivKey, ECDSAPub):\n    __privfields__ = ('s', )\n\n    def __privkey__(self):\n        ecp = ec.EllipticCurvePublicNumbers(self.x, self.y, self.oid.curve())\n        return ec.EllipticCurvePrivateNumbers(self.s, ecp).private_key(default_backend())\n\n    def _generate(self, oid):\n        if any(c != 0 for c in self):  # pragma: no cover\n            raise PGPError(\"Key is already populated!\")\n\n        self.oid = EllipticCurveOID(oid)\n\n        if not self.oid.can_gen:\n            raise ValueError(\"Curve not currently supported: {}\".format(oid.name))\n\n        pk = ec.generate_private_key(self.oid.curve(), default_backend())\n        pubn = pk.public_key().public_numbers()\n        self.x = MPI(pubn.x)\n        self.y = MPI(pubn.y)\n        self.s = MPI(pk.private_numbers().private_value)\n\n    def parse(self, packet):\n        super(ECDSAPriv, self).parse(packet)\n        self.s2k.parse(packet)\n\n        if not self.s2k:\n            self.s = MPI(packet)\n\n            if self.s2k.usage == 0:\n                self.chksum = packet[:2]\n                del packet[:2]\n\n        else:\n            ##TODO: this needs to be bounded to the length of the encrypted key material\n            self.encbytes = packet\n\n    def decrypt_keyblob(self, passphrase):\n        kb = super(ECDSAPriv, self).decrypt_keyblob(passphrase)\n        del passphrase\n\n        self.s = MPI(kb)\n\n    def sign(self, sigdata, hash_alg):\n        signer = self.__privkey__().signer(ec.ECDSA(hash_alg))\n        signer.update(sigdata)\n        return signer.finalize()",
                "retrieval_score": 0.8085654973983765
            },
            {
                "path": "src/leap/mx/vendor/pgpy/packet/fields.py",
                "identifier": "RSAPub",
                "snippet": "class RSAPub(PubKey):\n    __pubfields__ = ('n', 'e')\n\n    def __pubkey__(self):\n        return rsa.RSAPublicNumbers(self.e, self.n).public_key(default_backend())\n\n    def verify(self, subj, sigbytes, hash_alg):\n        # zero-pad sigbytes if necessary\n        sigbytes = (b'\\x00' * (self.n.byte_length() - len(sigbytes))) + sigbytes\n        verifier = self.__pubkey__().verifier(sigbytes, padding.PKCS1v15(), hash_alg)\n        verifier.update(subj)\n\n        try:\n            verifier.verify()\n\n        except InvalidSignature:\n            return False\n\n        return True\n\n    def parse(self, packet):\n        self.n = MPI(packet)\n        self.e = MPI(packet)",
                "retrieval_score": 0.6123365163803101
            }
        ],
        "import_statement": "import abc\nimport binascii\nimport calendar\nimport copy\nimport hashlib\nimport os\nimport re\nimport six\nfrom datetime import datetime\nfrom cryptography.hazmat.primitives import constant_time\nfrom cryptography.hazmat.primitives.asymmetric import padding\nfrom .fields import DSAPriv, DSAPub, DSASignature\nfrom .fields import ECDSAPub, ECDSAPriv, ECDSASignature\nfrom .fields import ECDHPub, ECDHPriv, ECDHCipherText\nfrom .fields import ElGCipherText, ElGPriv, ElGPub\nfrom .fields import OpaquePubKey\nfrom .fields import OpaquePrivKey\nfrom .fields import RSACipherText, RSAPriv, RSAPub, RSASignature\nfrom .fields import String2Key\nfrom .fields import SubPackets\nfrom .fields import UserAttributeSubPackets\nfrom .types import Packet\nfrom .types import Primary\nfrom .types import Private\nfrom .types import Public\nfrom .types import Sub\nfrom .types import VersionedPacket\nfrom ..constants import CompressionAlgorithm\nfrom ..constants import HashAlgorithm\nfrom ..constants import PubKeyAlgorithm\nfrom ..constants import SignatureType\nfrom ..constants import SymmetricKeyAlgorithm\nfrom ..constants import TrustFlags\nfrom ..constants import TrustLevel\nfrom ..decorators import sdproperty\nfrom ..errors import PGPDecryptionError\nfrom ..symenc import _decrypt\nfrom ..symenc import _encrypt\nfrom ..types import Fingerprint",
        "code": "    def pubalg(self):\n        return self._pubalg\n\n    @pubalg.register(int)\n    @pubalg.register(PubKeyAlgorithm)\n    def pubalg_int(self, val):\n        self._pubalg = PubKeyAlgorithm(val)\n        if self._pubalg in [PubKeyAlgorithm.RSAEncryptOrSign, PubKeyAlgorithm.RSAEncrypt, PubKeyAlgorithm.RSASign]:\n            self.signature = RSASignature()\n\n        elif self._pubalg == PubKeyAlgorithm.DSA:\n            self.signature = DSASignature()\n\n    @sdproperty\n    def halg(self):\n        return self._halg\n\n    @halg.register(int)\n    @halg.register(HashAlgorithm)\n    def halg_int(self, val):\n        try:\n            self._halg = HashAlgorithm(val)\n\n        except ValueError:  # pragma: no cover\n            self._halg = val\n\n    @sdproperty\n    def signer(self):\n        return self._signer\n\n    @signer.register(str)\n    @signer.register(six.text_type)\n    def signer_str(self, val):\n        self._signer = val\n\n    @signer.register(bytearray)\n    def signer_bin(self, val):\n        self._signer = binascii.hexlify(val).upper().decode('latin-1')\n\n    def __init__(self):\n        super(OnePassSignatureV3, self).__init__()\n        self._sigtype = None\n        self._halg = None\n        self._pubalg = None\n        self._signer = b'\\x00' * 8\n        self.nested = False\n\n    def __bytearray__(self):\n        _bytes = bytearray()\n        _bytes += super(OnePassSignatureV3, self).__bytearray__()\n        _bytes += bytearray([self.sigtype])\n        _bytes += bytearray([self.halg])\n        _bytes += bytearray([self.pubalg])\n        _bytes += binascii.unhexlify(six.b(self.signer))\n        _bytes += bytearray([int(self.nested)])\n        return _bytes\n\n    def parse(self, packet):\n        super(OnePassSignatureV3, self).parse(packet)\n        self.sigtype = packet[0]\n        del packet[0]\n\n        self.halg = packet[0]\n        del packet[0]\n\n        self.pubalg = packet[0]\n        del packet[0]\n\n        self.signer = packet[:8]\n        del packet[:8]\n\n        self.nested = (packet[0] == 1)\n        del packet[0]\n\n\nclass PrivKey(VersionedPacket, Primary, Private):\n    __typeid__ = 0x05\n    __ver__ = 0\n\n\nclass PubKey(VersionedPacket, Primary, Public):\n    __typeid__ = 0x06\n    __ver__ = 0\n\n    @abc.abstractproperty\n    def fingerprint(self):\n        \"\"\"compute and return the fingerprint of the key\"\"\"\n\n\nclass PubKeyV4(PubKey):\n    __ver__ = 4\n\n    @sdproperty\n    def created(self):\n        return self._created\n\n    @created.register(datetime)\n    def created_datetime(self, val):\n        self._created = val\n\n    @created.register(int)\n    def created_int(self, val):\n        self.created = datetime.utcfromtimestamp(val)\n\n    @created.register(bytes)\n    @created.register(bytearray)\n    def created_bin(self, val):\n        self.created = self.bytes_to_int(val)\n\n    @sdproperty\n    def pkalg(self):\n        return self._pkalg\n\n    @pkalg.register(int)\n    @pkalg.register(PubKeyAlgorithm)\n    def pkalg_int(self, val):\n        self._pkalg = PubKeyAlgorithm(val)\n\n        _c = {\n            # True means public\n",
        "next_line": "            (True, PubKeyAlgorithm.RSAEncryptOrSign): RSAPub,",
        "gold_snippet_index": 1,
        "origin_context": [
            {
                "path": "src/leap/mx/vendor/pgpy/packet/fields.py",
                "identifier": "ECDSAPub",
                "snippet": "class ECDSAPub(PubKey):\n    __pubfields__ = ('x', 'y')\n\n    def __init__(self):\n        super(ECDSAPub, self).__init__()\n        self.oid = None\n\n    def __len__(self):\n        return sum([len(getattr(self, i)) - 2 for i in self.__pubfields__] +\n                   [3, len(encoder.encode(self.oid.value)) - 1])\n\n    def __pubkey__(self):\n        return ec.EllipticCurvePublicNumbers(self.x, self.y, self.oid.curve()).public_key(default_backend())\n\n    def __bytearray__(self):\n        _b = bytearray()\n        _b += encoder.encode(self.oid.value)[1:]\n        # 0x04 || x || y\n        # where x and y are the same length\n        _xy = b'\\x04' + self.x.to_mpibytes()[2:] + self.y.to_mpibytes()[2:]\n        _b += MPI(self.bytes_to_int(_xy, 'big')).to_mpibytes()\n\n        return _b\n\n    def __copy__(self):\n        pkt = super(ECDSAPub, self).__copy__()\n        pkt.oid = self.oid\n        return pkt\n\n    def verify(self, subj, sigbytes, hash_alg):\n        verifier = self.__pubkey__().verifier(sigbytes, ec.ECDSA(hash_alg))\n        verifier.update(subj)\n\n        try:\n            verifier.verify()\n\n        except InvalidSignature:\n            return False\n\n        return True\n\n    def parse(self, packet):\n        oidlen = packet[0]\n        del packet[0]\n        _oid = bytearray(b'\\x06')\n        _oid.append(oidlen)\n        _oid += bytearray(packet[:oidlen])\n        # try:\n        oid, _  = decoder.decode(bytes(_oid))\n\n        # except:\n        #     raise PGPError(\"Bad OID octet stream: b'{:s}'\".format(''.join(['\\\\x{:02X}'.format(c) for c in _oid])))\n        self.oid = EllipticCurveOID(oid)\n        del packet[:oidlen]\n\n        # flen = (self.oid.bit_length // 8)\n        xy = bytearray(MPI(packet).to_mpibytes()[2:])\n        # xy = bytearray(MPI(packet).to_bytes(flen, 'big'))\n        # the first byte is just \\x04\n        del xy[:1]\n        # now xy needs to be separated into x, y\n        xylen = len(xy)\n        x, y = xy[:xylen // 2], xy[xylen // 2:]\n        self.x = MPI(self.bytes_to_int(x))\n        self.y = MPI(self.bytes_to_int(y))",
                "retrieval_score": 0.8119089603424072
            },
            {
                "path": "src/leap/mx/vendor/pgpy/packet/fields.py",
                "identifier": "ECDSAPriv",
                "snippet": "class ECDSAPriv(PrivKey, ECDSAPub):\n    __privfields__ = ('s', )\n\n    def __privkey__(self):\n        ecp = ec.EllipticCurvePublicNumbers(self.x, self.y, self.oid.curve())\n        return ec.EllipticCurvePrivateNumbers(self.s, ecp).private_key(default_backend())\n\n    def _generate(self, oid):\n        if any(c != 0 for c in self):  # pragma: no cover\n            raise PGPError(\"Key is already populated!\")\n\n        self.oid = EllipticCurveOID(oid)\n\n        if not self.oid.can_gen:\n            raise ValueError(\"Curve not currently supported: {}\".format(oid.name))\n\n        pk = ec.generate_private_key(self.oid.curve(), default_backend())\n        pubn = pk.public_key().public_numbers()\n        self.x = MPI(pubn.x)\n        self.y = MPI(pubn.y)\n        self.s = MPI(pk.private_numbers().private_value)\n\n    def parse(self, packet):\n        super(ECDSAPriv, self).parse(packet)\n        self.s2k.parse(packet)\n\n        if not self.s2k:\n            self.s = MPI(packet)\n\n            if self.s2k.usage == 0:\n                self.chksum = packet[:2]\n                del packet[:2]\n\n        else:\n            ##TODO: this needs to be bounded to the length of the encrypted key material\n            self.encbytes = packet\n\n    def decrypt_keyblob(self, passphrase):\n        kb = super(ECDSAPriv, self).decrypt_keyblob(passphrase)\n        del passphrase\n\n        self.s = MPI(kb)\n\n    def sign(self, sigdata, hash_alg):\n        signer = self.__privkey__().signer(ec.ECDSA(hash_alg))\n        signer.update(sigdata)\n        return signer.finalize()",
                "retrieval_score": 0.8085654973983765
            },
            {
                "path": "src/leap/mx/vendor/pgpy/packet/fields.py",
                "identifier": "DSAPriv",
                "snippet": "class DSAPriv(PrivKey, DSAPub):\n    __privfields__ = ('x',)\n\n    def __privkey__(self):\n        params = dsa.DSAParameterNumbers(self.p, self.q, self.g)\n        pn = dsa.DSAPublicNumbers(self.y, params)\n        return dsa.DSAPrivateNumbers(self.x, pn).private_key(default_backend())\n\n    def _generate(self, key_size):\n        if any(c != 0 for c in self):  # pragma: no cover\n            raise PGPError(\"key is already populated\")\n\n        # generate some big numbers!\n        pk = dsa.generate_private_key(key_size, default_backend())\n        pkn = pk.private_numbers()\n\n        self.p = MPI(pkn.public_numbers.parameter_numbers.p)\n        self.q = MPI(pkn.public_numbers.parameter_numbers.q)\n        self.g = MPI(pkn.public_numbers.parameter_numbers.g)\n        self.y = MPI(pkn.public_numbers.y)\n        self.x = MPI(pkn.x)\n\n        del pkn\n        del pk\n\n        self._compute_chksum()\n\n    def parse(self, packet):\n        super(DSAPriv, self).parse(packet)\n        self.s2k.parse(packet)\n\n        if not self.s2k:\n            self.x = MPI(packet)\n\n        else:\n            self.encbytes = packet\n\n        if self.s2k.usage in [0, 255]:\n            self.chksum = packet[:2]\n            del packet[:2]\n\n    def decrypt_keyblob(self, passphrase):\n        kb = super(DSAPriv, self).decrypt_keyblob(passphrase)\n        del passphrase\n\n        self.x = MPI(kb)\n\n        if self.s2k.usage in [254, 255]:\n            self.chksum = kb\n            del kb\n\n    def sign(self, sigdata, hash_alg):\n        signer = self.__privkey__().signer(hash_alg)\n        signer.update(sigdata)\n        return signer.finalize()",
                "retrieval_score": 0.7741719484329224
            },
            {
                "path": "src/leap/mx/vendor/pgpy/packet/fields.py",
                "identifier": "ECDHPub",
                "snippet": "class ECDHPub(PubKey):\n    __pubfields__ = ('x', 'y')\n\n    def __init__(self):\n        super(ECDHPub, self).__init__()\n        self.oid = None\n        self.kdf = ECKDF()\n\n    def __len__(self):\n        return sum([len(getattr(self, i)) - 2 for i in self.__pubfields__] +\n                   [3,\n                    len(self.kdf),\n                    len(encoder.encode(self.oid.value)) - 1])\n\n    def __pubkey__(self):\n        return ec.EllipticCurvePublicNumbers(self.x, self.y, self.oid.curve()).public_key(default_backend())\n\n    def __bytearray__(self):\n        _b = bytearray()\n        _b += encoder.encode(self.oid.value)[1:]\n        # 0x04 || x || y\n        # where x and y are the same length\n        _xy = b'\\x04' + self.x.to_mpibytes()[2:] + self.y.to_mpibytes()[2:]\n        _b += MPI(self.bytes_to_int(_xy, 'big')).to_mpibytes()\n        _b += self.kdf.__bytearray__()\n\n        return _b\n\n    def __copy__(self):\n        pkt = super(ECDHPub, self).__copy__()\n        pkt.oid = self.oid\n        pkt.kdf = copy.copy(self.kdf)\n        return pkt\n\n    def parse(self, packet):\n        \"\"\"\n        Algorithm-Specific Fields for ECDH keys:\n\n          o  a variable-length field containing a curve OID, formatted\n             as follows:\n\n             -  a one-octet size of the following field; values 0 and\n                0xFF are reserved for future extensions\n\n             -  the octets representing a curve OID, defined in\n                Section 11\n\n             -  MPI of an EC point representing a public key\n\n          o  a variable-length field containing KDF parameters,\n             formatted as follows:\n\n             -  a one-octet size of the following fields; values 0 and\n                0xff are reserved for future extensions\n\n             -  a one-octet value 01, reserved for future extensions\n\n             -  a one-octet hash function ID used with a KDF\n\n             -  a one-octet algorithm ID for the symmetric algorithm\n                used to wrap the symmetric key used for the message\n                encryption; see Section 8 for details\n        \"\"\"\n        oidlen = packet[0]\n        del packet[0]\n        _oid = bytearray(b'\\x06')\n        _oid.append(oidlen)\n        _oid += bytearray(packet[:oidlen])\n        # try:\n        oid, _  = decoder.decode(bytes(_oid))\n\n        # except:\n        #     raise PGPError(\"Bad OID octet stream: b'{:s}'\".format(''.join(['\\\\x{:02X}'.format(c) for c in _oid])))\n        self.oid = EllipticCurveOID(oid)\n        del packet[:oidlen]\n\n        # flen = (self.oid.bit_length // 8)\n        xy = bytearray(MPI(packet).to_mpibytes()[2:])\n        # xy = bytearray(MPI(packet).to_bytes(flen, 'big'))\n        # the first byte is just \\x04\n        del xy[:1]\n        # now xy needs to be separated into x, y\n        xylen = len(xy)\n        x, y = xy[:xylen // 2], xy[xylen // 2:]\n        self.x = MPI(self.bytes_to_int(x))\n        self.y = MPI(self.bytes_to_int(y))\n\n        self.kdf.parse(packet)",
                "retrieval_score": 0.7482081651687622
            },
            {
                "path": "src/leap/mx/vendor/pgpy/packet/fields.py",
                "identifier": "RSAPub",
                "snippet": "class RSAPub(PubKey):\n    __pubfields__ = ('n', 'e')\n\n    def __pubkey__(self):\n        return rsa.RSAPublicNumbers(self.e, self.n).public_key(default_backend())\n\n    def verify(self, subj, sigbytes, hash_alg):\n        # zero-pad sigbytes if necessary\n        sigbytes = (b'\\x00' * (self.n.byte_length() - len(sigbytes))) + sigbytes\n        verifier = self.__pubkey__().verifier(sigbytes, padding.PKCS1v15(), hash_alg)\n        verifier.update(subj)\n\n        try:\n            verifier.verify()\n\n        except InvalidSignature:\n            return False\n\n        return True\n\n    def parse(self, packet):\n        self.n = MPI(packet)\n        self.e = MPI(packet)",
                "retrieval_score": 0.6123365163803101
            }
        ],
        "prompt": "# Repo Name: leapcode/leap_mx\n# Path: src/leap/mx/vendor/pgpy/packet/fields.py\n# class ECDSAPriv(PrivKey, ECDSAPub):\n#     __privfields__ = ('s', )\n# \n#     def __privkey__(self):\n#         ecp = ec.EllipticCurvePublicNumbers(self.x, self.y, self.oid.curve())\n#         return ec.EllipticCurvePrivateNumbers(self.s, ecp).private_key(default_backend())\n# \n#     def _generate(self, oid):\n#         if any(c != 0 for c in self):  # pragma: no cover\n#             raise PGPError(\"Key is already populated!\")\n# \n#         self.oid = EllipticCurveOID(oid)\n# \n#         if not self.oid.can_gen:\n#             raise ValueError(\"Curve not currently supported: {}\".format(oid.name))\n# \n#         pk = ec.generate_private_key(self.oid.curve(), default_backend())\n#         pubn = pk.public_key().public_numbers()\n#         self.x = MPI(pubn.x)\n#         self.y = MPI(pubn.y)\n#         self.s = MPI(pk.private_numbers().private_value)\n# \n#     def parse(self, packet):\n#         super(ECDSAPriv, self).parse(packet)\n#         self.s2k.parse(packet)\n# \n#         if not self.s2k:\n#             self.s = MPI(packet)\n# \n#             if self.s2k.usage == 0:\n#                \n\n# Path: src/leap/mx/vendor/pgpy/packet/fields.py\n# class DSAPriv(PrivKey, DSAPub):\n#     __privfields__ = ('x',)\n# \n#     def __privkey__(self):\n#         params = dsa.DSAParameterNumbers(self.p, self.q, self.g)\n#         pn = dsa.DSAPublicNumbers(self.y, params)\n#         return dsa.DSAPrivateNumbers(self.x, pn).private_key(default_backend())\n# \n#     def _generate(self, key_size):\n#         if any(c != 0 for c in self):  # pragma: no cover\n#             raise PGPError(\"key is already populated\")\n# \n#         # generate some big numbers!\n#         pk = dsa.generate_private_key(key_size, default_backend())\n#         pkn = pk.private_numbers()\n# \n#         self.p = MPI(pkn.public_numbers.parameter_numbers.p)\n#         self.q = MPI(pkn.public_numbers.parameter_numbers.q)\n#         self.g = MPI(pkn.public_numbers.parameter_numbers.g)\n#         self.y = MPI(pkn.public_numbers.y)\n#         self.x = MPI(pkn.x)\n# \n#         del pkn\n#         del pk\n# \n#         self._compute_chksum()\n# \n#     def parse(self, packet):\n#         super(DSAPriv, self).parse(packet)\n#         self.s2k.parse(packet)\n# \n#         if not self.s2k:\n\n\n# Path: src/leap/mx/vendor/pgpy/packet/fields.py\n# class RSAPub(PubKey):\n#     __pubfields__ = ('n', 'e')\n# \n#     def __pubkey__(self):\n#         return rsa.RSAPublicNumbers(self.e, self.n).public_key(default_backend())\n# \n#     def verify(self, subj, sigbytes, hash_alg):\n#         # zero-pad sigbytes if necessary\n#         sigbytes = (b'\\x00' * (self.n.byte_length() - len(sigbytes))) + sigbytes\n#         verifier = self.__pubkey__().verifier(sigbytes, padding.PKCS1v15(), hash_alg)\n#         verifier.update(subj)\n# \n#         try:\n#             verifier.verify()\n# \n#         except InvalidSignature:\n#             return False\n# \n#         return True\n# \n#     def parse(self, packet):\n#         self.n = MPI(packet)\n#         self.e = MPI(packet)\n\n# Path: src/leap/mx/vendor/pgpy/packet/packets.py\nimport abc\nimport binascii\nimport calendar\nimport copy\nimport hashlib\nimport os\nimport re\nimport six\nfrom datetime import datetime\nfrom cryptography.hazmat.primitives import constant_time\nfrom cryptography.hazmat.primitives.asymmetric import padding\nfrom .fields import DSAPriv, DSAPub, DSASignature\nfrom .fields import ECDSAPub, ECDSAPriv, ECDSASignature\nfrom .fields import ECDHPub, ECDHPriv, ECDHCipherText\nfrom .fields import ElGCipherText, ElGPriv, ElGPub\nfrom .fields import OpaquePubKey\nfrom .fields import OpaquePrivKey\nfrom .fields import RSACipherText, RSAPriv, RSAPub, RSASignature\nfrom .fields import String2Key\nfrom .fields import SubPackets\nfrom .fields import UserAttributeSubPackets\nfrom .types import Packet\nfrom .types import Primary\nfrom .types import Private\nfrom .types import Public\nfrom .types import Sub\nfrom .types import VersionedPacket\nfrom ..constants import CompressionAlgorithm\nfrom ..constants import HashAlgorithm\nfrom ..constants import PubKeyAlgorithm\nfrom ..constants import SignatureType\nfrom ..constants import SymmetricKeyAlgorithm\nfrom ..constants import TrustFlags\nfrom ..constants import TrustLevel\nfrom ..decorators import sdproperty\nfrom ..errors import PGPDecryptionError\nfrom ..symenc import _decrypt\nfrom ..symenc import _encrypt\nfrom ..types import Fingerprint\n    def pubalg(self):\n        return self._pubalg\n\n    @pubalg.register(int)\n    @pubalg.register(PubKeyAlgorithm)\n    def pubalg_int(self, val):\n        self._pubalg = PubKeyAlgorithm(val)\n        if self._pubalg in [PubKeyAlgorithm.RSAEncryptOrSign, PubKeyAlgorithm.RSAEncrypt, PubKeyAlgorithm.RSASign]:\n            self.signature = RSASignature()\n\n        elif self._pubalg == PubKeyAlgorithm.DSA:\n            self.signature = DSASignature()\n\n    @sdproperty\n    def halg(self):\n        return self._halg\n\n    @halg.register(int)\n    @halg.register(HashAlgorithm)\n    def halg_int(self, val):\n        try:\n            self._halg = HashAlgorithm(val)\n\n        except ValueError:  # pragma: no cover\n            self._halg = val\n\n    @sdproperty\n    def signer(self):\n        return self._signer\n\n    @signer.register(str)\n    @signer.register(six.text_type)\n    def signer_str(self, val):\n        self._signer = val\n\n    @signer.register(bytearray)\n    def signer_bin(self, val):\n        self._signer = binascii.hexlify(val).upper().decode('latin-1')\n\n    def __init__(self):\n        super(OnePassSignatureV3, self).__init__()\n        self._sigtype = None\n        self._halg = None\n        self._pubalg = None\n        self._signer = b'\\x00' * 8\n        self.nested = False\n\n    def __bytearray__(self):\n        _bytes = bytearray()\n        _bytes += super(OnePassSignatureV3, self).__bytearray__()\n        _bytes += bytearray([self.sigtype])\n        _bytes += bytearray([self.halg])\n        _bytes += bytearray([self.pubalg])\n        _bytes += binascii.unhexlify(six.b(self.signer))\n        _bytes += bytearray([int(self.nested)])\n        return _bytes\n\n    def parse(self, packet):\n        super(OnePassSignatureV3, self).parse(packet)\n        self.sigtype = packet[0]\n        del packet[0]\n\n        self.halg = packet[0]\n        del packet[0]\n\n        self.pubalg = packet[0]\n        del packet[0]\n\n        self.signer = packet[:8]\n        del packet[:8]\n\n        self.nested = (packet[0] == 1)\n        del packet[0]\n\n\nclass PrivKey(VersionedPacket, Primary, Private):\n    __typeid__ = 0x05\n    __ver__ = 0\n\n\nclass PubKey(VersionedPacket, Primary, Public):\n    __typeid__ = 0x06\n    __ver__ = 0\n\n    @abc.abstractproperty\n    def fingerprint(self):\n        \"\"\"compute and return the fingerprint of the key\"\"\"\n\n\nclass PubKeyV4(PubKey):\n    __ver__ = 4\n\n    @sdproperty\n    def created(self):\n        return self._created\n\n    @created.register(datetime)\n    def created_datetime(self, val):\n        self._created = val\n\n    @created.register(int)\n    def created_int(self, val):\n        self.created = datetime.utcfromtimestamp(val)\n\n    @created.register(bytes)\n    @created.register(bytearray)\n    def created_bin(self, val):\n        self.created = self.bytes_to_int(val)\n\n    @sdproperty\n    def pkalg(self):\n        return self._pkalg\n\n    @pkalg.register(int)\n    @pkalg.register(PubKeyAlgorithm)\n    def pkalg_int(self, val):\n        self._pkalg = PubKeyAlgorithm(val)\n\n        _c = {\n            # True means public\n\n",
        "prompt_tokens": 1998,
        "tag": "python_cff"
    } 
    #{
    #     "repo_name": "jaywink/federation",
    #     "file_path": "federation/tests/entities/diaspora/test_utils.py",
    #     "context": "# Path: federation/entities/base.py\n# class Post(RawContentMixin, PublicMixin, CreatedAtMixin, ProviderDisplayNameMixin, BaseEntity):\n#     \"\"\"Reflects a post, status message, etc, which will be composed from the message or to the message.\"\"\"\n#     location = \"\"\n#     url = \"\"\n# \n#     _allowed_children = (Image,)\n#     _default_activity = ActivityType.CREATE\n# \n# class Profile(CreatedAtMixin, OptionalRawContentMixin, PublicMixin, BaseEntity):\n#     \"\"\"Represents a profile for a user.\"\"\"\n#     atom_url = \"\"\n#     email = \"\"\n#     gender = \"\"\n#     image_urls = None\n#     location = \"\"\n#     name = \"\"\n#     nsfw = False\n#     public_key = \"\"\n#     tag_list = None\n#     url = \"\"\n#     username = \"\"\n#     inboxes: Dict = None\n# \n#     _allowed_children = (Image,)\n# \n#     def __init__(self, *args, **kwargs):\n#         self.image_urls = {\n#             \"small\": \"\", \"medium\": \"\", \"large\": \"\"\n#         }\n#         self.inboxes = {\n#             \"private\": None,\n#             \"public\": None,\n#         }\n#         self.tag_list = []\n#         super().__init__(*args, **kwargs)\n#         # As an exception, a Profile does not require to have an `actor_id`\n#         self._required.remove('actor_id')\n# \n#     def validate_email(self):\n#         if self.email:\n#             validator = Email()\n#             if not validator.is_valid(self.email):\n#                 raise ValueError(\"Email is not valid\")\n# \n# Path: federation/entities/diaspora/entities.py\n# class DiasporaPost(DiasporaEntityMixin, Post):\n#     \"\"\"Diaspora post, ie status message.\"\"\"\n#     _tag_name = \"status_message\"\n# \n#     def to_xml(self):\n#         \"\"\"Convert to XML message.\"\"\"\n#         element = etree.Element(self._tag_name)\n#         properties = [\n#             {\"text\": self.raw_content},\n#             {\"guid\": self.guid},\n#             {\"author\": self.handle},\n#             {\"public\": \"true\" if self.public else \"false\"},\n#             {\"created_at\": format_dt(self.created_at)},\n#             {\"provider_display_name\": self.provider_display_name},\n#         ]\n#         if self.id and self.id.startswith(\"http\"):\n#             properties.append({\n#                 \"activitypub_id\": self.id,\n#             })\n#         struct_to_xml(element, properties)\n#         return element\n# \n# Path: federation/entities/diaspora/utils.py\n# def get_full_xml_representation(entity, private_key):\n#     \"\"\"Get full XML representation of an entity.\n# \n#     This contains the <XML><post>..</post></XML> wrapper.\n# \n#     Accepts either a Base entity or a Diaspora entity.\n# \n#     Author `private_key` must be given so that certain entities can be signed.\n#     \"\"\"\n#     from federation.entities.diaspora.mappers import get_outbound_entity\n#     diaspora_entity = get_outbound_entity(entity, private_key)\n#     xml = diaspora_entity.to_xml()\n#     return \"<XML><post>%s</post></XML>\" % etree.tostring(xml).decode(\"utf-8\")\n# \n# def format_dt(dt):\n#     \"\"\"\n#     Format a datetime in the way that D* nodes expect.\n#     \"\"\"\n#     return ensure_timezone(dt).astimezone(tzutc()).strftime(\n#         '%Y-%m-%dT%H:%M:%SZ'\n#     )\n# \n# def add_element_to_doc(doc, tag, value):\n#     \"\"\"Set text value of an etree.Element of tag, appending a new element with given tag if it doesn't exist.\"\"\"\n#     element = doc.find(\".//%s\" % tag)\n#     if element is None:\n#         element = etree.SubElement(doc, tag)\n#     element.text = value\n# \n# Path: federation/entities/utils.py\n# def get_base_attributes(entity):\n#     \"\"\"Build a dict of attributes of an entity.\n# \n#     Returns attributes and their values, ignoring any properties, functions and anything that starts\n#     with an underscore.\n#     \"\"\"\n#     attributes = {}\n#     cls = entity.__class__\n#     for attr, _ in inspect.getmembers(cls, lambda o: not isinstance(o, property) and not inspect.isroutine(o)):\n#         if not attr.startswith(\"_\"):\n#             attributes[attr] = getattr(entity, attr)\n#     return attributes\n\n",
    #     "import_statement": "import datetime\nimport re\nimport arrow\nfrom unittest.mock import patch, Mock\nfrom lxml import etree\nfrom federation.entities.base import Post, Profile\nfrom federation.entities.diaspora.entities import DiasporaPost\nfrom federation.entities.diaspora.utils import (\n    get_full_xml_representation, format_dt, add_element_to_doc)\nfrom federation.entities.utils import get_base_attributes",
    #     "code": "        attrs = get_base_attributes(entity).keys()\n        assert set(attrs) == {\n            \"created_at\", \"name\", \"email\", \"gender\", \"raw_content\", \"location\", \"public\",\n            \"nsfw\", \"public_key\", \"image_urls\", \"tag_list\", \"signature\", \"url\", \"atom_url\",\n            \"base_url\", \"id\", \"actor_id\", \"handle\", \"handle\", \"guid\", \"activity\", \"activity_id\", \"username\",\n            \"inboxes\", \"mxid\",\n        }\n\n\nclass TestGetFullXMLRepresentation:\n    @patch.object(DiasporaPost, \"validate\", new=Mock())\n    def test_returns_xml_document(self):\n        entity = Post()\n        document = get_full_xml_representation(entity, \"\")\n        document = re.sub(r\"<created_at>.*</created_at>\", \"\", document)  # Dates are annoying to compare\n        assert document == \"<XML><post><status_message><text></text><guid></guid>\" \\\n                           \"<author></author><public>false</public>\" \\\n                           \"<provider_display_name></provider_display_name></status_message></post></XML>\"\n\n\nclass TestFormatDt:\n    def test_formatted_string_returned_from_tz_aware_datetime(self):\n        dt = arrow.get(datetime.datetime(2017, 1, 28, 3, 2, 3), \"Europe/Helsinki\").datetime\n        assert format_dt(dt) == \"2017-01-28T01:02:03Z\"\n\n\ndef test_add_element_to_doc():\n    # Replacing value\n    doc = etree.fromstring(\"<comment><text>foobar</text><parent_author_signature>barfoo</parent_author_signature>\"\n                           \"</comment>\")\n",
    #     "prompt": "# Path: federation/entities/base.py\n# class Post(RawContentMixin, PublicMixin, CreatedAtMixin, ProviderDisplayNameMixin, BaseEntity):\n#     \"\"\"Reflects a post, status message, etc, which will be composed from the message or to the message.\"\"\"\n#     location = \"\"\n#     url = \"\"\n# \n#     _allowed_children = (Image,)\n#     _default_activity = ActivityType.CREATE\n# \n# class Profile(CreatedAtMixin, OptionalRawContentMixin, PublicMixin, BaseEntity):\n#     \"\"\"Represents a profile for a user.\"\"\"\n#     atom_url = \"\"\n#     email = \"\"\n#     gender = \"\"\n#     image_urls = None\n#     location = \"\"\n#     name = \"\"\n#     nsfw = False\n#     public_key = \"\"\n#     tag_list = None\n#     url = \"\"\n#     username = \"\"\n#     inboxes: Dict = None\n# \n#     _allowed_children = (Image,)\n# \n#     def __init__(self, *args, **kwargs):\n#         self.image_urls = {\n#             \"small\": \"\", \"medium\": \"\", \"large\": \"\"\n#         }\n#         self.inboxes = {\n#             \"private\": None,\n#             \"public\": None,\n#         }\n#         self.tag_list = []\n#         super().__init__(*args, **kwargs)\n#         # As an exception, a Profile does not require to have an `actor_id`\n#         self._required.remove('actor_id')\n# \n#     def validate_email(self):\n#         if self.email:\n#             validator = Email()\n#             if not validator.is_valid(self.email):\n#                 raise ValueError(\"Email is not valid\")\n# \n# Path: federation/entities/diaspora/entities.py\n# class DiasporaPost(DiasporaEntityMixin, Post):\n#     \"\"\"Diaspora post, ie status message.\"\"\"\n#     _tag_name = \"status_message\"\n# \n#     def to_xml(self):\n#         \"\"\"Convert to XML message.\"\"\"\n#         element = etree.Element(self._tag_name)\n#         properties = [\n#             {\"text\": self.raw_content},\n#             {\"guid\": self.guid},\n#             {\"author\": self.handle},\n#             {\"public\": \"true\" if self.public else \"false\"},\n#             {\"created_at\": format_dt(self.created_at)},\n#             {\"provider_display_name\": self.provider_display_name},\n#         ]\n#         if self.id and self.id.startswith(\"http\"):\n#             properties.append({\n#                 \"activitypub_id\": self.id,\n#             })\n#         struct_to_xml(element, properties)\n#         return element\n# \n# Path: federation/entities/diaspora/utils.py\n# def get_full_xml_representation(entity, private_key):\n#     \"\"\"Get full XML representation of an entity.\n# \n#     This contains the <XML><post>..</post></XML> wrapper.\n# \n#     Accepts either a Base entity or a Diaspora entity.\n# \n#     Author `private_key` must be given so that certain entities can be signed.\n#     \"\"\"\n#     from federation.entities.diaspora.mappers import get_outbound_entity\n#     diaspora_entity = get_outbound_entity(entity, private_key)\n#     xml = diaspora_entity.to_xml()\n#     return \"<XML><post>%s</post></XML>\" % etree.tostring(xml).decode(\"utf-8\")\n# \n# def format_dt(dt):\n#     \"\"\"\n#     Format a datetime in the way that D* nodes expect.\n#     \"\"\"\n#     return ensure_timezone(dt).astimezone(tzutc()).strftime(\n#         '%Y-%m-%dT%H:%M:%SZ'\n#     )\n# \n# def add_element_to_doc(doc, tag, value):\n#     \"\"\"Set text value of an etree.Element of tag, appending a new element with given tag if it doesn't exist.\"\"\"\n#     element = doc.find(\".//%s\" % tag)\n#     if element is None:\n#         element = etree.SubElement(doc, tag)\n#     element.text = value\n# \n# Path: federation/entities/utils.py\n# def get_base_attributes(entity):\n#     \"\"\"Build a dict of attributes of an entity.\n# \n#     Returns attributes and their values, ignoring any properties, functions and anything that starts\n#     with an underscore.\n#     \"\"\"\n#     attributes = {}\n#     cls = entity.__class__\n#     for attr, _ in inspect.getmembers(cls, lambda o: not isinstance(o, property) and not inspect.isroutine(o)):\n#         if not attr.startswith(\"_\"):\n#             attributes[attr] = getattr(entity, attr)\n#     return attributes\n\n\n# Path: federation/tests/entities/diaspora/test_utils.py\nimport datetime\nimport re\nimport arrow\nfrom unittest.mock import patch, Mock\nfrom lxml import etree\nfrom federation.entities.base import Post, Profile\nfrom federation.entities.diaspora.entities import DiasporaPost\nfrom federation.entities.diaspora.utils import (\n    get_full_xml_representation, format_dt, add_element_to_doc)\nfrom federation.entities.utils import get_base_attributes\n\n        attrs = get_base_attributes(entity).keys()\n        assert set(attrs) == {\n            \"created_at\", \"name\", \"email\", \"gender\", \"raw_content\", \"location\", \"public\",\n            \"nsfw\", \"public_key\", \"image_urls\", \"tag_list\", \"signature\", \"url\", \"atom_url\",\n            \"base_url\", \"id\", \"actor_id\", \"handle\", \"handle\", \"guid\", \"activity\", \"activity_id\", \"username\",\n            \"inboxes\", \"mxid\",\n        }\n\n\nclass TestGetFullXMLRepresentation:\n    @patch.object(DiasporaPost, \"validate\", new=Mock())\n    def test_returns_xml_document(self):\n        entity = Post()\n        document = get_full_xml_representation(entity, \"\")\n        document = re.sub(r\"<created_at>.*</created_at>\", \"\", document)  # Dates are annoying to compare\n        assert document == \"<XML><post><status_message><text></text><guid></guid>\" \\\n                           \"<author></author><public>false</public>\" \\\n                           \"<provider_display_name></provider_display_name></status_message></post></XML>\"\n\n\nclass TestFormatDt:\n    def test_formatted_string_returned_from_tz_aware_datetime(self):\n        dt = arrow.get(datetime.datetime(2017, 1, 28, 3, 2, 3), \"Europe/Helsinki\").datetime\n        assert format_dt(dt) == \"2017-01-28T01:02:03Z\"\n\n\ndef test_add_element_to_doc():\n    # Replacing value\n    doc = etree.fromstring(\"<comment><text>foobar</text><parent_author_signature>barfoo</parent_author_signature>\"\n                           \"</comment>\")\n",
    #     "next_line": "    add_element_to_doc(doc, \"parent_author_signature\", \"newsig\")",
    #     "prompt_tokens": 1415,
    #     "tag": "python_cff"
    # }
    code2 = construct_task_prompt(data, language, prompt_mode) 
    print(code2)



if __name__ == "__main__":
    example2()

