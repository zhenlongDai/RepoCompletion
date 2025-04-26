import os
import json
import hashlib
import pandas as pd
import numpy as np


class FileAlreadyExistsError(Exception):
    """自定义异常：文件已存在"""
    pass

def ensure_directory(file_path):
    """
    检查文件路径的目录是否存在，如果不存在则创建。
    
    :param file_path: 文件的完整路径
    """
    # 获取目录部分
    directory = os.path.dirname(file_path)
    
    # 如果目录不存在，则创建目录
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def save_list_to_parquet(data_list, file_path):
    """
    将列表对象按行存储到一个 Parquet 文件中。

    :param data_list: 要存储的列表对象
    :param file_path: Parquet 文件路径
    """
    # 将列表转为 DataFrame，每个元素作为一行
    df = pd.DataFrame(data_list)
    # 保存到 Parquet 文件
    df.to_parquet(file_path, engine='pyarrow')

def read_parquet_to_df(file_path):
    """
    从 Parquet 文件读取数据到列表中。

    :param file_path: Parquet 文件路径
    :return: 从文件中读取的DataFrame
    """
    # 读取 Parquet 文件为 DataFrame
    df = pd.read_parquet(file_path, engine='pyarrow')
    # 将 DataFrame 转为列表
    return df

def read_parquet_to_list(file_path):
    """
    从 Parquet 文件读取数据到列表中。

    :param file_path: Parquet 文件路径
    :return: 从文件中读取的列表
    """
    # 读取 Parquet 文件为 DataFrame
    df = pd.read_parquet(file_path, engine='pyarrow')
    # 将 DataFrame 转为列表
    return df.to_dict(orient='records')


def get_all_file_names(directory_path):
    """
    获取指定路径下的所有文件名字

    :param directory_path: 目录的路径
    :return: 文件名字列表
    """
    try:
        # 列出目录下的所有文件和文件夹
        entries = os.listdir(directory_path)
        # 过滤掉文件夹，只保留文件
        file_names = [entry for entry in entries if os.path.isfile(os.path.join(directory_path, entry))]
        return file_names
    
    except FileNotFoundError:
        print(f"Error: The directory '{directory_path}' does not exist.")
        return []
    except PermissionError:
        print(f"Error: Permission denied for accessing the directory '{directory_path}'.")
        return []
    
def remove_comments(code):
    """
    移除代码中的注释，包括单行注释和多行注释
    :param code: 包含注释的代码字符串
    :return: 移除注释后的代码字符串
    """
    # 定义用于匹配单行和多行注释的正则表达式
    single_line_comment_pattern = r'//.*?$|#.*?$'
    multi_line_comment_pattern = r'/\*.*?\*/|\'\'\'.*?\'\'\'|""".*?"""'
    
    # 组合正则表达式
    pattern = re.compile(
        single_line_comment_pattern + '|' + multi_line_comment_pattern,
        re.DOTALL | re.MULTILINE
    )

    # 使用正则表达式移除注释
    cleaned_code = re.sub(pattern, '', code)
    
    return cleaned_code

def calculate_md5(input_string):
    """
    计算并返回字符串的MD5哈希值

    参数:
    input_string (str): 需要计算MD5哈希值的字符串

    返回:
    str: 输入字符串的MD5哈希值
    """
    # 创建一个md5哈希对象
    md5_hash = hashlib.md5()
    
    # 更新哈希对象并计算哈希值
    md5_hash.update(input_string.encode('utf-8'))
    
    # 返回十六进制哈希值
    return md5_hash.hexdigest()

def check_catalogue_exists(filepath):
    """
    检查指定路径的文件是否存在。

    参数:
    filepath (str): 要检查的文件路径。

    返回:
    bool: 如果文件存在，返回True；否则，返回False。
    """
    return  os.path.exists(filepath)


def check_file_exists(filepath):
    """
    检查指定路径的文件是否存在。

    参数:
    filepath (str): 要检查的文件路径。

    返回:
    bool: 如果文件存在，返回True；否则，返回False。
    """
    return  os.path.isfile(filepath)

def read_python_file(file_path):
    """读取指定路径下的Python文件并返回其内容"""
    with open(file_path, 'r') as file:
        content = file.read()
    return content

def write_file_content_to_json(content, json_path):
    """将内容写入指定路径下的JSON文件"""
    data = {'file_content': content}
    with open(json_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def load_list_from_json(input_file_path):
        """从 JSON 文件读取列表"""
        with open(input_file_path, 'r') as json_file:
            data_list = json.load(json_file)
        return data_list
    
def save_list_to_json(lst, filepath):
    """
    将列表存储到指定路径的JSON文件中。
    
    参数:
    lst (list): 需要存储的列表。
    filepath (str): JSON文件的保存路径。
    """
    # 检查文件是否已存在
    if os.path.exists(filepath):
        raise FileAlreadyExistsError(f"文件 '{filepath}' 已存在。")
    
    ensure_dir(filepath)
    try:
        with open(filepath, 'w', encoding='utf-8') as json_file:
            json.dump(lst, json_file, ensure_ascii=False, indent=4)
        #print(f"列表已成功保存到 {filepath}")
    except Exception as e:
        print(f"保存列表时出错: {e}")

def load_list_from_jsonl(input_file_path):
    """
    读取一个jsonl文件，并返回一个列表，其中每个元素是一个JSON对象。

    :param file_path: jsonl文件的路径
    :return: 包含所有JSON对象的列表
    """
    data_list = []
    with open(input_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 移除可能的空行
            if line.strip():
                # 将每行解析为JSON对象
                json_obj = json.loads(line)
                data_list.append(json_obj)
    return data_list

def save_list_to_jsonl(data_list, file_path):
    """
    将列表保存为 .jsonl 文件。
    
    参数:
    - data_list: 需要保存的列表，每个元素都应是可以序列化为 JSON 的字典。
    - file_path: 保存文件的路径，应该以 .jsonl 结尾。
    """
    ensure_dir(file_path)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data_list:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')
    print(f"成功保存 {len(data_list)} 条记录到文件：{file_path}")    
        
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # 处理 numpy 数据类型
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super(CustomJSONEncoder, self).default(obj)

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory {directory} was created.")
    else:
        print(f"Directory {directory} already exists.")

def save_data_to_json(data, filepath):
    """
    将数据存储到指定路径的JSON文件中，确保特定列表字段不换行，其它字段换行。
    
    参数:
    data (list): 需要存储的列表数据。
    filepath (str): JSON文件的保存路径。
    """
    try:
        # 检查文件是否已存在
        if os.path.exists(filepath):
            raise FileAlreadyExistsError(f"文件 '{filepath}' 已存在。")
        
        ensure_dir(filepath)
        with open(filepath, 'w', encoding='utf-8') as json_file:
            # 使用 json.dumps 将整个数据结构序列化并设置缩进
            json.dump(data, json_file, cls=CustomJSONEncoder, ensure_ascii=False, indent=4)
        print(f"数据已成功保存到 {filepath}")
    except Exception as e:
        print(f"保存数据时出错: {e}")


if __name__ == '__main__':
    print("here")
    
    
   

    