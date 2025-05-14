import re

def extract_first_code_block(text):
    """
    提取文本中第一个代码块（由 ''' 包围的内容），
    即使代码块不完整，直到下一个 ''' 或文本结尾。
    
    :param text: 输入文本
    :return: 第一个代码块的内容，如果没有代码块则返回 None
    """
    # 正则表达式：匹配从第一个 ''' 开始，直到下一个 ''' 或文本结尾
    pattern = r"```(.*?)(?=```|$)"
    
    # 使用 re.DOTALL 使得 . 可以匹配换行符
    matches = re.findall(pattern, text, flags=re.DOTALL)
    
    # 取第一个匹配项（如果有）
    return matches[0] if matches else text

# 示例文本，包含多个代码块
s = "print('Hello')```\nprint('World')```\nprint"
# 调用函数，提取第一个代码块内容
first_code_block = extract_first_code_block(s)

# 输出提取到的第一个代码块
print(repr(first_code_block))
