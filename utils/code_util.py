import re
def comment_out(code, language):
    """
    给定编程语言和代码片段，将代码注释掉。
    
    :param language: 编程语言类型，如 "python", "java", "c++" 等
    :param code: 要注释的代码字符串
    :return: 注释后的代码
    """
    # 注释不同语言的代码
    if language.lower() == 'python':
        # Python 注释以 "#" 开头
        return '\n'.join([f'# {line}' for line in code.split('\n')])
    
    elif language.lower() == 'java' or language.lower() == 'c++':
        # Java 和 C++ 注释以 "//" 开头
        return '\n'.join([f'// {line}' for line in code.split('\n')])
    else:
        # 抛出异常并输出不支持的语言名称
        raise ValueError(f"Unsupported language: {language}")



def remove_language_tag(content, language): 
    # Make language case-insensitive for comparison
    language = language.lower()

    # Remove the language part from the start of the content
    if content.lower().startswith(language):
        # Remove the language and the ''' from the start
        content = content[len(language):]
    
    return content # Return the original string if no code blocks are found

def extract_content(s, language):
    try:
        # 1.提取<answer> 和 </answer> 之间的内容
        match = re.search(r'<answer>(.*?)</answer>', s, flags=re.DOTALL)
        if match:
            s = match.group(1)
        else:
            s = s.split('<answer>', 1)[1] if '<answer>' in s else s
            s = s.split('</answer>', 1)[0] if '</answer>' in s else s

        # 2. 提取所有代码块（```）并处理
        matches = re.findall(r"```(.*?)(?=```|$)", s, flags=re.DOTALL)
        s =  matches[0] if matches else s
        s = remove_language_tag(s, language)
        return s
    
    except Exception as e:
        print(f"verify failed: {e}, prediction: {s}")
        return s
    

if __name__ == "__main__":
    # 测试函数
    # code_sample = """def add(a, b):
    #     return a + b"""
    # print(comment_out(code_sample, 'python'))
    slist = [
        "hello...<answer>    \nprint(\"hhhh)</answer>  sss",
        "<answer>    print('Hello')</answer>",
        "   This is a test for \n</answer>print('Hello')",
        "```Python\nprint('Hello')\n\n```\nprint('World')\n```\nprint",
        "next <answer> ```  print('Hello')\n```\nprint('World')```\nprint</answer>  \n"
    ]
    for s in slist:
        result = extract_content(s, "python")
        print(repr(result))
   