def comment_code(language, code):
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
        return "Unsupported language for commenting."

if __name__ == "__main__":
    # 测试函数
    code_sample = """def add(a, b):
        return a + b"""

    print(comment_code('python', code_sample))