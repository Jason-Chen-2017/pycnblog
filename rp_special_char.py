import os

import re


def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()

    lines = content.split('\n')
    newLines = []
    # 遍历每一行过滤特殊字符
    for i in range(len(lines)):
        line = lines[i]
        # 去除文本行尾部的空白字符
        line = line.rstrip()

        # 替换特殊字符
        # 替换 \" 为 "
        line = line.replace('\\"', '"')
        # 替换 \\ 为 \  ，但是本行末尾的 \\ 不替换
        line = re.sub(r'\\\\(?!$)', r'\\', line)
        # 替换 \\[ 为 $$
        line = line.replace('\\[', '$$')
        # 替换 \\] 为 $$
        line = line.replace('\\]', '$$')

        newLines.append(line)

    newContent = '\n'.join(newLines)

    with open(filepath, 'w', encoding='utf-8') as file:
        file.write(newContent)


def process_directory(directory):
    for root, dirs, files in os.walk(directory):

        for file in files:
            filepath = os.path.join(root, file)

            process_file(filepath)
            print(filepath)


if __name__ == "__main__":
    # 替换为你要处理的目录路径
    directory_path = "/home/me/tools/pycnblog/blog"

    process_directory(directory_path)
