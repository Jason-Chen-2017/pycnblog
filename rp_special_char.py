import os

import re


def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()

    # 特殊关键字，不进行替换的部分
    special_keywords = ["\\\\nebula", "\\\\nabla", "\\\\neq", "\\\\not", "\\\\neg"]

    # 替换 \\n 为 \n，但保留特殊关键字

    def replace_newline(match):
        if match.group(0) in special_keywords:
            return match.group(0)

        return match.group(0).replace("\\n", "\n")

    content = re.sub(r'\\\\n|\\\\nebula|\\\\nabla|\\\\neq|\\\\not|\\\\neg', replace_newline, content)

    # 替换 \" 为 "
    content = content.replace('\\"', '"')
    # 替换 \\ 为 \
    content = content.replace('\\\\', '\\')
    # 替换 \\[ 为 $$
    content = content.replace('\\[', '$$')
    # 替换 \\] 为 $$
    content = content.replace('\\]', '$$')

    with open(filepath, 'w', encoding='utf-8') as file:
        file.write(content)


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
