import os
import re


def remove_think_tags(content):
    # 使用正则表达式替换 <think> 标签及其内容
    # (?s) 启用 DOTALL 模式,使 . 可以匹配换行符
    pattern = r'(?s)<think>.*?</think>'
    return re.sub(pattern, '', content)


def process_files(folder_path, file_extensions=None):
    if file_extensions is None:
        file_extensions = ['.md']  # 默认只处理 .md 文件
    # 遍历文件夹中的所有文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 检查文件扩展名
            if any(file.endswith(ext) for ext in file_extensions):
                file_path = os.path.join(root, file)
                try:
                    # 读取文件内容
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    # 处理内容
                    new_content = remove_think_tags(content)
                    # 如果内容有变化，则写回文件
                    if new_content != content:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        print(f"已处理文件: {file_path}")

                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {str(e)}")


# 使用示例
if __name__ == "__main__":
    # 替换为你要处理的文件夹路径
    folder_path = "/home/me/tools/pycnblog/blog/deepseek"
    process_files(folder_path)
