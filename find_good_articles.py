import os
import shutil


def find_good(date):
    global f, content
    # 定义原始目录和目标目录
    source_directory = f'/home/me/tools/pycnblog/articles/{date}'
    target_directory = f'/home/me/tools/pycnblog/articles_good/{date}'
    # 创建目标目录
    os.makedirs(target_directory, exist_ok=True)
    # 遍历原始目录中的所有文件
    for file_name in os.listdir(source_directory):
        # 确认文件是md文件
        if file_name.endswith('.md'):
            # 读取文件内容，统计行数
            file_path = os.path.join(source_directory, file_name)
            with open(file_path, 'r') as f:
                lines = f.readlines()
                content = ''.join(lines)
                # 文章的长度
                length = len(content)
                # 行数
                line_count = len(lines)
            # 如果文章的长度>4000 and 行数大于150，则复制文件到目标目录
            if length > 4000 and line_count > 150 and '结论与展望' in content:
                target_path = os.path.join(target_directory, file_name)
                shutil.copy(file_path, target_path)


dates = [
    # '20230612', '20230613', '20230614', '20230615',
    # '20230616', '20230617', '20230618', '20230619',
    # '20230620', '20230621', '20230622', '20230623',
    '20230624',
    '20230625',
    '20230626',
]

if __name__ == '__main__':
    for d in dates:
        find_good(d)
