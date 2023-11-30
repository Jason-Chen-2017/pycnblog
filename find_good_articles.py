import os
import re
import shutil

from fuzzywuzzy import fuzz

# 定义正则表达式
pattern = r'(.*外链图片转存中.*)|(.*\.png.*)|(.*\(https:\/\/.*)|.*(<img src=.*).*|(.*\(http:\/\/.*)'

similarity_threshold =  80

def check_similarity(text):
    count = 0
    lines = text.split('\n')
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):

            similarity = fuzz.ratio(lines[i], lines[j])

            # print("similarity:", similarity)
            # print("lines[i]:", lines[i])
            # print("lines[j]:", lines[j])

            if similarity > similarity_threshold:  # 设置阈值
                count = count + 1

    return count


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
                cleaned_lines = [line.strip() for line in lines if line.strip()]
                content = '\n'.join(cleaned_lines)
                sim_count = check_similarity(content)

                # 文章的长度
                length = len(content)
                # 行数
                line_count = len(cleaned_lines)

                # print("sim_count:", sim_count, "length:", length, "line_count:", line_count, "\t", file_path)

        # 如果满足条件,复制文件到目标目录
        if length >= 6000 and line_count >= 180 and sim_count < 600:
            target_path = os.path.join(target_directory, file_name)
            shutil.copy(file_path, target_path)


def find_draft(date):
    global f, content
    # 定义原始目录和目标目录
    source_directory = f'/home/me/tools/pycnblog/articles/{date}'
    target_directory = f'/home/me/tools/pycnblog/articles_draft/{date}'
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
                cleaned_lines = [line.strip() for line in lines if line.strip()]
                content = '\n'.join(cleaned_lines)
                sim_count = check_similarity(content)
                # 文章的长度
                length = len(content)
                # 行数
                line_count = len(cleaned_lines)

            # 如果文章的长度>4000 and 行数大于150，则复制文件到目标目录
            if 3000 < length < 6000 and 100 < line_count < 180 and sim_count < 1000:
                target_path = os.path.join(target_directory, file_name)
                shutil.copy(file_path, target_path)


if __name__ == '__main__':
    import datetime

    # 获取当前日期
    now = datetime.datetime.today()

    # 存储过去5天的日期
    dates = []
    for i in range(0, 6):
        # 减去一天的时间间隔
        date = now - datetime.timedelta(days=i)
        # 格式化成yyyyMMdd
        date_str = date.strftime('%Y%m%d')
        # 存储到数组中
        dates.append(date_str)

    for d in dates:
        try:
            find_good(d)
        except Exception as e:
            print(f"Error occurred while finding good: {e}")

        try:
            find_draft(d)
        except Exception as e:
            print(f"Error occurred while finding draft: {e}")
