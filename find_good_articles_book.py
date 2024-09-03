import os
import shutil
import threading

from fuzzywuzzy import fuzz

similarity_threshold = 80


def check_similarity(text):
    count = 0

    lines = text.split('\n')
    # O(n^2)次比较，性能较差;经测试重复相似度效果较好
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            similarity = fuzz.ratio(lines[i], lines[j])
            if similarity > similarity_threshold:  # 设置阈值
                count = count + 1

    return count



def is_good_content(content):
    # 包含关键字：$$ 表示有公式，```表示有代码
    keywords = [
        "$",
        "```",
        "背景介绍",
        "核心概念与联系",
        "核心算法原理",
        "数学模型和公式",
        "项目实践",
        # "实际应用场景",
        # "工具和资源",
    ]

    keywords2 = [
        "$",
        "```",
        "Background Introduction",
        "Core Concepts",
        "Core Algorithm",
        "Mathematical Model",
        "Project Practice",
        # "Practical Application",
        # "Tools and Resources",
    ]

    flag1 = True
    flag2 = True

    for keyword in keywords:
        if keyword not in content:
            flag1 = False

    for keyword in keywords2:
        if keyword not in content:
            flag2 = False

    return flag1 or flag2


def process_file(file_path, target_good_directory, target_draft_directory):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        content = '\n'.join(cleaned_lines)

        # sim_count = check_similarity(content)
        # is_not_similar = sim_count < 600

        length = len(content)
        line_count = len(cleaned_lines)
        short_lines_count_ration = len(
            [line for line in cleaned_lines if len(line) < 30 and line.startswith('##')]) / line_count

        print(f'{short_lines_count_ration} {length} {line_count} {file_path}')


    # target_good_directory
    if (length >= 2000 and
            line_count >= 80 and
            short_lines_count_ration < 0.6 and
            is_good_content(content)):
        file_name = os.path.basename(file_path)
        target_good_directory = os.path.join(target_good_directory, file_name)
        shutil.copy(file_path, target_good_directory)
        # print("process_good_file:", target_good_directory)

    # target_draft_directory
    if (1000 < length < 2000 and
            70 < line_count < 80 and
            short_lines_count_ration < 0.6 and
            is_good_content(content)):
        file_name = os.path.basename(file_path)
        target_draft_directory = os.path.join(target_draft_directory, file_name)
        shutil.copy(file_path, target_draft_directory)
        # print("process_draft_file:", target_draft_directory)


def find_articles(date):
    global f, content
    # 定义原始目录和目标目录
    source_directory = f'/home/me/tools/pycnblog/articles/{date}'
    target_good_directory = f'/home/me/tools/pycnblog/articles_good/{date}'
    target_draft_directory = f'/home/me/tools/pycnblog/articles_draft/{date}'
    # 创建目标目录
    os.makedirs(target_good_directory, exist_ok=True)
    os.makedirs(target_draft_directory, exist_ok=True)
    threads = []
    for file_name in os.listdir(source_directory):
        if file_name.endswith('.md'):
            file_path = os.path.join(source_directory, file_name)
            thread = threading.Thread(target=process_file,
                                      args=(file_path, target_good_directory, target_draft_directory))
            threads.append(thread)

    # 等待所有线程完成
    for t in threads:
        t.start()
        t.join()


if __name__ == '__main__':
    import datetime

    # 获取当前日期
    now = datetime.datetime.today()

    # 存储i天的日期； i=0 即为当天。
    dates = []
    for i in range(0, 2):
        # 减去i天的时间间隔
        date = now - datetime.timedelta(days=i)
        # 格式化成yyyyMMdd
        date_str = date.strftime('%Y%m%d')
        # 存储到数组中
        dates.append(date_str)

    for d in dates:
        try:
            find_articles(d)
        except Exception as e:
            print(f"Error occurred while finding good: {e}")
