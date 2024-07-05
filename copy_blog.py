# 读取 src 目录下面文件列表
# n个文件一组，组号从 start 开始，往后递增；
# 然后，分组复制到 dst 目录下面的子文件（文件名以组号命名）
# 如果 dst 下面的组号子目录不存在，则使用组号作为子文件名，创建 dst 子目录
# # 示例使用
# # copy_blog('path/to/src', 5, 'path/to/dst', 1)
#
# 这段代码定义了一个函数 `copy_blog`，它接受四个参数：源目录 `src`，每组文件数量 `n`，目标目录 `dst`，以及起始组号 `start`。函数的主要步骤如下：
#
# 1. 验证源目录和目标目录是否存在。
# 2. 读取源目录下的所有文件，并进行排序。
# 3. 按照指定的每组文件数量 `n` 进行分组。
# 4. 对于每一组文件，创建以组号命名的子目录。
# 5. 将当前组的文件复制到对应的组号子目录中。
#
# 请注意，这段代码假设源目录和目标目录已经存在，并且只会复制文件，不会复制子目录。如果需要处理子目录，代码需要进行相应的调整。

import os
import shutil


def copy_blog(src, n, dst, start):
    """
    分组复制文件到目标目录。

    :param src: 源目录路径
    :param n: 每组文件的数量
    :param dst: 目标目录路径
    :param start: 起始组号
    """
    # 确保源目录和目标目录存在
    if not os.path.exists(src) or not os.path.exists(dst):
        print("Source or destination directory does not exist.")
        return

    # 读取源目录下的文件列表
    files = [f for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
    files.sort()  # 对文件列表进行排序，确保顺序

    # 分组复制文件
    for i in range(0, len(files), n):
        # 计算当前组号
        group_number = start + (i // n)
        # 创建组号命名的子目录
        group_dir = os.path.join(dst, str(group_number))
        if not os.path.exists(group_dir):
            os.makedirs(group_dir)

        # 复制文件到组号命名的子目录
        for file in files[i:i + n]:
            shutil.copy(os.path.join(src, file), group_dir)


if __name__ == '__main__':
    src = '/Users/bytedance/ai/pycnblog/articles_good_mac/20240701'
    dst = '/Users/bytedance/ai/pycnblog/blog/ai'
    copy_blog(src, 15, dst, 150)
