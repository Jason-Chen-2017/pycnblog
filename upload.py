import asyncio
import html
import ssl
import sys
import xmlrpc.client

from img_transfer import *

# if len(sys.argv) == 1:
#     print('请输入markdown文件路径,带双引号哦')
#     exit(-1)


net_images = []  # 图片上传后url
image_count = 1  # 图片计数


def get_image_url(t):
    """回调，获取url"""
    global image_count
    url = t.result()['url']
    print(f'第{image_count}张图片上传成功,url:{url}')
    net_images.append(url)
    image_count += 1


def cancel_ssh_authentication():  # 取消全局ssl认证
    ssl._create_default_https_context = ssl._create_unverified_context


async def upload_tasks(local_images_, dir_name):
    tasks = []
    for li in local_images_:
        image_full_path = os.path.join(dir_name, li)
        task = asyncio.create_task(upload_img(image_full_path))
        task.add_done_callback(get_image_url)
        tasks.append(task)
    await asyncio.gather(*tasks)


def do_upload(md_path, dir_name, title):
    with open(md_path, encoding='utf-8') as f:
        md = f.read()
        print(f'markdown读取成功:{md_path}')
        local_images = find_md_img(md)

        if local_images:  # 有本地图片，异步上传
            asyncio.run(upload_tasks(local_images, dir_name))
            image_mapping = dict(zip(local_images, net_images))
            md = replace_md_img(md_path, image_mapping)
        else:
            print('无需上传图片')


        print(title)
        print(md)

        post = dict(description=md,
                    title=title,
                    categories="ai")

        server.metaWeblog.newPost(
            conf["blog_id"],
            conf["username"],
            conf["password"],
            post,
            conf["publish"]
        )

        print(f"markdown上传成功, 博客标题为'{title}', 状态为'{'已发布' if conf['publish'] else '未发布'}', "
              f"分类为:{conf['categories']} 请到博客园后台查看")



if __name__ == '__main__':
    cancel_ssh_authentication()

    md_path = '/home/me/tools/pycnblog/articles/12神经网络中的数学基础：梯度下降和反向传播.md'
    dir_name = os.path.dirname(md_path)
    title, _ = os.path.splitext(os.path.basename(md_path))  # 文件名作为博客标题

    do_upload(md_path, dir_name, title)
