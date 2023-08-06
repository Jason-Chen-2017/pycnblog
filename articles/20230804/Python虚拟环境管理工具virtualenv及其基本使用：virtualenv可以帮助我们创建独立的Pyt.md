
作者：禅与计算机程序设计艺术                    

# 1.简介
         

         Virtualenv（virtualenvwrapper）是一个很流行的Python虚拟环境管理工具，它可以帮助我们创建独立的Python开发环境，让我们的项目中使用的依赖包不影响全局环境。当然，virtualenv还有其他优点，比如隔离开发环境，让不同版本的Python或依赖包之间不会相互影响等。
     
         ## 安装virtualenv
         
         ```python
         pip install virtualenv 
         pip install virtualenvwrapper-win   （windows平台）
         ```
     
         ## 创建虚拟环境
     
         使用如下命令创建一个名为venv的虚拟环境，并激活该环境：
     
         ```python
         mkvirtualenv venv
         ```
     
         此时会自动在当前目录下生成一个名为venv的文件夹，里面包含独立的Python运行环境，其中有个activate脚本用于激活该环境。
     
         当我们想退出当前虚拟环境时，可以使用deactivate命令：
     
         ```python
         deactivate
         ```
     
         ## 在虚拟环境中安装依赖包
     
         激活虚拟环境后，我们可以使用pip命令安装所需依赖包：
     
         ```python
         (venv) $ pip install requests flask 
         Collecting requests 
           Downloading https://files.pythonhosted.org/packages/97/1a/6bfdb291ee9b827e0d26fd377a59e72ff7ed0692337ec4bddba9c3f4fa1f/requests-2.25.1-py2.py3-none-any.whl (61kB) 
               ... 
        Installing collected packages: certifi, chardet, idna, urllib3, requests
        Successfully installed certifi-2020.12.5 chardet-4.0.0 flask-1.1.2 idna-2.10 requests-2.25.1 urllib3-1.26.3 
         ```
     
         上述命令会将requests和flask这两个库安装到当前虚拟环境下的site-packages文件夹下。这样做的好处是，这些依赖包只对当前虚拟环境可用，不会影响系统中的其它任何东西。
     
         ## 切换虚拟环境
     
         有时候，我们需要在不同的项目中使用不同的虚拟环境，而每次都去修改路径、激活虚拟环境非常麻烦。所以virtualenvwrapper提供了一套方便快捷的方法，可以轻松地创建、删除、切换虚拟环境。
     
         下面介绍如何通过virtualenvwrapper实现创建、删除、切换虚拟环境：
     
         ### 创建虚拟环境
     
         如果要创建一个虚拟环境，只需在命令提示符下输入mkvirtualenv命令即可：
     
         ```python
         $ mkvirtualenv myproject
         New python executable in myproject\Scripts\python.exe
         Installing setuptools, pip, wheel...done.
         Running virtualenv with interpreter C:\Users\<your username>\AppData\Local\Programs\Python\Python38\python.exe

         (myproject)$ 
         ```
     
         上面的命令创建了一个名为myproject的虚拟环境，并自动激活了这个环境。进入myproject虚拟环境后，我们就可以像平时一样安装各种依赖包。
     
         ### 删除虚拟环境
     
         如果不需要某个虚拟环境 anymore，可以使用rmvirtualenv命令：
     
         ```python
         $ rmvirtualenv myproject
         ```
     
         ### 列出所有虚拟环境
     
         可以使用lsvirtualenv命令查看已有的虚拟环境：
     
         ```python
         $ lsvirtualenv
           py27        C:\Python27\python.exe
             bin     C:\Python27\Scripts
                 activate    activate_this.py
               include  C:\Python27\include
                     pip       pip3
                   python   python.exe
                  sqlite3  sqlite3.dll
              Scripts  C:\Python27\Scripts
                easy_install-3.4.exe   easy_install.exe
                idle               idle.bat
               iptestrunner.pyw   LICENSE.txt
            lib      C:\Python27\lib
                     site-packages
                     
                                  

            
         ```
     
         通过上面的输出可以看到，这里有一个名为py27的虚拟环境，它对应的Python解释器就是C:\Python27\python.exe。
     
         ### 切换虚拟环境
     
         如果我们需要进入某个虚拟环境，可以使用workon命令：
     
         ```python
         $ workon myproject
         (myproject)$ 
         ```
     
         如果我们想退出当前虚拟环境，可以使用deactivate命令：
     
         ```python
         (myproject)$ deactivate
         $ 
         ```