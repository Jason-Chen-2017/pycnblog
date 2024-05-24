
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Mydumper是阿里巴巴开源的一个MySQL数据备份和还原工具，可以将指定数据库的数据导出到本地文件或者导入到另一个MySQL服务器中。
# 2.安装部署
## 2.1 下载安装包并解压
从GitHub下载最新版Mydumper安装包 https://github.com/maxbube/mydumper/releases 。目前最新版本为v0.9.4。
```
wget -c https://github.com/maxbube/mydumper/archive/refs/tags/v0.9.4.tar.gz
tar xzf v0.9.4.tar.gz && cd mydumper-0.9.4
```
## 2.2 安装编译依赖库
Mydumper依赖于libgcrypt、openssl等第三方库，需要在编译前安装好。
```
sudo yum install libgcrypt-devel openssl-devel lz4-devel cmake
```
## 2.3 配置环境变量
打开`.bashrc`文件，添加以下两行：
```
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH # 将/usr/local/lib加入LD_LIBRARY_PATH
export PATH=$PWD/bin/:$PATH # 添加mydumper可执行文件的路径
```
刷新环境变量：
```
source ~/.bashrc
```
## 2.4 编译安装
执行make命令进行编译：
```
make -j$(nproc) && make install
```
如果出现编译错误，可能需要调整参数重新编译。如遇到问题，请联系作者QQ：719458762。
# 3.功能说明
## 3.1 指定数据库备份
为了指定备份哪个数据库，需要修改配置文件`mydumper.cnf`。配置项如下：
```
[mysql]
host=localhost # MySQL地址
port=3306 # MySQL端口号
user=root # MySQL用户名
password=example # MySQL密码
ssl-ca=./cacert.pem # 证书文件
ssl-cert=./client-cert.pem
ssl-key=./client-key.pem

[backup]
outputdir=~/backups # 备份输出目录
errorlog=~/logs/mydumper.err # 错误日志文件
loglevel=warning # 日志级别
databases=test # 指定要备份的数据库名称，多个数据库用逗号分隔，不指定则备份所有数据库
threads=4 # 线程数
compress=lz4 # 压缩方式，支持gzip、deflate、bz2、lz4
size=1G # 每个文件的最大值，单位MB或GB，超过该大小的文件会被拆分
lock-all-tables=true # 是否锁定所有表
events=true # 是否记录数据库的变化事件，用于增量备份
```
## 3.2 全量备份
对于全量备份来说，只需将上述配置文件中的`databases`项注释掉即可。比如，将`databases`项注释掉后，仅备份默认数据库：
```
[mysql]
host=localhost
port=3306
user=root
password=<PASSWORD>
ssl-ca=./cacert.pem
ssl-cert=./client-cert.pem
ssl-key=./client-key.pem

[backup]
outputdir=~/backups
errorlog=~/logs/mydumper.err
loglevel=warning
# databases=test
threads=4
compress=lz4
size=1G
lock-all-tables=true
events=true
```
## 3.3 增量备份
对于增量备份，需要在配置文件中设置`incremental=filename`，这里的`filename`代表增量备份之前的最新备份文件名（不包括扩展名）。按照下面的步骤进行增量备份：

1. 在第一次备份时，无需设置`incremental`选项；
2. 当第二次备份时，设置`incremental=latest_full.sql.lz4`，其中`latest_full.sql.lz4`代表上一次的全量备份文件名（不包括扩展名）；
3. 之后的每一次增量备份，都需要设置`incremental=last_incr.sql.lz4`，其中`last_incr.sql.lz4`代表上一次的增量备份文件名（不包括扩展名）。

例子如下：
```
[mysql]
host=localhost
port=3306
user=root
password=<PASSWORD>
ssl-ca=./cacert.pem
ssl-cert=./client-cert.pem
ssl-key=./client-key.pem

[backup]
outputdir=~/backups
errorlog=~/logs/mydumper.err
loglevel=warning
databases=test
threads=4
compress=lz4
size=1G
lock-all-tables=true
events=true
incremental=latest_full.sql.lz4 # 设置增量备份之前的最新备份文件名
```
# 4.具体操作步骤
## 4.1 执行备份
在备份目录下执行如下命令：
```
mydumper --defaults-file=mydumper.cnf --master-data=2 --skip-tz-utc --verbose=3
```
上面命令执行时，会先读取`mydumper.cnf`配置文件，然后将命令行参数覆盖掉。

选项说明：

* `--defaults-file=mydumper.cnf`：指定配置文件；
* `--master-data=2`：启用 master data 的备份模式；
* `--skip-tz-utc`：跳过UTC转换；
* `--verbose=3`：打印详细信息，包括进度条；

备份完成后，会生成多个.sql文件，每个文件对应一个数据库。这些.sql文件已经经过压缩和编码处理，并且按照`--size`指定的规则拆分成多个小文件。

## 4.2 恢复数据库
假设要恢复test数据库，首先需要创建一个空数据库，然后执行如下命令：
```
myloader --defaults-file=mydumper.cnf --database=test < test.sql.*.lz4
```
注意：一定要用 `myloader` 命令来恢复数据，而不是用 `mysql` 命令！否则可能会导致数据丢失或者数据损坏。

选项说明：

* `--defaults-file=mydumper.cnf`：指定配置文件；
* `--database=test`：指定数据库名称；
* `< file1.sql.*.lz4`：指定要恢复的文件列表，这里我们使用通配符来匹配所有的.sql.lz4文件。

恢复完成后，数据库中的数据即等于备份时的原始状态。