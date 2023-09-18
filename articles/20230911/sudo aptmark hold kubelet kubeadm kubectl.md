
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 什么是apt-mark?
APT全称Advanced Package Tool，即高级包管理工具。是一个用来处理软件安装、卸载、更新等事务的命令行工具，它的配置文件为/etc/apt/apt.conf，提供了丰富的选项来控制软件包管理方式。其中有个`apt-mark`，它用于对包进行标记和管理，例如标记为hold状态或unhold状态。
## 为什么要用apt-mark?
- 可以方便地禁止软件的升级，使得被标记的软件不会自动升级到最新版本。这样可以避免因系统缺少更新而导致运行出现问题。
- 可以方便地对软件进行分组，便于后续对这些分组中的软件进行统一管理。例如，可以将某个组织内部使用的所有软件都打上相同的标签。
- 对某些比较重要的软件也可以进行降级处理，从而在紧急情况下临时回滚至较旧的版本。
- 可以手动修改软件源列表，将某个包添加至特定组。这样可以按需灵活调整软件的安装顺序。

## 如何使用apt-mark?
### 命令格式
```bash
sudo apt-mark [COMMAND] PACKAGE...
```
### 参数说明
#### COMMAND
- `hold`: 把软件包标记为hold状态。不允许该软件包被自动或手动安装升级。只有手动执行`sudo apt-get -y install --no-install-recommends PACKAGE`命令才可安装该软件包。
- `unhold`: 把软件包恢复为可安装状态。如果之前已经把软件包hold住，此命令可取消其hold状态。
- `-d`, `--default-release RELEASE`: 设置默认软件源的release。当指定了该参数之后，APT会直接从该指定的release源中查找软件包。
- `-l`, `--list`: 查看当前所有已设置的软件包的hold状态。

### 使用示例
```bash
# 将kubelet和kubeadm标记为hold状态
sudo apt-mark hold kubelet kubeadm

# 检查当前所有已设置的软件包的hold状态
sudo apt-mark showhold

# 将kubelet和kubeadm恢复为可安装状态
sudo apt-mark unhold kubelet kubeadm

# 设置默认软件源的release为bionic
sudo apt-mark hold -d bionic
```
注：以上示例仅供参考。实际环境中应根据具体需求来决定是否对软件包进行分组并对不同分组采用不同的策略。

## 更多资源
- https://www.cnblogs.com/zhengbin/p/7792361.html
- https://blog.csdn.net/weixin_40795398/article/details/88806883
- https://helpmanual.io/help/ubuntu-apt-mark/
- https://askubuntu.com/questions/350972/what-is-the-difference-between-sudo-apt-mark-hold-and-echo-ignorepkg-baseu/351020#351020
- https://linuxize.com/post/how-to-use-apt-mark/




2021年7月1日|作者:苏宁云数据库团队|编辑:陈帅博|审阅者:甘浩然