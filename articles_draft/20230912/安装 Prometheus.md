
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Prometheus 是开源的服务监控系统和时间序列数据库。它最初是由 SoundCloud 的 CTO 在 2012 年启动的，是一个基于 Go 语言开发的可伸缩、高可用、多维度的监控系统，最初主要用于云计算平台的监控，后来逐渐成为主流监控系统。Prometheus 的架构采用 pull 模型从被监控目标拉取指标数据，并通过规则引擎对指标数据进行处理，然后将处理结果存储到时序数据库中，供查询和分析。
本文将详细阐述如何在 Linux 服务器上安装 Prometheus。

# 2.基本概念术语说明
## 2.1 Prometheus Server
Prometheus 服务端一般称为 Prometheus server。Prometheus server 根据配置文件中的配置信息，向所有需要收集监控数据的 targets（被监控对象）发起 HTTP 请求，获取指标数据。Server 通过 rules engine 对指标数据进行规则匹配，提取满足某些条件的指标数据，最终将这些数据保存在本地磁盘或远端的时间序列数据库中，供用户查询。Prometheus server 本身不存储任何原始数据，只提供查询接口。


## 2.2 Targets
Targets （被监控对象）是 Prometheus 中最基本的一种监控对象，一般指服务或者机器。Prometheus 通过配置文件告诉它应该监控哪些 targets ，并且指定相应的采集方式，包括 metrics endpoint 地址、采集周期等。根据不同监控类型，可以分成不同的类型 target 。比如 HTTP 服务的 target 可以通过指定 metrics 路径、采集周期、用户名密码的方式定义；TCP 服务的 target 可以通过指定 metrics endpoint 地址、采集周期的方式定义；其他类型的 target 也可以通过指定 metrics 路径、采集周期、用户名密码的方式定义。


## 2.3 Metrics Endpoint
Metrics endpoint 是指一个暴露应用系统内部状态信息的 API 接口。Prometheus 可以通过这个接口获取应用系统的各项性能指标数据，如 CPU 使用率、内存占用量、网络带宽利用率等。每个 metrics 都有一个唯一的名称，可以通过 labels 来区分不同 instance 或对象实例上的指标。不同的 metrics 有不同的含义，如 counters（计数器）、gauges（Gauge 表）、histograms（直方图）、summaries（摘要）。一般来说，counter 和 gauge 表示单调递增的值，而 histogram 和 summary 表示分布情况。


## 2.4 Rules Engine
Rules engine 是 Prometheus 中的关键组件之一，它用来处理 Prometheus 获取到的指标数据，对其进行处理，提取或聚合一些数据指标。比如可以使用正则表达式对 metrics 名称进行匹配，从而自动发现和监控新的业务功能或模块。除了自动发现，rules engine 还可以支持基于 PromQL 查询语法的复杂规则匹配和数据处理。


## 2.5 Storage
Storage 是 Prometheus 数据持久化的基础设施。Prometheus 支持多种类型的存储，包括本地磁盘存储、远程存储（如 AWS S3、Google Cloud Storage）、时间序列数据库（如 InfluxDB、OpenTSDB）等。建议将长期存储的数据（如历史数据）存放在较低价值的、有保证的存储设备上。


## 2.6 Alertmanager
Alertmanager 是一个独立的组件，用于管理 Prometheus 报警信息。当 Prometheus 报警产生时，它会将报警信息发送给 Alertmanager ，再由它负责转发和处理报警消息。Alertmanager 提供多种通知渠道，如电子邮件、webhook 调用、短信等。


## 2.7 Grafana
Grafana 是一个开源的数据可视化工具，支持 Prometheus 作为数据源，可以实时呈现 Prometheus 暴露出的监控数据。通过 Grafana 可以方便地构建仪表盘、面板及监控视图，让运营团队能够直观地查看、分析和管理系统运行状态。


# 3.安装 Prometheus
## 3.1 操作系统版本要求
目前 Prometheus 支持以下几类操作系统：
- Linux (amd64、i386、armv6l、armv7l)
- macOS (amd64 only)
- Windows (amd64 only)
- FreeBSD
- OpenBSD
- Solaris

对于大多数 Linux 发行版和容器环境，Prometheus 可以直接从官方仓库中安装。但对于其它操作系统，可以通过源码编译的方式安装。

本文假定读者的操作系统是 Ubuntu 18.04，如果读者使用的操作系统不是 Ubuntu，请按实际情况调整命令。

## 3.2 Prometheus 安装步骤
### 3.2.1 Prometheus 下载
首先，我们需要从 GitHub 上克隆 Prometheus 项目仓库：

```bash
git clone https://github.com/prometheus/prometheus.git
cd prometheus
```

### 3.2.2 配置文件配置
Prometheus 默认会加载配置文件 `prometheus.yml`，该文件包含了 Prometheus 的各项配置信息，包括端口号、存储路径、目标列表等。默认情况下，`prometheus.yml` 文件路径为 `$HOME/.prometheus/prometheus.yml`。如果你希望修改默认配置，可以拷贝一份模板文件到当前目录下，然后修改该配置文件：

```bash
cp /etc/prometheus/prometheus.yml.
```

编辑 `prometheus.yml` 文件，找到如下一行并将 `false` 修改为 `true`：

```yaml
scrape_configs:
  # The job name is added as a label `job=<job_name>` to any timeseries scraped from this config.
  - job_name: 'prometheus'

    # Override the global default and scrape targets from this job every 5 seconds.
    scrape_interval: 5s

    static_configs:
      - targets: ['localhost:9090']
```

改动后的 `scrape_configs` 部分包含了一个名为 `prometheus` 的 job，该 job 会每隔 5 秒抓取 Prometheus 本身的 metrics 数据，并将其保存到名为 `prometheus` 的 time series database 中。其中 `targets` 列表只包含 Prometheus 的监听端口 `9090`。

接着，我们需要将 Prometheus 以守护进程模式运行起来。

### 3.2.3 Prometheus 启动
```bash
./prometheus --config.file=prometheus.yml &
```

上面的命令会在后台运行 Prometheus ，并且把它的日志输出到屏幕上。

## 3.3 查看 Prometheus 状态
启动成功后，我们可以通过访问 http://localhost:9090/status 来查看 Prometheus 的状态。

如果看到类似于下面这样的内容，说明 Prometheus 已经正常工作：

```json
{
   "build_info":{
      "branch":"HEAD",
      "build_date":"20211208-14:05:12",
      "commit":"<PASSWORD>",
      "version":"(devel)"
   },
   "config_file":"/home/ubuntu/.prometheus/prometheus.yml",
   "storage":{
      "chunks_to_persist":0,
      "head":{
         "chunk_id":"mZbPjAAAAABXANpgAAACQ",
         "max_time":"1640134720001",
         "min_time":"1640134720001",
         "num_samples":1,
         "refreshed_at":"1640134720146"
      },
      "highest_completed_checkpoint_ts":"0",
      "lowest_unflushed_ts":"1640134720145",
      "open_chunk_refs":0,
      "pending_compactions":0,
      "retention_limit":1100000,
      "size_bytes":2661
   }
}
```

如果看到 `{"status":"success"}`，说明 Prometheus 已经正常运行，并且可以正常抓取 metrics 数据。

## 3.4 添加 targets
上面演示了 Prometheus 的基本操作，但是 Prometheus 需要有 target 才可以正常抓取 metrics 数据。所以，我们需要添加 targets 来让 Prometheus 抓取 metrics 数据。

一般来说，我们通过两种方式来添加 targets 到 Prometheus 中：
1. 静态配置：将 target 的相关信息写入配置文件中，重启 Prometheus 使之生效。
2. 动态发现：让 Prometheus 从注册中心或者服务发现机制中自动发现新加入集群的节点，并监测其是否健康。

这里，我们仅讨论第一种方法，即静态配置。我们可以在配置文件 `prometheus.yml` 的 `scrape_configs` 中增加相应的配置项，告诉 Prometheus 要抓取哪些 targets。例如，我们可以添加一个名为 `node_exporter` 的 job，用于监控 node exporter 的 metrics 数据：

```yaml
scrape_configs:

 ...

  # Example scrape configuration for probing third party resources.
  - job_name: 'node_exporter'
    # metrics_path defaults to '/metrics'
    # scheme defaults to 'http'.
    static_configs:
    - targets: ['192.168.0.10:9100', '192.168.0.11:9100']
```

这里，我们配置了一个名为 `node_exporter` 的 job，该 job 会通过 HTTP 协议抓取 `192.168.0.10` 和 `192.168.0.11` 两个主机的 `9100` 端口上的 metrics 数据。

然后，我们需要重启 Prometheus 使配置生效：

```bash
killall prometheus &&./prometheus --config.file=prometheus.yml
```

此时，我们可以通过访问 http://localhost:9090/targets 来查看当前的 targets 列表。如果出现刚才添加的 `node_exporter` 这个 job，且列表中的 targets 都处于 healthy 状态，则表示 Prometheus 已经正常工作。