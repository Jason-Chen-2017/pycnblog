## 背景介绍

Pulsar 是一个开源的分布式流处理系统，旨在为大规模数据流处理提供低延迟、高吞吐量和可靠性。Pulsar 是 Apache 项目的一个子项目，最初由 Yahoo 开发。Pulsar 的设计目标是为大规模数据流处理提供一个高效的平台，能够满足各种不同的场景需求，包括实时数据处理、批处理和事件驱动处理等。

Pulsar 的核心架构包括以下几个组件：

1. Pulsar Broker：负责接收客户端发送的数据和查询请求，并将其路由到正确的Pulsar Consumer和Pulsar Producer。
2. Pulsar Consumer：订阅Pulsar Topic的客户端，负责消费数据。
3. Pulsar Producer：发送数据到Pulsar Topic的客户端。
4. Pulsar Proxy：提供负载均衡和故障转移功能，提高系统的可用性和可靠性。

## 核心概念与联系

Pulsar 的核心概念是 Topic 和 Subscription。Topic 是一个数据流，Subscription 是消费者订阅的数据流的分支。每个 Topic 可以有多个 Subscription，每个 Subscription 只会消费一个分支的数据。

Pulsar 的流处理模型是基于 publish-subscribe 模式的。Pro