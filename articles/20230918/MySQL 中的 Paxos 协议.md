
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Paxos 是一个分布式系统协调一致性算法。其于2001年由麻省理工学院的布雷森、佩里·门罗、约翰·凯尔曼等一起提出。后被多家公司采用，包括 Google 和 Apache 的 Hadoop。从诞生之初就受到广泛关注。

本文将介绍 MySQL 中基于 Paxos 协议的分布式事务管理机制。Paxos 是一种被广泛使用的分布式一致性算法。它的特点在于简单而易于理解，适用于很多场景，比如共识（consensus）、分布式锁服务、分布式文件系统等。相比于其他协议，Paxos 比较简洁、通俗易懂，并且具有高度容错性。因此，它很适合作为 MySQL 中分布式事务的协调者。

为了实现 Paxos 对分布式事务的支持，MySQL 使用了两个模式：一个主服务器（Primary），和多个备份服务器（Replica）。主要的工作流程如下图所示：

1.客户端应用向主服务器发送请求；

2.如果该条目（Request）没有出现过（即不重复执行），那么主服务器会生成并返回一个编号（ProposalID）给客户端；

3.客户端收到 ProposalID 之后，开始向所有的副本服务器（Replica) 发送 Prepare 消息，询问是否可以执行这次操作；

4.如果所有副本都返回同意消息，那么主服务器会给予客户端最终的决策，同时向所有副本服务器发送 Accept 消息，表示已经接受该 Proposal，且执行相关操作；

5.当任意一个副本服务器接收到 Accept 消息时，如果发现自己也需要执行这次操作，那么会拒绝，因为冲突了。否则，会一直等待直到执行完成，然后返回结果给客户端。

Paxos 协议中的角色分为两种：Proposer 和 Acceptor 。Proposer 是发起提案的实体，Acceptor 是参与决策的实体。每个 Proposer 在开始的时候都会选择一个 ProposalID ，并向所有 Acceptor 发送 Prepare 请求，申请锁资源。如果超过半数的 Acceptor 同意通过提案，则认为提案获得通过，开始提交操作。如果半数以上 Acceptor 拒绝了提案，那么重新选择新的 ProposalID 继续尝试。

具体的流程如下图所示：


# 2.基本概念术语说明
## 2.1 节点角色
分布式数据库通常有三个角色：
- Coordinator: 协调者，负责处理 client 请求，分配事务 ID，记录事务日志等。
- Slave: 从库，读取主库的数据，提供读服务。
- Master: 主库，写入数据，提供写服务。

MySQL 为分布式事务提供了 Paxos 协议的支持，因此需要对上述节点进行分类：
- Paxos Group Leader: Paxos 分组领导者，负责产生全局事务 ID（GTID）。
- Paxos Group Member: Paxos 分组成员，负责参与者选举，记录事务日志，响应请求，参与事务协商。

## 2.2 GTID(Global Transaction ID)
MySQL 5.6 以后版本支持 GTID (Global Transaction ID)。每一次事务提交时，会生成一个唯一的 GTID，其中包括事务自身的 UUID 和序号。在 master-slave 集群中，master 会记录这些 GTID 信息。当 slave 需要读取某个位置之前的事务历史时，可以通过这个 GTID 找到对应的 binlog 文件和位置。

## 2.3 Paxos ID
Paxos ID 是由 Proposer 生成的一个递增编号，用于标识一个 Paxos Instance。每个 Paxos Instance 对应于一个完整的操作，包括事务准备、提交或回滚等。Proposer 只能对当前未完成的操作编号进行投票，不能跳过或改变已知编号的操作。Proposer 会依据自己的编号决定何时提交或回滚操作。

## 2.4 Ballot Number
Ballot Number 是由 Proposer 生成的一个编号，用来表明自己对某一项议题的态度。Proposer 可以首先向 Acceptor 发起 PROMISE 投票，也可以在超时后再次发起 PROPOSE 投票，或者投出弃权的 NOMINATE 投票。如果 Acceptor 赞成，则会产生一个确定值，告诉 Proposer 提案被批准。如果 Acceptor 不同意，则会产生一个否定值，告诉 Proposer 提案被否决。Proposer 在决定是否提交或回滚操作时，需要结合各个 Acceptor 的投票结果。