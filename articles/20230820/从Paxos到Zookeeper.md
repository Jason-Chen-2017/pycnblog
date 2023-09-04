
作者：禅与计算机程序设计艺术                    

# 1.简介
  

在分布式系统领域里，如何让多个节点就某一个事务达成一致性已经成为一个难题。业界曾经出现过多种分布式协调协议，如基于Paxos的实现，Raft、ZooKeeper等，但这些协议各有千秋，且各自设计精妙之处。为了更好地理解他们背后的原理及特点，以及它们为什么能够提供较高的可靠性，我们需要系统的学习、了解它们的区别与联系，并通过对比分析、实践试错，来找到最适合自己的分布式协调协议。本书将从“Paxos”协议开始，带领读者一起探讨其局限性，然后演进到“Zookeeper”，并对照两者之间的区别与联系，希望通过此书，读者可以更加清晰地认识和理解分布式协调协议。
# 2.Paxos算法
## 2.1 Paxos算法概述
Paxos算法是一个共识算法，它允许多个进程（可以是多个机器上的进程）在没有任何中心化的服务器或仲裁者的情况下达成数据一致性。在分布式系统中，通过选举一个“主导者”来决定某个值是否被接受。这个过程称作“投票”。当多个进程同时提议不同的值时，可能会出现冲突，这时需要重新选择。采用Paxos协议时，每个参与者都有两种选择：
- Propose ( 提案 ): 向大家宣布自己想要设置的值。
- Accept ( 接受 ): 如果一个值得到大家的广泛支持，那么它就被接受。
Paxos的目的是通过让整个过程尽可能简单、高效，并且容忍网络分区、节点失败等情况，保证最终所有参与者达成共识。
## 2.2 Paxos基本概念
### 2.2.1 Proposer （提案人）
Proposer 是指向集群中的其他成员发出建议的人。所有提案者都应该遵循一个共同的规范。他们会以轮流的方式选择值，并且只有一个提案者会被选中作为leader。他们不断尝试，直到达成共识。
### 2.2.2 Acceptor （接收人）
Acceptor 是指在整个集群中起关键作用，并且最终决定了最终结果的节点。一旦一个值被接受，它就会被所有接收方所知道。如果某个接收方发现自己没有被选中为leader，它会向另一个接收方发出请求，要求他帮助选举新的leader。Acceptor 可以处理两个角色：
- Leader：如果一个值被多数派Acceptor接受，那么该值就是最终的确定值。Leader负责广播它的决定给其他Acceptor。
- Follower：Follower只响应Leader的消息，而且只会在Leader失效或者故障转移时才选举新Leader。
### 2.2.3 Value （值）
Value 是指存储在数据库中要被协商一致的状态信息。一般来说，Value可以是一个具体的数字、字符串、对象等形式的数据结构。
### 2.2.4 Proposal Number （提案编号）
Proposal Number 是每个提案赋予的一个标识号。它使得不同的值具有唯一的编号。
### 2.2.5 Ballot （“表决卡”）
Ballot 是用来表示参与者支持哪个提案的序号。一张Ballot包括两个属性：
- A proposal number : 表示要进行表决的提案编号。
- An instance number : 每次对一个Ballot进行表决都会分配一个独一无二的实例编号，以记录每个提案都获得了一个固定的多数派选票。

一个Ballot由两部分构成：<proposal_number>:<instance_number> 。例如：<1:17> 表示编号为 1 的提案的第 17 个实例的投票。
### 2.2.6 PROMISE （承诺）
PROMISE （Promise）是一个消息，它被发送给Acceptors，以确认某个值是否被接受。它包括三部分：
- The value being promised : 承诺的那个值。
- The ballot number of the corresponding proposal : 对应的提案的ballot编号。
- The highest proposal number that has been accepted so far by any acceptor : 当前最高的已被接受的提案编号。
其中第一个值和第二个值是必需的；第三个值可用于优化：如果一个proposer提出的多个值具有相同的编号，那么它就可以直接跳过已经完成的投票阶段，而把中间过程省略掉。
### 2.2.7 ACCEPT （接受）
ACCEPT （Accept）是一个消息，它被发送给Acceptors，宣布一个值是否被接受。它包括四部分：
- The value being accepted : 被接受的值。
- The ballot number of the corresponding proposal : 对应的提案的ballot编号。
- The highest proposal number that has been accepted so far by any acceptor : 当前最高的已被接受的提案编号。
- The last known leader who voted for this value in its previous ballot : 上一次选举出leader后，同意接受值的提案者。
若两个或多个Acceptor同时收到了同样的Accept消息，那么它们可以通过比较ballot number来判断谁的提案的获胜，而无需等待一个Proposal Number被分配给某个值。
### 2.2.8 LEARNED （学习）
LEARNED （Learned）是一个消息，它被发送给Leader，宣布当前的leader已经得到超过半数派Acceptor的承认。它包括两个部分：
- The learned value : 被学习到的最新值。
- The highest proposal number that was ever proposed and still not accepted or rejected : 在之前的所有投票过程中，已经取得多数派支持的最新提案编号。