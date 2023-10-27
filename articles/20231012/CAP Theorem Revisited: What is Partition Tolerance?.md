
作者：禅与计算机程序设计艺术                    

# 1.背景介绍




The CAP theorem (Consistency, Availability and Partition tolerance) was introduced by <NAME> in his famous paper in 2000 to describe the trade-off between consistency, availability and partition tolerance in distributed systems that are networked together.

CAP theorem states that it’s impossible for a distributed system to simultaneously provide more than two of the following three guarantees: Consistency, Availability, and Partition tolerance. In simple terms, if you want your data to be consistent, then all nodes must see the same data at any point in time; If you want high availability, the system should continue operating even if some nodes fail or get disconnected from the network; And finally, Partition tolerance means being able to handle partial failures such as when one node goes down temporarily but other nodes remain operational. 

However, over the years, CAP has become an outdated concept because it’s difficult to guarantee strong consistency and high availability under real world conditions where networks break frequently due to various reasons such as power outages, natural disasters, hardware failures etc., leading to partitions and crashes in the system. This leads to situations where the system may appear inconsistent or unavailable despite meeting the basic requirements of consistency, availability and partition tolerance.  

To address these issues, many modern NoSQL databases have come up with their own solutions like Google's Spanner, Amazon's DynamoDB, Cassandra, HBase etc., which offer different levels of consistency and availability based on trade-offs made by designers and developers while ensuring optimal performance, scalability and reliability. Therefore, understanding how CAP works helps us to select the right database for our needs while also considering its pros and cons so we can choose the most suitable approach for our specific use case.



# 2. Core Concepts & Relationship 


## ACID vs BASE

In traditional relational databases, there are four main properties known as ACID properties:

* **Atomicity**: Transactions are executed in an “all or nothing” manner i.e. either they complete successfully or they don't. Each transaction takes effect as a single unit i.e. either everything happens or nothing happens.

* **Consistency**: A transaction will only leave the database in a valid state if all rules and constraints specified in the schema are followed correctly. Consistency ensures data integrity and allows users to rely upon the system.

* **Isolation**: Transactions do not interfere with each other. Isolation ensures that concurrent transactions do not interfere with each other and cause damage to the database.

* **Durability**: Once a transaction completes, changes made by it persist in the database permanently. Even if there is a power failure, the committed transactions would still be saved in the database. Durability ensures permanent storage of data.


But in cloud computing era, things change dramatically. Distributed systems, especially those designed to scale horizontally across multiple servers, introduce several new challenges. These challenges include non-determinism, latency, fault tolerance, and capacity planning. It becomes essential to ensure that these distributed systems meet the requirements of both ACID and BASE concepts since they apply equally to distributed systems regardless of whether they run on a single server or multiple servers connected via a network. For example, we need to make sure that even in the event of a server crash, all the updates performed during the crashed session are safely persisted in the database before making further writes. Similarly, we need to ensure that once a write operation succeeds, it cannot be rolled back without causing any harm to the application running above it. 



### BASE (Basically Available, Soft State, Eventually Consistent)

BASE stands for Basically Available, Soft State, Eventually Consistent. Let’s understand what these properties mean individually and why they differ from traditional ACID principles. 

#### Basically Available

This property says that every request receives a response about whether it succeeded or failed, without returning an error. However, it does not guarantee that reads will always return the latest written value, and it also does not guarantee that writes will always succeed. 

For instance, consider a key-value store where two clients perform read/write operations on the same key at the same time. One client performs a write operation after reading the previous value of the key, and the other client performs a read operation immediately after writing a new value. Since replication protocols could take some time to replicate the updated data to all replicas, it might happen that one replica gets the new value but another replica doesn't yet. Based on this scenario, the second read operation may return an old value instead of the newly written value, resulting in inconsistency between replicas.

Another example of this scenario occurs when multiple clients attempt to update the same key simultaneously using optimistic concurrency control techniques. When two or more clients try to write to the same key within a short window of time, it’s possible that the last writer wins and overwrites the existing values, resulting in conflicts. To avoid this conflict, systems often implement locking mechanisms that prevent conflicting writers from accessing the resource until they release the lock. Despite these measures, it remains possible that readers may observe stale data for a brief period of time, depending on the implementation and timing of the clock synchronization protocol used by the cluster.

Overall, although the BASE model promises greater availability, it does not guarantee consistent reads and consistent writes in all cases. Thus, it is recommended to combine this model with a technique called eventual consistency to build highly available and scalable systems that tolerate occasional inconsistencies due to network partitions and temporary failures.

#### Soft State 

In soft state model, the state of the system can change over time, even without input. This model allows for systems to have higher flexibility compared to hard state models, allowing for automatic recovery from unexpected events, such as machine failures or network partitions. Systems in this model maintain internal consistency without requiring immediate action from external entities.

An example of a system with a soft state model is a replicated graph database, where individual nodes and edges can rapidly evolve through addition, deletion, and modification. Despite being flexible, however, it can lead to inconsistent views of the graph, particularly in the face of frequent modifications. Systems implementing this type of model typically employ techniques such as vector clocks or leader election to detect and resolve inconsistencies automatically.

#### Eventual Consistency

Eventual consistency refers to the situation where all updates reach all copies of data eventually, but the order in which they occur is arbitrary. Within the context of distributed systems, this means that there is no global view of the system, and each component may have incomplete or incorrect information relative to others. While eventual consistency is desirable in certain scenarios, it can result in degraded performance and increased risk of unavailability in others.

One common example of eventual consistency involves replicating data across regions or geographies. Asynchronous replication techniques allow for near-real-time data synchronization, but it can potentially result in divergent views of the same data in different regions, resulting in inconsistencies. To mitigate this issue, applications can enforce strict ordering of updates by using sequence numbers or timestamps to determine the correct sequence of operations.