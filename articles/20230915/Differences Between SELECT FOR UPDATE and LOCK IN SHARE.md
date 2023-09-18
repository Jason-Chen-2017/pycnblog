
作者：禅与计算机程序设计艺术                    

# 1.简介
  

SELECT FOR UPDATE and LOCK IN SHARE MODE are two of the most commonly used locks available to prevent updates on tables by multiple sessions or users simultaneously. However, they have different behaviors that affect how they work and when should be used. This article will discuss these differences and their implications for building robust applications with high concurrency and scalability requirements. 

In this article, we will briefly introduce the concept of shared locks as well as explain what is meant by exclusive locks and lock modes in MySQL. We will then demonstrate some sample code demonstrating both SELECT FOR UPDATE and LOCK IN SHARE MODE. Finally, we will conclude with a discussion of potential drawbacks of using SELECT FOR UPDATE vs LOCK IN SHARE MODE depending on use case scenarios and constraints.

By the end of this article, you will understand the differences between the SELECT FOR UPDATE and LOCK IN SHARE MODE in MySQL and their advantages and limitations based on various use cases. It may help developers choose an appropriate locking mechanism based on their needs and constraints while building robust applications with high concurrency and scalability requirements. 

# 2. Basic Concepts
## Shared Locks: 
Shared locks allow concurrent access to rows but prohibit any modifications to those rows until the transaction completes. The other transactions can still read from the table without blocking. When no more reads are expected, the transaction holding the shared lock commits and unlocks the row(s). Only one session can hold a shared lock at a time on a given row. 

## Exclusive Locks: 
Exclusive locks block all other access to the locked rows until the transaction is complete. No other transactions can access the row(s) until the current transaction releases the lock. An exclusive lock prevents other sessions from accessing the row(s), even if they request shared locks. Attempting to acquire an exclusive lock blocks waiting threads until the lock becomes available. If a thread holding an exclusive lock tries to insert or update data that violates unique constraint violations, the database server automatically rolls back the entire transaction. Only one session can hold an exclusive lock on a given row at a time. 

## Lock Modes: In MySQL, there are three types of lock mode - READ (shared), WRITE (exclusive), and INTENTION EXCLUSIVE (transaction-level exclusive lock that does not conflict with regular exclusive locks). Each type has its own characteristics and restrictions. Some examples of lock modes include:

1. Read uncommitted – Transactions can see changes made by other transactions but cannot commit them.
2. Read committed – Transactions only see changes made by previous transactions which have been committed before the start of the current transaction.
3. Repeatable read – Transaction sees the same rows as it did during the first execution of the statement, even if new rows were inserted into the table after the initial query started running.
4. Serializable – A consistent view of the data is obtained at the cost of increased locking overhead and reduced concurrency. This is often used when dealing with highly complex queries that require full row level locking to avoid deadlocks.
 
The default behavior in MySQL is repeatable read for select statements and serializable for select for update statements. Therefore, SELECT FOR UPDATE effectively acquires an exclusive lock on each row selected so that no other transaction can modify those rows until the transaction is complete.

However, using SELECT FOR UPDATE allows you to perform other operations such as INSERT, UPDATE, DELETE within the same transaction. If another transaction wants to insert/update/delete data on any of the selected rows, it must wait until your transaction completes. You also need to keep track of the rows you want to modify so that you don't accidentally modify unrelated rows. On the other hand, LOCK IN SHARE MODE only grants shared locks, meaning other sessions can still read but cannot write to the locked rows. Other sessions can continue to read from the rows unless they obtain an exclusive lock on those rows. Hence, LOCK IN SHARE MODE works best when you need to share read access to certain rows without interfering with other sessions' ability to write to the same rows.

Both SELECT FOR UPDATE and LOCK IN SHARE MODE provide similar functionality but differ slightly in terms of performance, flexibility, and usage patterns. Both methods enable efficient handling of high volume concurrency and ensure data consistency. So, choosing the right locking mechanism depends on specific application requirements and constraints.