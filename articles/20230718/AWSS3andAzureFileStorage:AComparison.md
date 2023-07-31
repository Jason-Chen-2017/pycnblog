
作者：禅与计算机程序设计艺术                    
                
                
The Amazon Web Services (AWS) Cloud offers a wide range of services for cloud computing such as object storage that is suitable for storing large volumes of unstructured data. Two popular services in the market today are Amazon Simple Storage Service (S3) and Microsoft Azure File Storage. In this article we will compare these two services based on their features and benefits.

Object storage is one of the most commonly used storage solutions available in the cloud because it allows you to store and retrieve any type of file or object without worrying about how it will be structured, stored, managed, accessed, etc. It offers several advantages over traditional file systems including high scalability, low latency, cost effectiveness, and flexibility.

S3 provides simple, easy-to-use web service interfaces that allow developers to store and retrieve any amount of data from anywhere at any time, with no infrastructure management overhead. The service is designed to deliver 99.999% availability across all regions and supports multiple tiers of storage classes for different use cases, making it ideal for backups, media processing, content distribution, real-time analytics, and big data workloads.

Azure Files is another cloud-based solution offered by Microsoft that also has a strong focus on simplicity. Its primary feature is its ability to provide shared access to network-attached file shares, which can be mounted on Linux or Windows machines for easy integration into applications. The service integrates seamlessly with other Azure services like Azure Virtual Machines, Azure Kubernetes Service, and Azure Database for PostgreSQL/MySQL/MariaDB, making it a powerful choice for various use cases.

In summary, both S3 and Azure Files offer compelling advantages when it comes to simplifying the process of managing and storing large amounts of unstructured data. However, there are some key differences between the two services that should be considered before choosing one over the other. Let's get started comparing them!
# 2.Basic Concepts and Terms
## Object Storage vs File System
Before diving deeper into comparing S3 and Azure File Storage, let’s first understand the basic concepts of object storage and file system storage.

File system storage refers to physical storage devices connected to your computer, server, or cluster where files are organized in a hierarchical directory structure. It stores data as blocks of fixed size, typically around 4 KB each, called records or pages. This means that accessing specific parts of a file requires reading through the entire file, reducing efficiency.

On the contrary, object storage enables you to store and manage large amounts of unstructured data using small, flexible objects that can range in size from a few kilobytes up to petabytes. These objects are referred to as "objects," but they don't necessarily correspond directly to individual files. Instead, an object storage system breaks down large files into smaller pieces, aggregates them together, assigns unique identifiers to each piece, and manages them automatically so that users only need to know how to interact with the overall system.

A common analogy for object storage versus file system storage would be to consider block storage versus object storage. With file system storage, you might have a hard disk drive attached to your machine that contains many partitions. Each partition represents a separate volume, which can hold hundreds of gigabytes or even terabytes of data. To read or write a particular file, you need to mount the appropriate partition onto your file system and then navigate through the hierarchy until you reach the desired file. Block storage uses a similar approach, except instead of creating volumes, it creates individual blocks of storage space that can be combined to form larger files. Because individual blocks cannot be easily manipulated independently, it is not always efficient to work with them individually.

With object storage, however, the idea is to break down large files into smaller, more manageable chunks that are aggregated together under a single namespace. Once uploaded, these objects can be accessed quickly and efficiently using standard HTTP requests. This eliminates the need to manually map virtual drives to directories, which makes working with large datasets much easier. Additionally, since objects can be spread across multiple servers and distributed across different locations within a region, object storage systems scale well compared to file systems. Finally, object storage systems often support automatic replication and failover, enabling highly reliable storage capabilities that keep your data safe and accessible even if a failure occurs.

## Blobs vs Files
As mentioned earlier, object storage systems organize data into objects that can vary in size ranging from a few bytes to petabytes. Objects are identified using unique IDs and usually consist of metadata describing the contents of the object alongside the actual data itself. When you upload an object to object storage, you must specify additional properties such as encryption settings, access controls, versioning information, and expiration dates. You can also tag the object with keywords or categories to make it easier to search and find later.

Sometimes people refer to objects generically as blobs or simply "files" depending on who they are referring to. While both terms are technically accurate, they may still confuse readers because they refer to very different things. An object in object storage is just a blob while a file on a filesystem is a collection of related objects arranged logically. Just as objects in object storage represent real-world entities rather than logical units of code, files on a filesystem are meant to reflect the organization and relationships present in reality.

