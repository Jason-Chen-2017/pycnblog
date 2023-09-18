
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Schema matching has been a fundamental task in data integration, where it aims to identify common elements between two or more schemas that represent the same real-world entities. The major challenge of schema matching is to handle multiple sources with different attributes and constraints. However, existing approaches are mainly focused on handling simple type constraints such as string length, format, or even data types. In this paper, we propose an enhanced relational schema matching framework that takes into account both attribute and constraint information for better matching results. We further extend our proposed framework by incorporating user preferences into the matching process to obtain personalized matching results based on individual requirements. Our experiments show that our proposed framework significantly outperforms state-of-the-art techniques in terms of precision, recall, and F1 score, while achieving comparable performance on average similarity scores and effectiveness in dealing with diverse input scenarios.
# 2.相关工作
Schema matching is one of the most important tasks in data integration, which aims to match structured data from various sources without any errors. Previous research efforts have mostly focused on identifying similarities among schema components such as column names, data types, and domains. These methods do not consider complex constraints and preferences required when integrating heterogeneous datasets with varying physical structures and formats. 

In recent years, there have been many advances in database management systems (DBMSs) that support advanced features such as views and SQL functions. These technologies enable users to manipulate large amounts of data easily but also pose new challenges in integrating them together. To address these issues, several works have proposed algorithms to automatically discover potential schema matches using statistical and machine learning techniques. Nevertheless, they still rely heavily on content-based comparison strategies, and may fail to capture essential semantic differences between data sets. Moreover, these methods assume that all columns should be mapped exactly, which may not always be true under certain circumstances.

Thus, in view of the above mentioned limitations, there has been significant interest in developing new schema matching techniques that can effectively handle complex constraints and preferences involved in schema matching. Existing work often considers only specific aspects of the schema such as primary key selection or datatype matching, but fails to take full advantage of all available metadata. 

To tackle this problem, we propose an enhanced relational schema matching framework that utilizes both attribute and constraint information for better matching results. Specifically, we first construct a graph representation of each table's schema using its structure, relationships, and indexes. Then, we use the structural and content distances between tables' graphs to compute their pairwise similarity scores. To handle the complex constraints, we introduce a novel distance metric called Relative Constraint Distance (RCD), which measures the difference between the actual values in the tuples and the expected values given the data type and domain constraints. Finally, we incorporate user preferences into the matching process by allowing users to specify their desired mapping criteria. Within each criterion group, users can assign weights to different attributes, making personalization possible. We evaluate our approach using a variety of benchmark datasets, including social media sharing network datasets, e-commerce transaction logs, and medical records, and demonstrate significant improvements over existing schema matching techniques.
# 3.核心概念及术语
## 3.1 定义及关键术语
* **Attribute**: Attribute refers to an individual feature or property of an entity, such as name, age, email, etc. It is used to describe some aspect of an object, person, or place. For example, "name" is an attribute of people, "salary" is an attribute of employees, "title" is an attribute of books, etc. Attributes are identified by their respective names.

* **Entity**: An entity is anything that has identity and can be defined by a set of attributes. Examples of entities include persons, places, products, organizations, events, messages, etc. Entities are typically represented in tabular form, with rows representing instances/records and columns representing their corresponding attributes.

* **Database schema:** A database schema consists of a collection of tables, views, and other database objects that define the logical design of a relational database. Each table represents a collection of related data items, and contains fields (columns) describing each item’s properties and characteristics. A database schema includes:

  - Table definitions: Specifies the columns (attributes) and constraints (data types, uniqueness, foreign keys, etc.) associated with each table.
  - Indexes: Used to optimize search operations on specific columns. Typically used in conjunction with constraints to improve query performance. 
  - Views: Provide virtual representations of underlying tables, combining multiple tables to provide a consistent view of the data. Commonly used for complex queries involving joins and subqueries.
  - Stored procedures: Allow developers to encapsulate logic and calculations within the database server and reuse it across different parts of the application code. Can be written in PL/SQL, T-SQL, Java, C#, Python, etc.
  
* **Relational Database Management System (RDBMS):** A software system designed to manage relational databases. RDBMS stores, organizes, and retrieves data in tabular form. It allows users to create, modify, and delete tables, establish relationships between them, insert, update, and delete data, and execute queries against the stored data. Typical examples of RDBMS include Oracle, MySQL, PostgreSQL, Microsoft SQL Server, SQLite, MariaDB, IBM DB2, Couchbase, etc.

* **Table Graph:** A directed graph that represents the connectivity of a table. Nodes represent the columns and edges represent the dependencies between them. The weight of an edge indicates how strongly the source column depends on the target column. 

* **Structural Similarity Score (SSS):** A measure of the similarity between two tables based on their respective graph topologies. The higher the SSS, the closer the tables are related based on their column relationships. This metric relies on the assumption that two tables with identical column relationships will have a high structural similarity score, whereas tables with very dissimilar column relationships will have a low SSS.

* **Content Dissimilarity (CD):** A measure of the dissimilarity between two tables based on their respective row sets. CD is computed as the fraction of non-matching tuples between the two tables. A value of zero means that the tables contain identical row sets. Higher CD values indicate greater dissimilarity between the tables. 

* **Relative Constraint Distance (RCD):** A novel distance metric that captures the degree of violation of constraint rules specified in the database schema. For example, if the constraint specifies that a field must be a valid date between January 1st, 2020 and December 31st, 2020, then RCD would calculate the difference between the actual date and the expected range. If the actual value violates the constraint, RCD would increase accordingly. This metric addresses the issue of inconsistent expectations due to unclear or ambiguous constraints.

## 3.2 数据集与评价标准
We conducted experiments on three real-world datasets consisting of social media sharing network data, e-commerce transaction logs, and medical record data. All datasets were collected from public sources and follow different structure and sizes. We split each dataset randomly into training, validation, and test sets with approximately equal numbers of positive and negative pairs. The following metrics were used to evaluate the accuracy of the learned models: Precision, Recall, and F1 score. The SSS was calculated as the mean squared error between predicted SSS values and the ground truth values obtained from manual annotation. CD was calculated as the fraction of false matches in the top K predictions made by the model.

The experimental setup involves four steps:

1. Data preprocessing: Preprocess the raw data to convert it into standard forms suitable for analysis. This step converts dates to ISO format, removes invalid characters, encodes categorical variables, normalizes numerical data, and tokenizes text fields. 

2. Data exploration: Explore the preprocessed data to gain insights into the distribution and correlation patterns of the data. Identify interesting correlations between entities, examine class imbalances, and perform exploratory data analysis (EDA). 

3. Feature engineering: Extract relevant features from the preprocessed data using domain knowledge and statistical analysis techniques. Design customized transformers for continuous and categorical variables, preprocess text fields, and combine them to generate input vectors for neural networks.

4. Model training and evaluation: Train a deep learning model using the extracted features and train-validation splits. Evaluate the trained models using different evaluation metrics, including Precision, Recall, and F1 score, SSS, and CD. Use ensemble techniques to achieve better generalizability.