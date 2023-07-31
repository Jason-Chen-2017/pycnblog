
作者：禅与计算机程序设计艺术                    

# 1.简介
         
1.Python 是一种高级、易用且功能强大的编程语言。它具有广泛的标准库和第三方模块支持、丰富的数据处理能力、灵活的语法和可读性，可以有效地解决复杂问题。本速查表旨在帮助初级用户快速入门Python，提升个人技能水平。

         本速查表共分7个部分，分别对应了Python中常用的基础知识、数据结构、文件读写、图像处理、机器学习、统计建模和可视化等领域，并提供了详实的Python编程示例。

         如果您是Python初学者或者已经熟练掌握一定的编程技巧，但是对某些特定的知识还不是很了解的话，可以选择感兴趣的章节阅读，然后结合示例自己编写代码测试。

         欢迎评论、建议及意见。

        # 2. 基本概念术语说明
         1. 数据类型:
         - int : 整数
         - float : 浮点数
         - str : 字符串
         - bool : 布尔值
         - list : 列表
         - tuple : 元组
         - set : 集合
         - dict : 字典

         2. 运算符优先级顺序：
         从最高到最低：
         () []. (Attribute Access) ** (Exponentiation) ~ + - (Unary Positive/Negative) * / // % @ (Matrix Multiplication) + - (Arithmetic Operators) & | ^ << >> (Bitwise Operators) < <= > >=!= == (Comparison Operators) in not in (Membership Operators) is is not (Identity Operators) and (Logical AND) or (Logical OR)

         3. 控制语句:
         - if-elif-else
         - for loop
         - while loop
         - try-except-finally

         4. 函数:
         def function_name(parameter):
            """function doc string"""
            statement1
            return result

         5. 模块:
         A module can be a single python file or an importable package that contains multiple files. To use the functions defined inside a module, we need to import it first using the `import` keyword. There are different types of modules in Python such as built-in modules, user-defined modules, third party modules etc. Built-in modules like math, random, os, sys provide various mathematical, statistical and operating system functionalities which can be used directly without importing them separately. User-defined modules can be created by us or other programmers and imported into our code base. Third party modules usually come from external sources and require additional installation steps to install before they can be used. Modules can also contain constants, classes, variables, functions, decorators etc.

         We can use the help() function to get information about any module or function. For example, calling help('math') displays the documentation for all the functions available under the math module. Similarly, calling help('os.path') shows the documentation for the os.path submodule which provides common operations on filenames and paths. 

         6. 文件读写
         In Python, there are two ways to read data from a file:

         Method 1: Using with Statement
        ```python
        with open("file_name", "r") as f:
           contents = f.read()
        ```
         This method automatically closes the file after execution of the block and ensures proper resource cleanup. It's generally considered good practice to always close resources when they're no longer needed.

         Method 2: Using try-finally Block
        ```python
        try:
           f = open("file_name", "r")
           contents = f.read()
        finally:
           f.close()
        ```
         This method is useful when you want to handle errors related to opening or reading files. However, closing files manually after their usage is still important to ensure proper resource cleanup.

         Writing to Files
        ```python
        with open("output_file_name", "w") as output_file:
           output_file.write("Some text here.")
        ```

         Appending to Existing Files
        ```python
        with open("existing_file_name", "a") as existing_file:
           existing_file.write("
Additional text goes here.")
        ```

         Reading Text vs Binary Modes
        Text mode means that the file should be treated as a sequence of lines separated by newline characters ('
'). When we read from a text file in text mode, we get back strings rather than bytes. If the file does not end with a newline character, then the last line will not have one unless we add '
' ourselves. 

        On the other hand, binary mode treats the file as a sequence of bytes. If we read from a binary file in binary mode, we get back raw byte strings. Here, it's up to the programmer to interpret these bytes correctly based on the encoding scheme used by the file.
        
        Most commonly used encodings include UTF-8, ASCII, ISO-8859-1 and more. These encodings specify how each byte of the file maps to its corresponding character representation. Commonly encountered Unicode errors can occur if the wrong encoding is assumed. By default, text files are opened in text mode but this behavior can be overridden using the 'b' flag for binary mode.
          
         7. NumPy 
         NumPy (Numerical Python) is a library written in Python that provides support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays. The library has many features including linear algebra, Fourier transforms, and random number generation.

         Importing NumPy Module 
        ```python
        import numpy as np
        ```

         Array Creation
        ```python
        arr = np.array([1, 2, 3])    # creating array from list  
        arr = np.zeros((2, 3))      # creating zero matrix   
        arr = np.ones((2, 3))       # creating ones matrix     
        arr = np.empty((2, 3))      # creating empty matrix      
        arr = np.arange(start, stop, step)  
                                     # creating array with values from start to stop with given step size    
        arr = np.linspace(start, stop, num=50)  
                                     # creating evenly spaced numbers over a specified interval    
        arr = np.random.rand(2, 3)  # creating random matrix    
        arr = np.random.randn(2, 3) # creating random normal distribution matrix
        ```

         Array Indexing and Slicing
        ```python
        x[indices]              # indexing elements
        x[rows,cols]            # slicing rows and columns
        X[rows1:rows2, cols1:cols2]  # slicing specific submatrices       
        Y = X[:2,:3]             # copying a subset of a matrix
        ```

         Broadcasting
        ```python
        z = x+y                  # element-wise addition
        z = x*y                  # element-wise multiplication
        z = np.dot(x, y)         # dot product between two matrices
        ```

         Math Functions
        ```python
        z = np.abs(-3)           # absolute value
        z = np.sqrt(4)           # square root
        z = np.exp(2)            # exponential function
        z = np.log(np.e)         # natural logarithm
        z = np.sin(np.pi/2)      # trigonometric function
        ```

         Statistics Functions
        ```python
        z = np.mean(arr)         # mean of array
        z = np.median(arr)       # median of array
        z = np.std(arr)          # standard deviation of array
        z = np.var(arr)          # variance of array
        ```

         Linear Algebra Operations
        ```python
        z = np.linalg.inv(A)     # inverse of a matrix
        z = np.linalg.det(A)     # determinant of a matrix
        z = np.trace(A)          # trace of a matrix
        ```

         Data Types Conversion
        ```python
        z = arr.astype(int)      # converting array elements to integers type
        z = arr.astype(float)    # converting array elements to floats type
        ```

         8. Matplotlib
         Matplotlib (matplotlib.org) is a comprehensive library for creating static, animated, and interactive visualizations in Python. The library offers a vast range of graph types, including bar charts, scatter plots, histograms, heatmaps, contour plots, 3D plotting, and surface plotting among others. The figure class in matplotlib serves as a container for all the plot elements, and the axes class provides a way to manipulate the appearance of individual plot components. Matplotlib integrates closely with NumPy and pandas libraries, making it easier to create high quality data visualizations.

         Importing Matplotlib Library 
        ```python
        import matplotlib.pyplot as plt
        ```

         Creating Simple Plot
        ```python
        plt.plot([1, 2, 3], [4, 5, 6])
        plt.show()
        ```

         Adding Title, Labels, Legend, and Axis Limits
        ```python
        plt.title("Plot Title")
        plt.xlabel("X Label")
        plt.ylabel("Y Label")
        plt.legend(["Data Set 1"])
        plt.ylim([-10, 10])
        plt.xlim([-1, 4])
        plt.show()
        ```

         Setting Grid and Ticks
        ```python
        plt.grid(True)
        plt.xticks(range(1, 6), ["Label 1", "Label 2", "Label 3", "Label 4", "Label 5"])
        plt.yticks(range(1, 6), ["Value 1", "Value 2", "Value 3", "Value 4", "Value 5"])
        plt.show()
        ```

         Changing Line Style, Marker, and Color
        ```python
        plt.plot([1, 2, 3], [4, 5, 6], linestyle="--", marker='o', color='g')
        plt.show()
        ```

         Subplots and Multiple Plots
        ```python
        fig, ax = plt.subplots(nrows=2, ncols=2)  
                                                 # creating a 2x2 grid of subplots  
        ax[0][0].plot([1, 2, 3], [4, 5, 6])       
        ax[0][1].hist([1, 2, 3, 4, 5])               
        ax[1][0].scatter([1, 2, 3],[4, 5, 6])              
        ax[1][1].imshow([[1, 2], [3, 4]])                    
                                               # adding data to each axis  
        plt.tight_layout()                             # improving spacing between subplots  
        plt.show()                                   
        ```

         Bar Charts
        ```python
        objects = ['Item 1', 'Item 2', 'Item 3']
        values = [10, 20, 30]
        plt.bar(objects, values)
        plt.show()
        ```

         Histograms
        ```python
        plt.hist([1, 2, 3, 4, 5])
        plt.show()
        ```

         Box and Whisker Plots
        ```python
        data = [np.random.normal(0, std, 100) for std in range(1, 4)] 
                                                    # generating random data sets with varying standard deviations  
        plt.boxplot(data, labels=[' STD'+ str(i) for i in range(1, 4)]) 
                                                   # displaying box and whisker plots with custom labels  
        plt.show()
        ```

         Contour Plots
        ```python
        x, y = np.meshgrid(np.arange(-3, 3, 0.1), np.arange(-3, 3, 0.1))
        z = np.cos(np.sqrt(x**2 + y**2))
        plt.contourf(x, y, z, alpha=0.8)
        plt.colorbar()                              # adding a colorbar legend  
        plt.show()
        ```

         9. Pandas
         Pandas (pandas.pydata.org) is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool. It allows you to load your data into a DataFrame object, perform various operations on it, and export the results into various formats like CSV, JSON, or Excel. Pandas provides numerous methods for cleaning, transforming, and analyzing data, and makes working with datasets much simpler compared to traditional approaches.

         Importing Pandas Library 
        ```python
        import pandas as pd
        ```

         Loading Datasets From File
        ```python
        df = pd.read_csv("file_name.csv")           # loading CSV dataset  
        df = pd.read_excel("file_name.xlsx")        # loading MS Excel workbook  
        df = pd.read_json("file_name.json")         # loading JSON data  
        df = pd.read_sql("SELECT * FROM table_name;", engine)   
                                                  # connecting to SQL database and loading data
        ```

         Selecting Columns
        ```python
        selected_df = df[['column1', 'column2']]     # selecting specific columns  
        selected_df = df['column1':'column2']       # selecting column range  
        ```

         Filtering Rows
        ```python
        filtered_df = df[(df["column"]=='value')]    # filtering rows based on conditions  
        filtered_df = df[df["column"].isin(['value1','value2'])]    # filtering rows based on multiple values  
        ```

         Sorting Rows
        ```python
        sorted_df = df.sort_values("column", ascending=False)    # sorting rows based on a column  
        ```

         Grouping and Aggregating Data
        ```python
        grouped_df = df.groupby("column").sum()                   # grouping data and calculating sum  
        aggregated_df = df.agg({'column': ['min', max]})          # aggregating data across multiple columns  
        ```

         Renaming and Replacing Values
        ```python
        renamed_df = df.rename(columns={"old_column": "new_column"})    # renaming columns  
        replaced_df = df.replace({"old_value": "new_value"})          # replacing values  
        ```

         Handling Missing Data
        ```python
        missing_df = df.dropna()                        # removing missing values  
        filled_df = df.fillna(method='ffill')            # filling missing values forward  
        ```

         Merging Dataframes
        ```python
        merged_df = pd.merge(left_df, right_df, on='key')     # merging two dataframe based on a key column  
        ```

         Exporting Dataframes
        ```python
        df.to_csv("new_file.csv", index=False)             # exporting dataframe to CSV format  
        df.to_excel("new_file.xlsx", sheet_name='Sheet1')   # exporting dataframe to MS Excel format  
        ```

         10. Scikit-learn
         Scikit-learn (scikit-learn.org) is a machine learning library for Python that provides efficient implementations of several popular algorithms, including decision trees, random forests, k-means clustering, and neural networks. scikit-learn also includes various utility functions for loading and preprocessing data, evaluating performance metrics, and more.

         Importing Scikit-learn Library 
        ```python
        from sklearn import model_selection, tree, neighbors, svm, naive_bayes, discriminant_analysis, gaussian_process
        ```

         Decision Trees
        ```python
        clf = tree.DecisionTreeClassifier() 
                                             # initializing decision tree classifier  
        parameters = {'max_depth': [1, 2, 3]}  
                                             # setting hyperparameters  
        scores = model_selection.cross_val_score(clf, X, y, cv=5)  
                                             # performing cross validation  
        print("Accuracy:", round(scores.mean(), 2))  
                                             # printing accuracy score  
        ```

         Random Forests
        ```python
        clf = ensemble.RandomForestClassifier()  
                                            # initializing random forest classifier  
        parameters = {'n_estimators': [50, 100, 150]}   
                                            # setting hyperparameters  
        scores = model_selection.cross_val_score(clf, X, y, cv=5)  
                                            # performing cross validation  
        print("Accuracy:", round(scores.mean(), 2))  
                                            # printing accuracy score  
        ```

         K-Means Clustering
        ```python
        clf = cluster.KMeans(n_clusters=2) 
                                        # initializing K-Means clustering algorithm  
        predictions = clf.fit_predict(X) 
                                        # fitting and predicting clusters  
        ```

         Nearest Neighbors
        ```python
        clf = neighbors.KNeighborsClassifier()  
                                           # initializing nearest neighbor classifier  
        parameters = {'n_neighbors': [3, 5, 7]}   
                                           # setting hyperparameters  
        scores = model_selection.cross_val_score(clf, X, y, cv=5)  
                                           # performing cross validation  
        print("Accuracy:", round(scores.mean(), 2))  
                                           # printing accuracy score  
        ```

         Support Vector Machines
        ```python
        clf = svm.SVC()                          # initializing SVM classifier  
        parameters = [{'kernel': ['linear'], 'C': [1, 10]}, 
                      {'kernel': ['poly'], 'degree': [2, 3], 'C': [1, 10]}] 
                                          # setting hyperparameters  
        scores = model_selection.cross_val_score(clf, X, y, cv=5)  
                                          # performing cross validation  
        print("Accuracy:", round(scores.mean(), 2))  
                                          # printing accuracy score  
        ```

         Naive Bayes
        ```python
        clf = naive_bayes.GaussianNB()          
                                       # initializing Gaussian Naive Bayes classifier  
        scores = model_selection.cross_val_score(clf, X, y, cv=5)  
                                       # performing cross validation  
        print("Accuracy:", round(scores.mean(), 2))  
                                       # printing accuracy score  
        ```

         Discriminant Analysis
        ```python
        lda = discriminant_analysis.LinearDiscriminantAnalysis()  
                                                              # initializing LDA classifier  
        qda = discriminant_analysis.QuadraticDiscriminantAnalysis()
                                                           # initializing QDA classifier  
        ```

         Gaussian Process Regression
        ```python
        gpr = gaussian_process.GaussianProcessRegressor() 
                                                       # initializing Gaussian process regression  
        ```

