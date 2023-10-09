
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


Descriptive statistics (DESCRIPTIVE STATS) is one of the most fundamental and basic statistical methods used in data science to summarize data into a concise and useful information that can be easily understood by non-technical audience. In this chapter, we will cover descriptive statistics concepts such as population vs sample, mean, median, mode, variance, standard deviation, range, quartile, boxplot, histogram and frequency distribution. We also learn about the importance of data visualization techniques when dealing with large datasets and how they are used to identify patterns and relationships within the dataset. This part covers only a subset of these topics.


# 2.核心概念与联系
Populations and samples: Population refers to all possible outcomes or events from which a statistic is calculated, while Sample refers to a subset of a population or experiment where an analysis is made.

Mean: The average value of a set of numbers, usually denoted by the Greek letter μ (mu). It represents the central tendency of the distribution. 

Median: A number separating the higher half of a data sample from the lower half, it is often chosen because it does not skew the data towards any particular direction. If there are an even number of observations, then the median is the average of the two middle values.

Mode: The value(s) that occur most frequently in a set of data. Modelling refers to finding the pattern among variables in the dataset using different mathematical models like regression or decision trees.

Variance: The measure of how far each observation deviates from the mean. It describes the spread of the data around the mean. Variance is measured in squared units of the original variable.

Standard Deviation: The square root of the variance, it provides more informative measures of variability than variance since it takes into account both the magnitude and shape of the variation.

Range: The difference between the smallest and largest values in a set of data. Range shows whether the values have equal width or not.

Quartiles: Three values that divide the data into four parts similarly to percentiles but without any interpolation. First quarter (Q1), second quarter (Q2), third quarter (Q3), first decile (D1), tenth decile (D10) etc. Quartiles show the distribution of the data better than simple percentiles.

Box plot: An alternative way to represent the data visually, consisting of five components - minimum, first quartile (Q1), median (Q2), third quartile (Q3), maximum - connected by whiskers extending above and below them. Box plots help detect outliers and visualize the central tendency and dispersion of the data.

Histogram: One of the easiest ways to visualize the distribution of data is through histograms, which group numerical data into bins and graph their frequencies on the x axis versus their ranges on the y axis. Histograms show how many times each bin occurs in the data, allowing us to see if there are any unusual peaks or troughs in the distribution.

Frequency distribution: Another common representation of the distribution of data is a table showing the frequency count of each category found in the data. It summarizes the results obtained from various analyses, including correlation, covariances, and hypothesis tests.

Data visualization tools commonly used in data analytics include scatterplots, bar charts, line graphs, heat maps, and pie charts. These tools allow us to quickly and accurately identify trends, patterns, and clusters within our data sets. They enable us to formulate insights and make predictions based on the visual interpretation.


# 3.核心算法原理和具体操作步骤以及数学模型公式详细讲解
## Mean
The formula to find the mean of a sample is given by:

$$\bar{x}=\frac{\sum_{i=1}^{n}{x_i}}{n}$$

where $\bar{x}$ is the symbol for the mean and $n$ is the total number of elements in the sample.

To calculate the mean of a dataset, we need to add up all the values in the dataset and then divide by the total number of values. For example, let's say we want to calculate the mean of the following dataset:

$$\{5,7,9,2,6,1,8,4,3\}$$

We start by adding up all the values in the dataset:

5+7+9+2+6+1+8+4+3 = 45

Then we divide by the total number of values:

$\dfrac{45}{9}=5$

Therefore, the mean of the dataset is $5$.


## Median
The median is the midpoint value in an ordered list of numbers. If the size of the list is odd, the median is the middle value; if it is even, it is the average of the two middle values. To find the median of a sample, follow these steps:

1. Sort the sample in ascending order
2. Find the middle index (if the size of the sample is odd) or indices (if it is even)
3. Calculate the median (average of the two middle values if it exists)

Here's the code implementation in Python:

```python
def find_median(sample):
    n = len(sample)
    
    # sort the sample in ascending order
    sorted_sample = sorted(sample)

    # find the middle index or indices
    mid = n // 2
    left = right = mid
    
    if n % 2 == 0:
        left -= 1
        
    return (sorted_sample[left] + sorted_sample[right]) / 2
```

For instance, if we want to find the median of the following sample:

$$\{5,7,9,2,6,1,8,4,3\}$$

We first sort it:

$$\{1,2,3,4,5,6,7,8,9\}$$

Since its length is odd, the middle index is 4, so the two middle values are:

$$(6+7)/2=6.5$$

Therefore, the median of the sample is $6.5$.

If we want to find the median of a larger dataset, sorting it may take longer time than simply calculating the median directly. Therefore, some libraries provide built-in functions to compute the median efficiently. 

## Mode
The mode is the value that appears most frequently in a collection of data points. There could be multiple modes in a dataset depending upon the frequency of occurrence of values. Let's consider an example to understand this concept. Consider a binary dataset containing 100 records of the heights of students. Let's assume that there are two males with the same height (say 180 cm) and three females having the same height (say 160 cm). Then, the mode of the dataset would be either 180 cm or 160 cm, irrespective of which gender has the highest frequency.

Let's consider another example. Suppose we have collected data on the income level of individuals belonging to different age groups ranging from childhood (age 0 to 18) to adulthood (age > 65). The income levels of people fall in discrete intervals like [$0, 25, 50, 75, 100$]. Since there are more individuals in the lowest income bracket (between 0 and 25 dollars), it makes sense to call this bracket as 'child'. Similarly, the interval ($25, 50$) belongs to 'working' class, ($50, 75$) to'mature', and $(75, 100)$ to'senior'. 

Suppose we randomly select a person from each of these age brackets and observe their income levels. Based on this data, what might be the most frequent income bracket? Again, assuming uniform distributions across the age brackets, the answer depends on who is making the selection. 

In summary, the purpose of the mode is to identify the most commonly occurring item/value in a dataset. However, keep in mind that it may contain multiple items or values, depending on the context. Hence, before making any conclusion regarding the mode, additional research should be done to verify its accuracy. 


## Variance
The variance quantifies how much a random variable differs from its expected value. In other words, it tells you how much the actual values differ from the mean of the distribution. The formula to calculate the variance is:

$$\sigma^2=\frac{\sum_{i=1}^{n}(x_i-\mu)^2}{n-1}$$

where $\sigma^2$ is the symbol for the variance, $x_i$ is the value of the element at position $i$, $\mu$ is the mean of the entire dataset, and $n$ is the total number of elements in the dataset.

We use the sum of squares of differences between each value and the mean divided by the total number of values minus 1 to eliminate the bias due to dividing by small number of values. This approach gives more accurate estimates of the true variance compared to a single value of variance.