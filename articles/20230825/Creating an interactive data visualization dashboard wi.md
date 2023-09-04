
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## What is Interactive Data Visualization?

Interactive data visualization refers to a technique of presenting complex information in various ways that allow the user to interact with and explore it at a deeper level than static representations can offer. It involves using tools like charts, graphs, maps or tables to organize and display large amounts of data interactively on screen for easy analysis and understanding. The purpose of this article is to provide an introduction to creating interactive data visualizations using Tableau software. 

Tableau Software is one of the most popular business intelligence (BI) and data visualization platforms used by organizations worldwide. It provides users with the capability to create stunning and dynamic data visualizations within minutes while also enabling them to share their findings across teams and throughout an organization.

In this article we will discuss how to use Tableau Desktop, Tableau Server and Tableau Online to create interactive data visualizations from real-world datasets such as sales performance, customer behavior, inventory levels etc. We will be using publicly available datasets to demonstrate our examples but keep in mind that you can easily substitute these with your own data sets. By completing this tutorial you should have a good grasp of how to create interactive data visualizations using Tableau software and start designing more visually appealing dashboards that are much easier to understand and navigate.

# 2.Background Introduction

Before discussing technical details let's take a step back and talk about what exactly interactive data visualization is all about. Interactive data visualization is a powerful tool for analyzing and understanding complex datasets. However, simply looking at a dataset without any ability to interact with it becomes difficult. In order to make the data meaningful, insights must be gained through interactions between different views, charts and dimensions. These interactions enable analysts to quickly extract valuable patterns, trends, and relationships within the data set. Although there are many BI and data visualization platforms available today, some of the key features of Tableau include:

1. Ability to connect to multiple sources of data including CSV files, relational databases, web services, APIs etc.
2. Flexible and customizable layout which allows users to arrange worksheets and panes as per their preference.
3. Powerful calculations and formulas that allow users to perform numerical operations and comparisons on data points dynamically.
4. User-friendly interface and intuitive drag-and-drop functionality makes it very easy for novice users to create compelling visualizations. 
5. Easy sharing capabilities allow users to share their workbooks with other members of the team or external stakeholders for collaboration purposes. Additionally, Tableau has been recently acquired by Salesforce.com so it offers integration with other Salesforce applications like Marketing Cloud, Service Cloud etc.
6. Dashboard creation is just one click away and once created they can be shared instantly with anyone who has access to the workbook. 

Now that we know what interactive data visualization is all about, let's dive into the process of creating an interactive data visualization dashboard using Tableau.

# 3.Basic Concepts & Terminology

Let’s familiarize ourselves with some basic concepts and terminology associated with Tableau.

### Worksheet/Dashboard

A worksheet in Tableau represents a single view consisting of graphical representation of data. A dashboard consists of several individual worksheets organized together in a single window. Each worksheet can contain multiple views. A dashboard can be customized by adding additional worksheets and rearranging the layout to suit specific needs. A workspace can contain multiple dashboards, allowing users to group related reports or analytics together.



### Views

Views represent the type of data being displayed in a worksheet. There are five types of views in Tableau:

1. Sheet View - This displays each record in a table or dimension as a row or column. This view can be used to analyze tabular data.

2. Map View - Maps are commonly represented using map views in Tableau. They can show geographical data or links between locations.

3. Graph View - This is used to display quantitative data in terms of metrics. For example, bar graph, line chart, area chart etc.

4. Calendar View - Displays time-based data in a calendar format. This could be used to track events over a period of time.

5. Metric Summary - This summarizes important metrics based on selections made by the user. For example, this could show total sales amount for a product category or average order value for customers.

Each view contains one or more marks, which define its appearance. Marks include shapes, colors, lines, sizes, labels, and animations. Views can be filtered, sorted, aggregated, and styled according to user preferences.



### Filters

Filters allow users to select only certain records or categories of data to highlight during exploration. Filters can be applied to both worksheets and views. Filter options can range from simple text searches to sophisticated filters based on date ranges, logical operators, numeric values and calculated fields.

### Calculations

Calculations allow users to apply mathematical functions, such as sums, means, counts, correlations, and ratios, to data points or groups of data points. Calculation expressions can be entered directly into the GUI or added via drag-and-drop menus. Multiple calculations can be combined into a single formula to derive new insights.

### Dimensions

Dimensions represent categorical variables that categorize data points. Differentiation among dimensions enables users to segment data into subsets for more detailed analysis. Tableau supports up to seven dimensional hierarchies and lets users drill down into detail by selecting different combinations of dimensions.

### Measures

Measures represent quantitative variables that describe data points. They can be numbers, percentages, rates, totals, and counts. Measure values can be aggregated using various methods, such as sum, count, mean, median, mode, and standard deviation.

# 4.Core Algorithm & Operations

We now move onto discussing core algorithmic concepts behind interactive data visualization using Tableau.

## Step 1: Connecting to Data

The first step is to establish a connection between Tableau and the source of the data. This typically involves specifying the location of the file or database where the data resides, choosing the appropriate authentication method, and then clicking “Connect.” Depending on the size of the data set, this step may take a few moments. Once connected, the data appears in the left pane of the Tableau UI. You can preview the data by hovering over each field and observing the summary statistics and field names. If you need to modify the connection settings, go to the bottom right corner of the Tableau desktop application and click on the wrench icon. 

Once the data is loaded, you can begin exploring it by examining the different views provided by default. The following steps outline how to create and customize views in Tableau. 

## Step 2: Create New Views

To add a new view, click on the + sign next to the top left corner of the worksheet. This brings up a menu containing different types of views that you can choose from. Select the type of view you want to create and adjust the properties to suit your needs. For instance, if you want to visualize a scatter plot, drag a Scatter plot mark onto the canvas. Click on the mark to open the formatting panel, which allows you to change the color, shape, size, opacity, and labels of the markers. Also, you can drag additional measures into the y-axis and x-axis sections to further refine the visualization.  

You can also insert additional views below existing ones by dragging and dropping them anywhere along the canvas. To remove unwanted views, simply delete them by pressing Delete or by selecting them and clicking on the bin button. Alternatively, you can hide a view temporarily by clicking the eye dropper button near the top of the view toolbar.

After creating views, you can customize their appearance by modifying the marks, axes, legends, backgrounds, and tooltips. Use the Format dropdown to access these properties.

## Step 3: Filtering Data

To filter data, click on the Filter pane to expand it. Here, you can specify criteria to restrict the data shown in the current view. You can apply filters either to the entire worksheet or to individual views. Simply drag the desired field onto the Filter pane, select the operator, and enter the relevant value(s). You can combine multiple filters into a single expression by combining them with AND or OR statements. You can clear all active filters by clicking on the red x button at the top of the Filter pane.

## Step 4: Aggregation and Calculation

Aggregation allows you to condense larger volumes of data into smaller, manageable chunks. When working with large datasets, it can be helpful to aggregate the data by month, quarter, year, or some other time period. Tableau allows you to create calculated fields and measure groups to achieve this result. To create a calculated field, drag a calculation onto the Fields pane, select the function, and enter the necessary parameters. After defining a field, you can drag it onto a visualization to see the results. You can also calculate the results by applying aggregation functions and grouping data by a particular attribute.

## Step 5: Hierarchy Navigation

Hierarchy navigation allows you to break down complex data structures into simpler subsets. For instance, when analyzing customer behavior, breaking down the data by age group would enable you to examine differences between customers of different age brackets separately. Similarly, hierarchy navigators can be used to explore multidimensional data by traversing through the hierarchical structure. Tableau includes two types of hierarchy navigators: Slicer and Selector. 

Slicers allow you to slice data by selecting specific values from a list. For example, if you have a product catalogue, you can create a slicer that shows only those products that belong to a specified category or brand. 

Selectors allow you to select multiple values from a list simultaneously. For example, if you have a retail store, you can use a selector to compare sales across multiple regions or times of day. 

## Step 6: Linking Views

Linking views enables you to explore relationships between different data points. This can help identify patterns and trends not visible in isolated views. To link views, drag and drop the same field onto both the X-Axis and Y-Axis section of another view. Then, switch to the linked view and adjust the formatting as needed to reveal the relationship between the data points.

For even more advanced linking, you can use parameters to control the filtering and highlighting of the data points. Parameterized views allow you to create variations of a report or dashboard depending on user input. Parameters are defined in the parameters pane on the left side of the screen, and can be adjusted by the user before running the report or loading the dashboard. You can use parameterized views to compare sales across different stores, years, or demographics. 

Finally, you can create custom visualizations using Tableau’s extensive library of built-in data connectors and third party plugins. With careful selection and customization, you can develop intricate data visualizations that inform decision making and accelerate insights.