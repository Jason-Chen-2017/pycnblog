
作者：禅与计算机程序设计艺术                    

# 1.简介
  

> List is a basic data structure in computer science that can store and organize a collection of elements where each element can be accessed by an index. Lists are commonly used for storing collections of related data such as names, addresses, phone numbers, emails, etc., and allow for efficient organization and manipulation of the data. The most common types of list include arrays, linked lists, stacks, queues, trees, and graphs. In this article, we will learn about how to create bulletted lists using various symbols like *, +, -, and explain their syntax and usage. We will also cover some examples on creating different types of bulleted lists and discuss the pros and cons of each type. Finally, we'll explore some advanced topics like nested lists and item references in bulleted lists.

# 2.术语
Before getting into technical details, let's briefly define some terms:

- **List**: A collection of items that is ordered and changeable. In Python, lists are implemented as objects with several methods for adding, removing, and manipulating the items stored within them. 

- **Index**: An integer value representing the position of an item within a sequence (such as a string, tuple, or list). Indexes start at zero and go up to one less than the length of the sequence. For example, if you have a list called `fruits` containing three items "apple", "banana" and "orange", then `fruits[0]` refers to the first item ("apple"), `fruits[1]` refers to the second item ("banana"), and so on.

- **Bullet**: A symbol that represents the beginning of a new list item. There are two main types of bullets:

	- Hyphen (`-`): This bullet appears as a dash or minus sign before each item except for the very first item. It looks like this `- Item`. 

	- Plus (`+`): This bullet appears as a plus sign before each item except for the very first item. It looks like this `+ Item`. 

	- Star (`*`): This bullet appears as an asterisk before each item except for the very first item. It looks like this `* Item`. 

# 3.核心算法

## Creating Bulleted Lists

To create a bulleted list, simply start typing your text followed by the bullet symbol(s) and hit enter after each line. The number of spaces before the bullet symbol indicates the level of indentation, which determines the hierarchy of the list items. For example, here's what it might look like when created using stars (*):

```
* Item 1
  * Subitem 1
    * Subsubitem 1
  * Subitem 2
  * Subitem 3
* Item 2
  * Subitem 1
  * Subitem 2
* Item 3
```

This creates a top-level list item with the word "Item 1". Each subsequent sublist is indented once more relative to its parent item, making it clearer which item belongs under which other item. You can nest the bullet styles arbitrarily deep, but make sure not to mix them within the same list.

Here are some other ways you could format the same list using the other bullet symbols:

```
+ Item 1
  + Subitem 1
  + Subitem 2
  + Subitem 3
- Item 2
  - Subitem 1
  - Subitem 2
* Item 3
```

Note that the default style for Markdown files is set to a plus symbol (+), but it can be changed to any other bullet symbol using the syntax `*`, `#`, or `-`.

## Referencing Items

You can reference individual items in a bulleted list using their corresponding number or letter preceded by either a dot `.` or an open bracket `[ ]`. Here are some examples:

```
* Item 1
  * Subitem A
  * Subitem B
* Item 2
  * Subitem C [c]
  * Subitem D.d
```

In this case, the word "Subitem C" has been referenced multiple times throughout the document by enclosing it in square brackets. Additionally, ".d" follows directly behind the last bullet point, indicating that it should be treated as part of the previous list item.

By referencing specific items within the list, readers can quickly locate and access the information they need without having to read through all of it. However, referencing too many items in a single sentence can lead to confusion because readers may interpret them differently depending on context. To avoid ambiguity, try to limit the number of references per sentence or paragraph.