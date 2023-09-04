
作者：禅与计算机程序设计艺术                    

# 1.简介
         
　　　　在正则表达式中,锚定字符(Anchor)是一种元字符,它用来指定字符串的开头或结尾位置,或者某个位置是否是词首或词尾等.锚定字符是非常重要的,它可以帮助我们精确地定位一个字符串的位置,以及限定它的查找方向,从而提高正则表达式的执行效率。本文将介绍四个主要的锚定字符。
        # 2.基本概念术语
        ## 1.单词边界和非单词边界: 
        单词边界和非单词边界是锚定的两个概念。一般来说,单词边界就是指某些标点符号(如空格、标点、换行符等)后面跟着的一个单词的开头,或者句子中的第一个单词的开头;而非单词边界就是指某个位置不是单词边界.
        ```python
            re.match('word', 'Hello world')   # Match from start of string to the word boundary after 'Hello'
            re.search('world$', 'Hello world')    # Search for a match starting at any position and ending at the end of the line/string with 'world' as last word in it
            
            # same with \b - which matches word boundaries by default
            re.match('\bword', 'Hello world')   # Matches first occurrence of 'world'
            re.search('worldb?\b', 'Hello world')     # Search for words that have an optional 'b' before them
                
                # With positive lookbehind assertion, we can restrict search to words starting with 'a' or 'the' only
                    (?<!\b(?:the|a)\s)(?:\w+\W+)*\b(?:the|a)\w* 
                # The negative lookbehind assertion ensures that the word does not begin with 'the' or 'a'.
                    ^(?!\s*\b(?:the|a))
                # Positive lookahead assertion to ensure the word ends properly 
                    (?:(?:and|but|or|nor|yet|\.\.\.|\.{2,})[^.]*){0,}(\b(?:[A-Z][a-z]+)|\b\d+(st|nd|rd|th)?|(?<=\D)[a-zA-Z]\.|[^\w\s]|_)$  
                # Lookahead assertions are used to check if a pattern follows something else. In this case, it checks if a proper noun, number followed by a ordinal suffix, abbreviation using dot notation or non-alphanumeric characters exist till the end of the sentence. 
                    
               # We can use the ignorecase flag when searching in lowercase text also
                  re.compile("hello", re.IGNORECASE).search("Hello world") 
       ```
       ## 2.行首和行尾: 
        行首和行尾是在文本编辑器中经常看到的两个锚点,即字符串的开头或结尾.它可以用于定位整个文本中的某个位置.但是需要注意的是,当一个正则表达式搜索的目标字符串被多行整合到一起时,其锚点并不会随之改变,仍然只会匹配整个字符串的开头或结尾处。因此,如果你需要匹配文本中的某个位置,建议优先选择行首和行尾锚点。
        ```python
           import re
           line = "This is some sample text."
           print(re.findall("^T", line))        # Find all occurrences of T at the beginning of each line
           print(re.findall(".text$", line))      # Find all occurrences of.text at the end of each line

           text = '''Line 1
           Line 2
           Line 3'''
           
           print(re.findall("^L.*e$", text, flags=re.MULTILINE))  # Use multiline mode to match entire lines
        ```
        ## 3.词首和词尾: 
        词首和词尾指的是单词或短语的开头或结尾。它与单词边界、非单词边界的区别在于,词首只能出现在单词的开头,而词尾也可以出现在单词的末尾,例如像这样的词典中的“冰天雪地”、“借我千金”。虽然同样都是单词的最后一部分,但它们属于不同的词。
        ```python
           import re
           text = "The quick brown fox jumps over the lazy dog"
           print(re.findall("\bbr\w+", text))           # Finds all instances of "brown", "brooklyn", etc.
   
           text = """I hope you're doing well! Let's meet tomorrow."""
           print(re.findall("[a-zA-Z]+[\']?[a-zA-Z]+", text))    # Finds all personal names
        ```
        ## 4.多边界匹配: 
        多边界匹配允许正则表达式进行多次匹配,包括多个单词边界和多个非单词边界。通过重复使用这些锚点,可以实现对复杂数据结构的精准匹配。
        ```python
           import re
           text = "He said, \"What's up?\""
           print(re.findall("\".*?\"", text))          # Find all quoted strings containing letters and punctuation marks
       
           text = "<html><head></head><body><p>Some Text</p></body></html>"
           print(re.findall("<body>.*?</body>", text, re.DOTALL))  # Find everything between <body> tags
        ```
        # 3.核心算法原理
        锚定字符的作用是控制正则表达式的搜索范围。锚定字符有以下几种类型: 
        1. ^ 和 $ 字符。^ 表示字符串的开头，$ 表示字符串的结尾，用法如下所示：
        ```python
          re.match("^hello", "hello world")       # Matches hello at the beginning of the string
          re.match("^h.*rld$", "hello world")     # Matches hello world on its own line
          re.match("^abc|def$", "xyz def ghi")    # Matches either abc or def at the beginning of the string
          re.match("^abc|def$", "efg hij")       # Doesn't match anything since there isn't a full match
      ```
    2. \b 和 \B 字符。\b 是单词边界，\B 是非单词边界，用法如下所示：
    ```python
      re.search("\\bcat\\b", "The cat in the hat.")   # Matches the whole word 'cat'
      re.search("\\Bc(?:at|og)\\b", "The cat in the hat.")  # Matches both cat and cog but doesn't include partial matches like 'ca' or 'at'
  ```
  3. Word Boundaries（\b） 也是一种锚定字符，但它是一个比较特殊的锚定字符。单纯地使用 \b 进行匹配的话，有可能无法正确匹配所有词边界。比如：
  ```python
     re.search("\\bcat", "The cat in the hat.")       # This will match 'c' instead of 'cat' because it stops matching once it finds one character
     
     # To fix this issue, we need to add additional condition to make sure it includes complete words only
     re.search("\\b(?:\w+[-']\w+)+\\b", "The cat in the hat.")   # This expression matches all valid ways to write multiple words separated by hyphens or single quotes
  ```
  4. \G 字符。\G 表示一个完整匹配的起始点。如果前面已经匹配了一些字符，可以利用 \G 将其作为新的起点继续匹配：
  ```python
      re.sub("(foo)|(bar)", r"\1\2\Gbaz", "foobar foo bar baz qux")   # This will replace "foo" and "bar" with "foobarbaz" 
  ```
  
  # 4.具体操作步骤及代码实例  
  ## 1.确定匹配位置  
  通过查看文本示例和实际情况，我们通常可以确定正则表达式应该匹配哪些位置。比如，我们想要匹配数字串。那么，我们就可以按以下方法进行测试：

  ```python
  text = "The price was $19.99 per item."
  pattern = r'\d+'
  result = re.findall(pattern, text)
  print(result)   # Output: ['19']
  
  # Now let's try to get more precise results by specifying where digits should be matched.
  pattern = r'\d+(\.\d+)?'
  result = re.findall(pattern, text)
  print(result)   # Output: ['$19.99']
  
  # Finally, we can modify our pattern to extract currency symbols and amounts separately.
  pattern = r'[$£€]\d+(,\d{3})*(\.\d+)?'
  result = re.findall(pattern, text)
  print(result)   # Output: ['$19.99']
  ```
 
  从以上输出结果可以看出，通过设置正确的正则表达式模式，我们可以得到我们想要的结果。  
  
  ## 2.简单匹配模式  
  普通字符 `.` 可以匹配任何字符，加上 `*` 可以匹配零个或多个字符。所以，我们可以使用 `.*` 来匹配任意长度的字符串，如下面的例子所示：

  ```python
  text = "Python programming is fun!"
  pattern = r".*"
  result = re.findall(pattern, text)
  print(result)   # Output: ['Python programming is fun!']
  ```
  
  上述例子匹配所有文本。注意，`.*` 会捕获整个字符串，除非后面还有其他字符。如果你只想匹配单个字符，可以使用 `.`。
  
  ## 3.断言字符  
  有时候，我们不仅要匹配普通的字符，还需要对字符做一些限制。比如，我们希望匹配电话号码，其中第一位不能为零。那么，我们可以使用 `\d{3}` 来匹配三个数字，再加上一个 `?` 来表示可选的 `0`，最终的模式如下：

  ```python
  phone_number = "(\+?\d{2}?)-?\d{3}-?\d{4}"
  text = "Contact us at +91-234-567890, (+91)-456-789001 or 123-456-7890."
  result = re.findall(phone_number, text)
  print(result)   # Output: ['+91-234-567890', '+91-456-789001']
  ```
  
  这里，`?` 符号用来表示该字符可选，`()` 的作用是创建组，`.` 的行为类似 `.*`，`-` 符号用来匹配破折号。通过这种方式，我们成功地匹配了两个电话号码。  
  
  ## 4.更多的匹配模式  
  除了常用的断言字符外，还有一些匹配模式可以帮助我们更好地定位文本信息。比如，`\s` 可以匹配空白字符，`\S` 可以匹配非空白字符。由于 `\s` 包括空格、制表符、换行符等字符，所以 `\s+` 可以匹配至少有一个空白字符的字符串。另外，还有 `\w`、`\d`、`\b` 等字符，分别用来匹配英文字母、数字和单词边界、非单词边界等。通过各种组合，我们可以构造出各种不同的匹配模式。下面我们以匹配日期时间为例，演示如何用各种匹配模式来达到目的：

  ```python
  date_time = "\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}(am|pm)"
  text = "My birthday is 05/02/1990, and my appointment time is 9:00 am."
  result = re.findall(date_time, text)
  print(result)   # Output: ['05/02/1990 9:00am']
  ```
  
  这里，`\d{1,2}` 表示两位数字；`\d{1,2}` 表示两位数字；`\d{4}` 表示四位数字；`(am|pm)` 表示 am 或 pm 中的一个。`|` 表示或，括号内的内容匹配其中的一个。  
  
  此外，还有一些模式可以用来提取更丰富的信息。比如，我们可以用 `\w+` 来匹配单词，然后配合 `(?: )` 来分组。下面是一个例子：

  ```python
  name = r"([A-Za-z']+)\s([A-Za-z']+)"
  text = "My name is John Doe."
  result = re.findall(name, text)
  print(result)   # Output: [('John', 'Doe')]
  ```
  
  这里，`([A-Za-z']+)` 表示一个或多个由字母或单引号组成的单词，也就是名字；`\s` 表示空白字符，之后是一个空白字符；同样的，`([A-Za-z']+)` 表示另一个由字母或单引号组成的单词，也就是姓氏。然后，`(?: )` 表示创建了一个分组。通过这个分组，我们可以提取出名字和姓氏，而不是匹配到的整体。