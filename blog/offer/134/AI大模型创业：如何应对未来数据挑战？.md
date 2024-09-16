                 

### 主题：AI大模型创业：如何应对未来数据挑战？

#### 一、面试题库

**1. 什么是数据倾斜？如何解决数据倾斜问题？**

**答案：** 数据倾斜指的是在数据处理过程中，某些特征或者数据点占据了数据集的大部分，导致其他特征或数据点无法得到充分的利用。解决数据倾斜的方法包括：

- **采样法：** 对数据集进行采样，减少数据量，从而平衡数据分布。
- **特征加权法：** 对倾斜的特征进行加权处理，降低其影响。
- **数据转换法：** 对倾斜的特征进行转换，如离散化、标准化等。
- **缺失值处理法：** 对缺失值进行填充或删除，减少对数据倾斜的影响。

**2. 数据库性能优化的常见方法有哪些？**

**答案：** 数据库性能优化的常见方法包括：

- **索引优化：** 创建合适的索引，减少查询时间。
- **查询优化：** 改善查询语句，如使用 join 而不是子查询，避免笛卡尔积等。
- **缓存机制：** 使用缓存机制，如 Redis、Memcached 等，提高数据读取速度。
- **分库分表：** 将数据分散存储到多个数据库或表中，减少单表数据量，提高查询效率。
- **读写分离：** 将读操作和写操作分离到不同的服务器，提高并发能力。

**3. 数据预处理过程中常见的算法有哪些？**

**答案：** 数据预处理过程中常见的算法包括：

- **缺失值处理：** 使用均值、中位数、众数等方法进行填充或删除。
- **异常值处理：** 使用统计学方法（如 Z-Score、IQR）或机器学习方法（如聚类、回归）进行异常值检测和处理。
- **数据转换：** 进行数据归一化、标准化、离散化等操作，提高数据质量。
- **特征工程：** 通过构造新的特征、选择相关特征等方法，提高模型性能。

**4. 什么是数据降维？常用的数据降维算法有哪些？**

**答案：** 数据降维是指从高维数据中提取关键信息，减少数据维度，同时保持数据的代表性。常用的数据降维算法包括：

- **主成分分析（PCA）：** 通过最大化方差的方式提取主要成分，降低数据维度。
- **线性判别分析（LDA）：** 通过最大化类间方差和最小化类内方差的方式提取特征，适用于分类问题。
- **非负矩阵分解（NMF）：** 通过非负矩阵分解的方式提取特征，适用于非负数据。
- **t-SNE：** 通过非线性映射将高维数据映射到二维或三维空间，适用于可视化。

**5. 机器学习中常见的性能评估指标有哪些？**

**答案：** 机器学习中常见的性能评估指标包括：

- **准确率（Accuracy）：** 分类正确的样本数占总样本数的比例。
- **召回率（Recall）：** 真正属于某一类的样本中被正确分类的样本数占总样本数的比例。
- **精确率（Precision）：** 被正确分类为某一类的样本中被分类正确的比例。
- **F1 值（F1 Score）：** 精确率和召回率的调和平均值。
- **ROC 曲线：** 受试者操作特性曲线，通过计算真阳性率（真正率）和假阳性率（假正率）的交叉点评估模型性能。
- **AUC（Area Under Curve）：** ROC 曲线下方的面积，用于评估模型分类能力。

**6. 什么是过拟合？如何避免过拟合？**

**答案：** 过拟合是指模型在训练数据上表现良好，但在测试数据上表现不佳，即模型对训练数据的拟合程度过高，无法泛化到未知数据。

避免过拟合的方法包括：

- **正则化：** 通过在损失函数中加入正则项，限制模型复杂度。
- **交叉验证：** 使用交叉验证方法，将数据集划分为多个子集，训练和验证多个模型，选择性能最好的模型。
- **集成学习：** 将多个模型结合起来，提高模型泛化能力。
- **数据增强：** 通过添加噪声、旋转、翻转等方式增加数据多样性。
- **早停法（Early Stopping）：** 在训练过程中，当验证集性能不再提高时，提前停止训练。

**7. 如何处理不平衡数据集？**

**答案：** 处理不平衡数据集的方法包括：

- **重采样：** 通过随机过采样、随机欠采样、SMOTE 等，调整数据集比例。
- **代价敏感：** 在损失函数中加入权重，提高误分类代价，使模型更加关注少数类。
- **生成对抗网络（GAN）：** 使用生成对抗网络生成少数类样本，平衡数据集。
- **迁移学习：** 使用预训练模型，结合少量少数类样本，提高模型对少数类的识别能力。

**8. 什么是模型调参？常用的模型调参方法有哪些？**

**答案：** 模型调参是指调整模型超参数，以优化模型性能。常用的模型调参方法包括：

- **网格搜索（Grid Search）：** 预先设定超参数范围，遍历所有可能的组合，选择性能最好的组合。
- **随机搜索（Random Search）：** 从超参数范围内随机选择组合，寻找最优参数。
- **贝叶斯优化（Bayesian Optimization）：** 使用贝叶斯优化算法，根据先验知识和历史数据，选择下一个最有希望的超参数组合。

**9. 什么是卷积神经网络（CNN）？它适用于哪些任务？**

**答案：** 卷积神经网络是一种深度学习模型，主要用于图像识别、图像分类、目标检测等任务。CNN 通过卷积层、池化层和全连接层等结构，实现对图像特征的提取和分类。

**10. 什么是循环神经网络（RNN）？它适用于哪些任务？**

**答案：** 循环神经网络是一种基于时间序列数据的深度学习模型，可以处理变量长度序列数据。RNN 适用于自然语言处理、语音识别、时间序列预测等任务。

**11. 什么是生成对抗网络（GAN）？它适用于哪些任务？**

**答案：** 生成对抗网络是一种深度学习模型，由生成器和判别器组成。生成器生成虚假数据，判别器判断数据真假。GAN 适用于图像生成、图像超分辨率、数据增强等任务。

**12. 什么是深度强化学习（DRL）？它适用于哪些任务？**

**答案：** 深度强化学习是深度学习和强化学习的结合，使用神经网络作为代理，进行决策和优化。DRL 适用于智能博弈、自动驾驶、机器人控制等任务。

**13. 什么是迁移学习？它如何工作？**

**答案：** 迁移学习是指利用已有模型的知识，在新任务上提高模型性能。迁移学习通过在源任务和目标任务之间共享特征表示，实现知识迁移。

**14. 什么是数据流编程？它有哪些优势？**

**答案：** 数据流编程是一种编程范式，强调数据依赖关系和计算过程的动态调度。数据流编程的优势包括：

- **并行处理：** 数据流模型可以自动并行处理数据。
- **弹性伸缩：** 数据流编程支持动态调整计算资源。
- **易于维护：** 数据流编程的代码结构清晰，易于维护。

**15. 什么是图神经网络（GNN）？它适用于哪些任务？**

**答案：** 图神经网络是一种基于图结构的数据处理模型，可以捕捉节点和边之间的关系。GNN 适用于社交网络分析、推荐系统、图像分类等任务。

**16. 什么是图卷积网络（GCN）？它如何工作？**

**答案：** 图卷积网络是一种基于图结构的神经网络，通过卷积操作计算节点特征。GCN 将图中的节点看作卷积核，对邻居节点的特征进行加权求和，实现特征聚合。

**17. 什么是图注意力机制（GAT）？它如何工作？**

**答案：** 图注意力机制是一种基于图结构的注意力机制，通过计算节点之间的相似性，对邻居节点的特征进行加权。GAT 可以自适应地调整节点之间的权重，提高模型性能。

**18. 什么是自注意力机制（Self-Attention）？它如何工作？**

**答案：** 自注意力机制是一种基于序列数据的注意力机制，通过计算序列中每个元素之间的相似性，对元素进行加权。自注意力机制可以捕捉序列中的长距离依赖关系。

**19. 什么是Transformer 模型？它有哪些优势？**

**答案：** Transformer 模型是一种基于自注意力机制的深度学习模型，用于处理序列数据。Transformer 的优势包括：

- **并行处理：** Transformer 可以并行处理序列，提高计算效率。
- **长距离依赖：** Transformer 可以捕捉序列中的长距离依赖关系。
- **适应性：** Transformer 可以自适应地调整权重，提高模型性能。

**20. 什么是BERT 模型？它如何工作？**

**答案：** BERT（Bidirectional Encoder Representations from Transformers）是一种基于 Transformer 模型的预训练语言模型，通过双向编码器生成文本的表示。BERT 工作原理包括：

- **预训练：** BERT 在大量文本数据上进行预训练，学习文本的表示。
- **Masked Language Modeling（MLM）：** 在预训练过程中，随机遮挡部分文本，预测遮挡部分的内容。
- **Next Sentence Prediction（NSP）：** 在预训练过程中，预测两个句子之间的顺序。

#### 二、算法编程题库

**1. 寻找两个有序数组中的中位数。**

**题目描述：** 给定两个大小分别为 m 和 n 的有序数组 nums1 和 nums2，请你找出并返回这两个数组中的中位数。

**示例：**

```
nums1 = [1, 3]
nums2 = [2]

则中位数是 2。因此，返回 2。

```

**答案解析：** 本题可以使用二分查找的方法来解决。假设 nums1 和 nums2 的中点分别为 m1 和 m2，那么中位数可以表示为：

- 如果 m1 + m2 是偶数，中位数为 (m1 + m2) / 2；
- 如果 m1 + m2 是奇数，中位数为 (m1 + m2 + 1) / 2。

具体实现步骤如下：

1. 定义一个二分查找函数，用于在数组 nums1 中查找中点 m1；
2. 调用二分查找函数，在数组 nums2 中查找中点 m2；
3. 根据 m1 和 m2 的值，计算中位数。

**代码实现：**

```python
def findMedianSortedArrays(nums1, nums2):
    def findMedian(nums1, nums2, l1, r1, l2, r2):
        total_len = l1 + r1 + l2 + r2
        if l1 > r1:
            return (max(nums2[l2], nums1[l1]), True)
        if l2 > r2:
            return (max(nums1[l1], nums2[l2]), True)
        if l1 + r1 == 0:
            return (nums1[l1], False)
        if l2 + r2 == 0:
            return (nums2[l2], False)

        if (total_len + 1) % 2 == 0:
            mid = (total_len // 2) - 1
            l = min(mid, r1)
            r = max(mid, r1)
            if l == r:
                return (nums1[l], True)
            m1 = nums1[l] if nums1[l] > nums2[l2] else nums2[l2]
            l = min(mid + 1, r1)
            r = max(mid + 1, r1)
            m2 = nums1[l] if nums1[l] > nums2[l2] else nums2[l2]
            return (m1 + m2) / 2, True
        else:
            mid = (total_len + 1) // 2
            l = min(mid, r1)
            r = max(mid, r1)
            if l == r:
                return (nums1[l], True)
            m = max(nums1[l], nums2[l2])
            return m, True

    return findMedian(nums1, nums2, 0, len(nums1) - 1, 0, len(nums2) - 1)[0]
```

**2. 有效的括号。**

**题目描述：** 给定一个字符串，判断是否是有效的括号字符串。

**示例：**

```
输入：")(()))"
输出：false

```

**答案解析：** 使用栈实现。遍历字符串，遇到左括号入栈，遇到右括号出栈。如果字符串长度为偶数，且栈为空，则为有效括号字符串。

**代码实现：**

```python
class Solution:
    def isValid(self, s: str) -> bool:
        st = []
        for c in s:
            if c in '([{':
                st.append(c)
            else:
                if not st:
                    return False
                top = st.pop()
                if c == ')' and top != '(':
                    return False
                if c == ']' and top != '[':
                    return False
                if c == '}' and top != '{':
                    return False
        return not st
```

**3. 二进制中 1 的个数。**

**题目描述：** 给定一个非负整数，计算并返回其二进制表示中 1 的个数。

**示例：**

```
输入：n = 00000000000000000000000000001011
输出：3
```

**答案解析：** 使用位运算。通过不断将 n 右移，并判断最低位是否为 1，统计 1 的个数。

**代码实现：**

```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        cnt = 0
        while n:
            cnt += n & 1
            n >>= 1
        return cnt
```

**4. 合并两个有序链表。**

**题目描述：** 将两个升序链表合并为一个新的升序链表并返回。

**示例：**

```
输入：l1 = [1, 4, 5], l2 = [1, 3, 4]
输出：[1, 1, 3, 4, 4, 5]
```

**答案解析：** 使用虚拟头节点。创建一个虚拟头节点，遍历两个链表，比较当前节点值，将较小的节点添加到新链表中。

**代码实现：**

```python
class Solution:
    def mergeTwoLists(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0)
        curr = dummy
        while l1 and l2:
            if l1.val < l2.val:
                curr.next = l1
                l1 = l1.next
            else:
                curr.next = l2
                l2 = l2.next
            curr = curr.next
        curr.next = l1 if l1 else l2
        return dummy.next
```

**5. 盲人猜牌问题。**

**题目描述：** 有三个人（A，B，C）和两副牌（红桃和黑桃），现在将每副牌分成三份，分别给这三个人。然后让他们各自闭着眼睛拿走一份牌。由于每人的牌是随机分的，所以每人拿到的牌都是两张不同的牌。

接下来，他们每个人将手中的一张牌亮出，并告诉其他人手中没有这张牌。这样，每个人都能看到其他两人手中的牌，但看不到自己手中的牌。

现在，问题来了：你能推测出每张牌的颜色吗？

**答案解析：** 如果我们能通过逻辑推理确定每张牌的颜色，那么问题就可以解决。以下是一个可能的解决方案：

1. 假设 A 亮出的牌是红桃，那么 B 和 C 都知道 A 没有红桃。
2. 如果 B 亮出的牌是黑桃，那么 C 必须亮出红桃，因为 C 没有黑桃。
3. 如果 B 亮出的牌是红桃，那么 C 必须亮出黑桃，因为 C 没有红桃。
4. 如果 C 亮出的牌是黑桃，那么 A 必须亮出红桃，因为 A 没有黑桃。
5. 如果 C 亮出的牌是红桃，那么 A 必须亮出黑桃，因为 A 没有红桃。

通过以上推理，我们可以确定每张牌的颜色。

**6. 简化路径。**

**题目描述：** 给定一个字符串表示的简化路径，请你将其还原为原始路径。

**示例：**

```
输入："/a/./b/.."
输出："/a/b"
```

**答案解析：** 使用栈实现。遍历路径字符串，遇到路径分隔符（/）时，将当前路径入栈。遇到特殊字符（.）时，表示当前路径不变，直接跳过。遇到双点符（..）时，表示当前路径向上回退一级，将栈顶元素出栈。

**代码实现：**

```python
class Solution:
    def simplifyPath(self, path: str) -> str:
        st = []
        for part in path.split('/'):
            if part == '..':
                if st:
                    st.pop()
            elif part and part != '.':
                st.append(part)
        return '/' + '/'.join(st)
```

**7. 合并两个有序数组。**

**题目描述：** 给定两个有序数组 nums1 和 nums2，请你将 nums2 合并到 nums1 中，使得 nums1 成为一个有序数组。

**示例：**

```
输入：nums1 = [1,2,3,0,0,0], m = 3
nums2 = [2,5,6], n = 3
输出：[1,2,2,3,5,6]
```

**答案解析：** 从 nums1 的最后一个位置开始填充，比较两个数组的元素，将较大的元素填充到 nums1 的末尾。填充完成后，剩余的空位用 0 填充。

**代码实现：**

```python
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        i = m + n - 1
        m -= 1
        n -= 1
        while m >= 0 and n >= 0:
            if nums1[m] > nums2[n]:
                nums1[i] = nums1[m]
                m -= 1
            else:
                nums1[i] = nums2[n]
                n -= 1
            i -= 1
        while n >= 0:
            nums1[i] = nums2[n]
            i -= 1
            n -= 1
```

**8. 翻转字符串中的单词 III。**

**题目描述：** 给定一个字符串，你需要反转字符串中的每个单词。

**示例：**

```
输入："Let's take LeetCode contest"
输出："s'teL ekat edoCateL ecne"
```

**答案解析：** 使用栈实现。遍历字符串，遇到空格时，将当前单词入栈。遍历完成后，将栈中的单词依次出栈，用空格连接，形成新的字符串。

**代码实现：**

```python
class Solution:
    def reverseWords(self, s: str) -> str:
        st = []
        for c in s:
            if c != ' ':
                st.append(c)
            elif st:
                st.append(c)
                st.reverse()
        st.reverse()
        return ''.join(st)
```

**9. 验证回文串。**

**题目描述：** 给定一个字符串，请你确定是否为回文串。

**示例：**

```
输入："A man, a plan, a canal: Panama"
输出：true
```

**答案解析：** 使用双指针。初始化两个指针，一个指向字符串开头，一个指向字符串结尾。遍历字符串，比较两个指针指向的字符是否相同，同时跳过非字母和数字字符。如果两个指针相遇，则为回文串。

**代码实现：**

```python
class Solution:
    def isPalindrome(self, s: str) -> bool:
        i, j = 0, len(s) - 1
        while i < j:
            while i < j and not s[i].isalnum():
                i += 1
            while i < j and not s[j].isalnum():
                j -= 1
            if s[i].lower() != s[j].lower():
                return False
            i, j = i + 1, j - 1
        return True
```

**10. 两数之和。**

**题目描述：** 给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那两个整数，并返回他们的数组下标。

**示例：**

```
输入：nums = [2, 7, 11, 15], target = 9
输出：[0, 1]
解释：因为 nums[0] + nums[1] == 9，返回 [0, 1]。
```

**答案解析：** 使用哈希表。遍历数组，对每个元素，计算 target - nums[i]，然后判断差值是否在哈希表中。如果在，返回当前元素的索引和差值的索引。

**代码实现：**

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        m = {v: i for i, v in enumerate(nums)}
        for i, v in enumerate(nums):
            j = target - v
            if j in m and m[j] != i:
                return [i, m[j]]
        return []
```

**11. 最长公共前缀。**

**题目描述：** 编写一个函数来查找字符串数组中的最长公共前缀。

**示例：**

```
输入：strs = ["flower","flow","flight"]
输出："fl"
```

**答案解析：** 分而治之。将字符串数组分为两半，递归求解左右两半的最长公共前缀。然后将左右两半的前缀取公共部分。

**代码实现：**

```python
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        def lcp(left, right):
            if left > right:
                return ""
            mid = (left + right) // 2
            l = lcp(left, mid)
            r = lcp(mid + 1, right)
            return l if l == r else l[:min(len(l), len(r))]
        return lcp(0, len(strs) - 1)
```

**12. 两数相加。**

**题目描述：** 给出两个非空链表表示两个非负整数，每个节点包含一个数字。将这两个数相加并返回一个新的链表表示和。

**示例：**

```
输入：l1 = [2, 4, 3], l2 = [5, 6, 4]
输出：[7, 0, 7]
```

**答案解析：** 遍历两个链表，逐位相加，进位处理。

**代码实现：**

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0)
        cur = dummy
        carry = 0
        while l1 or l2 or carry:
            val1 = (l1 and l1.val) or 0
            val2 = (l2 and l2.val) or 0
            cur.next = ListNode((val1 + val2 + carry) % 10)
            carry = (val1 + val2 + carry) // 10
            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next
            cur = cur.next
        return dummy.next
```

**13. 合并两个有序链表。**

**题目描述：** 给定两个有序链表 l1 和 l2，将它们合并为一个新的有序链表并返回。新链表通过附加节点的方式构建。

**示例：**

```
输入：l1 = [1,2,4], l2 = [1,3,4]
输出：[1,1,2,3,4,4]
```

**答案解析：** 使用虚拟头节点。遍历两个链表，比较当前节点值，将较小的节点添加到新链表中。

**代码实现：**

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def mergeTwoLists(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0)
        cur = dummy
        while l1 and l2:
            if l1.val < l2.val:
                cur.next = l1
                l1 = l1.next
            else:
                cur.next = l2
                l2 = l2.next
            cur = cur.next
        cur.next = l1 or l2
        return dummy.next
```

**14. 翻转整数。**

**题目描述：** 给你一个 32 位的有符号整数 x，返回将 x 中的数字部分翻转后的结果。

**示例：**

```
输入：x = 123
输出：321
输入：x = -123
输出：-321
输入：x = 120
输出：21
```

**答案解析：** 负数反转问题。判断数字是否为负数，反转数字后判断是否越界。

**代码实现：**

```python
class Solution:
    def reverse(self, x: int) -> int:
        INT_MAX = 2**31 - 1
        INT_MIN = -2**31
        is_negative = x < 0
        x = abs(x)
        new_val = 0
        while x:
            if new_val > INT_MAX // 10 or (new_val == INT_MAX // 10 and x % 10 > 7):
                return 0
            new_val = new_val * 10 + x % 10
            x //= 10
        return -new_val if is_negative else new_val
```

**15. 合并两个有序数组。**

**题目描述：** 给你两个整数数组 nums1 和 nums2 ，请你将 nums2 合并到 nums1 中，使 nums1 成为一个有序数组。

**示例：**

```
输入：nums1 = [1,2,3,0,0,0]，m = 3
nums2 = [2,5,6]，n = 3
输出：[1,2,2,3,5,6]
```

**答案解析：** 双指针。从数组末尾开始填充，比较两个数组中的元素，将较大的元素填充到目标数组的末尾。

**代码实现：**

```python
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        i, j, k = m - 1, n - 1, m + n - 1
        while i >= 0 and j >= 0:
            if nums1[i] > nums2[j]:
                nums1[k] = nums1[i]
                i -= 1
            else:
                nums1[k] = nums2[j]
                j -= 1
            k -= 1
        while j >= 0:
            nums1[k] = nums2[j]
            j -= 1
            k -= 1
```

**16. 两数相加。**

**题目描述：** 给出两个 非空 的链表用来表示两个非负的整数。其中，它们各自的位数是相同的。如果一个链表数字较长，则在该链表中缺失的位数将
```
### 17. 颠倒字符串中的单词 III

**题目描述：** 给定一个字符串，你需要反转字符串中每个单词的字符顺序，单词之间用单个空格隔开。

**示例：**

```
输入："Let's take LeetCode contest"
输出："s'teL ekat edoCateL ecne"
```

**答案解析：** 使用两个栈。遍历字符串，遇到空格时，将当前单词入栈。遍历完成后，将栈中的单词依次出栈，用空格连接，形成新的字符串。

**代码实现：**

```python
class Solution:
    def reverseWords(self, s: str) -> str:
        st = []
        for c in s:
            if c != ' ':
                st.append(c)
            elif st:
                st.append(c)
                st.reverse()
        st.reverse()
        return ''.join(st)
```

**18. 爬楼梯。**

**题目描述：** 一个楼梯有 n 阶台阶，每次可以上一阶或两阶。请计算上楼梯的总方法数。

**示例：**

```
输入：n = 3
输出：3
解释：上楼梯的方法有：
1. 1 + 1 + 1 = 3
2. 1 + 2 = 3
3. 2 + 1 = 3
```

**答案解析：** 动态规划。定义一个数组 dp，dp[i] 表示上到第 i 阶台阶的方法数。状态转移方程为：dp[i] = dp[i - 1] + dp[i - 2]。

**代码实现：**

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        if n <= 2:
            return n
        dp = [0] * (n + 1)
        dp[1], dp[2] = 1, 2
        for i in range(3, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[n]
```

**19. 有效的括号。**

**题目描述：** 给定一个包含括号的字符串，判断其是否有效。

**示例：**

```
输入："()"
输出：true
输入：")("
输出：false
```

**答案解析：** 使用栈。遍历字符串，遇到左括号入栈，遇到右括号出栈。如果字符串长度为偶数，且栈为空，则为有效括号字符串。

**代码实现：**

```python
class Solution:
    def isValid(self, s: str) -> bool:
        st = []
        for c in s:
            if c in '([{':
                st.append(c)
            else:
                if not st:
                    return False
                top = st.pop()
                if c == ')' and top != '(':
                    return False
                if c == ']' and top != '[':
                    return False
                if c == '}' and top != '{':
                    return False
        return not st
```

**20. 数据流中的中位数。**

**题目描述：** 设计一个数据结构，能够添加元素并获取当前数据流的中位数。

**示例：**

```
stream MedianFinder()
stream.addNum(1)
stream.findMedian() -> 1
stream.addNum(2)
stream.findMedian() -> 1.5
stream.addNum(3)
stream.findMedian() -> 2
```

**答案解析：** 使用两个堆。一个最大堆存放较小的一半数据，一个最小堆存放较大的一半数据。添加元素时，将元素添加到较大堆中，然后根据大小关系调整堆。

**代码实现：**

```python
import heapq

class MedianFinder:
    def __init__(self):
        self.small = []  # 最大堆
        self.large = []  # 最小堆

    def addNum(self, num: int) -> None:
        heapq.heappush(self.large, -num)
        heapq.heappush(self.small, -heapq.heappop(self.large))
        if len(self.small) > len(self.large):
            heapq.heappush(self.large, -heapq.heappop(self.small))

    def findMedian(self) -> float:
        if len(self.small) == len(self.large):
            return (-self.small[0] + self.large[0]) / 2
        else:
            return float(-self.small[0])
```

### 二、算法编程题库

**1. 字符串转换整数 (atoi)**

**题目描述：** 请实现一个函数，将其转换为整数。

**示例：**

```
输入："42"
输出：42
输入："   -42"
输出：-42
输入："4193 with words"
输出：4193
```

**答案解析：** 遍历字符串，判断字符是否为数字，将数字字符转换为整数，同时处理正负号和溢出情况。

**代码实现：**

```python
def myAtoi(s: str) -> int:
    INT_MAX = 2**31 - 1
    INT_MIN = -2**31
    sign = 1
    result = 0
    i = 0
    while i < len(s) and s[i] == ' ':
        i += 1
    if i < len(s) and (s[i] == '+' or s[i] == '-'):
        sign = -1 if s[i] == '-' else 1
        i += 1
    while i < len(s) and s[i].isdigit():
        digit = ord(s[i]) - ord('0')
        if result > (INT_MAX - digit) // 10:
            return INT_MAX if sign == 1 else INT_MIN
        result = result * 10 + digit
        i += 1
    return result * sign
```

**2. 爬楼梯 II**

**题目描述：** 一个楼梯有 n 阶台阶，每次可以上一阶或两阶。请计算上楼梯的总方法数。

**示例：**

```
输入：n = 2
输出：2
解释：上楼梯的方法有：
1. 1 + 1 = 2
2. 2 = 2
```

**答案解析：** 动态规划。定义一个数组 dp，dp[i] 表示上到第 i 阶台阶的方法数。状态转移方程为：dp[i] = dp[i - 1] + dp[i - 2]。

**代码实现：**

```python
def climbStairs(n: int) -> int:
    if n <= 2:
        return n
    dp = [0] * (n + 1)
    dp[1], dp[2] = 1, 2
    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]
```

**3. 三数之和**

**题目描述：** 给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值 target 的三个整数，并返回索引。你可以按任意顺序返回三个下标。

**示例：**

```
输入：nums = [-1, 0, 1, 2, -1, -4], target = 0
输出：[-1, 0, 1], 2
解释：nums[2] + nums[4] + nums[5] = -1 + 0 + 1 = 0
```

**答案解析：** 双指针。先对数组进行排序，然后遍历数组，对于每个元素，使用双指针找到与该元素相加为目标值的两个元素。

**代码实现：**

```python
def threeSum(nums: List[int], target: int) -> List[List[int]]:
    nums.sort()
    result = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        left, right = i + 1, len(nums) - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total < target:
                left += 1
            elif total > target:
                right -= 1
            else:
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
    return result
```

**4. 合并两个有序链表**

**题目描述：** 给定两个有序链表 l1 和 l2，将它们合并为一个新的有序链表并返回。

**示例：**

```
输入：l1 = [1,2,4], l2 = [1,3,4]
输出：[1,1,2,3,4,4]
```

**答案解析：** 递归。递归合并两个链表，每次比较当前节点值，将较小的节点添加到新链表中。

**代码实现：**

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
def mergeTwoLists(l1, l2):
    if not l1:
        return l2
    if not l2:
        return l1
    if l1.val < l2.val:
        l1.next = mergeTwoLists(l1.next, l2)
        return l1
    else:
        l2.next = mergeTwoLists(l1, l2.next)
        return l2
```

**5. 删除链表的节点**

**题目描述：** 给定一个单链表的头节点 head 和一个整数 val，请删除链表中值为 val 的节点。

**示例：**

```
输入：head = [4,5,1,9], val = 5
输出：[4,1,9]
```

**答案解析：** 遍历链表，找到值为 val 的节点，将其前一个节点的 next 指针指向 val 节点的下一个节点，删除 val 节点。

**代码实现：**

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
def deleteNode(head: ListNode, val: int) -> ListNode:
    if head.val == val:
        return head.next
    prev = head
    while prev.next:
        if prev.next.val == val:
            prev.next = prev.next.next
            return head
        prev = prev.next
    return head
```

**6. 合并多个表**

**题目描述：** 给定多个有序表，合并它们并返回一个有序的列表。

**示例：**

```
输入：lists = [[1,4,5], [1,3,4], [2,6]]
输出：[1,1,2,3,4,4,5,6]
```

**答案解析：** 使用优先队列。将每个有序表的第一个元素加入优先队列，每次从优先队列中取出最小值，加入到结果列表中。取出最小值后，将下一个元素加入优先队列。

**代码实现：**

```python
from queue import PriorityQueue

def mergeKLists(lists):
    pq = PriorityQueue()
    for l in lists:
        if l:
            pq.put((l[0], l))
    result = []
    while not pq.empty():
        val, l = pq.get()
        result.append(val)
        if l and l.next:
            pq.put((l.next[0], l.next))
    return result
```

**7. 最长公共前缀**

**题目描述：** 给定一个字符串数组，找到它们的公共前缀。

**示例：**

```
输入：["flower","flow","flight"]
输出："fl"
```

**答案解析：** 遍历字符串数组，每次取第一个字符串和第二个字符串的最长公共前缀，然后将结果和第三个字符串继续比较，依次类推。

**代码实现：**

```python
def longestCommonPrefix(strs):
    if not strs:
        return ""
    prefix = strs[0]
    for s in strs[1:]:
        i = 0
        while i < len(prefix) and i < len(s):
            if prefix[i] != s[i]:
                break
            i += 1
        prefix = prefix[:i]
    return prefix
```

**8. 盲人猜牌问题**

**题目描述：** 有三个人（A，B，C）和两副牌（红桃和黑桃），现在将每副牌分成三份，分别给这三个人。然后让他们各自闭着眼睛拿走一份牌。由于每人的牌是随机分的，所以每人拿到的牌都是两张不同的牌。

接下来，他们每个人将手中的一张牌亮出，并告诉其他人手中没有这张牌。这样，每个人都能看到其他两人手中的牌，但看不到自己手中的牌。

现在，问题来了：你能推测出每张牌的颜色吗？

**答案解析：** 如果我们能通过逻辑推理确定每张牌的颜色，那么问题就可以解决。以下是一个可能的解决方案：

1. 假设 A 亮出的牌是红桃，那么 B 和 C 都知道 A 没有红桃。
2. 如果 B 亮出的牌是黑桃，那么 C 必须亮出红桃，因为 C 没有黑桃。
3. 如果 B 亮出的牌是红桃，那么 C 必须亮出黑桃，因为 C 没有红桃。
4. 如果 C 亮出的牌是黑桃，那么 A 必须亮出红桃，因为 A 没有黑桃。
5. 如果 C 亮出的牌是红桃，那么 A 必须亮出黑桃，因为 A 没有红桃。

通过以上推理，我们可以确定每张牌的颜色。

**9. 简化路径**

**题目描述：** 给定一个字符串表示的简化路径，请你将其还原为原始路径。

**示例：**

```
输入："/a/./b/.."
输出："/a/b"
```

**答案解析：** 使用栈。遍历路径字符串，遇到路径分隔符（/）时，将当前路径入栈。遇到特殊字符（.）时，表示当前路径不变，直接跳过。遇到双点符（..）时，表示当前路径向上回退一级，将栈顶元素出栈。

**代码实现：**

```python
def simplifyPath(path: str) -> str:
    stack = []
    for token in path.split('/'):
        if token == '..':
            if stack:
                stack.pop()
        elif token and token != '.':
            stack.append(token)
    return '/' + '/'.join(stack)
```

**10. 合并两个有序数组**

**题目描述：** 给定两个整数数组 nums1 和 nums2，请你将 nums2 合并到 nums1 中，使得 nums1 成为一个有序数组。

**示例：**

```
输入：nums1 = [1,2,3,0,0,0]，m = 3
nums2 = [2,5,6]，n = 3
输出：[1,2,2,3,5,6]
```

**答案解析：** 双指针。从数组末尾开始填充，比较两个数组中的元素，将较大的元素填充到 nums1 的末尾。

**代码实现：**

```python
def merge(nums1, m, nums2, n):
    i = m - 1
    j = n - 1
    k = m + n - 1
    while i >= 0 and j >= 0:
        if nums1[i] > nums2[j]:
            nums1[k] = nums1[i]
            i -= 1
        else:
            nums1[k] = nums2[j]
            j -= 1
        k -= 1
    while j >= 0:
        nums1[k] = nums2[j]
        j -= 1
        k -= 1
```

**11. 翻转单词序列**

**题目描述：** 输入一个字符串，输出该字符串的反序字符。

**示例：**

```
输入："I am a student."
输出："tuedents a ma I."
```

**答案解析：** 使用栈。遍历字符串，遇到空格时，将当前单词入栈。遍历完成后，将栈中的单词依次出栈，用空格连接，形成新的字符串。

**代码实现：**

```python
def reverseWords(s):
    st = []
    for c in s:
        if c != ' ':
            st.append(c)
        elif st:
            st.append(c)
            st.reverse()
    st.reverse()
    return ''.join(st)
```

**12. 字符串转换整数 (atoi)**

**题目描述：** 请实现一个函数，将其转换为整数。

**示例：**

```
输入："42"
输出：42
输入："   -42"
输出：-42
输入："4193 with words"
输出：4193
```

**答案解析：** 遍历字符串，判断字符是否为数字，将数字字符转换为整数，同时处理正负号和溢出情况。

**代码实现：**

```python
def myAtoi(s):
    INT_MAX = 2**31 - 1
    INT_MIN = -2**31
    sign = 1
    result = 0
    i = 0
    while i < len(s) and s[i] == ' ':
        i += 1
    if i < len(s) and (s[i] == '+' or s[i] == '-'):
        sign = -1 if s[i] == '-' else 1
        i += 1
    while i < len(s) and s[i].isdigit():
        digit = ord(s[i]) - ord('0')
        if result > (INT_MAX - digit) // 10:
            return INT_MAX if sign == 1 else INT_MIN
        result = result * 10 + digit
        i += 1
    return result * sign
```

**13. 最长公共前缀**

**题目描述：** 给定一个字符串数组，找到它们的公共前缀。

**示例：**

```
输入：["flower","flow","flight"]
输出："fl"
```

**答案解析：** 遍历字符串数组，每次取第一个字符串和第二个字符串的最长公共前缀，然后将结果和第三个字符串继续比较，依次类推。

**代码实现：**

```python
def longestCommonPrefix(strs):
    if not strs:
        return ""
    prefix = strs[0]
    for s in strs[1:]:
        i = 0
        while i < len(prefix) and i < len(s):
            if prefix[i] != s[i]:
                break
            i += 1
        prefix = prefix[:i]
    return prefix
```

**14. 翻转单词序列**

**题目描述：** 输入一个字符串，输出该字符串的反序字符。

**示例：**

```
输入："I am a student."
输出："tuednets a ma I."
```

**答案解析：** 使用栈。遍历字符串，遇到空格时，将当前单词入栈。遍历完成后，将栈中的单词依次出栈，用空格连接，形成新的字符串。

**代码实现：**

```python
def reverseWords(s):
    st = []
    for c in s:
        if c != ' ':
            st.append(c)
        elif st:
            st.append(c)
            st.reverse()
    st.reverse()
    return ''.join(st)
```

**15. 简化路径**

**题目描述：** 给定一个字符串表示的简化路径，请你将其还原为原始路径。

**示例：**

```
输入："/a/./b/.."
输出："/a/b"
```

**答案解析：** 使用栈。遍历路径字符串，遇到路径分隔符（/）时，将当前路径入栈。遇到特殊字符（.）时，表示当前路径不变，直接跳过。遇到双点符（..）时，表示当前路径向上回退一级，将栈顶元素出栈。

**代码实现：**

```python
def simplifyPath(path):
    stack = []
    for token in path.split('/'):
        if token == '..':
            if stack:
                stack.pop()
        elif token and token != '.':
            stack.append(token)
    return '/' + '/'.join(stack)
```

**16. 合并两个有序数组**

**题目描述：** 给定两个整数数组 nums1 和 nums2，请你将 nums2 合并到 nums1 中，使得 nums1 成为一个有序数组。

**示例：**

```
输入：nums1 = [1,2,3,0,0,0]，m = 3
nums2 = [2,5,6]，n = 3
输出：[1,2,2,3,5,6]
```

**答案解析：** 双指针。从数组末尾开始填充，比较两个数组中的元素，将较大的元素填充到 nums1 的末尾。

**代码实现：**

```python
def merge(nums1, m, nums2, n):
    i = m - 1
    j = n - 1
    k = m + n - 1
    while i >= 0 and j >= 0:
        if nums1[i] > nums2[j]:
            nums1[k] = nums1[i]
            i -= 1
        else:
            nums1[k] = nums2[j]
            j -= 1
        k -= 1
    while j >= 0:
        nums1[k] = nums2[j]
        j -= 1
        k -= 1
```

**17. 验证回文串**

**题目描述：** 给定一个字符串，请你确定是否为回文串。

**示例：**

```
输入："A man, a plan, a canal: Panama"
输出：true
```

**答案解析：** 使用双指针。初始化两个指针，一个指向字符串开头，一个指向字符串结尾。遍历字符串，比较两个指针指向的字符是否相同，同时跳过非字母和数字字符。

**代码实现：**

```python
def isPalindrome(s: str) -> bool:
    i, j = 0, len(s) - 1
    while i < j:
        while i < j and not s[i].isalnum():
            i += 1
        while i < j and not s[j].isalnum():
            j -= 1
        if s[i].lower() != s[j].lower():
            return False
        i, j = i + 1, j - 1
    return True
```

**18. 无重复字符的最长字串**

**题目描述：** 给定一个字符串 s ，找出其中不含有重复字符的最长子串的最长长度。

**示例：**

```
输入："abcabcbb"
输出：3
解释："abc" 是无重复字符的最长子串。
```

**答案解析：** 滑动窗口。初始化两个指针，一个指向窗口的左边界，一个指向窗口的右边界。遍历字符串，如果当前字符在窗口中已存在，则移动左边界，直到当前字符不在窗口中。更新最长子串的长度。

**代码实现：**

```python
def lengthOfLongestSubstring(s):
    left, right = 0, 0
    max_len = 0
    window = set()
    while right < len(s):
        if s[right] in window:
            left = max(left, window[s[right]] + 1)
        window.add(s[right])
        max_len = max(max_len, right - left + 1)
        right += 1
    return max_len
```

**19. 两数之和**

**题目描述：** 给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那两个整数，并返回他们的数组下标。

**示例：**

```
输入：nums = [2, 7, 11, 15], target = 9
输出：[0, 1]
解释：因为 nums[0] + nums[1] = 2 + 7 = 9，返回 [0, 1]。
```

**答案解析：** 哈希表。遍历数组，对每个元素，计算 target - nums[i]，然后判断差值是否在哈希表中。如果在，返回当前元素的索引和差值的索引。

**代码实现：**

```python
def twoSum(nums, target):
    m = {v: i for i, v in enumerate(nums)}
    for i, v in enumerate(nums):
        j = target - v
        if j in m and m[j] != i:
            return [i, m[j]]
    return []
```

**20. 最长公共子序列**

**题目描述：** 给定两个字符串 text1 和 text2，返回他们的最长公共子序列的长度。如果不存在公共子序列，返回 0。

**示例：**

```
输入：text1 = "abcde", text2 = "ace"
输出：3
解释：最长公共子序列是 "ace"，它的长度为 3。
```

**答案解析：** 动态规划。定义一个二维数组 dp，dp[i][j] 表示 text1 的前 i 个字符和 text2 的前 j 个字符的最长公共子序列长度。状态转移方程为：

- 如果 text1[i - 1] == text2[j - 1]，则 dp[i][j] = dp[i - 1][j - 1] + 1；
- 如果 text1[i - 1] != text2[j - 1]，则 dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])。

最终答案为 dp[m][n]，其中 m 和 n 分别为 text1 和 text2 的长度。

**代码实现：**

```python
def longestCommonSubsequence(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]
```

**21. 合并两个有序链表**

**题目描述：** 给定两个有序的链表 list1 和 list2，将它们合并成一个有序链表并返回。

**示例：**

```
输入：list1 = [1,2,4], list2 = [1,3,4]
输出：[1,1,2,3,4,4]
```

**答案解析：** 递归。递归合并两个链表，每次比较当前节点值，将较小的节点添加到新链表中。

**代码实现：**

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
def mergeTwoLists(list1, list2):
    if not list1:
        return list2
    if not list2:
        return list1
    if list1.val < list2.val:
        list1.next = mergeTwoLists(list1.next, list2)
        return list1
    else:
        list2.next = mergeTwoLists(list1, list2.next)
        return list2
```

**22. 旋转图像**

**题目描述：** 给定一个 n × n 的二维矩阵 matrix 表示一个图像。请你将图像顺时针旋转 90 度。

**示例：**

```
输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[[7,4,1],[8,5,2],[9,6,3]]
```

**答案解析：** 旋转图像可以通过以下步骤实现：

1. 先沿对角线进行翻转；
2. 再沿垂直中线进行翻转。

**代码实现：**

```python
def rotate(matrix):
    n = len(matrix)
    # 沿对角线进行翻转
    for i in range(n):
        for j in range(i, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    # 沿垂直中线进行翻转
    for i in range(n):
        matrix[i] = matrix[i][::-1]
    return matrix
```

**23. 盲人猜牌问题**

**题目描述：** 有三个人（A，B，C）和两副牌（红桃和黑桃），现在将每副牌分成三份，分别给这三个人。然后让他们各自闭着眼睛拿走一份牌。由于每人的牌是随机分的，所以每人拿到的牌都是两张不同的牌。

接下来，他们每个人将手中的一张牌亮出，并告诉其他人手中没有这张牌。这样，每个人都能看到其他两人手中的牌，但看不到自己手中的牌。

现在，问题来了：你能推测出每张牌的颜色吗？

**答案解析：** 如果我们能通过逻辑推理确定每张牌的颜色，那么问题就可以解决。以下是一个可能的解决方案：

1. 假设 A 亮出的牌是红桃，那么 B 和 C 都知道 A 没有红桃。
2. 如果 B 亮出的牌是黑桃，那么 C 必须亮出红桃，因为 C 没有黑桃。
3. 如果 B 亮出的牌是红桃，那么 C 必须亮出黑桃，因为 C 没有红桃。
4. 如果 C 亮出的牌是黑桃，那么 A 必须亮出红桃，因为 A 没有黑桃。
5. 如果 C 亮出的牌是红桃，那么 A 必须亮出黑桃，因为 A 没有红桃。

通过以上推理，我们可以确定每张牌的颜色。

**代码实现：**

这个问题需要通过逻辑推理来解决问题，无法直接用代码实现。但是，我们可以通过以下步骤来描述解题过程：

```python
def guess_cards(cards):
    red_hearts = []
    black_spades = []
    
    for card in cards:
        if card == 'R':
            red_hearts.append(card)
        else:
            black_spades.append(card)
    
    if red_hearts:
        for card in red_hearts:
            if card == 'R':
                black_spades.remove('B')
            else:
                black_spades.remove('R')
    else:
        for card in black_spades:
            if card == 'R':
                red_hearts.remove('B')
            else:
                red_hearts.remove('R')
    
    return red_hearts, black_spades
```

**24. 数据流中的中位数**

**题目描述：** 设计一个数据结构，能够添加元素并获取当前数据流的中位数。

**示例：**

```
stream MedianFinder()
stream.addNum(1)
stream.findMedian() -> 1
stream.addNum(2)
stream.findMedian() -> 1.5
stream.addNum(3)
stream.findMedian() -> 2
```

**答案解析：** 使用两个堆。一个最大堆存放较小的一半数据，一个最小堆存放较大的一半数据。添加元素时，将元素添加到较大堆中，然后根据大小关系调整堆。

**代码实现：**

```python
import heapq

class MedianFinder:

    def __init__(self):
        self.small = []  # 最大堆
        self.large = []  # 最小堆

    def addNum(self, num: int) -> None:
        heapq.heappush(self.large, -num)
        heapq.heappush(self.small, -heapq.heappop(self.large))
        if len(self.small) > len(self.large):
            heapq.heappush(self.large, -heapq.heappop(self.small))

    def findMedian(self) -> float:
        if len(self.small) == len(self.large):
            return (-self.small[0] + self.large[0]) / 2
        else:
            return float(-self.large[0])
```

**25. 打家劫舍**

**题目描述：** 你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。

给定一个代表每个房屋存放金额的非负整数数组，计算你在不触动警报装置的情况下，能够偷窃到的最高金额。

**示例：**

```
输入：[1,2,3,1]
输出：4
解释：偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。
     偷窃到的最高金额 = 1 + 3 = 4。
```

**答案解析：** 动态规划。定义一个数组 dp，dp[i] 表示前 i 个房屋能够偷窃到的最高金额。状态转移方程为：dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])。

**代码实现：**

```python
def rob(nums: List[int]) -> int:
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    dp = [0] * len(nums)
    dp[0], dp[1] = nums[0], max(nums[0], nums[1])
    for i in range(2, len(nums)):
        dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
    return dp[-1]
```

**26. 打家劫舍 II**

**题目描述：** 你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。

给你一个非空数组，表示每间房屋的存放金额，计算你在不触动警报装置的情况下，能够偷窃到的最高金额。

**示例：**

```
输入：[2,3,2]
输出：3
解释：你不能先偷窃 1 号房屋（金额 = 2），然后偷窃 3 号房屋（金额 = 2）, 因为他们是相邻的。
输入：[1,2,3,1]
输出：4
解释：你可以先偷窃 1 号房屋（金额 = 1），然后偷窃 3 号房屋（金额 = 3）。
```

**答案解析：** 动态规划。分两种情况：

1. 不偷窃最后一个房屋：dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])；
2. 偷窃最后一个房屋：dp[i] = max(dp[i - 1], dp[i - 2] + nums[i - 1])。

最终答案为两者的最大值。

**代码实现：**

```python
def rob(nums: List[int]) -> int:
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    if len(nums) == 2:
        return max(nums)
    dp = [0] * len(nums)
    dp[0], dp[1] = nums[0], max(nums[0], nums[1])
    for i in range(2, len(nums) - 1):
        dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
    dp[-1] = max(dp[-1], dp[-2] + nums[-1])
    return max(dp[-1], dp[-2])
```

**27. 三数之和**

**题目描述：** 给定一个包含 n 个整数的数组 nums，判断 nums 中是否含有三个元素 a，b，c ，使得 a + b + c = 0 ？找出满足条件的 a，b，c 的值。

**示例：**

```
输入：nums = [-1,0,1,2,-1,-4]
输出：[-1,0,1]
解释：当 a = -1，b = 0，c = 1 时，满足 a + b + c = 0 ，
```

**答案解析：** 双指针。先对数组进行排序，然后遍历数组，对于每个元素，使用双指针找到与该元素相加为目标值的两个元素。

**代码实现：**

```python
def threeSum(nums: List[int], target: int) -> List[List[int]]:
    nums.sort()
    result = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        left, right = i + 1, len(nums) - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total < target:
                left += 1
            elif total > target:
                right -= 1
            else:
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
    return result
```

**28. 最小路径和**

**题目描述：** 给定一个包含非负整数的 m x n 网格 grid ，找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。

**示例：**

```
输入：grid = [[1,3,1],[1,5,1],[4,2,1]]
输出：7
解释：因为路径 1→3→1→1→1 的总和最小。
```

**答案解析：** 动态规划。定义一个二维数组 dp，dp[i][j] 表示从左上角到 (i, j) 的最小路径和。状态转移方程为：dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]。

**代码实现：**

```python
def minPathSum(grid: List[List[int]]) -> int:
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = grid[0][0]
    for i in range(1, m):
        dp[i][0] = dp[i - 1][0] + grid[i][0]
    for j in range(1, n):
        dp[0][j] = dp[0][j - 1] + grid[0][j]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]
    return dp[-1][-1]
```

**29. 股票价格波动**

**题目描述：** 给定一个整数数组 prices，其中 prices[i] 是一支给定股票第 i 天的价格。

设计一个算法来计算从第一天到第 n 天的日用最小金额买一次卖一次该股票可以获得的利润。

**示例：**

```
输入：prices = [7,1,5,3,6,4]
输出：5
解释：在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     注意利润不能是 7-1 = 6，因为卖出价格需要大于买入价格；同时，你不能在买入前卖出股票。
```

**答案解析：** 双指针。定义两个指针，一个指向当前买入的位置，一个指向当前卖出的位置。遍历数组，比较两个指针之间的差值，更新最小差值。

**代码实现：**

```python
def minProfit(prices):
    n = len(prices)
    if n < 2:
        return 0
    min_diff = float('inf')
    for i in range(1, n):
        min_diff = min(min_diff, prices[i] - prices[i - 1])
    return min_diff
```

**30. 合并两个有序链表**

**题目描述：** 给定两个单链表，链表中元素的值已经按照递增顺序排列，要求将它们合并成一个有序的单链表。

**示例：**

```
输入：l1 = [1,2,4]，l2 = [1,3,4]
输出：[1,1,2,3,4,4]
```

**答案解析：** 递归。递归合并两个链表，每次比较当前节点值，将较小的节点添加到新链表中。

**代码实现：**

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
def mergeTwoLists(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    if not l1:
        return l2
    if not l2:
        return l1
    if l1.val < l2.val:
        l1.next = mergeTwoLists(l1.next, l2)
        return l1
    else:
        l2.next = mergeTwoLists(l1, l2.next)
        return l2
```

