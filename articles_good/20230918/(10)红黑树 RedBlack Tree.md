
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 什么是红黑树？
红黑树（英语：Red–black tree）是一种自平衡二叉查找树，它在20世纪70年代由Richard McMeek教授发明，是一种高度组织的数据结构。该树是为了解决普通的二叉查找树在最坏情况下可能出现的严重性能问题而设计出来的。

## 为什么要用红黑树？
红黑树是一种复杂的平衡数据结构，它的效率不及其他平衡数据结构如AVL树或伸展树，但却比一般平衡二叉树更适合文件系统、数据库索引、缓存等需要大量动态查询的数据结构。因此，对于要求高效查找、插入删除元素的应用场合，红黑树是一个值得考虑的选择。


# 2.基本概念术语说明
## 2.1 节点定义
**红色结点（red node）**：如果一个节点是红色的，那么它左右子节点都是黑色的。
**黑色结点（black node）**：如果一个节点是黑色的，那么它左右子节点都是红色的或者都为空（NIL）。
**叶子结点（leaf node）**：如果一个节点没有孩子节点，那么它就是一个叶子结点。
**空结点（NIL）**：空结点是树中的一种特殊节点，它既没有键也没有值，表示一个空白位置。
**路径长度（path length）**：从根节点到某一节点所经过边的数量称作该节点的路径长度。
## 2.2 基本操作定义
### 插入操作：将一个新节点插入到红黑树中。
### 删除操作：删除一个红黑树中的一个节点。
### 搜索操作：在红黑树中查找给定键值的节点。
### 旋转操作：当发生特定条件时，将红黑树的结构进行调整，使之重新平衡。
## 2.3 相关术语定义
**子女结点（child node）**：对于一个父结点，其某个孩子指向的节点，就叫做这个孩子的“子女结点”。
**兄弟结点（sibling node）**：对于同一层级上的两个结点，除了父结点外，其他所有结点，这些结点互称兄弟。
**祖先结点（ancestor node）**：设n为某节点，它的所有祖先指的是从根到n的路径上所有的节点。例如，某节点的祖先结点包括父节点、祖父节点、曾祖父节点等等。
**子孙结点（descendant node）**：设n为某节点，它的所有子孙指的是以n为根的子树中所有节点。
**堂兄弟结点（uncle node）**：对于一个结点，它在同样的父亲结点下的另一个子女结点，就叫做该结点的“堂兄弟结点”。
**分支因子（branching factor）**：对于一个具有n个结点的红黑树，它的分支因子是指根节点到任意叶子节点的最大距离。通常来说，分支因子越大，红黑树的结构越平衡；分支因子越小，红黑树的结构越不平衡。
**双亲指针（parent pointer）**：每个结点有且只有一个父结点，或者根节点没有父结点，表示着它的父节点为空。对每个非空结点，其父亲指针总是指向该结点的直接前驱。
**颜色属性（color attribute）**：每一个结点可以标记为红色或黑色，用来表征结点的状态。在RB-tree中，红色表示结点是红色的，黑色表示结点是黑色的。
**森林（forest）**：红黑树的集合称作红黑树的森林。森林可以由多个根结点组成，并且在相同时间内可能包含不同的红黑树。

# 3.核心算法原理和具体操作步骤以及数学公式讲解
## 3.1 插入操作
红黑树中插入一个新节点的过程如下图所示：


- **Step 1：**将新节点插入到红黑树中，将它的颜色设置为红色。
- **Step 2：**从新节点开始，沿着其父亲结点方向不断向上追溯直至根结点。
- **Step 3：**在每个结点上进行以下判断：
  - 如果遇到了已经为黑色的结点，则停止追溯，并将当前结点改为红色。
  - 如果新的红色结点已经连续相邻两个结点都是红色的，则将它们变为黑色。

经过以上两步后，红黑树的性质被保持了。

接下来，让我们看看如何利用数学方法证明红黑树的插入操作的时间复杂度是O(lgn)。

## 3.2 数学证明
假设红黑树上有n个结点，且每个结点的高度为h=lgn。我们将在任一结点处切分一个新结点，并根据这两个结点之间的关系进行颜色的调整，以使红黑树仍然平衡。

### 3.2.1 初始状态的红黑树
首先，考虑一棵初始状态的红黑树，其结构如下图所示：


因为它是一棵二叉树，因此它的高度为$log_2 n$。但是，由于不是完美平衡的树，因此红色结点多于黑色结点。这样会导致$lgn$个黑色结点对应一个红色结点。因此，红黑树的任何结点至少有一个黑色孩子，也就是说：

$$\text{黑色孩子数量} \geq lgn$$

因此，在初始状态下，红黑树的高度为$lgn+1$，此时一条链路长度为$\frac{\lambda}{2}\cdot lgn$。

### 3.2.2 插入操作
红黑树在插入操作中做出的修改主要有以下几点：

1. 一个新结点可能破坏红黑树的性质，因此需要修正它使之满足红黑树的性质。
2. 每次插入之后都会产生一次旋转操作，使得树保持平衡。

因此，红黑树的平均情况的时间复杂度为$O(\frac{lgn}{\sqrt{2}})$. 为了保证在最坏情况下的时间复杂度也为$O(lgn)$，我们必须采用旋转操作来保持红黑树的平衡。

#### 3.2.2.1 插入操作分析
插入操作的分析依赖于两个性质：

1. 在任何位置上插入一个新结点，必然使得树保持红黑树性质。
2. 经过一系列的旋转操作，可以使得树保持为平衡的状态。

因此，如果插入操作正确执行，则两个性质同时满足。我们首先分析如何在红黑树中插入一个新的结点。

#### 3.2.2.2 插入新结点的操作
插入一个新结点的操作的伪代码如下：

```
RBTreeInsert(T, z):
    y = nil[T] // initialize a new leaf node as the successor of z
    
    while T!= nil:
        y = parent[T]
        
        if key[z] < key[T]:
            T = left[T]
        else:
            T = right[T]
            
    add z to its proper position in the tree
        
    make z black
    
end RBTreeInsert()
```

在这个伪代码中，`nil[T]`是一个空的叶子节点。我们假设在插入操作中不会出现新的节点，只需要在已有的节点之间进行插入。

#### 3.2.2.3 插入操作正确性证明
**第一个性质**：插入操作必然保持红黑树性质。

因为新结点的父结点已经存在于树中，所以它至少有两个红色结点作为孩子。因此，由第四节的讨论可知，如果当前结点为根结点，则它的父结点必为黑色结点，且它至少有一个黑色孩子，即$lgn$个黑色结点至少有一个红色结点。

假设在某一结点t处插入了一个新的结点z，且z的父结点p存在，则在新增的结点后，结点p的子树可能失去平衡。因此，我们必须修正这一失衡状态。

设x是新插入的结点，它和p的相邻结点q形成了一条红色路径。设p是黑色结点，并且x、q是红色结点。由于红黑树的性质，黑色结点数量至少为$lgn$。由于p为黑色结点，其至少有$lgn-1$个相邻的黑色结点。因此，在新增的结点后，p的两个相邻结点至少有一个为黑色结点，而且还有一个孩子为红色结点。因此，有如下两种情形：

1. 当z是p的唯一孩子：
   - 这种情况是最简单的。因此，我们只需将z变为红色即可。
   - 此时的红黑树仍然保持平衡，只是缺少一条路径上的黑色结点。因此，我们只需将新的路径上的结点补充为黑色即可。

2. 当z也是p的右孩子，且新插入的结点在左侧：
   - 这种情况是稍微复杂一些。因此，我们只需再一次对z做一次左旋操作，然后将p涂黑，即可完成修正。

根据上述两个方案，我们证明插入操作一定保持红黑树性质。

**第二个性质**：插入操作必然对树的高度和红黑树的性质产生影响。

插入操作维护了红黑树的性质，从而可以有效减少树的高度。插入操作在新结点被添加到树之后，其高度必然大于等于原树的高度，所以树的高度将增加1。

因此，插入操作时间复杂度为$O(\log n)$，并且是最坏情况下的时间复杂度。

## 3.3 删除操作
红黑树中删除一个节点的过程如下图所示：


- **Step 1：**将被删掉的结点替换为它的后继结点，后继结点位于待删除结点的左子树或右子树中，它在此过程中被称为替身节点（substitution node）。
- **Step 2：**如果替身节点存在，则对它进行处理，以恢复红黑树性质。

下面，让我们详细探讨红黑树的删除操作。

### 3.3.1 删除操作定义
在红黑树中，删除一个节点通常称作“删除”。其定义如下：

1. 从红黑树中删除一个节点后，红黑树依然是一颗红黑树。
2. 被删除的节点的颜色必须为黑色。
3. 有三种类型的节点可以被删除：
   - 叶子节点（Leaf Node）：不含键值信息的节点，只占据自己一格。
   - 单个孩子的节点（One Child Node）：一个孩子的节点，它仅有左或右孩子。
   - 含有两个孩子的节点（Two Child Node）：有两个孩子的节点，它同时含有左右两个孩子。

### 3.3.2 删除叶子节点
在红黑树中删除叶子节点比较简单，只需要将其从树中移除就可以了。

比如，假设我们希望删除节点31，其左右孩子均为空，即它是一个叶子节点。可以将其直接从树中移除：


### 3.3.3 删除单个孩子节点
如果待删除的节点仅有一个孩子，则可以将其直接替换父节点的位置，也可以利用它的唯一孩子来取代自己的位置，然后删除唯一孩子。

比如，假设我们希望删除节点15，其右孩子为7。可以将15替换为7，然后将节点7移入树中。删除节点7后，树变为：


### 3.3.4 删除含有两个孩子节点
如果待删除的节点含有两个孩子，则不能直接删除，只能找到其后继节点，然后将其值替换到待删除节点上，最后删除后继节点。

比如，假设我们希望删除节点26，其左右孩子分别为25和33，找后继节点33，可以将其值复制给26，然后删除节点33，得到：


## 3.4 搜索操作
搜索操作在红黑树中通常使用二叉搜索树的方法实现。当我们在红黑树中搜索一个给定的键值时，我们可以沿着路径一直往下走，直到找到给定的键值或定位到一个空位置。

## 3.5 旋转操作
旋转操作用于将红黑树的结构调整为平衡状态。当红黑树的失衡使得搜索、插入或删除操作无法继续进行时，需要进行旋转操作来使红黑树重新平衡。

比如，当我们删除一个结点时，失衡会导致整个树失去平衡。因此，为了保持平衡，我们需要通过一次旋转操作来使树重新平衡。

旋转操作分为两类，一是单旋转（single rotation），二是双旋转（double rotation）。

### 3.5.1 单旋转
单旋转是指将某个结点的右旋或者左旋，使得整棵树重新满足红黑树性质。

**左旋操作**：将红色结点的右子树移动到它的左边，且将右子树的根结点（若存在）染色为红色：


左旋操作可以看作是将一条长连接从右边拉短，并从中间穿插上结点，最终拉出了一根新的左支线。

**右旋操作**：将红色结点的左子树移动到它的右边，且将左子树的根结点染色为红色：


右旋操作可以看作是将一条长连接从左边拉长，并从中间穿插上结点，最终拉出了一根新的右支线。

### 3.5.2 双旋转
双旋转是指在单旋转的基础上，进一步旋转结点的一条边。比如，对于右旋操作而言，可以顺便将整个子树的根结点移动到右边。

**右旋转后的左倾双旋操作**：


左倾双旋操作是指将一个双边结点的一边拉长，并将其分离成两个单边结点，其中一个单边结点带回左支线。

**右旋转后的右倾双旋操作**：


右倾双旋操作是指将一个双边结点的一边缩短，并将其分离成两个单边结点，其中一个单边结点带回右支线。

# 4.具体代码实例和解释说明
本节将以C++语言为例，介绍红黑树插入、删除和搜索的具体操作代码。

## 4.1 节点定义
红黑树中各节点的结构体定义如下：

```cpp
struct TreeNode {
    int val;
    bool color;   // true for red false for black
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : val(x), color(true), left(nullptr), right(nullptr) {}
};
```

- `val`：节点存储的值。
- `color`：节点颜色，true代表红色，false代表黑色。
- `left`，`right`：节点的左右孩子指针。

## 4.2 插入操作
红黑树的插入操作有以下步骤：

1. 将新节点插入到红黑树中。
2. 更新新增节点的父节点，以及新增节点的父节点的子树信息。
3. 对新增节点进行修正，以维持红黑树的性质。

插入操作的代码如下：

```cpp
TreeNode* insertNode(TreeNode* root, int val) {
    // create a new node with value and set it to red by default
    TreeNode* newNode = new TreeNode(val);

    // handle empty tree case
    if (!root) return newNode;

    // find the correct location for insertion
    TreeNode* current = root;
    while (current->val > val && current->left)
        current = current->left;

    // update children pointers
    if (current->val == val) delete currentNode; // already exists, replace old one instead of inserting a duplicate
    else if (current->val > val) current->left = newNode;
    else current->right = newNode;

    // recolor nodes after adding new node or updating existing node's child pointers
    newNode->left = nullptr;
    newNode->right = nullptr;
    newNode->color = RED;

    fixViolation(newNode);

    return root;
}


void fixViolation(TreeNode*& node) {
    while (node->parent && isRed(node->parent)) {
        TreeNode* grandParent = getParent(getParent(node));

        // check if uncle of this node is red
        if (isRed(grandParent->left)) {
            rotateRight(grandParent);    // perform right rotation on grandparent first

            // reset colors for grandparent and parent, we know that both are now BLACK due to right rotation
            grandParent->color = RED;
            node->parent->color = BLACK;
        } else if (isRed(grandParent->right)) {
            // nothing needs to be done here since our red node has no children with higher rank than us
            rotateLeft(node->parent);     // do not change directionality yet
            
            // set colors accordingly
            rotateRight(node);           // then do regular double rotation
            
        } else {
            // both uncles are black, we need to flip colors
            grandParent->color = RED;
            node->parent->color = BLACK;
            break;
        }
    }
    root->color = BLACK;    // root always remains black after all operations have completed
}


void rotateLeft(TreeNode*& root) {
    TreeNode* pivot = root->right;
    root->right = pivot->left;
    if (pivot->left) pivot->left->parent = root;
    pivot->parent = root->parent;
    if (!root->parent) root = pivot;      // special case when node becomes the root
    else if (root == root->parent->left) root->parent->left = pivot;
    else root->parent->right = pivot;
    pivot->left = root;
    root->parent = pivot;
}


void rotateRight(TreeNode*& root) {
    TreeNode* pivot = root->left;
    root->left = pivot->right;
    if (pivot->right) pivot->right->parent = root;
    pivot->parent = root->parent;
    if (!root->parent) root = pivot;       // special case when node becomes the root
    else if (root == root->parent->right) root->parent->right = pivot;
    else root->parent->left = pivot;
    pivot->right = root;
    root->parent = pivot;
}
```

插入操作的详细流程：

1. 创建一个新结点。
2. 寻找新增结点的正确位置，更新新增结点的父结点，以及父结点的子树信息。
3. 检查新增结点是否已经存在于红黑树中。
4. 设置新增结点的颜色为红色，默认值为红色。
5. 使用fixViolation函数检查新增结点是否违反红黑树性质，并修复该性质。

fixViolation函数用于在新增结点后修正树结构，以维持红黑树的性质。该函数的逻辑如下：

1. 获取新增结点的父结点的父结点。
2. 判断父结点的父结点是否为红色，如果是，则进行右旋操作。否则，判断父结点是否为红色，如果是，则进行左旋、右旋操作。
3. 如果双旋转后仍然存在违反红黑树性质的现象，则退出循环。否则，设置父结点和父结点的父结点的颜色，并结束修复动作。

## 4.3 删除操作
红黑树的删除操作有以下步骤：

1. 查找待删除的节点。
2. 用它的后继节点覆盖它。
3. 如果后继节点有左孩子，则找到最小值节点（后继节点的右子树中最左边的节点）。
4. 将后继节点的颜色设置为待删除节点的颜色。
5. 如果后继节点有左孩子，则让后继节点成为它的父节点的左孩子，否则让后继节点成为它的父节点的右孩子。
6. 删除后继节点。

删除操作的代码如下：

```cpp
TreeNode* removeNode(TreeNode* root, int val) {
    // find the node to be deleted and move its children to fill up gap
    TreeNode* currentNode = root;
    TreeNode* parentOfCurrentNode = nullptr;
    while (currentNode && currentNode->val!= val) {
        parentOfCurrentNode = currentNode;
        if (val < currentNode->val) currentNode = currentNode->left;
        else currentNode = currentNode->right;
    }

    if (!currentNode) return root;         // node doesn't exist, exit function

    // save values to be used later
    int replacementValue = currentNode->val;
    bool isRootBlack = isBlack(root);        // keep track of original color of root before removing any nodes
    TreeNode* temp = nullptr;

    if ((!currentNode->left ||!currentNode->right) &&!(currentNode->left && currentNode->right)) {    // node has at most one child
        temp = currentNode;                 // single child case
        if (temp->left) {
            temp->left->parent = temp->parent;
            temp->parent? (temp->parent->left = temp->left) : (root = temp->left);
        } else if (temp->right) {
            temp->right->parent = temp->parent;
            temp->parent? (temp->parent->right = temp->right) : (root = temp->right);
        } else {                               // temp has neither children, so just remove it from the tree
            temp->parent? ((temp->parent->left == temp)? (temp->parent->left = nullptr) : (temp->parent->right = nullptr))
                         : (root = nullptr);
            free(temp);                       // free memory associated with removed node
        }
    } else if (currentNode->left && currentNode->right) {     // node has two children
        temp = getSuccessor(currentNode);            // replace currrent with its successor
        replacementValue = temp->val;               // update deletion target node with its value
        if (temp->left) {                           // attach its left subtree to its predecessor's right side
            temp->left->parent = temp->parent;
            temp->parent? (temp->parent->right = temp->left) : (root = temp->left);
        }
        if (temp->right) {                          // attach its right subtree to its predecessor's right side
            temp->right->parent = temp->parent;
            temp->parent? (temp->parent->right = temp->right) : (root = temp->right);
        }
        *temp = *(currentNode);                      // copy information of currrent node to its successor
    }

    delete currentNode;                         // free memory associated with removed node

    // restore color of root node
    if (isRootBlack && root->left && root->right) root->color = RED;   // if both children of root are present, change root to red
    else if (root->left && isBlack(root->left)) root->color = RED;          // if only left child of root is black, change root to red
    else if (root->right && isBlack(root->right)) root->color = RED;        // if only right child of root is black, change root to red

    // start fixing violations starting from the new root
    if (root->color == RED) fixViolation(root);

    return root;                                // return updated root node
}


TreeNode* getSuccessor(TreeNode* delNode) {
    TreeNode* successor = nullptr;
    TreeNode* successorParent = nullptr;
    TreeNode* current = delNode->right;

    while (current) {                            // traverse right subtree to find minimum value node
        successorParent = successor;
        successor = current;
        current = current->left;
    }

    // special case where deleting node had only one right child, which will become root of tree again
    if (successorParent && isBlack(successor)) {
        // temporarily assign current node to successors right child to avoid modifying the data structure during iteration over keys
        successorParent->left = successor->right;
        successor->right = delNode->right;
        successor->right->parent = successor;
        successorParent = successor;
        successor = successor->right;
    }

    return successor;                             // return min value node
}


bool isRed(TreeNode* node) {
    return node && node->color == RED;
}


bool isBlack(TreeNode* node) {
    return node &&!node->color;
}
```

删除操作的详细流程：

1. 通过迭代的方式找到待删除结点，并记录待删除结点的父结点。
2. 如果待删除结点是叶子结点，则直接从树中删除该结点。
3. 如果待删除结点含有两个孩子，则找到后继结点（即待删除结点右子树中的最小结点），并复制待删除结点的颜色。
4. 删除后继结点，然后更新树结构。
5. 检查根结点的颜色，并开始修复违反红黑树性质的节点。
6. 返回更新的根结点。

## 4.4 搜索操作
红黑树的搜索操作基于二叉搜索树，其时间复杂度为O($log_2 n$)。搜索操作代码如下：

```cpp
TreeNode* searchNode(TreeNode* root, int val) {
    while (root && root->val!= val) {
        if (val < root->val) root = root->left;
        else root = root->right;
    }
    return root;
}
```

搜索操作的详细流程：

1. 初始化当前节点为根结点。
2. 不断比较当前节点的值和目标值，直至找到目标值或找不到目标值。
3. 返回找到的结点或nullptr。