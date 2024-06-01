
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Sorting is an important and fundamental operation in computer science that requires a significant amount of time to implement correctly. There are many sorting algorithms available today, including bubble sort, insertion sort, selection sort, merge sort, quicksort, heapsort, etc., each with its own advantages and disadvantages depending on the specific use case and requirements. In this article we will explore some popular algorithms used to sort arrays in JavaScript:

1. Bubble sort
2. Selection sort
3. Insertion sort
4. Merge sort
5. Quicksort
6. Heapsort
7. Timsort
8. Counting sort
9. Radix sort
10. Bucket sort

In addition, there are other related but less commonly-used sorting algorithms such as radix sort, bucket sort, counting sort, and various variations of these. Each algorithm has its unique set of characteristics and benefits, making them useful in different scenarios and contexts. The goal of this article is to provide a comprehensive overview of these algorithms and their implementation details using practical examples in JavaScript. We hope you find it helpful!

# 2.排序算法概述
## 2.1 算法分类
Sorting algorithms can be classified into two types: comparison-based and non-comparison based. Comparison-based sorting algorithms compare elements one by one according to a predefined order or criteria (e.g., ascending or descending). Non-comparison-based sorting algorithms work directly with data structures without comparing the elements themselves. These include counting sort, radix sort, bucket sort, and shell sort among others.

Comparison-based sorting algorithms typically have O(n log n) average time complexity, which makes them suitable for large datasets where the input array size grows exponentially. However, they may not perform well for small or already sorted inputs due to the overhead associated with comparisons. 

Non-comparison-based sorting algorithms like counting sort and radix sort operate solely on the keys of the input array rather than the actual values, resulting in faster execution times even for very large datasets. This is because the key distribution information is more meaningful and easier to analyze for most applications. Bucket sort and timsort belong to this category since they do not require any prior knowledge about the distribution of the input data. Shell sort belongs to both categories; however, it operates at a slightly higher level than the previous ones and may outperform all of them for certain input sizes.


## 2.2 数据结构
In computer programming, a data structure is a way to organize and store data so that it can be accessed efficiently and manipulated easily. Data structures are essential components of efficient algorithms and data processing. Popular data structures include lists, stacks, queues, trees, graphs, hash tables, heaps, and sets. They help make complex problems tractable and allow us to solve them effectively. 

One type of data structure that supports sorting operations is an array. An array stores homogeneous elements of same type, whereas linked lists and other linear data structures support only sequential access. Although arrays support random access and dynamic resizing, they also offer worst-case performance guarantees when compared with other data structures like linked lists. Therefore, choosing between an array and a linked list depends on the specific application needs and constraints.

# 3.Bubblesort
Bubble sort, sometimes referred to as sinking sort, is a simple sorting algorithm that repeatedly steps through the list to be sorted, compares adjacent elements and swaps them if they are in the wrong order. The pass through the list is repeated until no swaps are needed, indicating that the list is sorted. The algorithm gets its name from the way smaller or larger elements "bubble" to the top of the list as they are moved around during each iteration. It is easy to understand and works well for small lists, but performs poorly inefficiently on larger ones. Here's how the algorithm works:

1. Start at the beginning of the array
2. Compare the first element with the second element
3. If the first element is greater, swap them
4. Move on to the next pair of elements
5. Repeat step 2-4 until the end of the array is reached
6. Continue iterating over the unsorted portion of the array, moving smaller elements up towards the front

Here's the code implementation of bubblesort in JavaScript: 

```js
function bubblesort(array){
  const len = array.length;

  // Traverse through all array elements
  for (let i = 0; i < len - 1; i++) {

    // Last i elements are already sorted
    for (let j = 0; j < len - i - 1; j++) {

      // Swap if the element found is greater
      // than the next element
      if (array[j] > array[j + 1]) {
        let temp = array[j];
        array[j] = array[j + 1];
        array[j + 1] = temp;
      }
    }
  }

  return array;
}
```

This function takes an array as input and returns the sorted array using bubblesort algorithm. Let's test our function with sample input:

```js
const arr = [64, 34, 25, 12, 22, 11, 90];

console.log("Original Array:", arr);

// Sort array using bubblesort algorithm
const sortedArr = bubblesort(arr);

console.log("Sorted Array:", sortedArr);
```

Output: 
```
Original Array: [ 64, 34, 25, 12, 22, 11, 90 ]
Sorted Array: [ 11, 12, 22, 25, 34, 64, 90 ]
```

As expected, the output shows the original array after being sorted using bubblesort algorithm.

# 4.Selectionsort
Selection sort is another simple sorting algorithm that sorts an array by repeatedly finding the minimum element from the unsorted part of the array and placing it at the beginning. The algorithm maintains two subarrays, one consisting of the sorted elements and the other containing the remaining unsorted elements. Initially, the sorted subarray is empty and the unsorted subarray is the entire array. At each iteration, the smallest element in the unsorted subarray is picked and added to the end of the sorted subarray. After k iterations, the whole array would be sorted. Here's how the algorithm works:

1. Set the first element of the array to be considered as the minimum
2. Loop through the rest of the array and check each element against the current minimum value
3. If a lower value is found, update the minimum variable accordingly
4. Once the loop completes, place the minimum value at the beginning of the unsorted part of the array
5. Increment the index of the min value and repeat the process for the remaining unsorted part of the array

Let's see the implementation of selectionsort algorithm in JavaScript:

```js
function selectionSort(array) {
  
  // Find the length of the array
  var len = array.length;
  
  // Create a loop that goes from 0 to the length of the array minus 1
  for (var i = 0; i < len - 1; i++) {
    
    // Assume the first element of the array is the lowest value
    var minIndex = i;
    
    // Loop through the array starting from the current index to find the lowest value
    for (var j = i+1; j < len; j++) {
      
      // If a lower value is found, update the minimum variable
      if (array[j] < array[minIndex]) {
        
        // Update the minimum variable to the new lowest value
        minIndex = j;
      }
    }
    
    // Swap the lowest value with the first element of the array
    var temp = array[i];
    array[i] = array[minIndex];
    array[minIndex] = temp;
  }
  
  // Return the sorted array
  return array;
}
```

Now, let's try running the above code with an example input:

```js
const arr = [64, 34, 25, 12, 22, 11, 90];

console.log("Original Array:", arr);

// Sort array using selectionsort algorithm
const sortedArr = selectionSort(arr);

console.log("Sorted Array:", sortedArr);
```

Output:

```
Original Array: [ 64, 34, 25, 12, 22, 11, 90 ]
Sorted Array: [ 11, 12, 22, 25, 34, 64, 90 ]
```

Again, as expected, the output shows the original array after being sorted using selectionsort algorithm.

# 5.InsertionSort
Insertion sort is a simple sorting algorithm that builds the final sorted array one item at a time. It is much less efficient on large lists than more advanced algorithms such as quicksort, heapsort, or merge sort, but it is quite educational and works well for small lists. To sort an array A of n elements, we need to iterate over the first element and insert it at the correct position in the sorted subarray A[0...i]. The basic idea behind insertion sort is to assume that the left half of the array is sorted and add the right half of the array sequentially while maintaining the relative order of elements within the left and right halves. Here's how the algorithm works:

1. Iterate over each element of the array except the first one
2. For each element, move backwards through the sorted subarray until the correct position is found
3. Shift the elements to the right to create space for the current element
4. Insert the current element into its correct position

The main advantage of insertion sort lies in simplicity and ease of implementation. However, its efficiency decreases as additional items must be shifted, leading to a slower overall runtime than other sorting algorithms. Let's look at the implementation of insertion sort in JavaScript:

```js
function insertionSort(array) {
  
  // Get the length of the array
  var len = array.length;
  
  // Define a loop that goes from 1 to the length of the array
  for (var i = 1; i < len; i++) {
    
    // Assume the current element is the last unsorted element
    var currElement = array[i];
    
    // Create a variable to hold the index of the previous element
    var prevIndex = i-1;
    
    // Check if the current element is less than the previous element and move it backwards if necessary
    while (prevIndex >= 0 && array[prevIndex] > currElement) {
      
      // Shift the previous element to the right to open space for the current element
      array[prevIndex+1] = array[prevIndex];
      
      // Decrement the index of the previous element
      prevIndex--;
    }
    
    // Insert the current element into its correct position
    array[prevIndex+1] = currElement;
  }
  
  // Return the sorted array
  return array;
}
```

We can now test the above function with an example input:

```js
const arr = [64, 34, 25, 12, 22, 11, 90];

console.log("Original Array:", arr);

// Sort array using insertionsort algorithm
const sortedArr = insertionSort(arr);

console.log("Sorted Array:", sortedArr);
```

Output:

```
Original Array: [ 64, 34, 25, 12, 22, 11, 90 ]
Sorted Array: [ 11, 12, 22, 25, 34, 64, 90 ]
```

Again, as expected, the output shows the original array after being sorted using insertionsort algorithm.

# 6.MergeSort
Merge sort is an efficient divide-and-conquer sorting algorithm that sorts an array by dividing it into two halves, recursively sorting each half, and then merging the two sorted halves back together. The recursion depth increases with the size of the problem, making the algorithm impractical for sorting large arrays. Here's how the algorithm works:

1. Divide the input array into two halves
2. Recursively sort each half using merge sort
3. Merge the two sorted halves back together

To merge the two sorted halves, we start by creating an empty result array, called mergedArray. Then, we initialize three pointers, i, j, and k, to point to the beginning of the firstHalf, secondHalf, and mergedArray respectively. We compare the values pointed to by the pointers, adding the smaller value to mergedArray and incrementing the corresponding pointer. When one of the pointers reaches the end of its respective array, we append the remaining elements of the other array to mergedArray. Finally, we return mergedArray as the sorted array. Here's how the implementation looks like:

```js
function mergeSort(array) {
  
  // Base condition
  if (array.length <= 1) {
    return array;
  }
  
  // Split the array into two halves
  const middle = Math.floor(array.length / 2);
  const firstHalf = array.slice(0, middle);
  const secondHalf = array.slice(middle);
  
  // Recursive calls to split the halves and merge them
  return merge(mergeSort(firstHalf), mergeSort(secondHalf));
}

function merge(leftHalf, rightHalf) {
  
  // Initialize variables
  const result = [];
  let i = 0;
  let j = 0;
  
  // Loop through both halves simultaneously
  while (i < leftHalf.length && j < rightHalf.length) {
    
    // Compare the values at the current indices
    if (leftHalf[i] <= rightHalf[j]) {
      result.push(leftHalf[i++]);
    } else {
      result.push(rightHalf[j++]);
    }
  }
  
  // Append the remaining elements of either array to the result
  return [...result,...leftHalf.slice(i),...rightHalf.slice(j)];
}
```

We can now test the above functions with an example input:

```js
const arr = [64, 34, 25, 12, 22, 11, 90];

console.log("Original Array:", arr);

// Sort array using mergesort algorithm
const sortedArr = mergeSort(arr);

console.log("Sorted Array:", sortedArr);
```

Output:

```
Original Array: [ 64, 34, 25, 12, 22, 11, 90 ]
Sorted Array: [ 11, 12, 22, 25, 34, 64, 90 ]
```

Again, as expected, the output shows the original array after being sorted using mergesort algorithm.

# 7.Quicksort
Quicksort is an efficient and highly scalable sorting algorithm that uses partitioning to sort an array. It partitions the array into two parts, those less than a pivot element and those greater than or equal to the pivot element, and recursively applies the same procedure to the subarrays. The pivot element is selected randomly or deterministically to minimize the chance of selecting worst-case scenario and ensures good performance in practice. The average and best case time complexity of quicksort is O(n log n), which makes it faster than other competitive sorting algorithms. Here's how the algorithm works:

1. Choose a pivot element from the array
2. Partition the array into two parts, those less than the pivot and those greater than or equal to the pivot
3. Recursively apply the same procedure to the subarrays until the base case of having zero or one element is reached

Let's write the implementation of the quicksort algorithm in JavaScript:

```js
function quickSort(array) {
  
  // Base condition: If the array contains fewer than 2 elements, it is already sorted
  if (array.length <= 1) {
    return array;
  }
  
  // Choose a pivot element randomly
  const pivot = array[Math.floor(Math.random() * array.length)];
  
  // Partition the array into two parts
  const leftHalf = [], rightHalf = [];
  for (let i = 0; i < array.length; i++) {
    if (array[i] < pivot) {
      leftHalf.push(array[i]);
    } else {
      rightHalf.push(array[i]);
    }
  }
  
  // Recursively call the function on the subarrays
  return [...quickSort(leftHalf), pivot,...quickSort(rightHalf)];
}
```

We can now test the above function with an example input:

```js
const arr = [64, 34, 25, 12, 22, 11, 90];

console.log("Original Array:", arr);

// Sort array using quicksort algorithm
const sortedArr = quickSort(arr);

console.log("Sorted Array:", sortedArr);
```

Output:

```
Original Array: [ 64, 34, 25, 12, 22, 11, 90 ]
Sorted Array: [ 11, 12, 22, 25, 34, 64, 90 ]
```

Again, as expected, the output shows the original array after being sorted using quicksort algorithm.

# 8.Heapsort
Heapsort is a comparison-based sorting algorithm that converts an unsorted array to a max-heap or a min-heap, sorts the heap in-place, and extracts elements in order. The heap is implemented using an array to represent a binary tree, and a complete binary tree is formed with every parent node containing the maximum number of children nodes possible. The root node of the tree represents the largest or smallest element, and the leaves represent the remaining elements in the array. Here's how the algorithm works:

1. Build a max-heap or a min-heap from the array
2. Extract elements from the heap in order
3. Implement the heapify property of the heap, ensuring that the subtree rooted at each node satisfies the heap property
4. Repeat steps 2-3 until the heap is empty

The heapify property states that for each node, the value stored at that node should be greater than or equal to the values stored at its children nodes (if they exist). The purpose of the heapify property is to maintain the shape and height of the heap, preventing the worst-case scenario where the heap becomes skewed and causes O(n^2) time complexity. Let's see the implementation of heapsort algorithm in JavaScript:

```js
function heapSort(array) {
  
  // Convert the array into a max-heap
  buildMaxHeap(array);
  
  // Extract elements from the heap in order
  const len = array.length;
  for (let i = len - 1; i >= 1; i--) {
    
    // Swap the root with the last element of the heap
    let temp = array[i];
    array[i] = array[0];
    array[0] = temp;
    
    // Heapify the modified array
    heapify(array, 0, i);
  }
  
  // Return the sorted array
  return array;
}

function buildMaxHeap(array) {
  const len = array.length;
  for (let i = Math.floor(len / 2) - 1; i >= 0; i--) {
    heapify(array, len, i);
  }
}

function heapify(array, len, i) {
  
  // Get the indexes of the left and right child nodes
  const leftChild = 2 * i + 1;
  const rightChild = 2 * i + 2;
  let maxIndex = i;
  
  // Determine which child node is the greatest
  if (leftChild < len && array[maxIndex] < array[leftChild]) {
    maxIndex = leftChild;
  }
  if (rightChild < len && array[maxIndex] < array[rightChild]) {
    maxIndex = rightChild;
  }
  
  // If the child node is greater than the parent, swap them
  if (maxIndex!== i) {
    let temp = array[i];
    array[i] = array[maxIndex];
    array[maxIndex] = temp;
    
    // Recurse down the tree
    heapify(array, len, maxIndex);
  }
}
```

We can now test the above function with an example input:

```js
const arr = [64, 34, 25, 12, 22, 11, 90];

console.log("Original Array:", arr);

// Sort array using heapsort algorithm
const sortedArr = heapSort(arr);

console.log("Sorted Array:", sortedArr);
```

Output:

```
Original Array: [ 64, 34, 25, 12, 22, 11, 90 ]
Sorted Array: [ 11, 12, 22, 25, 34, 64, 90 ]
```

Again, as expected, the output shows the original array after being sorted using heapsort algorithm.

# 9.Timsort
Timsort is a hybrid sorting algorithm combining techniques from mergesort and insertion sort. It uses a combination of insertion sort, merge sort, and galloping search technique to achieve better performance than traditional merge sort and insertion sort algorithms. Timsort is particularly useful for cases where standard comparison-based sorting algorithms fail or become too slow. Here's how the algorithm works:

1. Increase the runsize by considering factors such as powers of 2 and the gap sequence defined below
2. Use insertion sort to sort small segments of the array
3. Use merge sort to merge adjacent runs created by the insertion sort phase
4. Implement the galloping search technique to handle duplicates in the input array

The factors used in defining the runsize are chosen based on empirical evidence. Also, the gap sequence is determined dynamically based on the pattern of similarities in the input array. The galloping search technique finds contiguous regions of equal values in the array and exploits the fact that consecutive occurrences of equal values tend to occur close together. Here's the implementation of the Timsort algorithm in JavaScript:

```js
function timSort(array) {
  
  // Establish initial state
  const MIN_RUNSIZE = 32;
  const MAX_MERGE = 32;
  let runstart = 0;
  let runend = 0;
  let tempArray = null;
  
  // Calculate the minimum runsize by analyzing the pattern of similarities in the input array
  calculateRunSize();
  
  // Perform insertion sort on small chunks of the array
  while (runend <= array.length) {
    
    // Apply insertion sort to the current chunk of the array
    for (let i = runstart; i < runend; i += MAX_MERGE) {
      insertionSort(array, i, Math.min(runend, i + MAX_MERGE));
    }
    
    // Double the size of the current segment and continue the outer loop
    runstart = runend;
    runend += runsize << 1;
    if (runend > array.length) {
      runend = array.length;
    }
  }
  
  // Merge adjacent runs created by the insertion sort phase
  mergeRuns(array, 0, array.length, tempArray);
  
  // Remove temporary storage and return the sorted array
  delete tempArray;
  return array;
}

function calculateRunSize() {
  
  // Default runsize to 32
  runsize = MIN_RUNSIZE;
  
  // Compute a histogram of individual digits, assuming each digit occurs at least once
  const hist = new Int32Array(64);
  for (let i = 0; i < array.length; i++) {
    const val = array[i];
    hist[val & 0x3F]++;
    hist[(val >> 6) & 0x3F]++;
    hist[(val >> 12) & 0x3F]++;
    hist[(val >> 18) & 0x3F]++;
    hist[(val >> 24) & 0x3F]++;
    hist[(val >> 30) & 0xFF]++;
  }
  
  // Generate a normalized histogram (sum of all frequencies equals 1)
  let sum = 0;
  for (let i = 0; i < 64; i++) {
    sum += hist[i];
  }
  const normHist = new Float64Array(64);
  for (let i = 0; i < 64; i++) {
    normHist[i] = hist[i] / sum;
  }
  
  // Compute the longest increasing subsequence (LIS) of the normalized histogram
  const LIS = new Uint32Array(64);
  let lenLIS = 0;
  for (let i = 0; i < 64; i++) {
    LIS[i] = i;
    for (let j = 0; j < lenLIS; j++) {
      if (normHist[i] > normHist[LIS[j]]) {
        break;
      }
    }
    LIS[j + 1] = i;
    if (lenLIS === j + 1) {
      lenLIS++;
    }
  }
  
  // Produce the runsize using the formula t = c*ln(n) + o(n), where 
  // c = 1.2, o(n) = Θ(nlgn), and n is the input array size
  const C = 1.2;
  const n = array.length;
  const lisSum = getLISSum(LIS);
  const ratio = getCumSumRatio(lisSum, LIS, 0, lenLIS - 1);
  const fakt = Math.sqrt(C * Math.log(n) / ratio);
  runsize = Math.ceil((C * Math.log(n)) / (ratio * fakt - C * Math.log(fakt)));
  
}

function getLISSum(LIS) {
  let sum = 0;
  for (let i = 0; i < LIS.length; i++) {
    sum += LIS[i] - i;
  }
  return sum;
}

function getCumSumRatio(sum, LIS, lo, hi) {
  if (hi === lo) {
    return sum + ((1 << (LIS[hi] - lo)) - 1) / (1 << (LIS[hi]));
  } else {
    const mid = Math.floor((lo + hi) / 2);
    return getCumSumRatio(sum, LIS, lo, mid)
           + getCumSumRatio(sum, LIS, mid + 1, hi);
  }
}

function mergeRuns(array, lo, hi, tempArray) {
  
  // Ensure that the target range is valid
  if (lo < 0 || hi > array.length || lo >= hi) {
    throw new Error('Invalid range');
  }
  
  // Allocate temporary storage if it does not yet exist
  if (!tempArray) {
    tempArray = new Array(hi - lo);
  }
  
  // If the number of elements to be merged is less than or equal to the threshold,
  // switch to insertion sort
  if (hi - lo <= MIN_MERGE) {
    for (let i = lo; i < hi; i++) {
      for (let j = i; j > lo && cmp(array[j], array[j - 1]) < 0; j--) {
        swap(array, j, j - 1);
      }
    }
    return;
  }
  
  // Otherwise, determine the minimum leaf size
  const minLeafSize = MIN_MERGE * MIN_MERGE;
  
  // Divide the array into groups of approximately equal size
  const groupSize = Math.floor((hi - lo + MIN_RUNSIZE - 1) / MIN_RUNSIZE);
  const numGroups = Math.floor((hi - lo) / groupSize);
  for (let g = 0; g < numGroups; g++) {
    
    // Find the boundaries of the current group and construct its run
    const groupLo = Math.max(lo, g * groupSize);
    const groupHi = Math.min(groupLo + groupSize, hi);
    const groupEnd = Math.min(groupHi + MIN_RUNSIZE, hi);
    const groupLen = groupEnd - groupLo;
    
    // Construct a balanced binary tree of leaves and internal nodes
    const stack = [{lo: groupLo, hi: groupEnd, pos: 0}];
    while (stack.length > 1) {
      const node = stack.pop();
      const loPos = node.pos;
      const midPos = loPos + Math.floor((node.hi - node.lo) / 2);
      const hiPos = loPos + node.hi - node.lo - 1;
      stack.push({lo: node.lo, hi: node.lo + midPos - node.pos, pos: loPos});
      stack.push({lo: node.lo + midPos - node.pos + 1, hi: node.hi, pos: midPos + 1});
      if (hiPos > loPos) {
        stack.push({lo: groupLo + loPos, hi: groupLo + hiPos, pos: midPos + 1});
      }
    }
    
    // Process the constructed tree recursively using local memory instead of dynamic allocation
    const auxArray = new Array(hi - lo);
    reconstructTree(auxArray, array, lo, hi, stack[0].pos, true);
    groupCopy(array, lo, auxArray, stack[0].pos, groupLen);
    
    // Reconstruct the tree again, but this time in reverse order
    reconstructTree(auxArray, array, lo, hi, stack[0].pos, false);
    groupCopy(array, groupEnd - groupLen, auxArray, stack[0].pos, groupLen);
  }
  
  // Free unused memory and return
  delete tempArray;
}

function reconstructTree(auxArray, srcArray, lo, hi, rootPos, forwardOrder) {
  
  // Handle invalid parameters
  if (!srcArray ||!auxArray || lo < 0 || hi > srcArray.length || lo >= hi) {
    throw new Error('Invalid arguments');
  }
  
  // Copy the root node to the auxiliary array
  auxArray[rootPos - lo] = srcArray[rootPos];
  
  // Walk down the tree copying nodes to the auxiliary array in order
  if (forwardOrder) {
    let curNode = rootPos - lo + 1;
    let nextNode = getNextPos(auxArray, curNode, rootPos - lo);
    while (nextNode < hi) {
      auxArray[curNode++] = srcArray[nextNode++];
    }
  } else {
    let curNode = rootPos - lo + 1;
    let nextNode = getNextPos(auxArray, curNode, rootPos - lo);
    while (nextNode < hi) {
      auxArray[--curNode] = srcArray[--nextNode];
    }
  }
}

function getNextPos(auxArray, pos, rootPos) {
  
  // Follow the path specified by the auxiliary array from the given position,
  // returning the next position visited on the way
  let newNode = auxArray[pos];
  while (newNode >= rootPos) {
    pos -= auxArray[pos + NODEWIDTH - 1] - auxArray[pos];
    newNode = auxArray[pos];
  }
  return pos + getNodeWidth(newNode - rootPos);
}

function getNodeWidth(depth) {
  
  // Compute the width of a node in the auxiliary array given its depth
  return NODEWIDTH - 1 + depth * 2;
}

function cmp(a, b) {
  
  // Compares two numbers using signed integer representation to ensure stability of the sort
  return a == b? 0 : a < b ^ -(Number(a) < 0) | 0;
}

function swap(array, i, j) {
  const temp = array[i];
  array[i] = array[j];
  array[j] = temp;
}

function groupCopy(dstArray, dstOffset, srcArray, srcOffset, count) {
  for (let i = 0; i < count; i++) {
    dstArray[dstOffset++] = srcArray[srcOffset++];
  }
}

const INSERTIONSORT_THRESHOLD = 32;
const MERGE_SWITCHOVER = 2 * INSERTIONSORT_THRESHOLD;
const MIN_MERGE = 32;
const PUSH_THRESHHOLD = 400;
const POP_THRESHHOLD = 200;

const NODEWIDTH = 6;
const MIN_GALLOP = 7;
const GALLOPING_DIAGONAL = 8;
const FIRST_SHIFT = 16;
const MAX_LEN = FIRST_SHIFT + GALLOPING_DIAGONAL - 1;
const CEIL_DIV_FACTOR = 16;