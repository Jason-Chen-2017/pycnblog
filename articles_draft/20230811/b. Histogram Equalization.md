
作者：禅与计算机程序设计艺术                    

# 1.简介
         

直方图均衡化（Histogram equalization）是一种图像增强的方法，其目的在于使得图像中所有像素值的分布出现较为均匀，从而增强图像的对比度、亮度、色调等视觉效果。它的基本思想是通过拉伸直方图，使其每个柱形区域中的灰度值相等，从而达到均匀分布的效果。

常用直方图均衡化算法包括均值滤波法、拉普拉斯滤波法、最小均方差滤波法以及光谱变换法等。

本文主要讨论直方图均衡化的两种方法：一是全局直方图均衡化（Global histogram equalization），二是局部直方图均衡化（Local histogram equalization）。

# 2.基本概念及术语说明

## 2.1. 图像和像素空间

### 2.1.1. 图像
图像（image）是一个二维矩阵，描述物体或物体的一部分在空间中的位置和颜色特征。图像由像素点组成，每个像素点都表示图像在某一坐标点上的亮度和色彩信息。图像通常可以分为三个主要属性：灰度级（Gray Level）、色彩深度（Color Depth）和尺寸。其中，灰度级指的是黑白之间取的值的个数；色彩深度一般为24-bit或者8-bit表示；尺寸则指的是图像的长宽。

### 2.1.2. 像素空间
图像的像素空间（Pixel space）是指图像中每一个像素对应的空间坐标系。它以图像所在参考坐标系的原点作为坐标原点，根据图像的分辨率表示为单位长度。图像中某一特定坐标点的像素值对应于该坐标的空间值。

### 2.1.3. 直方图
直方图（histogram）是图像的统计特性，它显示了不同灰度值的频率分布。直方图的横轴表示灰度级值（一般为0~255），纵轴表示相应的灰度值频率。

## 2.2. 全局直方图均衡化

全局直方图均衡化（Global histogram equalization）是指将整个图像的直方图拉伸到一个范围内，使其变为均匀分布。图像的全局直方图均衡化需要考虑图像的整体结构和内容，因此一般会花费更多的计算资源。以下将给出两种方法进行全局直方图均衡化：

1. 方法1：自适应直方图均衡化

在这种方法中，先求出图像的直方图，再按照一定规则改变图像中的灰度值，使其满足直方图均衡化的要求。这种方法的优点是计算量小，缺点是可能导致图像失真。

1.首先，将图像划分为若干个小区域，然后对每个小区域的直方图进行均衡化处理，再把均衡化后的各个区域的灰度值合并起来得到最终的输出结果。这种方法的基本思想是根据图像的结构，按照各个区域的直方图进行均衡化处理。

```python
def adaptive_eq(img):
# 获取图像的高度和宽度
height, width = img.shape[:2]

# 对每个小区域进行直方图均衡化
for y in range(height // blocksize + 1):
for x in range(width // blocksize + 1):
subimg = img[y*blocksize: (y+1)*blocksize,
  x*blocksize: (x+1)*blocksize]

hist = cv2.calcHist([subimg], [0], None, [256], [0, 256])
cdf = hist.cumsum()

for i in range(len(cdf)):
if cdf[i] == 0 or cdf[i] > len(subimg) * 255:
continue

j = np.argmax(hist <= cdf[i]) - 1
img[y*blocksize: (y+1)*blocksize, 
x*blocksize: (x+1)*blocksize][img[y*blocksize: (y+1)*blocksize,
                              x*blocksize: (x+1)*blocksize] == i] = j

return img
```

2. 方法2：直方图均衡化

通过对图像的直方图进行均衡化处理，使图像中各个灰度级值出现的概率相同。这种方法的优点是计算量低，缺点是可能使图像细节丢失，影响最终效果。

1.首先，计算图像的直方图，并将直方图划分为几个子区间，每个子区间的灰度级范围相同。
2.对于每个子区间，计算其累计分布函数（CDF），CDF记录着该子区间灰度级值出现的概率。
3.对于每一个像素的灰度值，找到其对应直方图下属的第一个大于等于该灰度值的灰度级值，并赋值给该像素。这样做的目的是减少像素值的变化幅度，保持原始图像的全局均衡性。

```python
def global_eq(img):
# 获取图像的直方图
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
cdf = hist.cumsum()

# 将CDF归一化
cdf_m = np.ma.masked_equal(cdf, 0)
cdf_norm = (cdf_m - cdf_m.min()) / (cdf_m.max() - cdf_m.min()) * 255

# 根据CDF映射图像
img_eq = cdf_norm[img].astype('uint8')

return img_eq
```

## 2.3. 局部直方图均衡化

局部直方图均衡化（Local histogram equalization）是指只对一部分区域的直方图进行均衡化，其余地方不变。由于局部的结构相似，因此可以在一定程度上消除噪声。局部直方图均衡化方法主要有两种：一是邻域直方图均衡化，二是基于局部像素的直方图均衡化。

### 2.3.1. 邻域直方图均衡化

邻域直方图均衡化（Neighborhood histogram equalization）是指只对邻域内的直方图进行均衡化，这么做可以消除一些噪声。最简单的邻域方式为领域接近算子（Neighborhood Closest Matched Operator，NCM），即只对邻域内具有相似灰度值的像素点进行均衡化。

```python
def NCM_eq(img):
# 初始化权重数组
w = [[1]*9 for _ in range(9)]

for y in range(1, 8):
for x in range(1, 8):
gray = int((img[y-1][x-1]+img[y-1][x]+img[y-1][x+1]+
img[y][x-1]+img[y][x]+img[y][x+1]+
img[y+1][x-1]+img[y+1][x]+img[y+1][x+1])/9.)

delta = abs(gray - img[y][x])
sumw = float(delta + max(delta-1, 0))

w[y][x] = sumw

# 更新权重数组
weight = np.array(w).flatten().reshape((-1, 1, 1)).astype("float32")

# 执行图像卷积
dst = cv2.filter2D(img, -1, weight/np.sum(weight))

return dst
```

### 2.3.2. 基于局部像素的直方图均衡化

基于局部像素的直方图均衡化（Local Pixel-Based Histogram Equalization，LBP-HE）是另一种常用的方法。该方法通过对图像不同大小的邻域内的像素值进行统计，找出不同的像素值分布模式，然后调整这些模式的权重，使得它们在全局看起来更加均匀。

```python
def LBP_HE(img, radius=5, points=None):
if points is None:
points = [(x, y) for x in range(-radius, radius+1) for y in range(-radius, radius+1)]

h, w = img.shape[:2]
out = np.zeros((h, w), dtype='float32')

for p in points:
Y = min(max(p[0]+int(round(radius)), 0), h-1)
X = min(max(p[1]+int(round(radius)), 0), w-1)

hist = {}

for dy in [-1, 0, 1]:
for dx in [-1, 0, 1]:
sy = Y + dy
sx = X + dx

if sy >= 0 and sy < h and sx >= 0 and sx < w:
g = img[sy][sx]

if not str(g) in hist:
hist[str(g)] = []

hist[str(g)].append((dy, dx))

areas = [len(v) for v in hist.values()]
median = np.median(areas)
count = {k: 0 for k in hist}

for key, val in hist.items():
if len(val) > median:
count[key] = 1

for y in range(Y-radius, Y+radius+1):
for x in range(X-radius, X+radius+1):
if y >= 0 and y < h and x >= 0 and x < w:
g = img[y][x]

if str(g) in count:
out[y][x] += count[str(g)]

maxv = np.max(out)
minv = np.min(out)

if maxv!= minv:
a = 255/(maxv-minv)
b = -a*minv

out = cv2.addWeighted(src1=out, alpha=a, src2=out, beta=0, gamma=b)

out = out.astype('uint8')

return out
```