
作者：禅与计算机程序设计艺术                    

# 1.简介
  

图像处理领域对几何变换(geometric transformation)是非常重要的任务之一。在这篇文章中，我们将会简要介绍图像处理中常用的两种正交变换——旋转变换、仿射变换。同时，对于仿射变换，我们也会介绍其几何意义及其用途。我们还会给出一些相关的数学知识和应用案例。
## 2.1 What is a Geometric Transformation?
Geometric transformations are used to transform an image from one coordinate system into another coordinate system or vice versa by applying some mathematical formulas on the pixel values of the image. In general, there are two types of geometric transformations: Affine and Projective transformations. These transformations can be categorized as linear transformations or non-linear transformations depending upon whether they preserve lines, planes, or both during transformation. The basic idea behind any type of geometric transformation is that it involves manipulating the positions of points in space so that their projections onto different planes are transformed to new locations in the same plane. This results in a modification to the shape of objects and images, which makes them more realistic and beautiful. Below we will discuss these two transformations separately with examples and use cases.
## 2.2 Rotation Transformation
A rotation transformation rotates an object around a specific axis without changing its position or direction. Mathematically, if $\boldsymbol{r}$ denotes the vector representing the point $P$ about which we want to rotate and $\theta$ is the angle of rotation, then the rotated vector $\hat{\boldsymbol{r}}$ is given by:

$$\hat{\boldsymbol{r}} = \cos(\theta)\boldsymbol{r} + \sin(\theta)n(\theta),$$ 

where $\boldsymbol{n}(\theta)$ is the unit normal vector of the reference frame passing through the origin along the rotation axis at an angle $\theta$.  

The effect of this rotation can be seen visually below:



In practice, we often represent rotation matrices using Rodrigues' formula:

$$R = I+\frac{sin(\theta)\mathbf{v}}{||\mathbf{v}||}, $$

where $I$ is the identity matrix, $\mathbf{v}$ is the rotation axis (normalized), and $||\mathbf{v}||$ is its magnitude. We can also find the equivalent formulation using quaternions:

$$Q_x(\theta)=\begin{pmatrix}\cos\frac{\theta}{2}\\e^{i\frac{\theta}{2}}\sin\frac{\theta}{2}\end{pmatrix}, Q_y(\theta)=\begin{pmatrix}\cos\frac{\theta}{2}\\e^{-i\frac{\theta}{2}}\sin\frac{\theta}{2}\end{pmatrix}, Q_z(\theta)=\begin{pmatrix}\cos\frac{\theta}{2}\\e^{i\frac{\theta}{2}}\sin\frac{\theta}{2}\end{pmatrix}$$

which correspond to the rotation around the x, y, z axes respectively. Quaternions provide a convenient alternative for efficient computation compared to other representations such as Euler angles or rotation vectors.

To perform a rotation transformation, we need to calculate the inverse of the original coordinates before and after the transformation. Then we multiply each corresponding pair of pixels with appropriate weights to obtain the final output. For example, to rotate the image counterclockwise by 45 degrees, we would first invert all the pixels and create a new canvas with dimensions equal to half those of the input image. Next, we would apply the rotation transform to the inverted canvas. Finally, we would rescale the resulting canvas back to the size of the original image and overlay the result on top of the original image using weighted blending techniques. 
## 2.3 Scaling Transformation
Scaling transforms the size of an object while preserving its orientation. Mathematically, scaling occurs in two ways: uniform scaling and non-uniform scaling. Uniform scaling refers to resizing the image proportionately along all directions. Non-uniform scaling allows us to stretch or shrink the image along individual directions. To achieve non-uniform scaling, we simply pass the center of the object as the focal point instead of the corner when performing the transformation. Here's how we can do this with Python:

```python
import cv2
import numpy as np
 
rows, cols = img.shape[:2]       # get number of rows and columns
 
 
# uniform scaling
scalingFactor = 1.5            # define scaling factor
 
 scaledImg = cv2.resize(img,(int(cols*scalingFactor), int(rows*scalingFactor)))    # resize image
cv2.imshow("Uniform Scaled",scaledImg)     # display resized image
 
# non-uniform scaling
fx = fy = 1.5                  # scale factors in X and Y directions
 
 # shift centre of mass to origin
 M = np.float32([[1,0,-cols/2],[0,1,-rows/2]])
 shiftedImg = cv2.warpAffine(img,M,(cols,rows))
 
 # apply non-uniform scaling
 dst = cv2.resize(shiftedImg,(0,0),fx=fx,fy=fy,interpolation=cv2.INTER_CUBIC)
 
 # shift centre of mass back to original location
 M = np.float32([[1,0,cols/2+0.5*(dst.shape[1]-cols)/fx],[0,1,rows/2+0.5*(dst.shape[0]-rows)/fy]])
 finalImg = cv2.warpAffine(dst,M,(cols,rows))
 
 cv2.imshow("Non-Uniform Scaled",finalImg)      # display final image
cv2.waitKey()                         # wait until user presses a key
```

The above code shows how we can perform uniform and non-uniform scaling using OpenCV functions `cv2.resize()` and `cv2.warpAffine()`. We start with the basics and gradually build up to the desired effects. Firstly, we load our image using `cv2.imread()`, and extract its height and width using `img.shape`. Next, we define the required scaling factor and call `cv2.resize()` function to produce the uniform scaled version of the image. We then use `cv2.imshow()` to show the resized image.

We proceed to the second part of the code where we simulate non-uniform scaling by shifting the center of mass of the image to the origin, calling `cv2.warpAffine()` to apply the necessary scaling, and finally shifting the center of mass back to its original location using a combination of translation and scaling transforms. Once again, we call `cv2.imshow()` to display the final result. Note that here, we have introduced an additional step of rounding off errors due to subpixel alignment issues during interpolation.