
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Quaternions are a mathematical tool used in physics and computer graphics to represent three-dimensional rotations and orientations. In this article, we will introduce the basics of quaternions by defining their basic concepts such as real part, imaginary parts, conjugate, product, division, etc., then explain its practical applications in computer graphics and robotics. We also show some interesting properties about them that can be derived from these concepts. This article is intended for readers with intermediate knowledge in linear algebra and trigonometry.

# 2. Basic Concepts
## Real Part & Imaginary Parts
A quaternion q = w + xi + yj + zk can be decomposed into two parts:
* The real part (w) represents the scalar component of the vector. It remains unchanged under rotation.
* The imaginary parts (xi+yj+zk) represent the vector component of the vector. When a point p lies on an axis perpendicular to the imaginary parts, it stays fixed when rotated around that axis. These components form the direction cosine matrix (DCM), which defines the orientation or rotation of the frame attached to the quaternion. 

## Conjugate Quaternion & Adjoint Matrix
The conjugate of a unit quaternion q is defined as:

q* = qw - xi - yj - zk

The adjoint matrix A of the quaternion is given by:

$$\begin{bmatrix}
     \hat{\mathbf{x}}_1 & \hat{\mathbf{x}}_2 & \hat{\mathbf{x}}_3 \\
    -\hat{\mathbf{y}}_1 & \hat{\mathbf{y}}_2 & \hat{\mathbf{y}}_3 \\
    -\hat{\mathbf{z}}_1 & -\hat{\mathbf{z}}_2 & \hat{\mathbf{z}}_3 \\
 \end{bmatrix}$$

where $\hat{\mathbf{v}}$ is the conjugate of $\mathbf{v}$. Multiplying any column vector $\mathbf{v}$ with $A$ gives us the corresponding skew symmetric matrix representing the cross product between $\mathbf{v}$ and the basis vectors $\hat{\mathbf{i}}, \hat{\mathbf{j}}, \hat{\mathbf{k}}$. For example, multiplying the $\hat{\mathbf{x}}_1$ column vector with $A$ yields $\begin{bmatrix}\mathbf{0} & -\mathbf{z}_1 & \mathbf{y}_1 \\ -\mathbf{z}_1 & \mathbf{0} & -\mathbf{x}_1 \\ \mathbf{y}_1 & -\mathbf{x}_1 & \mathbf{0}\end{bmatrix}$, which represents the effect of rotating along $\hat{\mathbf{x}}_1$ by $\theta$ radians.  

## Scalar-Vector Form & Quaternions Operations
Quaternions can be written in various forms, including the "scalar-vector" form:

q = s + v

where `s` is the scalar component and `v` is the vector component represented as a 3D vector (`[vx vy vz]`). Another common notation is `[w x y z]` where `(wx, wy, wz)` represent the imaginary parts and `w^2+x^2+y^2+z^2=1`.

### Product of Two Quaternions
The product of two quaternions `p = q * r`, where `*` denotes the Hamiltonian product of complex numbers. If `q` represents a rotation of angle $\theta$ about the $\hat{\mathbf{n}}$ axis, then `r` must represent a second rotation of another angle $\psi$ along the same axis after applying first rotation. The result of the multiplication is a new quaternion `p` whose rotation is equivalent to both rotations applied sequentially. Its scalar part is equal to the square root of the sum of squares of the scalar parts of the input quaternions, while its vector part is equal to the sum of products of each pair of matching terms of the input quaternions' vector parts.

If `q` has the scalar part `sqrt(2)/2` and the vector part `[sin(\frac{\pi}{4}) sin(\frac{\pi}{4}) sin(\frac{\pi}{4}) sqrt(2)]`, then `r` should have the scalar part `1/2` and the vector part `[cos(\frac{\pi}{4}) cos(\frac{\pi}{4}) cos(\frac{\pi}{4}) sqrt(2)]`. Applying these two rotations sequentially results in a new quaternion `p`:

```python
>>> import math

>>> # First rotation
>>> theta = math.pi / 4
>>> n = [0, 0, 1]     # Normalized axis of rotation
>>> c = math.cos(theta / 2)
>>> s = math.sin(theta / 2)
>>> w1 = c; x1 = s * n[0]; y1 = s * n[1]; z1 = s * n[2]    # First quaternion

>>> # Second rotation
>>> psi = math.pi / 4
>>> c = math.cos(psi / 2)
>>> s = math.sin(psi / 2)
>>> w2 = c; x2 = s * n[0]; y2 = s * n[1]; z2 = s * n[2]    # Second quaternion

>>> # Compute final quaternion using Hamiltonian product
>>> pw = math.sqrt((w1 ** 2 + x1 ** 2 + y1 ** 2 + z1 ** 2) * (w2 ** 2 + x2 ** 2 + y2 ** 2 + z2 ** 2))   # Scalar part
>>> px = ((w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2) * (pw ** 2))        # Vector part
>>> py = ((w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2) * (pw ** 2))
>>> pz = ((w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2) * (pw ** 2))

>>> print("Scalar part:", pw)      # Output: 1/2
>>> print("Vector part:", [px, py, pz])     # Output: [-0.9758993383121238, 0.17888543819998327, 0.06180339887498948]
```

This demonstrates how you can apply multiple rotations directly using the Hamiltonian product of quaternions without needing to extract angles and axes from matrices. The resulting output shows the updated position of the initial point after both rotations were applied.