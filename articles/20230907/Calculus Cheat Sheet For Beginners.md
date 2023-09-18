
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Calculus is the mathematical study of continuous change or rates of change in a function. It enables us to understand and manipulate functions that are defined by equations, such as those involving differential and integral calculus, which enable us to find extrema, areas under curves, etc. The calculus cheat sheet for beginners provides a concise reference guide with clear explanations of key concepts and formulas relevant to understanding and manipulating functions using this powerful tool. This resource will be useful for students, educators, data scientists, engineers, analysts, etc., who require a quick review of fundamental concepts related to calculus.

# 2.计算几何(Geometry)
## Coordinate Systems
A coordinate system refers to a set of axises used to define positions in space. There are two types of cartesian coordinate systems:

1. Cartesian coordinates - (x,y,z): X-axis pointing right, Y-axis pointing up, Z-axis pointing outwards. Origin point located at (0,0,0).
2. Polar Coordinates - (r,θ,ψ): Radius r points towards origin, θ is angle from positive x axis to line connecting origin to current position, ψ is angle between projection of radius vector on XY plane and z-axis. 

The standard notation for polar coordinates can also be written in Cartesian coordinates as (r cosθ, r sinθ, z), where (r,θ,z) represents the same values as (r,θ,ψ). In both cases, the radial distance r is measured along the positive direction of the X-axis.

## Vector Operations
Vectors are quantities with magnitude and direction. They are often represented using an arrow symbol, which shows the direction of motion or force acting on it. There are several basic vector operations, including addition, subtraction, dot product, cross product, and scalar multiplication/division. Additionally, there are some important properties of vectors like parallelism, perpendicularity, and normalization.

To calculate the length of a vector, use Pythagoras' theorem: sqrt(x^2 + y^2 + z^2). To find the unit vector, divide each component of the vector by its magnitude. To add or subtract vectors, simply add or subtract their components together respectively. To multiply or divide a vector by a scalar value, you need to apply the formula `a*v = [ax, ay, az]`, where `v` is the original vector and `a` is the scalar. Similarly, if we want to represent a vector v as a matrix `[vx vy vz]^T`, then multiplying it by a scalar a requires dividing all elements of the matrix by the scalar `a`. Dot products and cross products are calculated similarly but with different formulas. The dot product of two vectors u and v is given by `u. v = uv`, while the cross product is denoted by `u x v = (uyVz - uzVy, uzVx - uxVz, uxVy - uyVx)`.

## Parametric Equations
Parametric equations provide a way to graphically represent curved surfaces and trajectories. A curve parameterized by p(t), t∈[a,b], is a path that passes through every point in the domain [a,b]. Curves that have multiple tangents at any one point usually take more than one parameter value to describe them accurately. Parametric equations can also be derived from explicit equations relating the independent variable to dependent variables. Some examples of parametric equations include circular, elliptical, hyperbolic, parabolic, and spline curves. We can use Newton's method to solve these equations numerically for specific values of the parameters.

## Integrals
An integral is a function that gives the area underneath a curve or surface between two specified points. The most common type of integrals is the definite integral, which is obtained when we know the limits of integration. Common forms of definite integrals include triangles, rectangles, circles, and trapezoids. More complex integrals can be evaluated using techniques such as the Laplace transform, Fourier series expansion, and substitution. One commonly used property of integrals is the mean value theorem, which states that the average of a function over a finite interval approaches the integral of the function times the interval width.