
作者：禅与计算机程序设计艺术                    

# 1.简介
  

ElGamal算法是一种非对称加密算法，其公钥密码体制可以用来进行公开密钥加密、数字签名、密钥交换等，并满足密钥管理方面的需求。其特点在于它采用了一种新颖的公钥加密方案——椭圆曲线上的离散对数难题，该算法属于椭圆曲ulse密钥交换（ECDHE）方法。由于该算法能够实现安全且高效的公钥加密，因此被广泛地用于数字签名、密钥交换以及网络传输中的加/解密。
本文从ElGamal算法的设计者—伊莱恩·古纳德·米勒(J.A.E. Gutmann)和该算法的原理入手，对该算法进行详细阐述。
# 2.基本概念及术语说明
## 2.1 椭圆曲线和相关术语
### 2.1.1 椭圆曲线
椭圆曲线是两个已知点的相切线段上所成的图形，每条椭圆曲线都由一个参数方程来定义。该方程用两个变量表示，分别为横轴上的点坐标$x$和纵轴上的点坐标$y$，由下式给出：$y^2=x^3+ax+b\ (mod p)$，其中$p$是一个素数，而$a$和$b$则是选定的两个不全为零的整系数。在同一条椭圆曲线上，任何两点必定可以通过这条椭圆曲线上一点的切线相连。椭圆曲线上一点$(x_0, y_0)$的切线由下式给出：$y-y_0=\frac{x-x_0}{slope}$，其中$slope=(3ax+a^2)/(2y_0)$。一般来说，椭圆曲线都是由一个中心点、两个不共线的切向角和相关系数而确定，因此可以使用标准形式或点乘形式来描述。
### 2.1.2 椭圆曲线的标准型
对于一个椭圆曲线$y^2=x^3+ax+b$，若$4a^3+27b^2$不是整系数，则称其为非标准型椭圆曲线；反之，则称其为标准型椭圆曲线。当$4a^3+27b^2$不是整系数时，我们通常只考虑位于第一象限的椭圆曲线，即$0\leq x \leq sqrt(-3a^2-\delta)/2$,$0<y<sqrt{-3a^2+\delta}/2$，其中$\delta = -1 mod 27$。
### 2.1.3 离散对数难题（Discrete Logarithm Problem，DLP）
离散对数难题（DLP）是指对于椭圆曲线$y^2=x^3+ax+b$，求得某个点$P=(x,y)$，使得其关于$y$轴对称的点$Q=(x,-y)$也存在，并且满足$P+Q=(0,r)$，其中$r$是关于$p$的某一整数，称$P$为椭圆曲线$E$上的一个基点，而$Q$为椭圆曲线$E$上的对称点，而且$kP=(0,r)$对应着$r$值，则称这样的一个整数$k$为椭圆曲线$E$上的离散对数。如果椭圆曲线上没有多个相同的离散对数，则称椭圆曲线$E$为可重参数椭圆曲线。
## 2.2 公钥和私钥
公钥和私钥是密钥交换协议的重要组成部分，通过公钥进行加密，私钥进行解密。ElGamal算法中，公钥为椭圆曲线上的公共参数$(\lambda, P, Q,\mu,\gamma)$，其中$P=(x_p, y_p),Q=(x_q, -y_q), \lambda$是一个非负整数，代表了生成元。私钥为自然数$d$，并且有$dP=DP=(dx_p, dy_p)$。
## 2.3 椭圆曲线上的离散对数计算
椭圆曲线上离散对数问题的解决是一个十分困难的问题。但是，根据古纳德·米勒的论文“A Course in Computational Algebraic Number Theory”，可以通过一些启发式的方法来有效地解决椭圆曲线上的离散对数问题。古纳德·米勒的启发式方法包括：“在椭圆曲线上找一个多项式一次项系数为$a$”、“在椭圆曲线上找到一个特殊点”、“把椭圆曲线分解为两部分”、“暴力枚举法”。
### 2.3.1 在椭圆曲线上找一个多项式一次项系数为$a$
由于椭圆曲线$y^2=x^3+ax+b$的阶为$n$，所以至少要有$n$个不同的一次项系数才能使椭圆曲线成为可重参数椭圆曲线。因此，在椭圆曲线上选择一个多项式一次项系数为$a$，如$y^2=x^3+ax+b$, $y^2=x^3-5x+3$, $y^2=x^3-2x+1$均可，它们的$a$值分别为$-3, -5, -2$。随后可以利用离散对数问题来解决这些问题。
### 2.3.2 在椭圆曲线上找到一个特殊点
另一种解决椭圆曲线上的离散对数问题的方法是，首先找到椭圆曲线上一个特殊点$S$，并且令$d$的值足够小，那么$dP=DP=(dx_p, dy_p)$就等于$dS=DS=(ds_p, -ds_q)$。而$ds_p=-3ds_q-3\times d\times ds_q$是椭圆曲线$E$上的一个基点，它与椭圆曲线$E$上的其他基点不同。因此，可以在椭圆曲线上找到一个特殊点$S$，然后通过$S$的值，就可以快速地求得椭圆曲线上任意一个点的离散对数。
### 2.3.3 把椭圆曲线分解为两部分
第三种解决椭圆曲线上的离散对数问题的方法是把椭圆曲线分解为两部分，即$y^2=x^3+ax+b$可以分解为$y^2=x^3+(a/2)^2-xy+b$和$y^2=x^3-(a/2)^2-xy+b$。那么在椭圆曲线上，如果$y$值小于$(a/2)^2-xy+b$，那么$x$值落在第一部分椭圆曲线上，否则落在第二部分椭圆曲线上。
### 2.3.4 暴力枚举法
最后一种解决椭圆曲线上的离散对数问题的方法是暴力枚举法。暴力枚举法直接将椭圆曲线上所有点看作基点，然后尝试所有可能的取值$k$，判断是否满足$kP=(0,r)$，如果满足，则认为找到了一个整数$k$使得$kP=(0,r)$。如果遍历了一遍所有可能的$k$值之后还没找到，则说明椭圆曲线上不存在这种离散对数。当然，这个方法的时间复杂度太高，实际运行时间非常长。
# 3. 核心算法原理及操作步骤
## 3.1 椭圆曲线上的点加法和点减法
椭圆曲线上的点加法和点减法运算可以用如下的代数定义来表示：$P+Q=(x_{pq}, y_{pq})$，其中$P=(x_p, y_p),Q=(x_q, y_q)$，得到的结果为$P$和$Q$的联合点。$P-Q=(x_{pq}, y_{pq})$，其中$P=(x_p, y_p),Q=(x_q, y_q)$，得到的结果为$P$和$Q$的差点。
显然，点加法和点减法运算满足结合律、分配律和交换律。在相同椭圆曲线$E$上，两个点$P$和$Q$的和等于先加后减，即$P+Q-(Q+P)=Q+P-P-Q=0$。椭圆曲线上的点乘运算可以用如下的代数定义表示：$kPQ=(x_{kpq}, y_{kpq})$，其中$k$是一个非负整数，$P=(x_p, y_p),Q=(x_q, y_q)$。
## 3.2 生成元
为了保证椭圆曲线上的点加法和点减法运算在同一条椭圆曲线$E$上是封闭的，需要选择一组基点$P$，再选择生成元$\lambda$。假设椭圆曲线$E$上的基点$P=(x_p, y_p),Q=(x_q, -y_q)$，并且$x_p \neq x_q$，则生成元$\lambda$可以由以下方式得到：
$$
\begin{cases}
\lambda&=n+1 \\
y_q&\equiv (\lambda-1)(x_q-x_p)\pmod {p}\\
\end{cases}
$$
其中$n$是椭圆曲线$E$的阶，$p$是素数。
## 3.3 加密和解密
ElGamal算法就是基于椭圆曲线离散对数难题的公钥加密方案。它要求发送者和接收者事先已经建立好公钥和私钥。公钥由椭圆曲线的四个参数$(\lambda, P, Q,\mu,\gamma)$构成，而私钥仅有一个自然数$d$。加密过程如下：发送者先选择明文消息$M$，将$M$作为椭圆曲线上的一点$M'$的坐标，即$Mx'=y'$.发送者根据私钥$d$和$M'$，计算密文$C=(c_R, c_s)$，其中$c_R=(X_R, Y_R)$和$c_s=(s_R, s_s)$为随机值。发送者计算出$C$后，发送者发送$C$和$K$，其中$K$为$(x_p, y_p, d)$。接收者收到$C$和$K$后，他根据$K$和$C$，计算出明文$M''$，即$M''x'=-Y_R$.最终，接收者可以验证$M''$是否正确，如果正确，则认为接受到的消息没有遗漏，否则，接受到的消息有遗漏。
## 3.4 ElGamal签名算法
ElGamal签名算法采用椭圆曲线上的点运算和点乘运算的组合来实现签名功能。签名过程包括生成密钥对、签名计算和验证过程。签名算法生成密钥对的过程如下：由发送者生成一对密钥$(sk,pk)$，其中$pk=(\lambda, P, Q,\mu,\gamma)$是公钥，$sk$是一个非负整数为私钥。签名计算过程如下：发送者选择消息$M$，将消息$M$的哈希值$H(M)$作为椭圆曲线上的一点$M'$的坐标，即$Hx'=y'$。发送者根据$sk$和$M'$，计算签名$sig=(r,s)$，其中$r$是随机值，$s=r\cdot H(M)+(d\cdot r+1)\cdot M'$。签名计算完成后，发送者返回$sig$给接收者。验证过程如下：接收者收到签名$sig$和消息$M$后，他根据$sig$和$M$的哈希值$H(M)$，计算出椭圆曲LINE上的一点$M''$的坐标，即$Hx''=-Y_R/s$.接收者计算出$M''$后，比较$M$的哈希值$H(M)$和$M''$的坐标，如果二者相同，则认为接受到的消息没有遗漏，否则，接受到的消息有遗漏。
# 4. 具体代码实例
## 4.1 点加法及点减法
```python
import math

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return "({},{})".format(self.x, self.y)

    def __add__(self, other):
        if not isinstance(other, Point):
            raise TypeError("unsupported operand type(s) for +: 'Point' and '{}'".format(type(other)))

        # 同一条椭圆曲线上，两个点的和等于先加后减，即P+Q-(Q+P)=Q+P-P-Q=0
        if self == -other or (self!= Point(0,0) and ((self.x == other.x and self.y!= other.y) or math.isclose((self.y - other.y)/(self.x - other.x), (3*self.x**2+self.curve[0])/(2*self.y)))):
            slope = None
            if self.y > other.y:
                point = self
                diff = other
                is_zero = False
            else:
                point = other
                diff = self
                is_zero = True

            k = abs(diff.x)//abs(point.x)+1
            new_x = diff.x - point.x*(k-1)
            try:
                a = (point.curve[0] ** 3 - point.curve[2] * point.curve[1] ** 2) // (math.pow(point.curve[1], 2)*point.curve[3])
            except ZeroDivisionError:
                raise ValueError('Curve has singular Jacobian')
            
            lambda_ = (new_x ** 3 + a * new_x + point.curve[2]) % point.curve[3]
            new_y = (-point.curve[1]**2*new_x**3+point.curve[3]*new_x**2-point.curve[0]*new_x*lambda_-point.curve[2]*lambda_)//(point.curve[1]-lambda_)
            
            if is_zero:
                result = Point(new_x, new_y)
            elif slope >= point.y - new_y:
                result = Point(new_x, point.y)
            else:
                result = Point(new_x, new_y)
                
        else:
            if self.x == other.x and self.y == other.y:
                slope = -(3*self.x**2+self.curve[0])/(2*self.y)
                slope += math.copysign(1, slope)
                if self.y < other.y:
                    result = Point((-self.y-slope*self.x)//(1-slope**2), self.y)
                else:
                    result = Point((-other.y-slope*other.x)//(1-slope**2), other.y)
            else:
                slope = (3*self.x**2+self.curve[0])/(2*self.y)-self.curve[2]/self.curve[3]

                intercept = (3*self.curve[2]*self.y**2+2*self.curve[0]*self.y-self.curve[1]*self.x**3-2*self.curve[3]*self.x**2+self.curve[2]*self.x) // (2*self.curve[1]*self.y)
                
                # 两直线不相交时
                if not intersect(slope, intercept, *(self.coords()+other.coords())):
                    # print("the two lines don't intersect")
                    if slope < self.y - other.y:
                        result = Point((-self.y-slope*self.x)//(1-slope**2), self.y)
                    else:
                        result = Point((-other.y-slope*other.x)//(1-slope**2), other.y)

                # 两直线相交于点P'(x',y'),两点P和P'在P'切线方向上
                elif math.isclose(slope*self.x-intercept, self.y)*(self.x <= -intercept)**2:
                    result = Point(0, 0)
                    
                # 两直线相交于点P''(x'',y'')，两点P和P''在P''切线方向上
                elif math.isclose(slope*self.x-intercept, self.y)*(self.x > -intercept)**2:
                    result = Point(0, 0)
                    
                # 其他情况
                else:
                    slope -= math.copysign(1, slope)
                    if math.isclose(slope*self.x-intercept, self.y):
                        x3 = (-self.x-intercept)//slope

                        if math.isclose(slope*(x3+intercept)-self.y, 0) and math.isclose(slope*x3-intercept-self.y, 0):
                            result = Point(x3, self.y)
                        else:
                            result = Point(x3, (-slope*x3-intercept)%self.curve[3])
                            
                    else:
                        x1 = intersection((self.x, self.y),(slope*self.x+intercept, self.y))
                        x2 = intersection(((other.x, other.y)),(slope*other.x+intercept, other.y))
                        
                        slope1 = (slope*x1+intercept-self.y)/(x1-self.x)
                        slope2 = (slope*x2+intercept-other.y)/(x2-other.x)
                        
                        result = Point((slope1*x1-slope2*x2)/(slope2-slope1), (slope1*slope2*(x2-x1))+self.y)
    
    def __sub__(self, other):
        if not isinstance(other, Point):
            raise TypeError("unsupported operand type(s) for -: 'Point' and '{}'".format(type(other)))
        
        res = self.__add__(-other)
        
        if res.x == 0 and res.y == 0:
            res = Point(res.x, res.y)
            
        return res
        
    @property
    def curve(self):
        return self._curve
    
    def set_curve(self, params):
        if len(params)!= 5:
            raise ValueError("The length of the parameter should be 5.")
        self._curve = tuple(int(_) for _ in params)
    
    def coords(self):
        return [self.x, self.y]
        
def intersection(line1, line2):
    A1,B1,C1 = line1
    A2,B2,C2 = line2
    
    det = A1*B2-A2*B1
    
    if det == 0:
        return None
    
    x = (B1*C2-B2*C1)//det
    y = (A2*C1-A1*C2)//det
    
    return int(x)
    
def check_params(*args):
    flag = True
    for arg in args:
        if not hasattr(arg, '__len__'):
            continue
        if len(arg)!= 5:
            flag = False
            break
    return flag
    
if __name__ == "__main__":
    curve = [-3, -5, -2, 0, 7]
    
    if check_params(curve):
        pt1 = Point(1,2)
        pt1.set_curve(curve)
        pt2 = Point(3,4)
        pt2.set_curve(curve)
        res = pt1+pt2
        print(res.x, res.y)<|im_sep|>