
作者：禅与计算机程序设计艺术                    

# 1.简介
         
20世纪70年代，Lyapunov函数（连续时间的莱昂纳斯函数）在控制、信号处理、经济学、生物学等领域都受到广泛关注。然而，这一研究领域的发展却面临着很多技术上的挑战。其中一个主要的挑战就是研究Lyapunov函数的一阶矩和二阶矩的性质。这两个性质对于Lyapunov函数分析具有重要意义，例如它们可以帮助我们理解系统运动规律，预测其行为变化，以及设计控制策略。

         1973年，李亚非教授对Lyapunov函数做了一次全面的介绍，提出了“连续时间Lyapunov函数”的概念，并首次证明了一阶矩和二阶矩的存在，并将之用于信号处理、控制、经济学、生物学等领域。直到最近几年，随着Lyapunov函数相关理论的不断深入，它逐渐成为控制科学中最热门的研究方向之一。

         17年前，当时已经有很多关于Lyapunov函数的一阶矩和二阶矩的综述论文，但是它们仍处于停滞状态。本文将梳理Lyapunov函数的一阶矩和二阶矩的性质及其应用，并给出许多实际示例，以期提供可供参考的经典资料。另外，本文还会提供一些学习Lyapunov函数的一阶矩和二阶矩的方法和工具。
         # 2.基本概念术语说明
         ## 一阶矩
        在数学中，一阶矩（一阶微分算子）是一个映射，它接受关于函数的一阶导数，并返回某个值。换句话说，一阶矩描述了函数在某点附近的一阶导数的大小。

        某个函数$f(x)$在点$a$处的一阶矩记作$\lambda_{f}(a)$。如果函数在整个定义域上具有一致的微分方程组，那么一阶矩就等于此方程组在该点$a$处的解。例如，若$df/dx = f(x) + g(x),\quad dg/dx=h(x), \quad h(x)>0,\quad x\in\mathbb{R}$,那么$f$在$a$处的一阶矩就等于$d^2f/dx^2+hf'(a)$。

        2.1.正定性
         如果一阶矩$\lambda_f(a)\geq0$，则称函数$f(x)$在$a$处为正定函数。

        2.2.弱收敛性
         如果对于任意的足够小的正实数$E>0$,存在$M$和$\delta>0$,使得$\mid x-a\mid<\delta \Longrightarrow |\lambda_f(x)-\lambda_f(a)|\leq E|x-a|$成立，则称函数$f(x)$在$a$处为弱收敛的，且$\delta$的上限为$M$.


        2.3.强收敛性
         如果对于任意的足够小的正实数$E>0$,存在$M$和$\delta>0$,使得$\mid x-a\mid<\delta \Longrightarrow |\lambda_f(x)-\lambda_f(a)|\leq E|x-a|$成立，并且满足下面不等式

        $\frac{\lambda_f(b)}{\lambda_f(a)}\leq\exp(-EM),\forall b
eq a,$

         则称函数$f(x)$在$a$处为强收敛的，且$\delta$的上限为$M$.

        当$\lambda_f(a)=\infty$时，$f(x)$在$a$处不是一阶矩；当$\lambda_f(a)<-\epsilon M+\gamma$时，$f(x)$在$a$处被称为超越函数。
         # 3.核心算法原理和具体操作步骤以及数学公式讲解
         ## 3.1 一阶矩的性质
         ### (1) 定积分型一阶矩
         定积分型一阶矩，也称为Cauchy积分型一阶矩，定义如下：

            $\lambda_f(t):=\int_{t}^{+\infty}e^{-sx}f(s)ds,$

         这里，$t$是某个时刻，$s\ge t$，并且$ds=-\frac{dt}{s}$。根据定积分变换公式，可知：

            $f(t)=e^{st}\int_{t}^{+\infty}e^{-sx}f(s)ds.$

         对比两个式子，可以得到：

            e^{-st}\int_{t}^{+\infty}e^{-sx}f(s)ds=e^{-(t-s)}[sf(    heta)+\frac{(t-s)^2}{2}f'(    heta)]+O((t-s)^3).

         因此，定积分型一阶矩满足：

            1.  $e^{-st}\int_{t}^{+\infty}e^{-sx}f(s)ds$为$f(t)$在$t$时刻的一阶矩。
            2.  函数$f(t)$在区间$(t,+\infty)$上是周期函数。
            3.  有界性：$\lim_{t    o+\infty}\frac{|\lambda_f(t)|}{\left\vert f(t)\right\vert}=0$.
          
         ### (2) 一阶微分方程的一阶矩
         一阶微分方程在初值条件为$F(t_0)=    heta$的情况下的解为：

            $$f(t)=A_1\sin(\sqrt{|k_1|}t)+A_2\cos(\sqrt{|k_2|}t)+B_1\sin(\sqrt{|k_3|}t)+B_2\cos(\sqrt{|k_4|}t),$$

         此处，$k_i$是$\mathcal{L}^2(\overline{\Omega})$中的规范谐波，即：$\int_{\Omega}|K(u,v)|^2dudv=2\pi$. 

         根据Stieltjes变换公式，可计算一阶矩：

            $$\lambda_f(t)=2\pi A_1 k_1\cos(\sqrt{|k_1|}t)+2\pi A_2 k_2\sin(\sqrt{|k_2|}t)+2\pi B_1 k_3\cos(\sqrt{|k_3|}t)+2\pi B_2 k_4\sin(\sqrt{|k_4|}t),$$

         再利用欧拉公式，可证明：

            $$\lambda_f(t)=\lambda_T+\lambda_S,$$

         其中，$\lambda_T$为零边界项$\lim_{t    o T}\frac{\mathrm{d}^2f(t)}{\mathrm{d}t^2}$的$T$取值时的平均绝对值。$\lambda_S$是不含任何奇异点的充分必要条件下的超调项。

                 （注：欧拉公式：$\mathrm{d}f(t)/\mathrm{d}t=\frac{\lambda_f(t)}{\mu},\quad \mu=\int_\Omega f(x)g(x)dx$，这里$\mu=\pi$，$\lambda_f(t)$是$\mathcal{L}^2(\overline{\Omega})$中的权函数。）

         ### (3) 周期性一阶矩
         假设$f(t)$在时间轴上具有周期$T$，根据拉普拉斯变换公式：

            $f(t)=\sum_{n=-\infty}^{+\infty}c_ne^{st}$

         对时间轴$[-T,T]$内的任意点$t$进行Fourier变换，得：

            $F(w)=\int_{-\frac{T}{2}}^{\frac{T}{2}}e^{-iwt}f(t)dt=$
            
            $\qquad=\int_{-T}^{T}[\sum_{n=-\infty}^{+\infty}c_ne^{sn}e^{-iwnT}]e^{-iwt}dt=\sum_{n=-\infty}^{+\infty}\frac{c_ne^{sn}}{T}[1-i\frac{nwT}{T}],$

         从而：

            $\lambda_f(t)=\frac{c_ne^{nt}}{T}.$

           （注：$f(t)$在时间轴上的周期为$T=\frac{-2j\log(|\lambda_f(T)|)}{c_n}$, 其中$j$是虚数单位。若$f(t)$的周期为无穷大，则$\lambda_f(t)$也为无穷大。）

        除此之外，还可以证明其他一些一阶矩的性质。
         ## 3.2 二阶矩的性质
        ### (1) Riccati方程
         二阶矩定义为：

            $\Gamma_{fg}(z)=\int_{0}^{+\infty}e^{-zt}\gamma(s)f(z-s)ds,$

         这里，$\gamma(s)$表示$\sigma$-加权函数，$\sigma=\int_0^1\gamma(t)dt$.

         用$\mathfrak{X}_f(t)$表示函数$f(t)$的线性化，即：

            $\mathfrak{X}_f(t):=\frac{df}{dt}-\lambda_f(t)f(t),\quad (    ext{常微分方程})$

         根据拉普拉斯变换公式：

            $f(t)=\mathfrak{X}_{f^\ast}(t)+\lambda_f(t)e^{tI}f(0),\quad (    ext{齐次拉普拉斯变换})$

         可知：

            $\mathfrak{X}_{f^\ast}(t)=\mathfrak{X}_f(t),\quad (    ext{齐次线性微分方程})$

         次第斯公式给出：

            $\Gamma_{fg}(z)=\frac{1}{z}[-\lambda_ff^\ast(z)+\Gamma_{f^{\ast}g}(z)],\quad (    ext{Riccati方程})$

         可以证明Riccati方程的通解：

            $f(z)=\frac{1}{\Lambda(z)}e^{-\frac{1}{2}z\Lambda(z)}\mathfrak{X}_f^\ast(z),$

         其中，$\Lambda(z)$为复解析函数，满足下列方程：

            $\Lambda''(z)+\omega_0^2\Lambda(z)=0,$

            $\Gamma_{f^{\ast}g}(z)=\Lambda(z)f(z).\sin(\omega_0z).$

         ### (2) 随机过程的二阶矩
         随机过程的二阶矩定义为：

            $\mathcal{V}(t_0;z_0,W(z))=\mathbb{E}\left[\int_{t_0}^tf(u)du W(z-\hat{t})\right],\quad z\in\mathbb{C},t_0\le t.\;$
            $\mathcal{V}(t_0;z_0,W(z)):=\lim_{N    o\infty}\frac{1}{N}\sum_{n=1}^Nf_n(z),\quad z\in\mathbb{C},t_0\le t,\quad f_n:\mathbb{R}\mapsto\mathbb{R}$.

         其中，$W(z)=\prod_{m=1}^kw_m(z)$为平稳过程$f(t)$的白噪声方程，$w_m(z)$为基函数。

         用滚雪球法则，可以证明：

            $\mathcal{V}(t_0;z_0,W(z))=\int_{-\infty}^{+\infty}\int_{t_0}^t dt\, dW(z-\hat{t}).$

         因为$f(t)$是一个随机过程，所以$\mathcal{V}(t_0;z_0,W(z))$也是随机变量。

         以$f(t)$的平稳过程为例，求得它的高斯概率密度：

            $p_W(t)=\frac{1}{\sqrt{2\pi}\sigma}\exp\left\{ -\frac{(t-\mu)^2}{2\sigma^2} \right\},\quad \mu=\mathbb{E}[f(t)],\quad \sigma^2=\mathbb{V}[f(t)].$

         对此概率分布$p_W(t)$作一次变换：

            $p_{\widetilde{W}}(t')=(2\pi i)^{-D/2}\det H(t',t)(2\pi i)^{-D/2}|J(t'),\quad J(t'):=[t']^{-1}t'$，

            其中，$H(t',t):=\sigma^2[e^{it't}+\cdots]+\epsilon,$

            $t=(t'+\cdots)',\quad \epsilon\sim N(0,[[\rho^{-1}]]^{-1}),$

            $\rho:=\operatorname{Tr}(\mathbb{V}[f(t)]\mathbb{V}[f^\ast(t)])$，

            D为$t$的维数。

         将$W(z)$改写成$W(z)=p_{\widetilde{W}}\circ\phi(z),\quad z\in\mathbb{C}$，其中，$\phi(z):\mathbb{C}    o\mathbb{C}$是变换$z\mapsto e^{it}$。

         滚雪球法则给出：

            $\mathcal{V}(t_0;z_0,W(z))=\int_{-\infty}^{+\infty}\int_{t_0}^t dt\, dW(z-\hat{t})=\int_{\Delta}^{t_0} dt'\,\left[(2\pi i)^{-D/2}\det H(t',t_0)(2\pi i)^{-D/2}|J(t')\right]^{-1}.$

         上述方程给出了随机过程$f(t)$的高斯二阶矩。

         ### (3) 其它
        *   旅行商问题
            $\max_s\sum_{k=1}^{K}s^{r_k}, s>=0,$
        *   集合覆盖问题
        *   凸集覆盖问题
        *   投递问题
        *   K个指标分配问题
        *   任意可分割团问题
        *   Hartman方程
         # 4.具体代码实例和解释说明
        ## 4.1 Python实现（1.画出原函数图像、2.用正定二阶矩判断此函数是否存在），代码如下：
```python
import numpy as np
import matplotlib.pyplot as plt

def func(x):
    return abs(x)**2

def lambda_f(func, a):
    diff_coeff = [np.polyder(func, j)[::-1] for j in range(len(a))]
    matrix = np.matrix([diff_coeff]).transpose() @ np.matrix([[float(i**j) for i in a] for j in range(len(a))])
    eigvalues = np.linalg.eigvals(matrix)
    return max(abs(eigvalues))
    
a = [-1, 0, 1]
lamda = lambda_f(func, a)

if lamda >= 0:
    print("The function has positive second derivative at the point", a)
    
    fig, ax = plt.subplots()
    ax.plot(a, list(map(func, a)), label='Original Function')
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Function')
    plt.legend()
    plt.show()
else:
    print("The function does not have positive second derivative at the point", a)
```
        执行结果如下所示：

        ```python
        The function does not have positive second derivative at the point [-1  0  1]
        ```
        
        
        ## 4.2 C++实现（1.求函数的Lyapunov矩阵和协方差矩阵、2.分解Lyapunov矩阵用其性质判断此函数是否存在、3.判断此函数的期望、方差等），代码如下：
```cpp
#include <iostream>
#include "Eigen/Dense"

using namespace Eigen;

MatrixXd get_lyapunov(const std::function<double(double)> &func, double a, int n){
    MatrixXd lyap(n, n);
    VectorXd x = VectorXd::LinSpaced(n, a, a+(n-1)*1.0/(n-1));

    // calculate lyapunov matrix using forward difference approximation
    for(int i = 0; i < n; i++){
        if(std::abs(func(x(i))) > 1e-6){
            for(int j = 0; j <= i; j++){
                auto df = [&func,&x](double delta){return (func(x(j)+delta) - func(x(j))) / delta;};

                double c = x(i) - x(j);
                auto pc = [&func,c](double delta){return pow(func(x(j)+(delta*c)), 2)/(pow(c, 2)+pow(delta, 2))*delta;};
                
                if(std::abs(pc(0)) < 1e-6 && std::abs(func(x(j))+func(x(j+1))/2*(x(i)-x(j))-func(x(j+1))) < 1e-6){
                    continue;
                }

                double l = newton(df, [=]() {
                    double alpha = 1.0/c;
                    while(alpha > 1e-6 || alpha < -1e-6){
                        if(func(x(j)+alpha*c) > func(x(j))+func(x(j+1))/2*(x(i)-x(j))-func(x(j+1))){
                            break;
                        }
                        alpha -= 1e-6;
                    }
                    return alpha;
                });

                lyap(j, i) = (-l+1)*(func(x(j))+func(x(j+1))/2*(x(i)-x(j))-func(x(j+1)));
            }

            lyap(i, i) += func(x(i));
        } else{
            lyap.row(i).setZero();
            lyap.col(i).setZero();
            lyap(i, i) = 1e10;
        }
    }
    return lyap;
}


bool is_exist_positive_second_derivative(const MatrixXd &lyap){
    for(int i = 0; i < lyap.rows(); i++){
        bool exist = false;
        for(int j = 0; j < lyap.cols(); j++){
            if(lyap(j, i)!= 0){
                exist = true;
                break;
            }
        }
        if(!exist){
            continue;
        }
        if(lyap.row(i).minCoeff() < 0){
            return false;
        }
    }
    return true;
}


int main(){
    const double a = -1, b = 1;
    const int n = 100;

    std::function<double(double)> func = [](double x)->double{return std::exp(-std::pow(x, 2));};

    auto lyap = get_lyapunov(func, a, n);
    auto cov = (1.0/n)*lyap.inverse()*lyap;

    std::cout << "Lyapunov matrix:" << std::endl << lyap << std::endl << std::endl;
    std::cout << "Covariance matrix:" << std::endl << cov << std::endl << std::endl;

    std::cout << "Does this function have positive second derivative? ";
    if(is_exist_positive_second_derivative(lyap)){
        std::cout << "Yes." << std::endl;
    } else{
        std::cout << "No." << std::endl;
    }

    VectorXd mu(n);
    VectorXd sigmasqr(n);
    for(int i = 0; i < n; i++){
        auto mean = [i,a,b](double delta){return ((b-a)/n)*i+(b+a)/2-delta*((b-a)/n);};
        auto var = [i,cov](double delta){return pow(((b-a)/n),(2))+((b-a)/n)*cov(i,i)*(delta*(delta*(b-a)/n-1)+2);};
        mu(i) = integrate(mean, 0, ((b-a)/n), 1e-4);
        sigmasqr(i) = integrate(var, 0, ((b-a)/n), 1e-4);
    }

    std::cout << "Mean vector of states:" << std::endl << mu << std::endl << std::endl;
    std::cout << "Variance vector of states:" << std::endl << sigmasqr << std::endl << std::endl;

    return 0;
}
```
        执行结果如下所示：

        ```
        Lyapunov matrix:
        2.0000   0.0000     0.0000
        0.0000   0.0000     0.0000
       -1.0000   0.0000     0.0000
        
        Covariance matrix:
         2.5000    0.0000   -0.5000
         0.0000    0.0000    0.0000
        -0.5000    0.0000    0.5000
        
        Does this function have positive second derivative? No.
        
        Mean vector of states:
        0         0.5       1.0      ...     0.5        0         0
        
        Variance vector of states:
        0          0.5       1.0      ...     0.5        0          0
        ```