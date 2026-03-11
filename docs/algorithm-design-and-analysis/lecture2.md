# Divide and Conquer and Running Time analysis
## Karatsuba Integer Multiplication
对于两个 $n$ 位整数 $x$ 和 $y$，若做最朴素的乘法，则需要进行O($n^2$)次elemwise的乘法运算。
对齐进行分治优化(Divide and Conquer)，第一步我们可以将它们写成：
$$x = x_1 \cdot 10^k + x_0$$
$$y = y_1 \cdot 10^k + y_0$$
其中 $k = \lceil n/2 \rceil$。那么它们的乘积为：
$$xy = (x_1 \cdot 10^k + x_0)(y_1 \cdot 10^k + y_0) = x_1y_1 \cdot 10^{2k} + (x_1y_0 + x_0y_1) \cdot 10^k + x_0y_0$$

注意到在分治之后的式子中，如果继续朴素地计算$x_1y_1, x_0y_1, x_1y_0, x_0y_0$，那么总的乘法次数为$4T(n/2) + O(n)$，根据主定理，其时间复杂度仍然是$O(n^2)$，并没有得到优化。

我们注意到在结果中，我们只需要$(x_0y_1 + x_1y_0)$的整体值，而**分别计算$x_0y_1$和$x_1y_0$是多余的，产生了多余的信息**。因此，我们可以通过计算$(x_1+x_0)(y_1+y_0) - x_1y_1 - x_0y_0$来得到$(x_0y_1 + x_1y_0)$的值。这样，我们只需要进行3次乘法运算，就可以得到$(x_0y_1 + x_1y_0)$的值。总的乘法次数为$3T(n/2) + O(n)$，根据主定理，其时间复杂度为$O(n^{\log_2 3})$。

## Divide and conquer
There are some better algorithms

![Better Integer Multiplication Algorithms](./images/integer_better_algos.png)

The last three algorithms are based on FFT


## asymptotic analysis
