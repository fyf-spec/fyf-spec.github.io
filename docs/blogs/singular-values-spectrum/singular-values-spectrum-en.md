---
title: "Singular Values and Spectral Decomposition"
description: A geometric route from directional stretching to singular-value spectra, low-rank approximation, and the spectral norm.
date: 2026-07-12
lang: en-US
outline: deep
---

# Singular Values and Spectral Decomposition

Consider two matrices:

$$
A_1=
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix},
\qquad
A_2=
\begin{bmatrix}
1&0\\
0&10^{-3}
\end{bmatrix}.
$$

Both have rank two. Yet they do not preserve a two-dimensional signal equally well. $A_1$ leaves the unit circle unchanged, while $A_2$ squeezes it into a needle: the second coordinate still exists in exact arithmetic, but a small amount of noise can overwhelm it.

Rank cannot express this difference. It tells us whether a direction has vanished, but not how close that direction is to vanishing. What we want instead is a description of a matrix in terms of the input directions it acts on, the amount of stretching along each direction, and the output directions those inputs become.

That description is the singular value decomposition. We will build it from the two-dimensional picture, then use the same picture to understand the singular-value spectrum, $A^TA$, low-rank approximation, spectral decomposition, and the spectral norm.

## Singular value decomposition

For a real matrix $A\in\mathbb{R}^{m\times n}$, the singular value decomposition is

$$
A=U\Sigma V^T.
$$

$U$ and $V$ are orthogonal matrices, and $\Sigma$ is rectangular and diagonal. Its diagonal entries are the singular values,

$$
\sigma_1\ge \sigma_2\ge\cdots\ge 0.
$$

The product becomes useful when we read it in the order in which it acts on a vector:

$$
x
\xrightarrow{\;V^T\;}
V^Tx
\xrightarrow{\;\Sigma\;}
\Sigma V^Tx
\xrightarrow{\;U\;}
U\Sigma V^Tx.
$$

Let $v_i$ be the $i$-th column of $V$. The $i$-th entry of $V^Tx$ is

$$
(V^Tx)_i=v_i^Tx,
$$

so $V^T$ reads the coordinate of $x$ along the input direction $v_i$. The diagonal matrix $\Sigma$ multiplies that coordinate by $\sigma_i$, and $U$ places the result along the output direction $u_i$. Equivalently,

$$
Av_i=\sigma_i u_i.
$$

This equation contains the geometry of the whole decomposition. The $v_i$ form orthogonal input directions; the $u_i$ form orthogonal output directions; and $\sigma_i$ records how much length survives between them. If $\sigma_i=0$, the entire $v_i$ direction is erased.

Move $\theta_V$, $\sigma_1$, $\sigma_2$, and $\theta_U$ below; the panels follow a fixed vector through $x$, $V^Tx$, $\Sigma V^Tx$, and $U\Sigma V^Tx$.

<SvdExplorer locale="en" />

The unit sphere now has a simple fate. $V^T$ and $U$ preserve lengths because they are orthogonal. Only $\Sigma$ changes length, so $A$ maps the unit sphere to an ellipsoid whose semiaxis lengths are the singular values and whose semiaxis directions are the columns of $U$.

## What singular values measure

The opening example has singular values $(1,1)$ for $A_1$ and $(1,10^{-3})$ for $A_2$. Rank sees only that both pairs contain two nonzero numbers:

$$
\operatorname{rank}(A)=\#\{i:\sigma_i>0\}.
$$

The singular values tell us what rank leaves out. A value of zero means that a direction is gone. A very small value means that the direction survives algebraically but is fragile numerically, because recovering it requires division by a small number.

For a full-rank square matrix, this fragility is summarized by the two-norm condition number,

$$
\kappa_2(A)=\frac{\sigma_1}{\sigma_n}.
$$

The condition number of $A_2$ is $10^3$: an error aligned with its weak direction can be amplified by a factor on that scale when the map is inverted. If $\sigma_n=0$, the inverse does not exist and the condition number is infinite.

## Singular-value spectrum

The ordered list $(\sigma_1,\sigma_2,\ldots)$ is called the **singular-value spectrum**. The qualifier matters. The spectrum of a square matrix usually means its eigenvalues, which may be negative or complex; singular values exist for rectangular matrices and are always nonnegative real numbers.

In high dimensions, the tail of this list is often more informative than exact rank. Given a threshold $\varepsilon$, we can define an effective rank

$$
r_\varepsilon=\#\{i:\sigma_i\ge\varepsilon\}.
$$

Given a truncation point $k$, we can also measure how much squared singular-value mass is retained:

$$
E_k=
\frac{\sum_{i=1}^{k}\sigma_i^2}
{\sum_i\sigma_i^2}.
$$

A flat spectrum spreads scale across many directions. A decaying spectrum concentrates it near the front, so a matrix may have full exact rank while behaving like a much lower-dimensional map.

The example below uses $\sigma_i=e^{-\alpha(i-1)}$; adjust $\alpha$, $\varepsilon$, and $k$ to see how spectral decay changes effective rank, retained energy, and truncation error.

<SpectrumMicroscope locale="en" />

This is the quantitative advantage of singular values: they turn the binary statement “a direction exists” into a graded description of how strongly that direction survives.

## SVD from AᵀA

The SVD tells us that the important input directions are the columns of $V$. How can we recover those directions from $A$?

The useful observation is that squared output length is a quadratic form:

$$
\lVert Ax\rVert_2^2=x^TA^TAx.
$$

The matrix $A^TA$ therefore records how much $A$ stretches every input direction, without retaining the final orientation of the output. Substituting the SVD makes this cancellation explicit:

$$
\begin{aligned}
A^TA
&=(U\Sigma V^T)^T(U\Sigma V^T)\\
&=V\Sigma^TU^TU\Sigma V^T\\
&=V\Sigma^T\Sigma V^T.
\end{aligned}
$$

Because $U^TU=I$, the left singular directions disappear. What remains is an orthogonal spectral decomposition of $A^TA$:

$$
A^TAv_i=\sigma_i^2v_i.
$$

Thus the right singular vectors are eigenvectors of $A^TA$, and the singular values are the square roots of its eigenvalues. For every positive singular value we can then recover the corresponding left singular vector from

$$
u_i=\frac{Av_i}{\sigma_i}.
$$

When $\sigma_i=0$, this formula is undefined; the remaining columns of $U$ are supplied by an orthonormal completion. The same argument on the output side gives

$$
AA^Tu_i=\sigma_i^2u_i.
$$

There is one common shortcut worth avoiding. For an arbitrary vector $x$,

$$
A^TAx=V\Sigma^T\Sigma V^Tx,
$$

not $\Sigma^2x$. The diagonal action appears only after $x$ has been expressed in the $V$ basis, or when $x$ itself is one of the $v_i$.

This eigendecomposition route explains the mathematics, but it is not the preferred numerical implementation of SVD. Explicitly forming $A^TA$ squares the condition number and can destroy information in weak directions; stable algorithms work on $A$ more directly, typically through bidiagonalization.

## Spectral decomposition

The relation $Av_i=\sigma_i u_i$ describes what $A$ does to one special direction. To recover its action on an arbitrary vector, expand

$$
x=\sum_i(v_i^Tx)v_i.
$$

Applying $A$ gives

$$
Ax=\sum_i\sigma_i(v_i^Tx)u_i.
$$

Since this holds for every $x$, the matrix itself can be written as

$$
A=\sum_i\sigma_i u_iv_i^T.
$$

Each rank-one term is a directional channel: $v_i^T$ reads one input coordinate, $\sigma_i$ scales it, and $u_i$ writes it into the output space. Keeping only the first $k$ channels gives the truncated SVD,

$$
A_k=\sum_{i=1}^{k}\sigma_i u_iv_i^T.
$$

The Eckart–Young theorem says that $A_k$ is the best rank-$k$ approximation to $A$ in both the spectral and Frobenius norms. In the spectral norm, the error has an especially clean form:

$$
\lVert A-A_k\rVert_2=\sigma_{k+1}.
$$

This is why the tail of the bar chart matters: the first discarded bar is exactly the worst remaining directional error.

The rank-one expansion is sometimes loosely called a spectral decomposition, but the distinction is useful. A real symmetric matrix has an orthogonal eigendecomposition

$$
A=Q\Lambda Q^T
=\sum_i\lambda_iq_iq_i^T.
$$

Here the same direction $q_i$ appears on both sides. For a general matrix, the input and output directions $v_i$ and $u_i$ need not agree, so its SVD is not an orthogonal spectral decomposition of $A$ itself.

If $A$ is symmetric positive semidefinite, then its eigenvalues are nonnegative and its eigendecomposition aligns with its SVD. If $A$ is symmetric but indefinite, its singular values are $|\lambda_i|$ and the signs are absorbed into one set of singular vectors. For an arbitrary matrix, it is $A^TA$ and $AA^T$ that always have genuine orthogonal spectral decompositions:

$$
A^TA=\sum_i\sigma_i^2v_iv_i^T,
\qquad
AA^T=\sum_i\sigma_i^2u_iu_i^T.
$$

## Spectral norm

One final quantity falls out of the same picture. The spectral norm asks for the largest factor by which $A$ can stretch a unit vector:

$$
\lVert A\rVert_2
=\max_{\lVert x\rVert_2=1}\lVert Ax\rVert_2.
$$

Set $y=V^Tx$. Orthogonality gives $\lVert y\rVert_2=\lVert x\rVert_2=1$, and therefore

$$
\begin{aligned}
\lVert Ax\rVert_2^2
&=\lVert U\Sigma V^Tx\rVert_2^2\\
&=\lVert\Sigma y\rVert_2^2\\
&=\sum_i\sigma_i^2y_i^2\\
&\le\sigma_1^2\sum_i y_i^2\\
&=\sigma_1^2.
\end{aligned}
$$

The bound is attained by choosing $x=v_1$, so

$$
\boxed{\lVert A\rVert_2=\sigma_1}.
$$

We can now answer the question posed at the start. A matrix is best understood as a collection of orthogonal directional channels,

$$
v_i\xrightarrow{\;\sigma_i\;}u_i.
$$

Rank counts how many channel gains are nonzero. The singular-value spectrum records all their strengths. Truncation removes the weak channels in the best possible way, and the spectral norm is the gain of the strongest channel. This picture applies to every real matrix; only when the matrix is symmetric do its input and output directions align into an orthogonal spectral decomposition of the matrix itself.
