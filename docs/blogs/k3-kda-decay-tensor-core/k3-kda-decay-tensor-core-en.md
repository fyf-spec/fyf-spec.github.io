---
title: "K3's KDA Precision Problem: Why Lower-Bounded Decay Is Necessary"
description: "Understanding the joint design of decay lower bounds, 16-token tiles, and Tensor Cores through the split-exponent and Neumann-inverse precision paths."
date: 2026-08-06
lang: en-US
outline: deep
---

# K3's KDA Precision Problem: Why Lower-Bounded Decay Is Necessary

In the Kimi K3 paper, the implementation of KDA adds a lower bound to the log-decay $g$, restricting a quantity that could otherwise approach negative infinity to $(-5, 0)$. The authors state that this addresses precision problems on diagonal tiles during matrix multiplication, allowing a diagonal tile that would otherwise require position-by-position computation to be evaluated efficiently on Tensor Cores. But how does the lower bound solve this problem, and does it also improve numerical precision elsewhere?

Based on the K3 report, the [FlashKDA repository](https://github.com/MoonshotAI/FlashKDA), and a set of mechanism experiments on an H200, I reached two preliminary conclusions about how the lower bound works:

1. It bounds the absolute range of $|g_i-g_j|$, preventing $\exp(G_i-G_j)$ in diagonal-tile matrices from exploding or disappearing and destabilizing the matrix multiplication.
2. It controls the decay rate of intermediate values such as $L^2$, $L^4$, and $L^8\ldots$ while inverting the lower-triangular system, preventing numerical underflow.

**The first point determines whether a diagonal tile can use dense Tensor Core operations. The second determines whether the FP16 Neumann expansion remains controlled, and helps explain why the final chunk size is 16.**

## How FlashKDA computes tiles

Let us temporarily ignore decay and begin with the most basic query-key multiplication inside a chunk. Suppose the chunk length is $C=128$ and the head dimension is $d=128$:

$$
Q,K\in\mathbb{R}^{128\times128}.
$$

The first dimension indexes token positions, and the second contains the 128 features of each token. A Tensor Core does not process the entire causal matrix as one indivisible matrix. Instead, $Q$ and $K$ are sliced along the token dimension into panels of 16 rows. The $r$-th query panel and the $c$-th key panel are

$$
Q^{(r)},K^{(c)}\in\mathbb{R}^{16\times128},
$$

Multiplying them produces one score tile:

$$
B^{(r,c)}
=Q^{(r)}\left(K^{(c)}\right)^\top
\in\mathbb{R}^{16\times16}.
$$

Thus, multiplying two 16×128 panels computes only one 16×16 block of the 128×128 score matrix. When $C=128$, there are $8\times8$ such blocks.

![How a 128×128 causal matrix is computed from 16×128 Q/K panels as 16×16 tiles](./kda-tiled-causal-matrix.png)

The causal mask divides these tiles into three categories:

- When $c<r$, all 16 positions in the query tile occur after all 16 positions in the key tile, so the entire 16×16 block is valid. This is an **off-diagonal dense tile**.
- When $c>r$, every key comes from the future, so the whole block is masked.
- When $c=r$, the queries and keys come from the same group of 16 positions, and only the lower triangle satisfying $i\ge j$ is valid. This is a **diagonal tile**.

> The 128-token chunk above illustrates the computation in which a large chunk is subdivided into 16-token secondary tiles. FlashKDA v1 currently uses a 16-token kernel chunk directly, so one kernel chunk is exactly one 16×16 numerical tile.

#### How the per-channel decay factor $\exp(G)$ enters tile computation

KDA computes more than an ordinary $QK^\top$. It must also decay earlier keys according to the retention at every step. Let the log-decay at step $t$ be $g_t<0$. The cumulative log-decay from the beginning of the chunk through position $i$ is

$$
G_i=\sum_{t=1}^{i}g_t,
\qquad
\Gamma_i=\exp(G_i).
$$

After collecting the $\Gamma_i$ values into a scaling vector $\Gamma$, the causal score and output inside a chunk can be written as

$$
A=\operatorname{Tril}\!\left[
(\Gamma\odot Q)
\left(K\oslash\Gamma\right)^\top
\right],
$$

For a particular position pair $(i,j)$, the scaled query-key product is

$$
(\Gamma_i q_i)\cdot\left(\frac{k_j}{\Gamma_j}\right)
=q_i\cdot k_j\,\exp(G_i-G_j).
$$

This explains how decay can be folded into a GEMM. First scale row $i$ of $Q$ by $\exp(G_i)$, then scale row $j$ of $K$ by $\exp(-G_j)$. An ordinary matrix multiplication will then produce the correct relative decay for every position pair.

For tile $(r,c)$, the same process is

$$
A^{(r,c)}
=
\left(\Gamma^{(r)}\odot Q^{(r)}\right)
\left(K^{(c)}\oslash\Gamma^{(c)}\right)^\top.
$$

When $i\ge j$, every $g_t<0$, so $G_i\le G_j$ and therefore

$$
0<\exp(G_i-G_j)\le1.
$$

In other words, **the relative decay in the result matrix cannot explode.** The numerical risk is not the final $\exp(G_i-G_j)$. It arises because, to execute the GEMM, the kernel splits that quantity into two independent operands:

$$
\exp(G_i-G_j)=\exp(G_i)\exp(-G_j).
$$

If $G_i$ is very negative, the left factor $\exp(G_i)$ may underflow to zero, while the right factor $\exp(-G_j)$ may overflow to $\infty$. This is the first central tension:

> Forming $\exp(G_i-G_j)$ directly is numerically safe, but it requires a separate calculation for each position pair. Splitting it into $\exp(G_i)$ and $\exp(-G_j)$ lets the pairwise work collapse into a GEMM, but requires both factors to be individually representable at low precision.

## Why a diagonal tile needs lower-bounded decay

As described above, matrix computation splits the target $\exp(G_i-G_j)$ into $\exp(G_i)$ and $\exp(-G_j)$ and assigns them to the $Q$ and $K$ tiles. Either operand may therefore underflow or overflow. In FlashKDA, every step has $g_t<0$, so the cumulative quantity $G_t$ decreases monotonically with position. For an off-diagonal tile, all key positions $j$ precede all query positions $i$. We can therefore choose one shared boundary $b$ between the two tiles such that

$$
G_i\le G_b\le G_j.
$$

The relative decay can then be split around this boundary:

$$
\exp(G_i-G_j)
=\exp(G_i-G_b)\exp(G_b-G_j).
$$

Both exponents are non-positive, so both factors lie in $(0,1]$. The same $G_b$ serves the entire off-diagonal tile, making it possible to form a dense GEMM directly.

The diagonal tile is different: $i$ and $j$ lie in the same tile. Although every valid position pair satisfies $j\le i$, there is no fixed boundary $b$ that lies between every pair $(j,i)$. If $b$ is fixed at the start of the tile, then $G_b-G_j\ge0$; if it is fixed at the end, then $G_i-G_b\ge0$. One side always becomes a potentially large positive exponent. Keeping both sides at most 1 would require the boundary to change with every position pair. That degenerates into Kimi Linear's position-pair diagonal path, which cannot be computed efficiently on Tensor Cores.

This is where lower-bounded decay matters. With

$$
g\in(-5,0),
\quad
16\lvert g_{\min}\rvert=80.
$$

the difference between any $G$ and a fixed $G_b$ inside a 16-token tile is bounded by $(-80,80)$. The corresponding exponential is at most $\exp(80)$, which remains within BF16's dynamic range. The diagonal tile no longer needs a separate boundary for every $(i,j)$ pair and can use a dense Tensor Core GEMM just like an off-diagonal tile, substantially improving runtime performance.

## Matrix inversion in FlashKDA

The delta updates inside a KDA chunk form a strictly lower-triangular matrix $L$, and the implementation must compute

$$
\mathrm{INV}=(I+L)^{-1}.
$$

For a general dense $C\times C$ matrix, direct inversion costs $O(C^3)$. But $I+L$ has a special lower-triangular structure that permits a Neumann-series expansion.

A Neumann series can be understood as the matrix version of a geometric series. For a scalar $x$ with $\lvert x\rvert<1$,

$$
\frac{1}{1-x}=1+x+x^2+x^3+\cdots.
$$

Replacing the scalar $x$ with a matrix $L$ gives

$$
(I+L)^{-1}=I-L+L^2-L^3+\cdots.
$$

This expansion is called a Neumann series. In general, it requires the spectral radius of $L$ to be less than 1, meaning that every eigenvalue has absolute value below 1. Here, however, $L$ is not a general matrix. It is a $C\times C$ strictly lower-triangular matrix, so its spectral radius is exactly zero. More directly, every strictly lower-triangular matrix is [nilpotent](https://en.wikipedia.org/wiki/Nilpotent_matrix): each multiplication by $L$ moves a nonzero entry down by at least one diagonal, and after $C$ multiplications,

$$
L^C=0.
$$

Therefore, every term of degree at least $C$ in the Neumann series of $L$ is exactly zero. The infinite series becomes a finite expansion that must terminate and is exact:

$$
(I+L)^{-1}
=I-L+L^2-L^3+\cdots+(-L)^{C-1}.
$$

This finite sequence can be simplified further. FlashKDA does not add the terms one by one; it uses a repeated-squaring factorization. For $C=16$,

$$
(I+L)^{-1}
=(I-L)(I+L^2)(I+L^4)(I+L^8).
$$

The kernel only needs to compute $L^2$, $L^4$, and $L^8$, rewriting the term-by-term expansion as three rounds of dense GEMMs suitable for Tensor Cores. This reduces the serial depth of the inversion path and improves hardware parallelism. Under classical matrix multiplication, however, each GEMM still costs $O(C^3)$. The gain comes mainly from the parallel mapping at a fixed $C=16$, not from reducing the asymptotic complexity to $O(C^2)$.

## The role of lower-bounded decay in matrix inversion

The previous section showed that the depth of the repeated-squaring expansion is determined by the chunk length $C$. The connection between lower-bounded decay and matrix inversion is also established through $C$.

K3 restricts each step's log-decay to $g\in(-5,0)$. If every step is close to the strongest decay, a chunk of length $C$ must handle the following worst-case cumulative scales:

$$
\exp(-5C)
\quad\text{and}\quad
\exp(5C).
$$

From the exponent range of BF16/FP32, $\exp(\pm85)$ remains finite at $C=17$, whereas $\exp(\pm90)$ at $C=18$ has crossed the overflow and underflow boundaries. From the dynamic range of cumulative decay alone, the upper bound on a usable chunk length is therefore around 17.

Meanwhile, FlashKDA's inverse constructs $L^2,L^4,L^8,\ldots$ by repeated squaring, so powers of two form natural implementation boundaries. Sixteen is exactly the largest power of two inside the exponent-safe region above:

$$
\begin{aligned}
C=16 &: \quad L^{16}=0,\quad \text{the highest power actually formed is }L^8,\\
C=17 &: \quad L^{16}\ne0\ \text{ in general},\quad \text{requiring an additional }(I+L^{16}).
\end{aligned}
$$

Thus, increasing the chunk length from 16 to 17 does not yet exceed the exponent range, but it requires the low-precision kernel to form $L^{16}$ for the first time. This does not mean that $L^{16}$ must overflow, but it adds another path for intermediate-value growth and rounding-error amplification. Increasing the length further to 18 also reaches the dynamic-range boundary of cumulative decay.

By bounding the worst-case cumulative decay, lower-bounded decay and the Neumann repeated-squaring expansion jointly determine a sensible chunk boundary: **16 is the largest power of two inside the exponent-safe region, and it lets the inverse stop after $L^8$.**

## Numerical experiments on H200

To separate the split-exponent problem from the Neumann-inverse problem, the experiment first constructs a `floor_e-5` input: every position uses the strongest decay $g=-5$, and the key arithmetic path of FlashKDA K1 is extended across different chunk lengths.

The experimental procedure is:

1. Convert $\exp(G)$ and $\exp(-G)$ to BF16.
2. Construct $L$ with BF16 operands and FP32 accumulation.
3. Store $L$ in FP16.
4. Use FP16 accumulation on the H200 to simulate the Neumann inverse.
5. Use FP64 construction and direct inversion as the gold result.

Each configuration is run with three seeds. The errors below are medians, while counts of non-finite scales use the worst observed value.

### Exponent boundary

| Chunk C | inverse finite | decay = 0 | reciprocal = inf | INV total | INV power |
|---:|---:|---:|---:|---:|---:|
| 8 | 100% | 0 | 0 | $8.80\times10^{-7}$ | $5.16\times10^{-8}$ |
| 16 | 100% | 0 | 0 | $1.23\times10^{-6}$ | $8.51\times10^{-8}$ |
| 17 | 100% | 0 | 0 | $1.07\times10^{-6}$ | $6.69\times10^{-8}$ |
| 18 | 100% | 1 | 1 | $7.86\times10^{-6}$ | $6.62\times10^{-8}$ |
| 19 | 0% | 2 | 2 | $\infty$ | $6.60\times10^{-8}$ |
| 32 | 0% | 15 | 15 | $\infty$ | $6.51\times10^{-8}$ |

Under the worst-case decay $g_{\min}=-5$, $C=16$ remains in the safe range. The expected overflow and underflow begin around 18--19 tokens and eventually propagate into the valid region of $L$. At the same time, INV power stays near $10^{-8}$ throughout, showing that the failure observed here comes from the exponent range rather than high powers in the Neumann expansion.

### High-power stress input

To exclude exponent overflow and underflow, the experiment constructs `weak_decay_correlated` inputs:

- All keys are identical and normalized, producing highly correlated update directions.
- $g=-0.01$, keeping $\exp(\pm G)$ finite throughout.
- $\beta=0.5$, preventing multi-step propagation from disappearing too quickly.

| Chunk C | Highest power formed | Max power magnitude | INV total | INV power | Finite |
|---:|---:|---:|---:|---:|---:|
| 8 | $L^4$ | $1.40$ | $1.51\times10^{-3}$ | $2.13\times10^{-4}$ | 100% |
| 16 | $L^8$ | $1.96\times10^1$ | $3.27\times10^{-3}$ | $3.93\times10^{-3}$ | 100% |
| 17 | $L^{16}$ | $2.43\times10^1$ | $6.75\times10^{-3}$ | $5.64\times10^{-3}$ | 100% |
| 24 | $L^{16}$ | $5.32\times10^2$ | $7.55\times10^{-2}$ | $5.94\times10^{-2}$ | 100% |
| 32 | $L^{16}$ | $5.87\times10^3$ | $1.33$ | $1.21$ | 100% |
| 48 | $L^{32}$ | $\infty$ | $\infty$ | $\infty$ | 0% |

The error gradually increases with the power. $L^{16}$ and $L^{32}$ become new entry points for intermediate-value growth and FP16 error amplification. This is only a simulation, however, and the decay setting $g=-0.01$ is deliberately extreme. Actual behavior still depends on key correlation, $\beta$, decay strength, and the structure of $L$ in the trained model.

### Ordinary random input

The stress experiment exposes a mechanism boundary; it should not be treated as a typical model distribution. Random `model_like` inputs provide an important counterexample:

| Chunk C | Finite | INV total | INV power | Max power magnitude |
|---:|---:|---:|---:|---:|
| 8 | 100% | $1.78\times10^{-5}$ | $1.46\times10^{-6}$ | $4.50\times10^{-2}$ |
| 16 | 100% | $3.01\times10^{-5}$ | $1.51\times10^{-6}$ | $2.82\times10^{-2}$ |
| 32 | 100% | $2.61\times10^{-5}$ | $1.34\times10^{-6}$ | $7.84\times10^{-2}$ |
| 48 | 0% | $\infty$ | $1.93\times10^{-6}$ | $7.46\times10^{-2}$ |

In more realistic conditions, the extreme behavior above is less likely to occur, and error does not necessarily or monotonically worsen as the chunk grows.

Choosing a chunk size of 16 is therefore a conservative design point that covers extreme decay, highly correlated keys, low-precision inversion, and the hardware layout.

#### Experiment code

The excerpt below retains the three input families, the stable reference, the kernel's numerical path, FP16 accumulation, and single-run metrics. Result aggregation, plotting, table generation, and file output are omitted.

<details>
<summary>Show experiment code</summary>

```python
#!/usr/bin/env python3
"""Measure FlashKDA (I + L)^-1 accuracy as chunk size grows."""

import ctypes
import math
from pathlib import Path

import torch


LOG2E = math.log2(math.e)
_CUBLAS = None
_CUBLAS_HANDLE = ctypes.c_void_p()
_FP16_ONE = ctypes.c_ushort(0x3C00)
_FP16_ZERO = ctypes.c_ushort(0x0000)


def exp2_ftz_f32(x):
    y = torch.exp2(x.float())
    return torch.where(y < torch.finfo(torch.float32).tiny, 0.0, y)


def cublas_fp16_accum_mm(a, b):
    """CUDA FP16-accumulator GEMM, matching the kernel precision path."""
    global _CUBLAS
    if _CUBLAS is None:
        try:
            _CUBLAS = ctypes.CDLL("libcublas.so")
        except OSError:
            import nvidia.cublas

            lib_dir = Path(nvidia.cublas.__file__).parent / "lib"
            candidates = sorted(lib_dir.glob("libcublas.so.*"))
            if not candidates:
                raise
            _CUBLAS = ctypes.CDLL(str(candidates[0]))
        assert _CUBLAS.cublasCreate_v2(ctypes.byref(_CUBLAS_HANDLE)) == 0

    m, k = a.shape
    k2, n = b.shape
    assert k == k2 and a.is_contiguous() and b.is_contiguous()

    out = torch.zeros(m, n, dtype=torch.float16, device=a.device)
    torch.cuda.synchronize(a.device)
    status = _CUBLAS.cublasGemmEx(
        _CUBLAS_HANDLE,
        0,
        0,
        ctypes.c_int(n),
        ctypes.c_int(m),
        ctypes.c_int(k),
        ctypes.byref(_FP16_ONE),
        ctypes.c_void_p(b.data_ptr()),
        ctypes.c_int(2),
        ctypes.c_int(n),
        ctypes.c_void_p(a.data_ptr()),
        ctypes.c_int(2),
        ctypes.c_int(k),
        ctypes.byref(_FP16_ZERO),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int(2),
        ctypes.c_int(n),
        ctypes.c_int(64),  # CUBLAS_COMPUTE_16F
        ctypes.c_int(0),   # CUBLAS_GEMM_DEFAULT
    )
    assert status == 0, f"cublasGemmEx failed: {status}"
    torch.cuda.synchronize(a.device)
    return out


def fp16_accum_mm(a, b):
    """Use CUDA FP16 accumulation, with a deterministic CPU fallback."""
    if a.is_cuda:
        return cublas_fp16_accum_mm(a, b)

    out = torch.zeros(
        a.shape[0], b.shape[1], dtype=torch.float16, device=a.device
    )
    for k in range(a.shape[1]):
        product = a[:, k:k + 1].float() * b[k:k + 1].float()
        out = (out.float() + product).half()
    return out


def neumann_inverse_fp16(l):
    """Compute (I-L)(I+L^2)(I+L^4)... with FP16 accumulation."""
    n = l.shape[0]
    inv = (torch.eye(n, dtype=torch.float16, device=l.device) - l).half()
    power = l
    exponent = 1
    max_power = float(l.abs().max())

    while exponent * 2 < n:
        power = fp16_accum_mm(power, power)
        exponent *= 2
        max_power = max(max_power, float(power.abs().max()))
        inv = (inv + fp16_accum_mm(inv, power)).half()

    return inv, exponent, max_power


def make_inputs(case, chunk, dim, seed, device):
    generator = torch.Generator(device="cpu").manual_seed(seed)

    if case == "weak_decay_correlated":
        key = torch.randn(1, dim, dtype=torch.float64, generator=generator)
        key = key / key.norm(dim=-1, keepdim=True)
        keys = key.expand(chunk, -1)
        log_decay = torch.full((chunk,), -0.01, dtype=torch.float64)
        beta = torch.full((chunk,), 0.5, dtype=torch.float64)
    else:
        keys = torch.randn(
            chunk, dim, dtype=torch.float64, generator=generator
        )
        keys = keys / keys.norm(dim=-1, keepdim=True)
        if case == "floor_e-5":
            log_decay = torch.full((chunk,), -5.0, dtype=torch.float64)
        else:
            random_gate = torch.randn(
                chunk, dtype=torch.float64, generator=generator
            )
            log_decay = -5.0 * torch.sigmoid(random_gate)
        beta = torch.sigmoid(
            torch.randn(chunk, dtype=torch.float64, generator=generator)
        )

    return (
        keys.to(device=device, dtype=torch.bfloat16),
        log_decay.to(device),
        beta.to(device),
    )


def stable_l(keys, log_decay, beta):
    keys64 = keys.double()
    cumsum = log_decay.cumsum(0)
    dots = keys64 @ keys64.t()
    decay_ratio = torch.exp(
        (cumsum[:, None] - cumsum[None, :]).clamp(max=0.0)
    )
    return torch.tril(
        dots * decay_ratio * beta[:, None], diagonal=-1
    )


def kernel_l(keys, log_decay, beta):
    cumsum_log2 = log_decay.cumsum(0) * LOG2E
    decay = exp2_ftz_f32(cumsum_log2).bfloat16()
    inv_decay = exp2_ftz_f32(-cumsum_log2).bfloat16()

    k_decayed = keys * decay[:, None]
    k_inverse = keys * inv_decay[:, None]
    l = (k_decayed.float() @ k_inverse.float().t()).half()
    l = torch.tril(l, diagonal=-1)
    l = (l * beta.half()[:, None]).half()
    return l, decay, inv_decay


def rel_rmse(actual, expected):
    if not torch.isfinite(actual).all():
        return math.inf
    error = (actual.double() - expected.double()).square().mean().sqrt()
    scale = expected.double().square().mean().sqrt()
    return float(error / (scale + 1e-30))


def run_one(case, chunk, dim, seed, device):
    keys, log_decay, beta = make_inputs(
        case, chunk, dim, seed, device
    )
    l_ref = stable_l(keys, log_decay, beta)
    identity = torch.eye(chunk, dtype=torch.float64, device=device)
    inv_ref = torch.linalg.inv(identity + l_ref)
    l_kernel, decay, inv_decay = kernel_l(keys, log_decay, beta)

    stable_inv, stable_power, stable_power_max = (
        neumann_inverse_fp16(l_ref.half())
    )
    kernel_inv, kernel_power, kernel_power_max = (
        neumann_inverse_fp16(l_kernel)
    )
    residual = rel_rmse(
        (identity + l_ref) @ kernel_inv.double(), identity
    )

    if torch.isfinite(l_kernel).all():
        inv_of_kernel_l = torch.linalg.inv(identity + l_kernel.double())
        neumann_only = rel_rmse(kernel_inv, inv_of_kernel_l)
    else:
        neumann_only = math.inf

    return {
        "case": case,
        "chunk": chunk,
        "seed": seed,
        "decay_zeros": int((decay == 0).sum()),
        "inv_decay_nonfinite": int((~torch.isfinite(inv_decay)).sum()),
        "l_finite": int(torch.isfinite(l_kernel).all()),
        "inv_finite": int(torch.isfinite(kernel_inv).all()),
        "l_rel_rmse": rel_rmse(l_kernel, l_ref),
        "inv_total_rel_rmse": rel_rmse(kernel_inv, inv_ref),
        "inv_neumann_only_rel_rmse": neumann_only,
        "inv_stable_l_rel_rmse": rel_rmse(stable_inv, inv_ref),
        "inverse_residual": residual,
        "highest_power": kernel_power,
        "kernel_power_max": kernel_power_max,
        "stable_l_highest_power": stable_power,
        "stable_l_power_max": stable_power_max,
    }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = run_one(
        case="floor_e-5",
        chunk=16,
        dim=128,
        seed=0,
        device=device,
    )
    print(result)
```

</details>

## References

- Kimi Team, [Kimi K3: Open Frontier Intelligence](https://arxiv.org/abs/2607.24653), 2026. See Section 2.1.1 and Figure 3 in particular.
- MoonshotAI, [FlashKDA](https://github.com/MoonshotAI/FlashKDA). See the chunk-size selection discussion, `fwd_kernel1.cuh`, and `utils.cuh` in particular.
