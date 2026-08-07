---
title: "K3's KDA Precision Problem: From the Decay Lower Bound to Tensor Core Utilization"
description: "How split-exponent range, diagonal tiles, finite Neumann inverses, and an H200 benchmark converge on a 16-token tile."
date: 2026-08-07
lang: en-US
outline: deep
---

# K3's KDA Precision Problem: From the Decay Lower Bound to Tensor Core Utilization

One of Kimi K3's key changes to KDA looks almost trivial: it adds a lower bound to the gate, restricting a log-decay that could previously approach negative infinity to `(-5, 0)`. What this change really fixes, however, is not merely “numerical stability.” It allows diagonal tiles, which previously required explicit position-pair computation, to return to the dense matrix multiplications that Tensor Cores handle well.

This design is easy to summarize incorrectly. **K3 does not choose a tile size of 16 only to prevent a Neumann series from exploding, and it does not set the decay lower bound to -5 only to align with Tensor Cores.** A more accurate account is that K3 constrains two different low-precision failure modes at the same design point:

1. the dynamic range of `exp(G)` and `exp(-G)`;
2. the growth of intermediate powers `L², L⁴, L⁸, ...` during the triangular solve.

The first determines whether a diagonal tile can use dense Tensor Core operations. The second determines whether an FP16 Neumann expansion remains under control. This post combines the K3 technical report, the FlashKDA implementation, and a set of H200 mechanism experiments to explain why these two paths meet at `16`.

## 1. Where the Precision Problem Enters KDA

KDA stores history in a fixed-size state matrix. The token-by-token recurrence is easy to understand, but it cannot fully exploit GPU matrix-multiplication throughput. Training and prefill therefore group `C` consecutive tokens into a chunk.

The output inside a chunk can be written as

```text
A = Tril[(Γ ⊙ Q)(K / Γ)ᵀ]
O = (Γ ⊙ Q)S + A Ṽ
```

where

- `Q` and `K` are the query and key matrices inside one chunk;
- `S` is the historical state entering the current chunk;
- `Ṽ` is the delta-corrected pseudo-value;
- `Γ` is the retention accumulated from the beginning of the chunk to each position;
- `Tril` keeps the lower triangle and the diagonal, ensuring that output position `i` can only see input positions `≤ i`.

Let the per-step log-decay be `gᵢ < 0`. The cumulative log-decay is

```text
Gᵢ = g₁ + g₂ + … + gᵢ
Γᵢ = exp(Gᵢ)
```

The matrix multiplication uses an elegant factorization:

```text
(Γᵢ qᵢ) · (kⱼ / Γⱼ)
= qᵢ · kⱼ · exp(Gᵢ - Gⱼ)
```

When `i ≥ j`, we have `Gᵢ - Gⱼ ≤ 0`, so the relative decay that the mathematics actually requires, `exp(Gᵢ-Gⱼ)`, always lies in `(0,1]`. The mathematical result is safe. The danger comes from how it is implemented: to rewrite all position pairs as one GEMM, the kernel materializes `exp(Gᵢ)` and `exp(-Gⱼ)` separately.

One factor may underflow to `0` while the other overflows to `inf`. Even if their product should be a perfectly normal relative decay over the real numbers, finite-precision hardware cannot recover the correct result from `0 × inf`.

KDA's first precision conflict is therefore:

> Computing `exp(Gᵢ-Gⱼ)` directly is stable but requires position-pair work. Splitting it into `exp(Gᵢ)exp(-Gⱼ)` enables GEMM but may create 0 and inf.

## 2. Why Kimi Linear Stalls on Diagonal Tiles

The K3 report explicitly states that Kimi Linear uses an unbounded negative-Softplus log-decay:

```text
g = -exp(A) · Softplus(z),    g ∈ (-∞, 0)
```

Consequently, no matter how small the secondary tile is, there is no strict worst-case guarantee on the dynamic range. Kimi Linear handles this by

1. computing relative decay in log space;
2. dividing a larger chunk into secondary 16-token tiles;
3. using dense Tensor Core GEMMs for off-diagonal tiles;
4. using explicit position-pair computation for diagonal tiles.

Here, a diagonal tile does not mean the few scalar elements on the matrix diagonal. It means an entire `16 × 16` block on the main diagonal of the blocked matrix.

A four-block sketch makes the distinction clear:

| Output tile \ input tile | 0 | 1 | 2 | 3 |
|---|---:|---:|---:|---:|
| 0 | Diag | - | - | - |
| 1 | Dense | Diag | - | - |
| 2 | Dense | Dense | Diag | - |
| 3 | Dense | Dense | Dense | Diag |

Inside an off-diagonal causal tile, every query in the later tile may read every key in the earlier tile. The entire block contains valid work and maps naturally to dense GEMM. A diagonal tile still contains a fine-grained causal boundary: only position pairs satisfying `i ≥ j` are valid.

The fact stated explicitly by the authors is that Kimi Linear's diagonal tiles require an explicit position-pair path, that this path is the main intra-chunk bottleneck, and that off-diagonal tiles can run on Tensor Cores.

The formulas support a further inference about the mechanism. When `g` has no lower bound, the two split exponents cannot be materialized safely even inside a diagonal tile. Computing the stable expression `exp(Gᵢ-Gⱼ)` directly means forming a relative decay for every valid position pair and applying the causal condition at that granularity. This path contains more scalar exponentials, elementwise operations, and masking, and it cannot collapse the entire diagonal tile into one regular dense Tensor Core path.

This explains why the question “the tile is already 16, so why can it not just use Tensor Cores?” cannot be answered from shape alone. **Matrix dimensions that align with the hardware are necessary, but the relative decay must also admit a numerically representable two-sided factorization before dense GEMM is valid.**

## 3. K3 Turns the Decay Bound into a Hardware Contract

K3 changes the log-decay to a scaled sigmoid:

```text
g = g_min · Sigmoid(exp(A) · z)
g_min = -5
g ∈ (-5, 0)

α = exp(g)
α ∈ (exp(-5), 1)
```

The easiest point to confuse is that `-5` is the lower bound on log-decay, not the lower bound on the retention `α`. The actual per-step retention lower bound is

```text
exp(-5) ≈ 6.7 × 10⁻³
```

A channel can still forget more than 99% of the old state in one token, but its log-decay can no longer become arbitrarily negative.

For a 16-token tile, the worst-case cumulative range becomes

```text
-80 < G < 0

exp(G)  > exp(-80) ≈ 1.80 × 10⁻³⁵
exp(-G) < exp( 80) ≈ 5.54 × 10³⁴
```

BF16 and FP32 both use an 8-bit exponent, with a maximum finite value of approximately `3.39 × 10³⁸`. The current FlashKDA implementation also uses `ex2.approx.ftz.f32`, so results below the minimum normal value are flushed to zero. `exp(±80)` stays within both boundaries, which means both scaling factors remain finite normal values.

The K3 report draws a direct systems conclusion: the finite split-exponent range allows both diagonal and off-diagonal causal tiles to use dense Tensor Core matrix multiplication, eliminating the position-pair diagonal path.

The decay lower bound is therefore not a conventional training regularizer. It can be understood as a numerical contract between the model parameterization and the kernel:

```text
Model guarantee: |cumulative log-decay| < 80 over 16 tokens
Kernel consequence: exp(G) and exp(-G) can be materialized safely
System payoff: every causal tile can use dense Tensor Core operations
```

Removing the lower bound would not make the model mathematically undefined, but the kernel would need to restore a log-domain, position-pair, or segmented-rescaling path. Without one of those alternatives, it would have no worst-case precision guarantee.

## 4. The H200 Experiment Verifies the Exponent Boundary

To separate split-exponent failures from Neumann-inverse failures, the experiment first constructs a `floor_e-5` input: every position uses the strongest allowed decay, `g=-5`, and the key arithmetic path of FlashKDA K1 is generalized to different chunk lengths.

The experiment performs the following steps:

1. convert `exp(G)` and `exp(-G)` to BF16;
2. construct `L` with BF16 operands and FP32 accumulation;
3. store `L` in FP16;
4. emulate the Neumann inverse on an H200 with FP16 accumulation;
5. construct `L` and compute the inverse directly in FP64 as the gold result.

Each configuration uses three seeds. The table reports median errors and the worst number of non-finite scales.

### Key Benchmark Implementation

The functions below are not simplified pseudocode. They are the functions that determine the numerical path in the benchmark. The shared dependencies and constants are listed first so that the code blocks can be read independently:

```python
import ctypes
import math
from pathlib import Path

import torch


LOG2E = math.log2(math.e)
_CUBLAS = None
_CUBLAS_HANDLE = ctypes.c_void_p()
_FP16_ONE = ctypes.c_ushort(0x3C00)
_FP16_ZERO = ctypes.c_ushort(0x0000)
```

The first function constructs the three input families. `floor_e-5` fixes every step at the worst-case log-decay bound. `model_like` uses random gates and random normalized keys. `weak_decay_correlated` fixes identical keys and weak decay to expose the risk from intermediate Neumann powers.

```python
def make_inputs(case, chunk, dim, seed, device):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if case == "weak_decay_correlated":
        key = torch.randn(1, dim, dtype=torch.float64, generator=generator)
        key = key / key.norm(dim=-1, keepdim=True)
        keys = key.expand(chunk, -1)
        log_decay = torch.full((chunk,), -0.01, dtype=torch.float64)
        beta = torch.full((chunk,), 0.5, dtype=torch.float64)
    else:
        keys = torch.randn(chunk, dim, dtype=torch.float64, generator=generator)
        keys = keys / keys.norm(dim=-1, keepdim=True)
        if case == "floor_e-5":
            log_decay = torch.full((chunk,), -5.0, dtype=torch.float64)
        else:
            log_decay = -5.0 * torch.sigmoid(
                torch.randn(chunk, dtype=torch.float64, generator=generator)
            )
        beta = torch.sigmoid(torch.randn(chunk, dtype=torch.float64, generator=generator))

    return keys.to(device=device, dtype=torch.bfloat16), log_decay.to(device), beta.to(device)
```

The next functions construct `L`. `stable_l` forms the safe relative decay `exp(Gᵢ-Gⱼ)` directly in FP64. `kernel_l` deliberately reproduces the kernel's split-exponent path and materializes both sides in BF16. `exp2_ftz_f32` also models the flush-to-zero behavior for results below the FP32 normal range.

```python
def exp2_ftz_f32(x):
    y = torch.exp2(x.float())
    return torch.where(y < torch.finfo(torch.float32).tiny, 0.0, y)


def stable_l(keys, log_decay, beta):
    keys64 = keys.double()
    cumsum = log_decay.cumsum(0)
    dots = keys64 @ keys64.t()
    decay_ratio = torch.exp((cumsum[:, None] - cumsum[None, :]).clamp(max=0.0))
    return torch.tril(dots * decay_ratio * beta[:, None], diagonal=-1)


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
```

This comparison is essential. If the `stable_l` path remains normal while `kernel_l` becomes non-finite, the failure comes from the split exponents rather than from the relative decay required by KDA itself.

The third group reproduces FP16-accumulation matrix multiplication and the finite Neumann doubling chain. On CUDA, it calls cuBLAS with `CUBLAS_COMPUTE_16F`; on the CPU, it rounds the result back to FP16 after every multiply-add. The next section explains why the loop forms only `L², L⁴, L⁸, ...`.

```python
def cublas_fp16_accum_mm(a, b):
    """CUDA fp16-accumulator GEMM, matching tests/torch_ref.py."""
    global _CUBLAS
    if _CUBLAS is None:
        try:
            _CUBLAS = ctypes.CDLL("libcublas.so")
        except OSError:
            import nvidia.cublas
            candidates = sorted((Path(nvidia.cublas.__file__).parent / "lib").glob("libcublas.so.*"))
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
        0, 0,  # CUBLAS_OP_N
        ctypes.c_int(n), ctypes.c_int(m), ctypes.c_int(k),
        ctypes.byref(_FP16_ONE),
        ctypes.c_void_p(b.data_ptr()), ctypes.c_int(2), ctypes.c_int(n),
        ctypes.c_void_p(a.data_ptr()), ctypes.c_int(2), ctypes.c_int(k),
        ctypes.byref(_FP16_ZERO),
        ctypes.c_void_p(out.data_ptr()), ctypes.c_int(2), ctypes.c_int(n),
        ctypes.c_int(64),  # CUBLAS_COMPUTE_16F
        ctypes.c_int(0),   # CUBLAS_GEMM_DEFAULT
    )
    assert status == 0, f"cublasGemmEx failed: {status}"
    torch.cuda.synchronize(a.device)
    return out


def fp16_accum_mm(a, b):
    """Use real CUDA fp16 accumulation, with a deterministic CPU fallback."""
    if a.is_cuda:
        return cublas_fp16_accum_mm(a, b)
    out = torch.zeros(a.shape[0], b.shape[1], dtype=torch.float16, device=a.device)
    for k in range(a.shape[1]):
        out = (out.float() + a[:, k:k + 1].float() * b[k:k + 1].float()).half()
    return out


def neumann_inverse_fp16(l):
    """Kernel factorization: (I-L)(I+L^2)(I+L^4)... in fp16."""
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
```

The final two functions define the error metric and orchestrate one complete run. `inv_total_rel_rmse` includes all error from split exponents, construction of `L`, and the Neumann inverse. `inv_stable_l_rel_rmse` starts from a stable `L` and observes only the low-precision Neumann path. Their difference is what lets the experiment isolate the two precision problems.

```python
def rel_rmse(actual, expected):
    if not torch.isfinite(actual).all():
        return math.inf
    error = (actual.double() - expected.double()).square().mean().sqrt()
    scale = expected.double().square().mean().sqrt()
    return float(error / (scale + 1e-30))


def run_one(case, chunk, dim, seed, device):
    keys, log_decay, beta = make_inputs(case, chunk, dim, seed, device)
    l_ref = stable_l(keys, log_decay, beta)
    inv_ref = torch.linalg.inv(torch.eye(chunk, dtype=torch.float64, device=device) + l_ref)
    l_kernel, decay, inv_decay = kernel_l(keys, log_decay, beta)

    stable_inv, stable_power, stable_power_max = neumann_inverse_fp16(l_ref.half())
    kernel_inv, kernel_power, kernel_power_max = neumann_inverse_fp16(l_kernel)
    identity = torch.eye(chunk, dtype=torch.float64, device=device)
    residual = rel_rmse((identity + l_ref) @ kernel_inv.double(), identity)

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
```

The functions `summarize`, `print_summary`, `write_csv`, and `main` are omitted because they only aggregate seeds, format output, and dispatch command-line arguments. They do not change the numerical path of any individual run. The complete executable version remains in the original `chunk_size_accuracy.py`.

| Chunk C | Inverse finite | Decay values equal to 0 | Reciprocal values equal to inf | INV total | INV power |
|---:|---:|---:|---:|---:|---:|
| 8 | 100% | 0 | 0 | `8.80e-7` | `5.16e-8` |
| 16 | 100% | 0 | 0 | `1.23e-6` | `8.51e-8` |
| 17 | 100% | 0 | 0 | `1.07e-6` | `6.69e-8` |
| 18 | 100% | 1 | 1 | `7.86e-6` | `6.62e-8` |
| 19 | 0% | 2 | 2 | `inf` | `6.60e-8` |
| 32 | 0% | 15 | 15 | `inf` | `6.51e-8` |

The results match the dynamic-range calculation:

- at `C=17`, `exp(±85)` is still representable;
- at `C=18`, one `0` and one `inf` appear for the first time;
- at `C=19`, `0 × inf` enters the valid strict lower triangle and contaminates `L`;
- at `C=32`, many positions are already outside the representable range.

Why does the final inverse remain finite at `C=18` even though 0 and inf have appeared? The first `0 × inf` pair lands exactly on the matrix diagonal and is later removed by the strict causal mask. At `C=19`, a new zero appears in a later row while an earlier infinity remains in an earlier column. Their product falls inside the valid strict lower triangle and can no longer be masked away.

The more important evidence is that `INV power` remains near `10⁻⁸` from `C=16` through `C=32`. The failure therefore occurs before or during the construction of `L`, not in the high powers of the Neumann expansion.

This experiment directly supports K3's lower-bound argument: **the bound first guarantees that the inputs to split-exponent GEMM are finite, which in turn makes the diagonal tile's Tensor Core path possible.**

## 5. The Neumann Expansion Is a Separate Risk

The delta update inside a KDA chunk forms a strictly lower-triangular matrix `L` and requires

```text
INV = (I + L)⁻¹
```

Because `L` is a `C × C` strictly lower-triangular matrix,

```text
Lᶜ = 0
```

The Neumann series here is therefore not an infinite series that relies on a spectral radius below one. It is a finite expansion that must terminate:

```text
(I + L)⁻¹ = I - L + L² - L³ + … + (-L)ᶜ⁻¹
```

FlashKDA does not add the terms one by one. It uses a doubling factorization. For `C=16`,

```text
(I + L)⁻¹ = (I - L)(I + L²)(I + L⁴)(I + L⁸)
```

The kernel needs to compute only `L²`, `L⁴`, and `L⁸`. `L¹⁶=0` proves that the final remainder vanishes; `L¹⁶` itself is never materialized.

This corrects a common misconception: **chunk 16 does not “compute up to L¹⁶ and then explode.” It is exactly the size at which L¹⁶ does not need to be computed.** Once `C>16`, covering the higher-order dependencies in the strict lower triangle requires an additional `(I+L¹⁶)` factor. Once `C>32`, the kernel must continue to `L³²`.

Although `L` is eventually nilpotent, its lower powers are not guaranteed to decrease monotonically. A non-normal matrix can exhibit transient power growth: `L²`, `L⁴`, and `L⁸` may grow before a sufficiently high power becomes zero. FP16 Tensor Cores must represent these intermediate results. A bounded final inverse therefore does not guarantee a safe computation path.

## 6. A Stress Test Isolates High-Power Growth

To remove exponent underflow and overflow from the picture, the experiment constructs `weak_decay_correlated`:

- every key is identical and normalized, creating highly correlated update directions;
- `g=-0.01`, keeping every `exp(±G)` finite;
- `β=0.5`, preventing multi-step propagation from disappearing too quickly.

This is not a typical model input. It is a mechanism stress test designed specifically to activate growth in the high powers of `L`.

| Chunk C | Highest materialized power | Max power magnitude | INV total | INV power | Finite |
|---:|---:|---:|---:|---:|---:|
| 8 | `L⁴` | `1.40e0` | `1.51e-3` | `2.13e-4` | 100% |
| 16 | `L⁸` | `1.96e1` | `3.27e-3` | `3.93e-3` | 100% |
| 17 | `L¹⁶` | `2.43e1` | `6.75e-3` | `5.64e-3` | 100% |
| 24 | `L¹⁶` | `5.32e2` | `7.55e-2` | `5.94e-2` | 100% |
| 32 | `L¹⁶` | `5.87e3` | `1.33e0` | `1.21e0` | 100% |
| 48 | `L³²` | `inf` | `inf` | `inf` | 0% |

This curve is completely different from the previous one:

- at `C=16`, the highest materialized power is `L⁸`, whose largest intermediate element is about `19.6`;
- at `C=17`, the path forms `L¹⁶` for the first time, and the error increases without immediately exploding;
- at `C=32`, `L¹⁶` reaches approximately `5.87×10³`, while isolated-inverse RMSE reaches `1.21`;
- at `C=48`, the path must form `L³²` and overflows in FP16.

The claim “forming `L¹⁶` necessarily fails” would still be too strong. `L¹⁶` introduces a new risk, but the actual error depends on key correlation, `β`, decay strength, and the non-normal structure of `L`.

This also shows that the decay lower bound cannot solve every precision problem by itself. It guarantees finite split exponents, but it does not guarantee that powers in the Neumann expansion will not grow temporarily. Another role of `C=16` is to limit the highest power formed by the current implementation to `L⁸`.

## 7. Random Inputs Do Not Degrade Monotonically with Chunk Size

The stress test reveals a mechanism boundary, but it should not be mistaken for a typical model distribution. The `model_like` random input provides an important counterexample:

| Chunk C | Finite | INV total | INV power | Max power magnitude |
|---:|---:|---:|---:|---:|
| 8 | 100% | `1.78e-5` | `1.46e-6` | `4.50e-2` |
| 16 | 100% | `3.01e-5` | `1.51e-6` | `2.82e-2` |
| 32 | 100% | `2.61e-5` | `1.34e-6` | `7.84e-2` |
| 48 | 0% | `inf` | `1.93e-6` | `7.46e-2` |

The end-to-end errors at `C=16` and `C=32` are both on the order of `10⁻⁵`; they do not increase monotonically with chunk size. The failure at `C=48` comes from the randomly accumulated gate eventually exceeding the exponent range. Once `L` is constructed stably, `INV power` remains only `1.93×10⁻⁶`.

This experiment supports a conservative worst-case design, not the claim that every `C=32` input loses precision. K3 and FlashKDA choose 16 to cover strong decay, highly correlated keys, a low-precision inverse, and hardware layout simultaneously—not because every real input is expected to reach the stress test's extreme conditions.

## 8. Why the Design Settles on 16

Looking only at exponent range, `C=17` is still safe and `C=18` is where values first leave the range. The BF16 maximum alone therefore does not uniquely derive 16.

Six boundaries align to make 16 the natural design point:

| Constraint | What happens at `16` | Nature of the evidence |
|---|---|---|
| Split exponent | Both `exp(±80)` are finite normal values | K3 report and H200 experiment |
| Diagonal tile | Both scales can be materialized safely, enabling dense GEMM | Explicitly stated in the K3 report |
| Neumann inverse | `L¹⁶=0`; the implementation forms powers only through `L⁸` | Strictly triangular algebra and code |
| Power of two | 16 is the largest power of two before the exponent boundary | Numerical derivation |
| MMA mapping | A `16×16` triangular system matches one base matrix tile | FlashKDA implementation |
| Storage cost | `L`, `INV`, and `Mqk` all grow with `C²` | Workspace layout |

The current FlashKDA implementation uses a `16×8×16` MMA atom, and the complete `16×16` `L`/`INV` can be handled by one warp-level path. At `C=64`, a `64×64` matrix must be divided into sixteen `16×16` sub-blocks, forcing a redesign of the inverse, register fragments, shared memory, and synchronization.

The per-tile workspace in the code is approximately

```text
workspace(C) = 3·C·D·2 + D·4 + 2·C²·2 bytes
```

For `D=128`,

```text
C=16: 13,824 bytes ≈ 13.5 KiB
C=64: 66,048 bytes ≈ 64.5 KiB
```

K2 also uses a three-stage input pipeline while keeping a `128×128` state resident. Increasing `C` quickly raises shared-memory pressure per CTA and can reduce occupancy. This is an engineering inference supported by the implementation structure, not an independent performance ablation reported by K3.

Two levels must also be kept separate. The K3 report discusses a 16-token secondary tile inside a larger chunk. FlashKDA v1 directly fixes `CHUNK=16`, so one kernel chunk happens to equal one numerical secondary tile. They have the same size in the current implementation, but they are not the same concept.

## 9. Evidence Boundaries

The H200 results come from a generalized arithmetic experiment, not from a collection of real, variable-chunk FlashKDA kernels benchmarked end to end. I checked every provided function in `chunk_size_accuracy.py` and verified that the code blocks above match the benchmark's key numerical path. I did not perform a new H200 run or independently inspect the original per-seed CSV.

The experiment reproduces the important rounding points:

- BF16 split exponents;
- BF16 operands with FP32 accumulation when constructing `L`;
- FP16-accumulation doubling for the inverse;
- an FP64 gold inverse.

It does not reproduce every reduction order, register layout, workspace data path, or K2 recurrence in the official one-warp kernel. It also does not measure the actual throughput gain from moving the diagonal tile to Tensor Cores. The conclusions should therefore be separated as follows:

- **The authors state:** lower-bounded decay removes the position-pair diagonal path and lets all causal tiles use dense Tensor Core matrix multiplication.
- **The experiment supports:** `g_min=-5` combined with a 16-token tile keeps the split exponents in BF16 range, with the predicted 0/inf failure appearing near 18–19 tokens.
- **The experiment supports:** even when every exponent is finite, a larger chunk can still amplify FP16 inverse error through transient power growth in `L`.
- **The evidence does not establish:** that every real model input loses precision for `C>16`, or that the lower bound by itself produces a specific end-to-end speedup.

## 10. Conclusion

K3's precision design can be compressed into two complementary constraints:

```text
decay lower bound
  → bounds the dynamic range of exp(G) and exp(-G)
  → permits split-exponent dense GEMM on diagonal tiles
  → increases Tensor Core coverage

16-token tile
  → bounds worst-case cumulative log-decay to (-80, 0)
  → makes a 16th-order strictly lower-triangular matrix satisfy L¹⁶=0
  → requires materializing only L², L⁴, and L⁸ for the inverse
  → aligns with a 16×16 MMA tile
```

`g_min=-5` and `tile=16` are therefore not two unrelated magic numbers. Together they form an algorithm–numerics–hardware co-design point: the lower bound makes the matrix factorization legal in low precision, while 16 lets that guarantee cover one complete Tensor Core tile and keeps intermediate powers in the triangular inverse under control.

Using a larger chunk in the future would require more than changing one constant. At minimum, it would require redesigning

1. exponent rebasing or segmented rescaling inside the chunk;
2. higher-order Neumann expansion or another stable triangular solve;
3. lower-triangular dependency propagation across multiple MMA tiles;
4. shared-memory, register, and workspace layouts;
5. joint numerical-accuracy and end-to-end-throughput benchmarks.

Sixteen is not a mathematical constant independent of the implementation. It is the engineering balance defined jointly by the current gate lower bound, BF16 exponent range, FP16 inverse, and Tensor Core tile shape.

## References

- Kimi Team, [Kimi K3: Open Frontier Intelligence](https://arxiv.org/abs/2607.24653), 2026. See Section 2.1.1 and Figure 3 in particular.
- MoonshotAI, [FlashKDA](https://github.com/MoonshotAI/FlashKDA). See the chunk-size selection discussion, `fwd_kernel1.cuh`, and `utils.cuh`.
- The H200 numerical experiment in this post comes from `why-flashkda-chunk-16-exploration.md` and `chunk_size_accuracy.py`, dated 2026-08-05. The benchmark source has been checked, but the original per-seed CSV and execution log were not provided and the H200 experiment was not rerun. The results are therefore not presented as an independent reproduction or an end-to-end kernel benchmark.
