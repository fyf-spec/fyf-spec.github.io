# Lecture 7 Parallelize basics

## 目录

- [网络与集合通信基础](#网络与集合通信基础)
  - [单 GPU 的局限性](#单-gpu-的局限性)
  - [集合通信原语 (Collective Communication)](#集合通信原语-collective-communication)
- [数据并行 (Data Parallelism)](#数据并行-data-parallelism)
  - [朴素数据并行 (Naïve DP)](#朴素数据并行-naïve-dp)
  - [ZeRO 系列：消除内存冗余](#zero-系列消除内存冗余)
  - [ZeRO 各阶段内存记账](#zero-各阶段内存记账)
- [模型并行 (Model Parallelism)](#模型并行-model-parallelism)
  - [流水线并行 (Pipeline Parallel, PP)](#流水线并行-pipeline-parallel-pp)
  - [张量并行 (Tensor Parallel, TP)](#张量并行-tensor-parallel-tp)
  - [序列并行 (Sequence Parallel, SP)](#序列并行-sequence-parallel-sp)
- [激活显存与重计算](#激活显存与重计算)
- [3D 并行策略与实战](#3d-并行策略与实战)
  - [训练方案选择逻辑](#训练方案选择逻辑)
  - [Llama 3 405B 的实战配置 (16384 GPUs)](#llama-3-405b-的实战配置-16384-gpus)
  - [总结：资源平衡](#总结资源平衡)

## 网络与集合通信基础

### 单 GPU 的局限性

随着模型参数量的指数级增长，单块 GPU 面临两大瓶颈：

1. **计算瓶颈 (Compute Limit)**：虽然硬件算力（FLOPS）在增长，但单卡处理 Exaflops 级训练仍需集群协作。
2. **内存瓶颈 (Memory Limit)**：GPT-3 (175B) 需要约 350GB 显存仅用于存储参数（fp16），远超单块 A100 (80GB) 的容量。

**核心目标**：实现 **线性显存缩放**（模型规模随 GPU 增加而线性增加）和 **线性计算缩放**（吞吐量线性增加）。

### 集合通信原语 (Collective Communication)

理解并行方案前，需掌握 GPU 间交换数据的基本操作：

| 原语               | 描述                                                 | 备注                         |
| ------------------ | ---------------------------------------------------- | ---------------------------- |
| **Broadcast**      | Root Rank 将数据发送给所有 Rank                      | 用于权重初始化同步           |
| **Reduce**         | 将所有 Rank 的数据按操作（如 SUM）汇总到某一个 Rank  | -                            |
| **All-Reduce**     | 归约汇总结果并分发给所有 Rank                        | **DDP 的核心**               |
| **All-Gather**     | 每个 Rank 贡献一部分，最终所有 Rank 拥有完整拼接结果 | **ZeRO-3/FSDP 前向传播核心** |
| **Reduce-Scatter** | 归约结果并按 Rank 拆分，每个 Rank 拿走一部分         | **ZeRO-2/3 梯度同步核心**    |

> [!IMPORTANT] 
> **带宽受限下的最优等价关系**：
> $$\text{All-Reduce} = \text{Reduce-Scatter} + \text{All-Gather}$$
> 这意味着在带宽有限的情况下，直接进行 All-Reduce 与先散播归约再收集的效果在效率上是等同的。

## 数据并行 (Data Parallelism)

### 朴素数据并行 (Naïve DP)

- **逻辑**：每块 GPU 存储全量模型参数 $\theta$，仅将 Batch $B$ 拆分为 $M$ 份。
- **同步**：计算完梯度 $\nabla f$ 后，通过 All-Reduce 同步梯度。
- **内存记账**：对于参数量为 $\Psi$ 的模型，显存开销极大：
  - 2 Bytes (fp16/bf16 参数) + 2 Bytes (梯度) + 12 Bytes (Adam 状态：fp32 权重 + 一阶矩 + 二阶矩) = **16 Bytes/参数**。

> [!WARNING] 
> **Naïve DP 的死穴**：内存冗余。每块 GPU 都存了相同的 16 份状态，浪费了集群的总内存潜力。

### ZeRO 系列：消除内存冗余

ZeRO (Zero Redundancy Optimizer) 的核心思想是 **分片 (Sharding)**。

- **ZeRO-1 (**$P_{os}$**)**：仅对 **优化器状态** (Optimizer States) 进行分片。每块 GPU 只更新自己那部分参数。
- **ZeRO-2 (**$P_{os+g}$**)**：在 Stage 1 基础上，对 **梯度** (Gradients) 也进行分片。计算完某层梯度后立即 Reduce-Scatter 并释放冗余。
- **ZeRO-3 / FSDP (**$P_{os+g+p}$**)**：对 **参数** (Parameters) 也进行分片。前向传播时，按需 All-Gather 获取参数，计算完立即释放。

### ZeRO 各阶段内存记账

假设模型参数量 $\Psi = 7.5B$，使用 $N_d = 64$ 块 GPU：

| 阶段         | 内存占用公式                        | 估算显存 (GB) | 通信成本 (相对量)          |
| ------------ | ----------------------------------- | ------------- | -------------------------- |
| **Baseline** | $(2+2+K)\Psi$                       | 120GB         | $2 \times$ #params         |
| **ZeRO-1**   | $2\Psi + 2\Psi + \frac{K\Psi}{N_d}$ | 31.4GB        | $2 \times$ #params (Free!) |
| **ZeRO-2**   | $2\Psi + \frac{(2+K)\Psi}{N_d}$     | 16.6GB        | $2 \times$ #params         |
| **ZeRO-3**   | $\frac{(2+2+K)\Psi}{N_d}$           | **1.9GB**     | **3** $\times$ **#params** |

> [!IMPORTANT] 
> **ZeRO-1 是“免费”的**：它的通信量与朴素 DP 完全相同，但显存显著降低。因此实际训练中应默认开启。

## 模型并行 (Model Parallelism)

当模型单层都塞不下单卡，或需要极致降低单卡激活显存时，需要切分模型。

### 流水线并行 (Pipeline Parallel, PP)

- **逻辑**：按层（Depth-wise）切分模型。GPU 0 处理层 0-7，GPU 1 处理层 8-15。
- **气泡问题 (The Bubble)**：
  - 朴素实现中，GPU 在等待其他层完成前向/反向传播时处于闲置状态。
  - **优化策略**：使用 **Micro-batches** (Gpipe) 和 **1F1B Schedule**。
- **通信**：仅传输层间的激活值（Activations），适合跨节点（Inter-node）带宽较窄的情况。

### 张量并行 (Tensor Parallel, TP)

- **逻辑**：在层内（Width-wise）切分矩阵乘法（如 MLP 或 Attention 的线性层）。
- **实现**：
  - $A$ 矩阵按列切分为 $[A_1, A_2]$，分别存在两块 GPU 上。
  - $B$ 矩阵按行切分为 $[B_1; B_2]$。
- **通信**：每次矩阵乘法的前向和反向都需要 All-Reduce。
- **限制**：通信频率极高，通常限制在 **节点内 (Intra-node)** 使用高速互联（NVLink）。

### 序列并行 (Sequence Parallel, SP)

- **发现**：LayerNorm 和 Dropout 是按 Token 进行的点对点操作，不需要全量序列信息。
- **方法**：将原本无法被 TP 缩减的 10sbh 激活显存项，通过在序列轴上切分并使用 Reduce-Scatter/All-Gather 来线性降低。

## 激活显存与重计算

显存消耗并非静态，**激活值 (Activations)** 往往是训练大 Batch 或长序列时的瓶颈。

| 配置                       | 每一层 Transformer 的激活显存         |
| -------------------------- | ------------------------------------- |
| **无并行**                 | $sbh(34 + 5 \frac{as}{h})$            |
| **TP + SP**                | $sbh(\frac{34}{t} + 5 \frac{as}{ht})$ |
| **TP + SP + 选择性重计算** | $sbh(\frac{34}{t})$                   |

> [!TIP] 
> **选择性重计算 (Selective Recomputation)**：只重计算 Attention 的计算密集型部分，可以以极小的计算开销（~10%）换取巨大的显存空间，从而支持更大的 Batch Size。

## 3D 并行策略与实战

在万卡集群上，通常将 DP、PP、TP 三者结合，即 **3D Parallelism**。

### 训练方案选择逻辑

```
flowchart TD
    A[训练大规模模型] --> B{模型能塞入单卡显存?}
    B -- Yes --> C[使用 ZeRO-1/DP 扩展算力]
    B -- No --> D[使用 TP 切分算力, 上限 8 GPU]
    D --> E{TP 后能塞下吗?}
    E -- No --> F[使用 PP 跨机切分层]
    E -- Yes --> G[剩余 GPU 用于 DP/ZeRO]
    F --> G
```

### Llama 3 405B 的实战配置 (16384 GPUs)

| 参数                       | TP (张量) | PP (流水线) | DP / FSDP | 序列长度 | 吞吐量/GPU |
| -------------------------- | --------- | ----------- | --------- | -------- | ---------- |
| **Stage 1 (Pretrain)**     | 8         | 16          | 64        | 8,192    | 430 TFLOPs |
| **Stage 2 (Long Context)** | 8         | 16          | 128       | 131,072  | 380 TFLOPs |

**Llama 3 并行顺序优化**：按照 `[TP, CP, PP, DP]` 排序。TP 在最内层（最高带宽需求），DP 在最外层（最能容忍网络延迟）。

### 总结：资源平衡

- **内存受限**：优先用 ZeRO-3 或 PP。
- **带宽受限**：优先用 PP，避免在跨机链路用 TP。
- **Batch Size 小**：PP 会带来严重的气泡，应增加梯度累加。

