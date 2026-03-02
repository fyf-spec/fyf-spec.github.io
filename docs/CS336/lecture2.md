# CS 336 Lecture 2: PyTorch 与资源统计 (Resource Accounting)

## 目录

- [内存统计：数据类型 (Memory Counting: Datatypes)](#内存统计数据类型-memory-counting-datatypes)
- [计算统计：Einops 库 (Compute Counting: Einops)](#计算统计einops-库-compute-counting-einops)
  - [einsum：爱因斯坦求和约定](#einsum爱因斯坦求和约定)
  - [reduce：张量降维](#reduce张量降维)
  - [rearrange：维度变换与重塑](#rearrange维度变换与重塑)
- [FLOPs 统计 (FLOPs Counting)](#flops-统计-flops-counting)
  - [张量乘法的前向传播](#张量乘法的前向传播)
  - [计算时间与利用率 (MFU)](#计算时间与利用率-mfu)
  - [梯度反向传播 (Gradients FLOPs)](#梯度反向传播-gradients-flops)
- [训练资源管理与优化](#训练资源管理与优化)
  - [模型初始化 (Model Initialization)](#模型初始化-model-initialization)
  - [优化器内存占用 (Optimizer Memory)](#优化器内存占用-optimizer-memory)
  - [显存记账公式 (Memory Accounting)](#显存记账公式-memory-accounting)
  - [精度选择策略](#精度选择策略)

## 内存统计：数据类型 (Memory Counting: Datatypes)

在深度学习中，显存的占用很大程度上取决于所选用的数据类型（Precision）。

- **float32 (fp32 / 单精度)**
  - 占用 **4 Bytes**。
  - 结构：1 bit 符号位，8 bits 指数位，23 bits 尾数位。
  - 特点：深度学习中的最高精度标准，通常用于权重更新的核心计算。
- **float16 (fp16 / 半精度)**
  - 占用 **2 Bytes**。
  - 结构：1 bit 符号位，5 bits 指数位，10 bits 尾数位。
  - 特点：节省显存，但表示范围较窄，容易出现数值溢出（Overflow）。
- **bfloat16 (bf16 / 大脑半精度)**
  - 占用 **2 Bytes**。
  - 结构：1 bit 符号位，8 bits 指数位，7 bits 尾数位。
  - 特点：由 Google 提出，指数位与 fp32 一致，虽然精度稍低，但动态范围与 fp32 相同，训练稳定性远好于 fp16。
- **float8 (fp8)**
  - 占用 **1 Byte**。
  - 变体：**E4M3**（高精度，用于前向/反向传播）和 **E5M2**（宽范围，用于梯度/状态）。
- **混合精度训练 (Mixed Precision Training)**
  - 由于低精度训练可能带来不稳定性，通常采用前向传播使用低精度（如 bf16/fp8），而权重更新和某些核心累加使用高精度（fp32）的方案。

## 计算统计：Einops 库 (Compute Counting: Einops)

`einops`（Einstein Summation Notation）提供了一种简洁的方式来标注和操作 PyTorch 张量的维度信息。

### einsum：爱因斯坦求和约定

相比传统的转置与矩阵乘法，`einsum` 能够自动处理维度的求和与广播。

```
x: Float[torch.Tensor, "batch seq1 hidden"] = torch.ones(2,3,4)
y: Float[torch.Tensor, "batch seq2 hidden"] = torch.ones(2,3,4)

# 传统方式
z = x @ y.transpose(-2,-1) # [batch, seq1, seq2]

# einops 方式
# einsum 会对结果中没有提及的维度（hidden）自动求和
z = einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")

# 使用 ... 代表广播任意维
z = einsum(x, y, "... seq1 hidden, ... seq2 hidden -> ... seq1 seq2")
```

### reduce：张量降维

`reduce` 可以直接对指定维度进行聚合操作（如 mean, sum, max）。

```
x: Float[torch.Tensor, "batch seq hidden"] = torch.ones(2,3,4)

# 传统方式
y = x.mean(dim = -1)

# einops 方式
y = reduce(x, "... hidden -> ...", "mean")
```

### rearrange：维度变换与重塑

这是 einops 最强大的功能，可以直观地处理维度的拆分与合并。

```
x: Float[torch.Tensor, "batch seq1 total_hidden"] = torch.ones(2,3,8)
w: Float[torch.Tensor, "hidden1 hidden2"] = torch.ones(4,4)

# 将 total_hidden 拆分为多头形式
x = rearrange(x, "... (heads hidden1) -> ... heads hidden1", heads=2)

# 进行矩阵变换
x = einsum(x, w, "... hidden1, hidden1 hidden2 -> ... hidden2")

# 将多维合并回一维
x = rearrange(x, "... heads hidden1 -> ... (heads hidden1)")
```

## FLOPs 统计 (FLOPs Counting)

### 张量乘法的前向传播

对于矩阵乘法 $B \times D$ 和 $D \times K$，每一个结果元素都需要 $D$ 次乘法和 $D$ 次加法。 因此，实际浮点运算次数为：

$$\text{actual\_num\_flops} = 2 \times B \times D \times K$$

在神经网络（如 MLP）中，若 $B$ 代表 Token 数量，$(D, K)$ 代表参数矩阵大小，则一次前向传播的 FLOPs 约为：

$$\text{FLOPs (forward)} = 2 \times (\# \text{tokens}) \times (\# \text{parameters})$$

### 计算时间与利用率 (MFU)

$$\text{actual\_time} = \frac{\text{FLOPs\_needed}}{\text{FLOPS (Peak)}}$$

- **FLOPS**：每秒浮点运算次数，取决于硬件性能和数据类型。例如 H100 在 FP32 下仅为 $6.7 \times 10^{13}$。
- **MFU (Model FLOPs Utilization)**：模型 FLOPs 利用率。
  - 定义：$mfu = \frac{\text{actual\_flop\_per\_sec}}{\text{promised\_flop\_per\_sec}}$
  - 经验值：MFU $\ge 0.5$ 被视为非常优秀，通信开销和内存墙是主要的性能限制。

<img src="C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20260214161756866.png" alt="Hardware Performance Data" style="zoom:60%;" />

### 梯度反向传播 (Gradients FLOPs)

在反向传播中，我们需要计算：

- **权重梯度 (Weight Grad)**：计算复杂度约为 $2 \times (\# \text{tokens}) \times (\# \text{parameters})$。
- **激活梯度 (Activation Grad)**：计算复杂度约为 $2 \times (\# \text{tokens}) \times (\# \text{parameters})$。

因此，**反向传播的 FLOPs 是前向传播的 2 倍**：

$$\text{FLOPs (backward)} = 4 \times (\# \text{tokens}) \times (\# \text{parameters})$$

**训练总 FLOPs 统计**：

$$\text{Total FLOPs} = 6 \times (\# \text{tokens}) \times (\# \text{parameters})$$

## 训练资源管理与优化

### 模型初始化 (Model Initialization)

为了防止输出在多层叠加后数值爆炸或消失，通常使用 **Xavier 初始化**（或其他缩放方案）：

```
w = nn.Parameter(torch.randn(input_dim, hidden_dim) / np.sqrt(input_dim))
```

### 优化器内存占用 (Optimizer Memory)

在训练结束后，可以通过设置 `set_to_none=True` 来释放梯度占用的显存：

```
optimizer.zero_grad(set_to_none=True)
```

### 显存记账公式 (Memory Accounting)

总显存消耗取决于参数、激活值、梯度以及优化器状态的总和：

- $num\_parameters = D \times D \times num\_layers$
- $num\_activations = B \times D \times num\_layers$
- $num\_gradients = num\_parameters$
- $num\_optimizer\_states = num\_parameters$（例如 Adam 需要 2 份状态）

**总内存估算**：

$$\text{Total Memory} \approx 4\text{Bytes} \times (num\_parameters + num\_activations + num\_gradients + num\_optimizer\_states)$$

### 精度选择策略

- **前向传播 (Forward)**：建议使用 **{bf16, fp8}** 以减少显存和加速计算。
- **反向传播与权重更新 (Backward & Update)**：建议在关键累加环节使用 **{float32}** 以保证数值稳定性。
- **策略总结**：低精度训练难度较大（需要 Loss Scaling 等技巧），但推理时的量化（Quantization）相对容易。