# Assignment 1: BPE Tokenizer & Transformer Basics

## Part 1: BPE Tokenizer

### 1.1 为什么需要 BPE？

Unicode 建立了从 character 到 code points (integers) 的映射，但是不能直接拿来训练模型，因为 Unicode 包含大概 150K 个 integer-character 映射，而大部分 character 在模型的训练中是 rare 的。因此直接使用 Unicode 编码的缺点是**稀疏性 (sparsity)** 和**词表过大 (large vocabulary)**。

**Byte-level tokenization**
UTF-8, UTF-16, UTF-32 将 Unicode character 编码成字节（Bytes），范围缩减到 0-255，减少了词表大小，这被称为 byte-level tokenization。

**Subword-level tokenization (BPE)**
按字节 tokenization 的优势是词表范围可控，但是随之而来的问题是产生的 token 序列过长。例如对于汉字，可能一个 unicode character 要转化成 2/3 字节，UTF-32 编码的话更是所有 character 都需要 4 字节，这使得 tokenize 方法的压缩比大幅降低。

> 如果 byte sequence `b'the'` 经常出现在我们的原始训练数据中，为其在词表中分配一个条目（entry）将把这个 3-token 序列减少为一个 single token。

对于基于 UTF-8 编码的 token 序列，词表范围在 0-255，BPE 方法会找到出现频率最高的 token 对，用 unused 的 index 来表示新的对。BPE 能够帮忙找到常见的单词（如 the），前后缀（如 un-，pre-）甚至是词组，对于识别句子的语义特征很有帮助。构造 BPE tokenizer 词表的过程被称为 "training" the BPE tokenizer。

### 1.2 BPE 训练流程

1. **Vocabulary Initialization**: 使用 UTF-8 字节，初始化 0-255 基础词汇。
2. **Pre-tokenization**: 对原始语料进行初步切分。
3. **Compute BPE Merges**: 统计相邻 token 对频率，合并最高频的拍对。

*Computational expensive?* Naively，对于每次合并，我们都要重新过一遍 corpus 去统计 token 对的出现次数。但在实际实现中，通常会维护一个 pair frequencies 的字典来加速这一过程。

### 1.3 Pre-tokenization 的作用与正则解析

如果不加限制地直接合并，如 `"dog"` 合并为 `"dog!"`，两者在语义上可能完全相近，但是却使用了完全不同的 token，这可能带来一些问题。

针对上面的问题，我们进行 pre-tokenization：在计算频率前先按照空格、标点等切分片段。这样，当我们计算字符 't' 和 'e' 相邻频率时，我们看到单词 'text' 有 't' 和 'e' 相邻，可以直接将该对计数加 10，而不需要遍历整个 corpus。

**作用： 保护了语义边界，防止非法合并；预计算了一些 token 对出现的次数，只需要在 pre-tokenize 出来的结果里加即可。**

```python
# GPT-2 的经典 Pre-tokenization 正则
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

| **正则片段** | **语法逻辑** | **匹配目标示例** |
|--------------|--------------|------------------|
| **`'(?:[sdmt]\|ll\|ve\|re)`** | 匹配 `'` 加上特定的英文缩写后缀。 | `'s`, `'ll`, `'re` |
| **` ?\p{L}+`** | ` ?` 表示**可选的空格**。后面接一个或多个**字母**。 | ` hello`, `世界` |
| **` ?\p{N}+`** | 可选空格 + 一个或多个**数字**。 | ` 2024`, ` 100` |
| **` ?[^\s\p{L}\p{N}]+`** | 可选空格 + 一个或多个**符号**。 | ` !!!`, ` +++` |
| **`\s+(?!\S)`** | 匹配空格，前提是后面**没有**非空格内容。 | 段落末尾的空格 |
| **`\s+`** | 匹配一个或多个空格。 | 单词间的普通空格 |

```python
>>> import regex as re
>>> re.findall(PAT, "some text that i'll pre-tokenize")
['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
```

### 1.4 特殊 Token (Special Tokens)

Special tokens 像 `<|endoftext|>` 应该**永远不被 split**。

```text
Example:
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
and the vocabulary has a special token <|endoftext|>

## Vocabulary    
Initialize vocabulary with special token <|endoftext|> and 256 byte values.

## Pre-tokenization
Assume in this simple example that pre-tokenization simply splits on whitespace(\s)
{low: 5, lower: 2, widest: 3, newest: 6}

这可以表示为 dict[tuple[bytes], int], 例如 {(l,o,w): 5}。注意即使是单个字节在 Python 中也是 bytes object。

## Merges
第一轮我们得到 {lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 9, st: 9, ne: 6, ew: 6}。
'es' 和 'st' tied，我们取字典序更大的 'st'。最终学到的 merge 序列可能是：
['s t', 'e st', 'o w', 'l ow', 'w est', 'n e']
```

### 1.5 BPE 实现细节 (Python 技巧)

> [!TIP]
> 记录一些在实现 BPE 过程中非常有用的 Python 内置函数和库方法：

```python
re.escape(special_token) # 用来将字符串中的 [、]、|、\ 等转义
re.split(pattern, chunk) # 将 chunk 按照 pattern 中的正则匹配的方式 split，返回 list[str]
re.finditer(pattern, chunk) # 返回 list[Match]。Match 可调用 .group(), .start(), .end()
re.findall(pattern, chunk) # 返回 list[str]

# 字典的合并
pre_token_counts: dict[tuple, int] = Counter()
for cc in chunk_counts:
    pre_token_counts.update(cc)
       
# max 中 key 的高级用法 (处理 Tie-breaking)
max(pair_counts, key=lambda p: (pair_counts[p], p)) 
# key 用一个二元 tuple 表示：将 pair_counts 值作为第一比较优先级，p 本身的字典序作为第二优先级

dict.pop(key, default) # 取出并删除，如果没有 key 返回 default
```

### 1.6 大文件内存优化 (Memory Mapping)

What if the dataset is too big to load into memory? 

We can use a Unix systemcall named **mmap** which maps a file on disk to virtual memory, and lazily loads the file contents when that memory location is accessed. Thus, you can “pretend” you have the entire dataset in memory. 

如果在处理超大语料（如进行 BPE 编码时）：
- Numpy 实现了这个机制：`np.memmap` 或在 `np.load` 时加上标志 `mmap_mode='r'`。
- 它返回一个 numpy array-like 对象，会在你访问时**按需 (on-demand)** 加载条目。
- 确保指定的 `dtype` 匹配，并可以验证 memory-mapped 的数据没有越界。

而在 Tokenizer 编码大文件时，我们可以使用**生成器**实现惰性求值：
```python
def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
    # 使用 yield from 逐行处理并释放内存
    for text in iterable:
        yield from self.encode(text)
```

---

## Part 2: Transformer 基础构建块 (nn.py)

### 2.1 初始化策略 (Truncated Normal)

不良的初始化会导致梯度消失或爆炸。对于现代 Transformer，通常使用**截断正态分布**（Truncated Normal）进行初始化。

- **Linear Weights**: $\mathcal{N}(\mu=0, \sigma^2 = \frac{2}{d_{in} + d_{out}})$，并在 $[-3\sigma, 3\sigma]$ 截断。
- **Embedding**: $\mathcal{N}(\mu=0, \sigma^2 = 1)$，并在 $[-3, 3]$ 截断。

在 PyTorch 中使用 `nn.init.trunc_normal_(tensor, a=-3*std, b=3*std)` 来实现。

### 2.2 线性层 (Linear) 与 einsum

PyTorch 在内存中默认是**行主序（row-major）**，这意味着数学上的列向量乘法 $y = Wx$ 在代码中通常对应于行向量乘法 $y = xW^T$。

使用 `einops.einsum` 可以完全屏蔽底层存储布局的差异，并且让意图极其可读：

```python
class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        # 权重维度：(out_features, in_features) 
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 代替传统的：return x @ self.weight.T
        # '... d_in' 代表输入，'d_out d_in' 代表权重，
        # 在 d_in 维度进行 contraction (缩并)，输出 '... d_out'
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
```

### 2.3 词嵌入层 (Embedding)

Embedding 的本质**不是矩阵乘法，而是一个巨大的寻址查找表 (Lookup Table)**。

PyTorch 天然支持高级索引 (Advanced Indexing)，所以我们可以直接把 `token_ids` 当作索引传给 `weight` 矩阵，它会自动把任意维度的索引映射对应的向量。

```python
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # 形状变化: (...) -> (..., embedding_dim)
        return self.weight[token_ids]
```

### 2.4 RMSNorm (均方根层归一化)

相比于传统的 LayerNorm，RMSNorm (Zhang & Sennrich, 2019) 去掉了中心化（减去 mean）的步骤，只除以均方根，效率提升显著且模型表现相当（例如被 LLaMA 采用）：

$$ \text{RMSNorm}(a_i) = \frac{a_i}{\sqrt{\frac{1}{d_{model}}\sum_{j=1}^{d_{model}} a_j^2 + \epsilon}} \cdot g_i $$

其中 $g_i$ 是可学习的增益参数 (gain parameter)，初始为 1，维度为 $d_{model}$。

> [!WARNING]
> **数值稳定性警告**：在求平方和时 (`x ** 2`)，极易发生溢出（Overflow）。  
> 在计算 RMS 之前，**必须**先将输入向上转换（Upcast）到 `torch.float32`，完成归一化后再向下转换（Downcast）回原来的 `dtype`。

```python
in_dtype = x.dtype
x = x.to(torch.float32)

rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
result = (x / rms) * self.weight

return result.to(in_dtype)
```

---

## Part 3: 激活函数与前馈网络 (FFN)

### 3.1 SiLU (Swish) 激活函数

SiLU 是一种平滑版本的 ReLU，定义为：
$$ \text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}} $$

优势在于**它在零点附近是平滑可导的**，能提供更好的梯度流。通常直接使用 `x * torch.sigmoid(x)` 实现以保证数值稳定性，而不是手动实现公式。

### 3.2 SwiGLU 门控前馈网络

最初在 PaLM 和 LLaMA 等现代大模型中，**SwiGLU** 彻底取代了传统的两层感知机 FFN。其核心优势是引入了类似 LSTM/GRU 的**门控（Gating）机制**。

$$ \text{FFN}(x) = \text{SwiGLU}(x, W_1, W_2, W_3) = W_2 (\text{SiLU}(W_1 x) \odot W_3 x) $$

1. `gate = SiLU(x @ W1.T)`：充当控制信息流动的**门控信号**（d_ff 维度）。
2. `value = x @ W3.T`：充当真正的**候选值序列**（d_ff 维度）。
3. 二者逐元素相乘 `gate * value` 后，再由 `W2` 聚合并降维回到 `d_model`。

> [!TIP]
> **参数量守恒与硬件对齐**：  
> 传统的 FFN 使用 2 个矩阵（上投影和下投影），中间维度 $d_{ff} = 4 \times d_{model}$。而 SwiGLU 使用了 3 个矩阵（$W_1, W_2, W_3$）。  
> 为了保持与传统 FFN **总参数量完全一致**，算法设计将内部维度调整为：$d_{ff} = \frac{8}{3} d_{model}$。  
> 此外，在代码实现中，计算出的 $d_{ff}$ 还会**向上取整到 64 的倍数**（`((d_ff + 63) // 64) * 64`），这能极大优化 GPU 尤其是 Tensor Core 的显存存取效率和矩阵乘法速度。

---

## Part 4: 旋转位置编码 (RoPE)

### 4.1 RoPE 的核心原理

标准的自注意力机制自身是**排列不变** (permutation invariant) 的，没有位置概念。相对位置编码 (RoPE, Su et al., 2021) 通过**在复数空间/二维平面中旋转**每一对相邻的 query / key 维度，将位置信息自然地注入点积之中。

给定 token 位置 $i$，对于 Query / Key 的第 $k$ 对维度，旋转特定角度：
$$ \theta_{i,k} = i \cdot \frac{1}{\Theta^{2k/d_{k}}} \quad (k \in \{0, \dots, \frac{d_k}{2}-1\}) $$

每一对 `(x_even, x_odd)` 独立做 2D 旋转矩阵相乘：
$$
\begin{bmatrix} x_{even}' \\ x_{odd}' \end{bmatrix} = 
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}
\begin{bmatrix} x_{even} \\ x_{odd} \end{bmatrix}
$$
在代码中等于：`rotated_even = cos * x_even - sin * x_odd` 和 `rotated_odd = sin * x_even + cos * x_odd`。

**RoPE 的绝妙之处**：当你计算位置 $i$ 和 $j$ 的 Q、K 的点积时，其数学结果仅取决于这两者的**相对距离 $\theta_{i,k} - \theta_{j,k} \propto (i - j)$**，完全满足我们期待模型学习到的“相对位置感”。

### 4.2 缓存与性能优化

位置信息只与 Sequence Length 有关，跟 Batch 大小或模型可学习的参数毫无关系，属于纯粹的**确定性推导公式**。

> [!IMPORTANT]
> 计算 $\cos$ 和 $\sin$ 是非常昂贵的三角函数浮点运算。我们必须在模块的 `__init__` 中提前为模型能接受的 `max_seq_len` 预计算好 $\cos$ 和 $\sin$ 的缓存矩阵！
>
> **不该存进 state_dict**：使用 `self.register_buffer("cos_cached", ..., persistent=False)` 机制来储存这些缓存，它具有以下优点：
> 1. Tensor 会随整个 `nn.Module` 同步移动到对应的 GPU。
> 2. PyTorch 引擎知道它不在梯度图内。
> 3. `persistent=False` 告诉系统保存模型检查点 (`checkpoint.pt`) 时**忽略这个张量**，大大节省硬盘空间，因为它在模型 load 之后随时可以由超参数无损即时重构。
