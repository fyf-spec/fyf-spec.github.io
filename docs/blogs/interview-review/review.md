<NoteVisual topic="interview-review" />

今天上来做了一个自我介绍，没说太多，问了我很多项目背后的理解，几乎是一个都没答上来。有很多意想不到的问题，应该是需要理解这背后整个发展的历程才能回答上来。将部分问题记录在此以供复盘（PS：由于问题全部从个人项目出发，因此不具有任何参考价值）

**Q:** RoPE是怎么实现的，为什么需要RoPE？RoPE的强大之处在哪?

**A:** Transformer 的 Attention 本身对 token 的顺序不敏感，如果不显式加入位置信息，模型只能看到一组 token，不知道它们谁在前谁在后。RoPE（Rotary Position Embedding）就是一种把位置信息注入到 Attention 里的方式，它不是把一个位置向量加到 hidden states 上，而是在计算 Attention 前，对每一层的 **Query 和 Key** 按当前位置做旋转变换。

实现上可以把 head dimension 两两配对，例如 $(x_{2i}, x_{2i+1})$ 看作一个二维向量。对位置 $m$ 和频率 $\theta_i$，做如下旋转：


工程实现中一般会提前缓存不同位置、不同维度频率对应的 `cos` 和 `sin`，然后通过 `rotate_half` 这类操作完成：

```python
q = q * cos + rotate_half(q) * sin
k = k * cos + rotate_half(k) * sin
```

RoPE 的关键性质是：旋转后的 $q_m^\top k_n$ 会自然包含相对位置 $m-n$ 的信息。也就是说，模型不只是知道“这个 token 在第几个位置”，更容易学到“两个 token 之间相隔多远”。这比单纯的绝对位置编码更适合语言建模，因为语言里的依赖关系很多时候是相对位置关系，比如前后搭配、括号匹配、长距离引用。

它的强大之处主要有几点：

1. **相对位置能力自然进入 Attention 分数**：RoPE 不是额外拼接或相加位置特征，而是直接改变 $QK^\top$ 的几何关系。
2. **无额外可学习参数**：相比 learned positional embedding，RoPE 更轻量，也不受训练时最大位置表的硬限制。
3. **和 KV Cache 很兼容**：解码时历史 token 的 Key 已经按当时的位置旋转并缓存，新 token 只需要旋转当前 Query/Key，再和缓存里的 Key 做 Attention。
4. **长上下文扩展空间大**：后续很多长上下文方法，如 NTK scaling、YaRN、位置插值，本质上都可以看作对 RoPE 频率或位置尺度的调整。

如果面试继续追问不足，可以补一句：RoPE 不是天然解决无限外推问题。训练长度之外的位置会遇到频率分布不匹配，所以长上下文模型通常还要配合 RoPE scaling、长上下文继续训练或稀疏注意力设计。

**Q**: 在SFT训练已有数据的时候，可以采用什么方式加速?

**A:** SFT 的目标是让模型在给定 prompt 的情况下拟合人工或教师模型给出的 answer。这里的“已有数据”通常指已经收集好的 instruction-response 数据，所以加速可以从 **数据利用效率**、**训练计算效率** 和 **参数更新方式** 三个角度讲。

第一类是数据侧加速：

1. **预先 tokenize 并缓存**：避免每个 epoch 重复做文本解析、chat template 拼接和 tokenizer 计算。
2. **按长度分桶（bucketing）和动态 batch**：把长度相近的样本放在一起，减少 padding 浪费。
3. **sequence packing**：把多个短样本拼成一个长序列训练，并用 attention mask / loss mask 隔开样本边界。SFT 数据常常有很多短问答，packing 对吞吐提升很明显。
4. **只对 answer 部分算 loss**：prompt 部分只作为条件，不参与 loss，可以减少无效监督信号，也让训练目标更贴近“学会回答”。
5. **清洗、去重和筛选高质量样本**：如果已有数据里有大量重复、低质、格式错误样本，直接训练会浪费 token budget。

第二类是训练侧加速：

1. **混合精度训练**：使用 bf16/fp16，降低显存和带宽压力。
2. **FlashAttention / fused kernels**：减少 Attention 的显存读写，提高长序列训练吞吐。
3. **梯度累积 + ZeRO/FSDP**：在显存有限时维持较大的 global batch，同时切分 optimizer state、gradient、parameter。
4. **activation checkpointing**：用额外计算换显存，允许更大 batch 或更长序列。
5. **合理设置 max length**：不是所有数据都需要按最大长度训练。可以截断异常长样本，或者分阶段逐步增加上下文长度。

第三类是参数更新方式：

1. **LoRA / QLoRA**：如果只是领域适配或指令风格适配，不一定要全参数训练。LoRA 只训练低秩增量矩阵，QLoRA 还可以在量化基座上训练 adapter。
2. **冻结部分层**：在任务差异不大时，只更新高层或 adapter，降低显存和优化器开销。
3. **先小规模试跑再放大**：用小模型、小数据子集或较短序列验证数据格式、loss mask 和学习率，避免大规模训练才发现问题。

如果面试中只能答一句，我会说：SFT 加速最直接有效的是 **预 tokenize + 长度分桶 + sequence packing + FlashAttention + LoRA/ZeRO**。其中 sequence packing 是很多人容易漏掉但非常关键的一点，因为 SFT 数据长度分布通常很不均匀。

**Q** 在RL的时候如果想要结果做的token efficiency，在训练的时候可以怎么设计，比如我现在已经有了一些数据?
（其实我还是不能理解为什么SFT和RL硕能给一些数据，这些不都应该是模型实时生成然后给反馈吗）

**A:** 先澄清一个关键点：SFT 和 RL 都可以“给一些数据”，但数据的角色不一样。

SFT 的数据是监督样本，形式通常是：

```text
prompt -> reference answer
```

训练时模型不需要实时生成完整回答再被打分，而是直接在 reference answer 上做 teacher forcing，学习每一步下一个 token 应该是什么。

RL 的数据更多是 prompt、偏好、reward、历史 trajectory 或 verifier 反馈。在线 RL 确实需要模型自己生成 response，然后用 reward model、规则 verifier、人类反馈或环境反馈打分。但这不代表 RL 完全不能利用已有数据。已有数据可以用于：

1. 先做 SFT warm-up，让策略模型别从很差的分布开始探索。
2. 训练 reward model 或 process reward model。
3. 做 DPO/IPO/KTO/ORPO 这类偏好优化，相当于用离线偏好数据进行 RLHF 的替代或前置阶段。
4. 构造 prompt 集合，让在线 RL 在固定任务分布上采样。
5. 做 rejection sampling，把模型生成的多个答案中高分答案筛出来再 SFT。

如果目标是 **token efficiency**，可以理解成两件事：一是训练时用更少生成 token 获得更大收益；二是最终模型回答时更少废话、更短路径地得到正确结果。

训练设计上可以这样做：

1. **从已有高质量数据做 SFT 或 DPO warm-up**

   不要让 RL 从原始 base model 开始探索。先用已有数据把模型拉到一个可用分布，再做 RL。这样生成的样本更少是垃圾，reward 信号更有效。

2. **用已有数据训练 reward model / verifier**

   如果有偏好对，可以训练 reward model；如果是数学、代码、工具调用任务，可以设计规则 verifier，例如单测是否通过、答案是否匹配、工具调用是否成功。verifier 比纯人工反馈便宜很多，也更适合大规模 RL。

3. **使用 group sampling 提高每个 prompt 的信息量**

   例如 GRPO/RLOO 这类方法会对同一个 prompt 采样多个回答，然后用组内相对奖励估计 advantage。这样不一定需要 critic，也能比较稳定地知道哪个回答更好。

4. **加入长度惩罚或预算约束**

   如果希望模型 token efficient，reward 不能只奖励“答对”，还要把长度纳入目标。例如：

   $$
   R = R_{\text{task}} - \lambda \cdot \text{len(response)}
   $$

   或者设计成“答对且更短得分更高”。但长度惩罚不能太强，否则模型可能学会过早结束、跳步或输出不完整答案。

5. **做过程级 reward，而不是只做最终 reward**

   对复杂推理任务，只在最后给 0/1 奖励会很稀疏，token 利用率低。可以用 process reward model、单步 verifier、工具调用结果等方式，让模型知道中间步骤哪里好、哪里坏。

6. **复用历史样本，但要控制 off-policy 偏差**

   已有 trajectory 可以放进 replay buffer，或者用于离线偏好优化。但 PPO/GRPO 这类方法通常假设样本来自当前或相近策略，所以复用旧样本时需要 importance sampling、KL 约束或限制样本过期程度。

7. **课程学习和 hard prompt mining**

   不要平均地在所有 prompt 上浪费采样。可以把已有数据按难度、长度、错误率分桶，优先训练当前模型会错但又不是完全不会的样本。

8. **把“少 token 做对”写进数据和评价**

   如果已有数据里答案普遍啰嗦，模型很难自动学会简洁。可以构造短答案偏好对，例如同样正确时偏好更短、更直接、更少重复的回答。

面试里可以这样总结：已有数据在 RL 里不是替代在线采样，而是减少无效探索。先用 SFT/DPO 把模型放到好分布，再用 verifier/reward model 做在线采样优化，同时把长度惩罚、过程奖励、group-relative advantage 和 hard prompt mining 加进去，就能提高 token efficiency。

**Q** 在KV Cache做的过程中，有哪些思考，现有方法有什么不足，怎么加速推理?

**A:** KV Cache 解决的是自回归解码时的重复计算问题。生成第 $t$ 个 token 时，前面 $1 \sim t-1$ 个 token 的 Key 和 Value 在每一层其实已经算过了。如果每一步都把完整上下文重新跑一遍，计算会非常浪费。KV Cache 会把历史 token 在每层的 $K,V$ 存下来，下一步只计算新 token 的 $Q,K,V$，然后用当前 $Q$ 去 attend 历史缓存的 $K,V$。

它带来的收益是：decode 阶段每生成一个 token，不需要重复计算历史 token 的 K/V 和中间层表示。但它也引入了新的瓶颈：**显存容量和显存带宽**。

KV Cache 的大小大致和下面这些量成正比：

```text
num_layers * sequence_length * num_kv_heads * head_dim * 2(K,V) * dtype_size
```

所以 batch 越大、上下文越长、层数越多，KV Cache 越容易成为推理的主要显存占用。对于长上下文和高并发服务，瓶颈往往不是算力，而是不断从显存读取历史 K/V 的带宽。

我在做 KV Cache 时会重点考虑这些问题：

1. **Cache layout 是否适合连续读写**

   常见布局会围绕 `[batch, layer, kv_head, seq, head_dim]` 或其变体设计。decode 时每一步都要读历史 seq 维度的数据，所以 layout 会影响 coalesced memory access 和 kernel 效率。

2. **变长请求怎么管理**

   在线 serving 中每个请求长度不同，如果直接为每个请求分配连续最大长度 buffer，会有大量碎片和浪费。PagedAttention / block-based KV Cache 会把 cache 切成固定大小的 block，通过 block table 管理逻辑连续的序列。

3. **prefill 和 decode 是两个不同瓶颈**

   prefill 阶段一次处理完整 prompt，Attention 计算量大，适合 FlashAttention、chunked prefill、prefix cache。decode 阶段每次只生成一个或少量 token，通常更受 KV 读取带宽、batch 调度和 kernel launch 开销影响。

4. **Cache 是否可复用**

   对系统提示词、RAG 前缀、多轮对话历史，可以做 prefix caching。相同前缀不必重复 prefill，直接复用对应的 KV Cache。

现有 KV Cache 方法的不足主要是：

1. **显存线性增长**：上下文长度增长时，KV Cache 按 token 数线性增长。
2. **长上下文 decode 带宽压力大**：每生成一个 token 都要读越来越长的历史 K/V。
3. **动态 batch 管理复杂**：请求不断进入、结束、扩容，容易产生碎片和调度开销。
4. **压缩或量化会有精度损失**：KV quantization、cache eviction、滑动窗口都会影响模型对长距离信息的使用。
5. **对 prefill 帮助有限**：KV Cache 主要优化 decode，长 prompt 的首次 prefill 仍然很贵，除非有 prefix cache 或 chunked prefill。

加速推理可以从模型结构、kernel、系统调度三层回答：

1. **减少 KV Cache 本身**

   - MQA/GQA：多个 Query head 共享更少的 KV head。
   - MLA / latent KV cache：把 KV 压到低维 latent 表示，需要时再恢复。
   - KV quantization：把 KV Cache 从 fp16/bf16 压到 int8/fp8，降低显存和带宽。
   - sliding window / sparse attention：只 attend 局部窗口或重要 token。

2. **提高 cache 访问效率**

   - PagedAttention：用分页块管理变长 KV，减少碎片，提升 serving 吞吐。
   - Flash-Decoding / fused attention kernels：减少中间读写和 kernel launch。
   - 更合理的内存 layout：让 seq/head_dim 方向的读取更连续。

3. **减少需要解码的步数**

   - speculative decoding：小模型先草拟多个 token，大模型批量验证。
   - multi-token prediction：一次预测多个未来 token，提高每次 forward 的产出。
   - early stop 和更好的 EOS 训练：避免模型无意义拖长回答。

4. **提高服务端吞吐**

   - continuous batching：把不同请求动态合批，提高 GPU 利用率。
   - chunked prefill：把长 prompt 的 prefill 切块，避免阻塞 decode 请求。
   - prefix caching：复用系统 prompt、few-shot prompt、RAG 共享前缀。
   - CPU/GPU 分层 cache 或 offload：在极长上下文场景下用容量换延迟。

面试中可以最后总结成一句：KV Cache 把自回归解码从“重复算历史”变成“读取历史缓存”，所以主要矛盾从算力转移到了显存容量、显存带宽和动态调度。优化方向就是减少 KV、压缩 KV、提高 KV 访问效率，以及减少需要生成的 token 数。

