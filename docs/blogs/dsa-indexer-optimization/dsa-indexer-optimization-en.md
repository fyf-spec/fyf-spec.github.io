---
title: "After DSA: Two Routes for Optimizing the Indexer"
description: From redundancy in index-score patterns to redesigning the candidate generator.
date: 2026-07-31
lang: en-US
outline: deep
---

# After DSA: Two Routes for Optimizing the Indexer


## After Sparse Attention, Why Is There Still a Quadratic Cost?

When I first looked at DSA, I had a natural question. Suppose a model faces a 128K-token context but ultimately lets each query read only 2K of those tokens. It appears to have skipped most of the history, so why does a new performance bottleneck still emerge at long context lengths? Looking one step earlier in the execution pipeline makes the answer straightforward: the model must first know which 2K tokens it should read, and “finding those 2K tokens” itself requires computation.

DeepSeek Sparse Attention (DSA) answers this problem by adding a lightweight indexer outside the main attention operation. The indexer first computes the relevance between the current query and every historical key, selects the top-$k$, and only then lets the actual Sparse MLA read those tokens.

For the query at position $t$ and a historical position $s$, the DSA indexer score can be summarized as

$$ I_{t,s}=\sum_{j=1}^{H_I}w^I_{t,j}\operatorname{ReLU}\!\left(q^I_{t,j}\cdot k^I_s\right) $$

Here $H_I$ is the number of indexer heads, $q^I_{t,j}$ and $k^I_s$ are low-dimensional indexing queries and keys, and $w^I_{t,j}$ determines the contribution of each head to the current query.

This is much cheaper than running full attention directly, but it still contains one unavoidable operation: **every query first scans all historical tokens.** If the sequence length is $L$, the main attention cost has already fallen from $O(L^2)$ to $O(Lk)$, yet the indexer still produces $O(L^2)$ query–key scores. As Sparse MLA becomes faster, the indexer can turn from a “small accessory” into the new dominant cost. This is also the starting point for how I understand the work that follows DSA:

> If the indexer's job is to help attention look at fewer tokens, can the indexer itself also look at fewer tokens?

I roughly organize the current work into two routes. The first accepts the basic form of the DSA indexer and then examines the dimensions along which its scores are redundant. The second takes a step back and rethinks how candidate addresses should be generated in the first place:

1. **Start from index-score patterns.** Observe where the full index scores or top-$k$ results are redundant, then make the existing indexer compute less, select less, or reuse existing results.
2. **Start from the indexer architecture.** Stop assuming that “a low-dimensional query performs one flat scan over all keys” is the only possible form, and redesign candidate generation, training supervision, and the system execution path.


## Finding Redundancy in Score Patterns

We can first write the work of one indexer approximately as

$$ W_{\mathrm{index}}\propto U_L\times H_I\times Q_I\times N_I $$

- $U_L$: the number of layers that independently generate indices;
- $H_I$: the indexer heads that participate in exact scoring;
- $Q_I$: the queries that must initiate retrieval separately;
- $N_I$: the historical keys scanned by each retrieval.

Traditional DSA is almost “full” along all four dimensions: every layer recomputes the index, every head participates, every query retrieves independently, and every retrieval scans the complete history. Subsequent work did not immediately overturn the entire indexer. It first returned to actual routing maps to look for redundancy: are neighboring queries repeatedly selecting the same tokens, and are neighboring layers also repeating similar retrieval results? The two plots below provide an intuitive slice.

<figure style="margin: 1.75rem auto 2rem; width: 100%;">
  <div style="display: flex; gap: 14px; align-items: flex-start; justify-content: center; flex-wrap: wrap;">
    <div style="flex: 1 1 360px; min-width: 0;">
      <img src="./images/route-layer-23.png" alt="DSA-style indexer routing heatmap at Layer 23" style="display: block; width: 100%; height: auto; border: 1px solid rgba(127, 127, 127, 0.24); border-radius: 8px;" />
      <div style="margin-top: 0.4rem; text-align: center; color: #737373; font-family: 'Noto Serif', Georgia, serif; font-size: 0.84rem; line-height: 1.5;">(a) Layer 23</div>
    </div>
    <div style="flex: 1 1 360px; min-width: 0;">
      <img src="./images/route-layer-18.png" alt="DSA-style indexer routing heatmap at Layer 18" style="display: block; width: 100%; height: auto; border: 1px solid rgba(127, 127, 127, 0.24); border-radius: 8px;" />
      <div style="margin-top: 0.4rem; text-align: center; color: #737373; font-family: 'Noto Serif', Georgia, serif; font-size: 0.84rem; line-height: 1.5;">(b) Layer 18</div>
    </div>
  </div>
  <figcaption style="margin: 0.8rem auto 0; max-width: 94%; color: #666; text-align: left; font-family: 'Noto Serif', Georgia, serif; font-size: 0.9rem; line-height: 1.75; letter-spacing: 0.01em;">
    <strong style="font-weight: 600; color: #4f4f4f;">Figure 1 | Routing patterns of a DSA-style indexer.</strong> A DSA-style indexer is attached to a dense Llama-3.2-3B, warmed up on about 1B tokens, and then trained for about another 16B tokens. The left and right panels show routing at Layer 23 and Layer 18. Red denotes a fixed local window or attention sink; yellow denotes agreement between the indexer prediction and the oracle; cyan denotes tokens selected only by the indexer, i.e. false positives; green denotes tokens selected only by the oracle, i.e. false negatives; and blue denotes unrouted regions.
  </figcaption>
</figure>

The exact false-positive and false-negative positions are not identical in the two layers, but their overall structure is very similar. Neighboring queries often select similar tokens along contiguous regions, and different layers retain roughly the same routing skeleton. This observation cannot by itself prove which computations can safely be removed, but it provides a common starting point for subsequent work: if index patterns may repeat along the token, head, layer, and query dimensions, we can ask in turn which scores require exact computation and which searches can be coarsened, shared, or reused.

### Token Axis: Not Every Token Needs an Exact Score

[HISA: Efficient Hierarchical Indexing for Fine-Grained Sparse Attention](https://arxiv.org/abs/2603.28458) observes that important tokens are often not scattered across history without any structure. Neighboring tokens exhibit some local consistency, so the model can first identify potentially important regions at a coarser block granularity and then perform the original token-level DSA scoring inside a small number of selected blocks.

![HISA's block-level coarse retrieval and token-level precise reranking](./images/hisa.png)

The figure can be read from top to bottom. The full set of indexing keys is first pooled into $L/B$ block representatives, and the query selects only the top-$m$ blocks. Those blocks are then expanded, and a second scoring pass selects the final top-$k$ tokens. The blue output remains a set of token indices, so HISA changes candidate generation without changing the input interface of Sparse MLA.

It turns one flat search,

$$ N\text{ exactly scanned tokens}, $$

into

$$ \frac{N}{B}\text{ coarsely screened blocks}+C\text{ precisely screened candidate tokens},\qquad C\ll N. $$

This can be understood as looking for one sentence in a library. The original indexer inspects the entire library page by page; HISA first uses a summary of each shelf to rule out most shelves, then checks the remaining shelves page by page. The key technique is not a vague switch to “block attention,” but **making the block mean responsible only for candidate recall while returning final ranking to the original DSA token score**. Concretely, HISA partitions indexing keys into fixed-size blocks of size $B$ and averages them, uses all indexer heads to score the $L/B$ block representatives, expands at most $mB$ original tokens from the selected $m$ blocks, and applies exactly the same score as DSA to obtain a token-level top-$k$. What it passes to Sparse MLA is therefore still a set of fine-grained token indices rather than a block mask.

The benefit is that coarse screening can be cheap while precise screening does not waste the budget on an entire block. The paper's implementation also forcibly retains the first block and the last valid block, using deterministic rules to protect attention sinks, local context, and packed-sequence boundaries. Its irreversible risk is equally clear: if the block mean averages away a region with only one strong signal, the second stage can never see that token. HISA compresses $N_I$ from the full prefix to a candidate set, but every exact score it saves assumes sufficient first-stage block recall.

### Head Axis: Not Every Indexer Head Is Equally Important

DSA uses multiple indexer heads because different low-dimensional subspaces may capture different relevance patterns. But “having many heads” does not imply that “every query must invoke every head.” The central observation of [MISA: Mixture of Indexer Sparse Attention](https://arxiv.org/abs/2605.07363) is that, for a particular query, only a small number of heads often determine the top-$k$. It first uses block-level key summaries to estimate the contribution of each head:

![MISA adds a head router before the DSA indexer](./images/misa.png)

The green path in the figure is the original DSA indexer, while the orange Router is the lightweight branch added by MISA. The Router reads the indexer queries, their corresponding gates, and pooled indexing keys, then outputs an active-head mask. The following Top-$k$ Selector still receives token-level index scores. In other words, MISA does not replace the multi-head indexer with a single head; it turns multi-head computation into query-dependent conditional computation.

$$ E_{t,j}=\frac{1}{M}\sum_b\left|w^I_{t,j}\operatorname{ReLU}\!\left(q^I_{t,j}\cdot\widetilde k^I_b\right)\right| $$

The crucial technique is that **the head router cannot look only at the gate $w_{t,j}$ or the query norm; each head must first interact cheaply with historical block summaries**. Looking only at the query says how “loud” a head is, but not whether the current prefix actually contains content it is good at matching. $E_{t,j}$ considers the query, gate, and historical summaries together, so it estimates whether the head is useful for this retrieval. After selecting the top-$h$, only those heads scan the complete token sequence. The more conservative MISA$^\dagger$ first lets a small number of heads recall a larger set of $k'$ candidates, then uses all heads to recover the full DSA ranking inside that candidate set.

The difference from HISA is clear:

- HISA retains all heads but reduces the number of tokens each head scans precisely;
- MISA retains the full token prefix but reduces the number of heads participating in exact scanning.

MISA therefore primarily reduces $H_I$, not $N_I$. In a representative configuration from the paper, DeepSeek-V3.2 can go from 64 indexer heads to 8, but those 8 active heads still have to read the complete key sequence. If the system bottleneck comes from KV/index-key bandwidth rather than head arithmetic, the reduction in kernel FLOPs may not translate proportionally into end-to-end gains.

### Layer Axis: Neighboring Layers Need Not Find Addresses Again

[IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse](https://arxiv.org/abs/2603.12201) observes that the top-$k$ token sets produced by neighboring layers usually overlap substantially. It therefore divides layers into two kinds: a small number of Full layers compute indices normally, while subsequent Shared layers skip the indexer and directly reuse the token positions of the nearest preceding Full layer. What is reused is only “where to read,” not “how much weight to assign after reading.” A Shared layer still uses its own query, key, and value to recompute attention over the reused support. If all $L$ layers originally ran an indexer but only $M$ anchor layers now generate routes, $U_L$ falls from $L$ to $M$.

![Cross-layer top-k overlap and sharing groups in IndexCache](./images/indexcache.png)

The IndexCache paper does not provide a separate Full/Shared execution diagram; this heatmap is the most important figure for understanding its structure. Both axes are layers, the color indicates top-$k$ overlap between two layers, and the red boxes show sharing groups found by loss-based greedy search. Notice that the boxes do not mechanically segment the brightest local regions. This is exactly the point: high overlap motivates reuse, but end-to-end loss still determines which layers become Full anchors.

In my view, the most important IndexCache technique to explain is not reuse itself, but **how the authors decide which layers can be removed**. A seemingly natural solution is to compute the cosine similarity of neighboring layers' indices, attention outputs, or score vectors and merge the most similar layers. The paper does try such similarity proxies, but its final training-free method uses a greedy loss search that attempts to remove layers one at a time. Starting with every layer in Full mode, each step temporarily changes every candidate layer to Shared, evaluates the complete model on a fixed calibration set, and permanently removes the layer whose removal yields the lowest validation loss:

$$ \ell^*=\arg\min_{\ell\in\mathcal R}\operatorname{EvalLoss}\!\left(M,\mathcal D,\mathbf c\vert_{c_\ell\rightarrow S}\right) $$

Here $M$ is the frozen model, $\mathcal D$ is the calibration data, and $\mathcal R$ is the set of layers that remain removable. After one layer is removed, the next round must reevaluate the other candidates under the new sharing structure because removal decisions interact. Consider a four-layer example: if removing layers 2, 3, and 4 gives losses of 1.03, 1.01, and 1.05, the first step removes layer 3. The second step cannot reuse the old scores; it must test layers 2 and 4 again in a model where layer 3 is already Shared.

Why not use cosine similarity? The paper explains that it measures only directional similarity and is insensitive to magnitude and a small number of high-scoring tokens, whereas a top-$k$ index is determined precisely by relative token-level score magnitudes and the ranking boundary. Two vectors can have very high cosine similarity yet exchange several nearly tied tokens around rank $k$, and those tokens may be crucial to the downstream task. More fundamentally, local index similarity cannot tell us whether the remaining Transformer will amplify or cancel the error. In the paper, the similarity proxy is even weaker than uniform sharing. Loss search is expensive, but it directly measures whether the final model distribution is damaged. The training-aware version takes another route: one Full indexer is distilled against the attention teachers of several following layers so that uniform sharing becomes usable after training.

### Query Axis: Neighboring Queries Need Not Scan the Same Prefix Separately

[PIVOT: Efficient Query-Group Indexing for Token-Level Sparse Attention](https://arxiv.org/abs/2607.24593) turns to the query axis, which earlier work addressed less often. It finds that neighboring queries have highly overlapping top-$k$ sets. In the paper's measurements, adjacent queries share about 0.8–0.9 of their selections. Even when four queries form a group, the union of their top-$k$ sets is usually only slightly larger than $k$ and far smaller than the worst case of $4k$. PIVOT therefore aggregates a group of $g$ queries into one proxy query and performs only one full-prefix scan:

![PIVOT serves a group of queries with one proxy scan](./images/pivot.png)

On the left, DSA performs a separate full scan for each of $q_1$ through $q_4$. On the right, neighboring prefill queries or queries from the same MTP decoding step form a group, and only the proxy query scans the complete prefix. In the upper Refine path, each query reranks the proxy's top-$c$; in the lower Reuse path, all queries directly share the proxy's top-$k$. The speed–accuracy difference is visible directly in the figure.

$$ \bar q_j=\operatorname{Mean}_{t\in G}q^I_{t,j} $$

There are then two ways to proceed:

- PIVOT-Reuse: the entire group directly shares the proxy's top-$k$;
- PIVOT-Refine: the proxy first selects top-$c$ candidates, then every query reranks them using the original DSA score.

PIVOT's key technique is to **separate “sharing the scan” from “sharing the final selection.”** The paper's ablation shows that mean pooling clearly outperforms using the first or last query in a group as the proxy; the mean better represents the common needs of the group. PIVOT-Reuse directly shares the proxy top-$k$ for maximum speed. By default, PIVOT-Refine first takes roughly $c=2k$ shared candidates and then uses each query's original multi-head DSA indexer to rescore them. The proxy is therefore responsible only for high recall, while query-specific scores determine final ranking. This follows the same coarse-to-fine idea as HISA, but sharing occurs on the query axis. Group complexity changes from $O(gL)$ to $O(L)$ for Reuse or $O(L+gc)$ for Refine.

A minimal example is four queries that respectively need $\{1,4\}$, $\{1,5\}$, $\{1,4\}$, and $\{1,6\}$. Although they make eight selections in total, their union contains only four tokens. The proxy can first recall those four candidates, and Refine can then restore the two results for each query, eliminating the need to scan the full prefix four times. PIVOT therefore does more than add another optimization axis. It reveals a more general principle: **the indexer's output is not a random tensor whose four axes are independent; it has simultaneous structure across tokens, heads, layers, and queries.**

## Redesigning the Indexer Architecture

The pattern route asks, “Which of the original scores need not be computed?” The architecture route goes further: “Why must we use a low-dimensional query to scan all keys flatly and then take a top-$k$?” Once we ask the latter question, the indexer is no longer merely a small MLP in front of attention. It can be understood as a retrieval system that must define candidate representations, search algorithms, supervision, reuse scope, and hardware execution. The methods below look quite different, but each changes a basic interface of that retrieval system.

### Louver: Rewrite Top-$k$ as a Range Query

[Sparse Attention as a Range Searching Problem (Louver)](https://arxiv.org/abs/2605.06763) takes a very different entry point. Instead of directly asking “which $k$ keys have the highest scores?”, it asks for all keys satisfying

![Louver's threshold oracle, geometric index, buffer, and sparse-attention data flow](./images/louver.png)

In the figure, one copy of the query enters the Louver Index, while another lets the Threshold Oracle estimate $\tau$ from sampled scores. The index returns candidate $K^*,V^*$ for attention. Newly produced keys first enter a dense buffer, and when the buffer is full, the geometric index is updated asynchronously. The notable point is that Louver not only proposes an offline search algorithm; it also completes the online update loop required as an autoregressive KV cache continues to grow.

$$ \langle q,k\rangle\ge\tau. $$

It partitions keys into small clusters and stores a center $c$ and radius $\rho$ for each. By Cauchy–Schwarz, the maximum possible score of any key in a cluster is bounded by

$$ \langle q,k\rangle\le \langle q,c\rangle+\rho\|q\|_2. $$

If a cluster's upper bound is below $\tau$, the entire cluster can be safely excluded, and only the remaining candidates need exact dot products. The key technique is that Louver does not arbitrarily allocate $\tau/S$ to each subspace. Instead, it uses the Threshold Algorithm to scan cluster bounds synchronously across several subspaces. Let $f_{s,\sigma_s(d)}$ be the upper bound of the next cluster in subspace $s$ at scan depth $d$. The full-dimensional inner product of every unseen key is then no greater than

$$ U(d)=\sum_s f_{s,\sigma_s(d)}. $$

The search can safely stop the first time $U(d)<\tau$. Cluster members encountered before that point receive complete exact dot products to remove false positives. The synchronized stopping rule matters because a relevant key may contribute little in one subspace and a great deal in another; pruning each subspace independently can create false negatives. Louver therefore does not reuse one axis of an existing score tensor. It changes the mathematical interface of the search problem from fixed-budget top-$k$ to threshold range search with a geometric upper-bound certificate. Its advantage is that filtering correctness can be guaranteed relative to a given threshold. Its limitation lies in exactly the same place: returning every key above $\tau$ does not automatically guarantee that $\tau$ preserves enough attention mass.

### MiniMax Sparse Attention: Make the Indexer a Native GQA Branch

[MiniMax Sparse Attention (MSA)](https://arxiv.org/abs/2606.13392) does not directly adopt DSA's multi-head token top-$k$ indexer. It assigns one low-dimensional index-query head to each GQA group and shares the index key across all groups. It first computes inexpensive token scores, max-pools each 128-token block, and finally selects 16 blocks. The Main Branch still uses standard GQA queries to compute exact softmax attention over the selected 2,048 tokens. Three constraints jointly determine the indexer's granularity:

![The Index Branch and Main Branch of MiniMax Sparse Attention](./images/minimax-sparse-attention.png)

The left side clearly separates the two paths. The green Index Branch uses $Q_{idx}K_{idx}^{\top}$, block max pooling, and Top-$k$ only to produce KV-block addresses. The blue Main Branch retains standard $Q,K,V$ and performs exact sparse softmax over the selected blocks. The right side shows that different GQA groups may choose different remote blocks, while query heads in the same group share one access mask.

1. Per-GQA-group selection preserves retrieval differences between KV groups;
2. block-level output provides contiguous KV access and better GPU compatibility;
3. low-dimensional, shared index keys reduce the constant factor of the full scan.

The genuinely difficult and crucial technique is **how to train a pure selector with a non-differentiable top-$k$**. LM loss can update the selected Main Branch but cannot directly tell the indexer which unselected block should have ranked higher. MSA first goes through a full-attention warmup, allowing the random indexer to observe a complete teacher without damaging the backbone. It then uses the mean attention distribution of multiple Main heads in the same GQA group as a KL teacher and applies stop-gradient to the hidden state and teacher, so the auxiliary KL updates only the index projections. This prevents the backbone from making its main attention artificially simpler to accommodate a small indexer. In the paper's ablation, failing to detach leads to stalled LM loss and gradient spikes, showing that this is not a decorative training detail.

MSA still contains a quadratic Index Branch, so it does not eliminate the full scan asymptotically. It uses low dimensionality, few heads, a shared index key, and regular block access to reduce the constant. It represents a different architectural approach: **rather than optimizing an expensive token indexer after the fact, design the indexer interface, supervision, and kernel together during pretraining so that they are hardware-executable.**

### CLSA: Make Shared Routing Part of the Model Architecture

IndexCache searches for reusable indices in an ordinary layer-by-layer Transformer. [You Only Index Once: Cross-Layer Sparse Attention with Shared Routing (CLSA)](https://arxiv.org/abs/2606.06467) goes further. Built on YOCO's shared-KV architecture, it lets multiple cross-decoder layers read the same KV cache and generates only one common token route for those layers. Every cross layer still uses its own query to recompute attention weights over the selected tokens, but a common indexer decides “where to read.”

![CLSA generates one token route on a shared-KV architecture](./images/clsa.png)

The lower half of the figure is the self-decoder, which produces Full KV only once. The common Top-$k$ Indexer on the left also runs only once, producing Sparse KV for all cross-attention layers in the upper half. “Routing Once, Cache Once” is a paired constraint: the common positions have a clearly shared object precisely because the cross layers read the same KV.

Its most important training technique is **not making the shared indexer imitate an arbitrary single layer, but fitting it to the mean dense-attention distribution across all cross layers and attention heads**:

$$ \bar A=\frac{1}{L_cH_a}\sum_{l=1}^{L_c}\sum_{h=1}^{H_a}\operatorname{softmax}\!\left(Q^{(l,h)}K^{(h)\top}\right) $$

If two layers concentrate most of their probability on tokens 1 and 3 respectively, the mean teacher asks the common route to cover both instead of favoring one anchor layer. Training first freezes the backbone and warms up only the indexer with KL, then jointly optimizes LM loss and distillation loss so that a random shared route does not damage the entire cross-decoder at the start. This turns layer-axis reuse from an empirical optimization into an architectural invariant. Instead of noticing that neighboring layers look similar and then attempting reuse, it first makes the layers share an information source and trains a route representing their common needs. The gain is more complete, but so is the cost: CLSA is not a drop-in patch for an existing DSA model. It requires a shared-KV backbone and large-scale training adaptation, and the mean teacher may dilute tokens that matter only to a few layers.

### SparDA: Let the Indexer Work One Layer Ahead

[SparDA: Sparse Decoupled Attention for Efficient Long-Context LLM Inference](https://arxiv.org/abs/2606.04511) addresses not whether index results are repeated, but why selection must block attention. It adds a Forecast $F_l$ at layer $l$ to predict the KV blocks needed by layer $l+1$ in advance:

![SparDA decouples next-layer block selection from current-layer execution](./images/sparda.png)

In panel (a), the baseline cannot select blocks until the current-layer $Q_l$ appears. In panel (b), $F_{l-1}$ has already prepared the route for layer $l$. Panel (c) further puts the next-layer route produced by $F_l$ and CPU KV prefetching on a side path parallel to current-layer execution. The dashed cross-layer arrow is the core of SparDA: Forecast supplies addresses early, while the real query remains responsible for attention.

$$ \mathcal B_{l+1}=\mathcal B_{\mathrm{init}}\cup\mathcal B_{\mathrm{local}}\cup\operatorname{TopK}\!\left(F_l\widetilde K_{l+1}^{\top},k\right) $$

The actual attention at layer $l+1$ is still performed by $Q_{l+1}$. Forecast is responsible only for addresses, so the system can fetch the next layer's KV from CPU while the current layer is computing. The key technique is that **Forecast learns the set distribution of the next layer's original multi-head query selector, and its objective explicitly adds an “other blocks” bucket**. The teacher's top-$k$ entries remain separate, while all remaining probability mass is combined into the $(k+1)$-th entry before KL. If supervision covered only the ordering inside the top-$k$, the student would not be penalized for simultaneously assigning high probability to many incorrect blocks. The “other” bucket also gives out-of-set logits a gradient and directly constrains the total mass of wrong candidates.

SparDA redesigns the temporal position of the indexer: “select, transfer, then compute in the current layer” becomes “predict in the previous layer, overlapping selection and prefetching with current computation as much as possible.” This is why its benefit cannot be measured by FLOPs alone; shortening the critical path can matter more than reducing total work. It predicts the next layer's access addresses, not its KV content, and it requires training Forecast together with a specialized CPU–GPU offload pipeline. It therefore cannot be treated as a training-free scheduling trick.

## References

- DeepSeek-AI. DeepSeek Sparse Attention / DeepSeek-V3.2.
- HISA: Efficient Hierarchical Indexing for Fine-Grained Sparse Attention. https://arxiv.org/abs/2603.28458
- IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse. https://arxiv.org/abs/2603.12201
- MISA: Mixture of Indexer Sparse Attention for Long-Context LLM Inference. https://arxiv.org/abs/2605.07363
- PIVOT: Efficient Query-Group Indexing for Token-Level Sparse Attention. https://arxiv.org/abs/2607.24593
- Sparse Attention as a Range Searching Problem: Towards an Inference-Efficient Index for KV Cache. https://arxiv.org/abs/2605.06763
- MiniMax Sparse Attention. https://arxiv.org/abs/2606.13392
- Scout Before You Attend: Sketch-and-Walk Sparse Attention for Efficient LLM Inference. https://arxiv.org/abs/2602.07397
- You Only Index Once: Cross-Layer Sparse Attention with Shared Routing. https://arxiv.org/abs/2606.06467
- SparDA: Sparse Decoupled Attention for Efficient Long-Context LLM Inference. https://arxiv.org/abs/2606.04511
