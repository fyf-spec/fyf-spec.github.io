<NoteVisual topic="arpo" />

# ARPO 论文笔记

## 1\. 论文定位

**论文名**：*Agentic Reinforced Policy Optimization*  
**缩写**：ARPO  
**作者单位**：中国人民大学 + 快手科技  
**任务场景**：训练能够进行**多轮工具调用**的 LLM Agent，例如搜索、浏览器、代码解释器等。([arXiv](https://arxiv.org/html/2507.19849v1 "Agentic Reinforced Policy Optimization"))

* * *

## 2\. 要解决的问题

传统 GRPO / DAPO / REINFORCE++ 这类方法主要做 **trajectory-level rollout**：

> 对同一个问题，从头采样多条完整轨迹，然后根据最终答案 reward 更新模型。

问题在于：

- 多轮 Agent 的关键决策往往发生在**工具调用之后**；
- 工具返回的信息会改变模型的上下文和不确定性；
- 传统完整轨迹采样容易忽略这些细粒度的 step-level tool-use 行为。([arXiv](https://arxiv.org/html/2507.19849v1 "Agentic Reinforced Policy Optimization"))

* * *

## 3\. 核心观察

论文发现：LLM 在收到工具反馈后，生成的前 **10–50 个 token** 熵会显著升高。

这说明模型在工具返回之后最不确定，也最值得探索。搜索工具带来的不确定性通常比 Python 这类确定性输出更强。

* * *

## 4\. 核心创新 idea

ARPO 的核心思想可以概括为：

> **不要总是从问题开头重新采样完整轨迹，而是在工具调用后模型最不确定的位置进行分叉采样。**

也就是：

1. 先采样若干条全局轨迹；
2. 每次工具调用后，计算 token entropy；
3. 如果工具反馈后 entropy 明显升高，就从当前节点分叉出多条 partial rollout；
4. 用 advantage attribution 区分共享前缀和分支路径的贡献。

* * *

## 5\. 关键公式

### 5.1 Token entropy：衡量模型不确定性

$$  
H_t=-\sum_{j=1}^{V}p_{t,j}\log p_{t,j}  
$$

其中：

# $$  
\mathbf{p}_t

# \pi_\theta(\cdot \mid \mathcal{R}_{<t},x;T)

\operatorname{Softmax}\left(\frac{\mathbf{z}_t}{\tau}\right)  
$$

含义：

- (H_t)：第 (t) 步 token 分布的熵；
- (V)：词表大小；
- (\mathbf{z}_t)：softmax 前的 logits；
- (\tau)：解码温度；
- 熵越高，说明模型对下一个 token 越不确定。

* * *

### 5.2 工具调用后的熵变化

# $$  
\Delta H_t

\operatorname{Normalize}  
\left(  
H_t-H_{\text{initial}}  
\right)  
$$

含义：

- (H_{\text{initial}})：初始轨迹前若干 token 的 entropy；
- (H_t)：第 (t) 次工具调用后生成 token 的 entropy；
- (\Delta H_t>0)：工具反馈后不确定性上升；
- (\Delta H_t<0)：工具反馈后不确定性下降。

* * *

### 5.3 自适应分叉概率

$$  
P_t=\alpha+\beta\cdot \Delta H_t  
$$

$$  
\operatorname{Action}(P_t)=  
\begin{cases}  
\operatorname{Branch}(Z), & P_t>\tau \  
\operatorname{Continue}, & \text{otherwise}  
\end{cases}  
$$

含义：

- (\alpha)：基础采样概率；
- (\beta)：控制 entropy 影响强度的参数；
- (\tau)：分叉阈值；
- (Z)：从当前工具调用节点扩展出的分支数量。

这就是 ARPO 的核心 rollout 机制：**高熵工具调用节点多探索，低熵节点继续当前轨迹。**

* * *

### 5.4 Hard advantage attribution

对分支后的 individual token，使用 group-normalized reward：

# $$  
\hat{A}_{i,t}

\frac{  
r_i-\operatorname{mean}({R_i}*{i=1}^{G})  
}{  
\operatorname{std}({R_i}*{i=1}^{G})  
}  
$$

对共享前缀 token，取共享该片段的多条轨迹 advantage 平均：

# $$  
\hat{A}^{\text{shared}}_{i,t}

\frac{1}{d}  
\sum_{i=1}^{d}  
\hat{A}_{i,t}  
$$

含义：

- 共享前缀：多条分支共用的 reasoning token；
- individual token：分叉之后各路径独有的 token；
- ARPO 试图让模型知道：**到底是共享步骤有用，还是某个分支步骤带来了收益。**

* * *

### 5.5 Soft advantage：GRPO-style 目标函数

论文最终默认采用 soft setting，用 GRPO 形式隐式地区分共享 token 和分支 token：

# $$  
J_{\mathrm{GRPO}}(\theta)

## \mathbb{E}  
\left[  
\frac{1}{G}  
\sum_{i=1}^{G}  
\frac{1}{|y_i|}  
\sum_{t=1}^{|y_i|}  
\min  
\left(  
r_{i,t}(\theta)\hat{A}*{i,t},  
\operatorname{clip}  
\left(  
r*{i,t}(\theta),1-\epsilon,1+\epsilon  
\right)  
\hat{A}_{i,t}  
\right)

\beta D_{\mathrm{KL}}  
\left(  
\pi_\theta  
\Vert  
\pi_{\mathrm{ref}}  
\right)  
\right]  
$$

其中 importance ratio 为：

# $$  
r_{i,t}(\theta)

\frac{  
\pi_\theta(y_{i,t}\mid x,y_{i,<t})  
}{  
\pi_{\mathrm{ref}}(y_{i,t}\mid x,y_{i,<t})  
}  
$$

如果两条轨迹共享前缀：

$$  
y_{i,<t}=y_{j,<t}  
\Rightarrow  
r_{i,t}(\theta)=r_{j,t}(\theta)  
$$

如果已经分叉：

$$  
y_{i,<t}\neq y_{j,<t}  
\Rightarrow  
r_{i,t}(\theta)\neq r_{j,t}(\theta)  
$$

这意味着 ARPO 不一定要手动给每个共享 token 单独设计 advantage；通过分叉结构和 importance ratio，本身就能让共享部分和独立分支部分获得不同更新信号。

* * *

## 6\. 和普通 GRPO 的区别

| 
维度

 | 

普通 GRPO

 | 

ARPO

 |
| --- | --- | --- |
| 

rollout 方式

 | 

从头采多条完整轨迹

 | 

先全局采样，再在高熵工具节点分叉

 |
| 

探索重点

 | 

完整答案轨迹

 | 

工具调用后的高不确定步骤

 |
| 

token credit

 | 

主要基于整条轨迹 reward

 | 

区分共享前缀和分支路径

 |
| 

适用场景

 | 

单轮或普通 reasoning

 | 

多轮 tool-use Agent

 |
| 

核心优势

 | 

简单稳定

 | 

更节省工具调用预算，更聚焦关键交互步骤

 |

* * *

## 7\. 实验结论

论文在计算推理、知识推理、深度搜索等 **13 个 benchmark** 上评估 ARPO，作者报告 ARPO 相比传统 trajectory-level RL 更强，并且只需要约一半的工具调用训练预算。([arXiv](https://arxiv.org/html/2507.19849v1 "Agentic Reinforced Policy Optimization"))

* * *

## 8\. 一句话总结

**ARPO 的核心是：在工具调用后模型最“迷茫”的地方多采样、多比较、多学习，而不是从头盲目生成更多完整轨迹。**

