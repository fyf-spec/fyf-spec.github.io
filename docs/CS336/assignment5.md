

## Chain-of-Thought Reasoning and Reasoning RL

### Different types of reasoning

- **Chain-of-Thought (CoT) Reasoning with LLMs**: 
    Early researches use "srcatchpad" method to break the problem into intermediate steps. 
    Later, other work prompts a strong model to "think step by step", which was found significantly improving.

- **Reason with expert iteration** :
    The Self-Taught Reasoner (STaR) [Zelikman et al., 2022]
    frames reasoning as a bootstrapping loop: a pretrained model first samples diverse chains-of-thought (CoTs),
    keeps only those that lead to correct answers, and then finetunes on these “expert” traces. Iterating this
    cycle can improve the LM’s reasoning capabilities and solve rate. STaR demonstrated that this version of
    expert iteration [Anthony et al., 2017] using automatic, string match–based verification of generated answers
    can bootstrap reasoning skills without human-written reasoning traces.

- **Reasoning RL with verified rewards, o1 and R1**:
    OpenAI o1, Deepseek R1, KIMI1.5, using **Policy gradient methods** to  train on math and code tasks where
    string matching or unit tests verify correctness.
    

## SFT
Through SFT, we observed that we can improve the performance of our SFT model by filtering out bad examples from the SFT data.

## Expert Iteration


## Policy gradient
for LM, given $s_t$ as an input or state, $a_t$ as an output under the state, we can view LM as a *categorical stochastic policy*.

$$ a_t \sim \pi_\theta(·|s_t) ,  \pi_\theta(a_t | s_t) = [softmax(f_\theta(s_t))]_{a_t}$$

### Trajectory
We call $s_{t+1} = s_t \parallel a_t$ as a trajectory (or episodes or rollouts).

### Rewards and Return

A scalar reward $r_t = R(s_t, a_t)$ judges the immediate quality of the action taken at state $s_t$. 

$$r_T = R(s_T, a_T) := \begin{cases} 1 & \text{if the trajectory } s_T \parallel a_T \text{ matches the ground-truth according to our reward function} \\ 0 & \text{otherwise.} \end{cases}$$

 *finite-horizon undiscounted returns*:

$$R(\tau) := \sum_{t=0}^T r_t $$

and *infinite-horizon discounted returns*:

$$R(\tau) := \sum_{t=0}^\infty \gamma^t r_t, \quad 0 < \gamma < 1. $$

In our case, we will use the undiscounted formulation since episodes have a natural termination point (end-of-text or max generation length).
The objective of the agent is to maximize the expected return:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)], $$

leading to the optimization problem:

$$\theta^* = \arg \max_\theta J(\theta).$$

In one word, **最大化策略的回报期望**

### Vanilla Policy Gradient

Next, let us attempt to learn policy parameters $\theta$ with gradient ascent on the expected return:

$$\theta_{k+1} = \theta_k + \alpha \nabla_\theta J(\theta_k). $$

The core identity that we will use to do this is the REINFORCE policy gradient, shown below:

$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t | s_t) R(\tau) \right].$$

### Deriving the policy gradient
How did we get this equation? For completeness, we will give a derivation of this identity below. We will make use of a few identities.

1. The probability of a trajectory is given by
   $$P(\tau | \theta) = \rho_0(s_0) \prod_{t=0}^T P(s_{t+1} | s_t, a_t) \pi_\theta(a_t | s_t) $$
   Therefore, the log-probability of a trajectory is:
   $$\log P(\tau | \theta) = \log \rho_0(s_0) + \sum_{t=0}^T [\log P(s_{t+1} | s_t, a_t) + \log \pi_\theta(a_t | s_t)] $$

2. The log-derivative trick:
   $$\nabla_\theta P = P \nabla_\theta \log P $$

3. The environment terms are constant in $\theta$. $\rho_0, P(\cdot | \cdot)$ and $R(\tau)$ do not depend on the policy parameters, so
   $$\nabla_\theta \rho_0 = \nabla_\theta P = \nabla_\theta R(\tau) = 0 $$

Applying the facts above:
$$
\begin{aligned}
\nabla_\theta J(\theta) &= \nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)] \\
&= \nabla_\theta \sum_{\tau} P(\tau | \theta) R(\tau) \\
&= \sum_{\tau} \nabla_\theta P(\tau | \theta) R(\tau) \\
&= \sum_{\tau} P(\tau | \theta) \nabla_\theta \log P(\tau | \theta) R(\tau) \quad \text{(Log-derivative trick)} \\
&= \mathbb{E}_{\tau \sim \pi_\theta} [\nabla_\theta \log P(\tau | \theta) R(\tau)] \\
&= \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) R(\tau) \right]
\end{aligned}
$$

Intuitively, this gradient will increase the log probability of every action in a trajectory that has high return, and decrease them otherwise.

**Sample estimate of the gradient.** Given a batch of $N$ rollouts $\mathcal{D} = \{\tau^{(i)}\}_{i=1}^N$ collected by sampling a starting state $s_0^{(i)} \sim \rho_0(s_0)$ and then running the policy $\pi_\theta$ in the environment, we form an unbiased estimator of the gradient as:

$$\hat{g} = \frac{1}{N} \sum_{i=1}^N \sum_{t=0}^{T_i} \nabla_\theta \log \pi_\theta(a_t^{(i)} | s_t^{(i)}) R(\tau^{(i)})$$

**where $T_i$ is the length of the $i$-th trajectory, which may vary across samples**. This vector is used in the gradient-ascent update: $\theta \leftarrow \theta + \alpha \hat{g}$.

## Policy Gradient Baseline
当我们引入了Vanilla Policy Gradient（REINFORCE）基础策略梯度算法后，我们开始考虑它的不足。一个很大的问题是， 策略梯度不稳定，方差（variance）非常大，这会导致收敛缓慢。

为了减小方差，一个常用的技巧是引入 **Baseline (基准)** $b(s_t)$。

### 引入 Baseline 的策略梯度
带 Baseline 的策略梯度公式如下：
$$B = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t | s_t) (R(\tau) - b(s_t)) \right]$$

一个合理的 Baseline 选择是**状态值函数 (On-policy value function)** $V^\pi(s) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau) | s_t = s]$。直观上，$(R(\tau) - V^\pi(s_t))$ 衡量了实际观测到的回报比预期的好多少。

### 无偏性证明
只要 Baseline $b(s_t)$ 仅依赖于状态 $s_t$ 而不依赖于具体的动作 $a_t$，引入它就不会产生偏差。我们可以通过展开期望来证明这一点：
$$B = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t | s_t) R(\tau) \right] - \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t | s_t) b(s_t) \right]$$

我们要证明减号后面的那一项等于 0。根据全期望公式，我们可以把那一项重写为：
$$\mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t | s_t) b(s_t) \right] = \sum_{t=0}^T \mathbb{E}_{s_t} \left[ b(s_t) \mathbb{E}_{a_t \sim \pi_\theta(\cdot | s_t)} [\nabla_\theta \log \pi_\theta(a_t | s_t)] \right]$$

这里最关键的数学恒等式是：**Score Function 的期望为 0**。

#### Score Function 期望为 0 的证明
对于任何概率分布 $P_\theta(x)$，关于其参数 $\theta$ 的记分函数 (Score Function) $\nabla_\theta \log P_\theta(x)$ 在原分布下的期望恒等于 0：
$$\mathbb{E}_{x \sim P_\theta} [\nabla_\theta \log P_\theta(x)] = 0$$

**证明过程如下：**
$$
\begin{aligned}
\mathbb{E}_{x \sim P_\theta} [\nabla_\theta \log P_\theta(x)] &= \int P_\theta(x) \nabla_\theta \log P_\theta(x) dx \\
&= \int P_\theta(x) \frac{\nabla_\theta P_\theta(x)}{P_\theta(x)} dx \\
&= \int \nabla_\theta P_\theta(x) dx \\
&= \nabla_\theta \int P_\theta(x) dx \\
&= \nabla_\theta (1) \\
&= 0
\end{aligned}
$$

由于 $\mathbb{E}_{a_t \sim \pi_\theta(\cdot | s_t)} [\nabla_\theta \log \pi_\theta(a_t | s_t)] = 0$，整个基准项的期望确实为 0。
这意味着 $B = \nabla_\theta J(\theta)$，即引入基准后，梯度的期望值保持不变（无偏），但方差可以显著降低。

### pg_loss

pg_loss is not a loss in the canonical sense—it’s not meaningful to report pg_loss on the train or
validation set as an evaluation metric, and **a good validation pg_loss doesn’t indicate that our model is
generalizing well**. Instead, it is a surrogate objective function whose gradient is the policy gradient estimator. 
$$\text{pg\_loss}(\theta) = \frac{1}{N} \sum_{i=1}^N \sum_{t=0}^{T_i} \log \pi_\theta(a_t^{(i)} | s_t^{(i)}) (R(\tau^{(i)}) - b(s_t^{(i)}))$$

**When doing RL, you should always log and report train and validation rewards. These are the
“meaningful” evaluation metrics and what we are attempting to optimize with policy gradient method**

## Off-Policy Policy Gradient
之前的vanilla policy以及添加了baseline的方法都是典型的**on-policy**方法，即每次更新策略参数$\theta$后，都需要根据policy model新的 $\theta$ 重新收集一批新的数据来计算梯度。这导致了数据利用率低的问题。为了解决这个问题，我们可以使用**off-policy**方法.

较为典型的有 PPO 和 GRPO ，他们都是通过收集旧policy的 rollouts来更新新policy的参数。
$$\hat{g}_{off\_policy} = \frac{1}{N} \sum_{i=1}^N \sum_{t=0}^{T_i} \dfrac{\pi_\theta(a_t^{(i)} | s_t^{(i)})}{\pi_{\theta_{old}}(a_t^{(i)} | s_t^{(i)})} \nabla_\theta \log \pi_\theta(a_t^{(i)} | s_t^{(i)}) R(\tau^{(i)})$$

当$\pi_\theta$ 和 $\pi_{\theta_{old}}$相差不大时，上面的权重系数是合理的

## Group Realtive Plocy Optimization(GRPO)
Baseline设置：对于一个问题，我们采样多条rollouts，计算它们的group-normalized reward. For a question $q$ and group outputs ${o^{(i)}}^G_{i=1} \sim \pi_\theta(\cdot | q)$, let $r^{(i)} = R(q, o^{(i)})$.
$$A = r_{norm}^{(i)} = \frac{r^{(i)} - mean_ r^{(j)}}{std_j r^{(j)} + \epsilon}$$

对于同一个output $i$中的所有token, 他们的$A^{(i)}$是相同的

### GRPO-Clip Objective

Let us first write out the full GRPO-Clip objective, and then we can build some intuition on what the clipping does:

$$J_{\text{GRPO-Clip}}(\theta) = \mathbb{E}_{q \sim \mathcal{D}, \{o^{(i)}\}_{i=1}^G \sim \pi_{\theta}(\cdot|q)} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o^{(i)}|} \sum_{t=1}^{|o^{(i)}|} \underbrace{\min \left( \frac{\pi_\theta(o_t^{(i)} | q, o_{<t}^{(i)})}{\pi_{\theta_{\text{old}}}(o_t^{(i)} | q, o_{<t}^{(i)})} A^{(i)}, \text{clip} \left( \frac{\pi_\theta(o_t^{(i)} | q, o_{<t}^{(i)})}{\pi_{\theta_{\text{old}}}(o_t^{(i)} | q, o_{<t}^{(i)})}, 1-\epsilon, 1+\epsilon \right) A^{(i)} \right)}_{\text{per-token objective}} \right]$$

The hyperparameter $\epsilon > 0$ controls how much the policy can change. To see this, we can rewrite the per-token objective in a more intuitive way following Achiam [2018a, b]. Define the function

$$g(\epsilon, A^{(i)}) = \begin{cases} (1+\epsilon)A^{(i)} & \text{if } A^{(i)} \ge 0 \\ (1-\epsilon)A^{(i)} & \text{if } A^{(i)} < 0 \end{cases}$$

We can rewrite the per-token objective as

$$\text{per-token objective} = \min \left( \frac{\pi_\theta(o_t^{(i)} | q, o_{<t}^{(i)})}{\pi_{\theta_{\text{old}}}(o_t^{(i)} | q, o_{<t}^{(i)})} A^{(i)}, g(\epsilon, A^{(i)}) \right)$$

We can now reason by cases. When the advantage $A^{(i)}$ is positive, the per-token objective simplifies to

$$\text{per-token objective} = \min \left( \frac{\pi_\theta(o_t^{(i)} | q, o_{<t}^{(i)})}{\pi_{\theta_{\text{old}}}(o_t^{(i)} | q, o_{<t}^{(i)})}, 1+\epsilon \right) A^{(i)}$$

Since $A^{(i)} > 0$, the objective goes up if the action $o_t^{(i)}$ becomes more likely under $\pi_\theta$, i.e., if $\pi_\theta(o_t^{(i)} | q, o_{<t}^{(i)})$ increases. The clipping with $\min$ limits how much the objective can increase: once $\pi_\theta(o_t^{(i)} | q, o_{<t}^{(i)}) > (1+\epsilon)\pi_{\theta_{\text{old}}}(o_t^{(i)} | q, o_{<t}^{(i)})$, this per-token objective hits its maximum value of $(1+\epsilon)A^{(i)}$. So, the policy $\pi_\theta$ is not incentivized to go very far from the old policy $\pi_{\theta_{\text{old}}}$.

Analogously, when the advantage $A^{(i)}$ is negative, the model tries to drive down $\pi_\theta(o_t^{(i)} | q, o_{<t}^{(i)})$, but is not incentivized to decrease it below $(1-\epsilon)\pi_{\theta_{\text{old}}}(o_t^{(i)} | q, o_{<t}^{(i)})$ (refer to Achiam [2018b] for the full argument).