---
title: "Glitch Token and MiniMax-M2"
description: My notes on the MiniMax-M2 glitch token failure, LM head drift, and post-training mismatch.
date: 2026-05-09
outline: deep
aside: left
---

# Glitch Token and MiniMax-M2

MiniMax-M2 had a visible failure around `马嘉祺`: it could often describe the person, but the generated name was unstable， `嘉琪` was replaced by `俊杰` or `佳琪`, the token was avoided by llm purposely. MiniMax-M2.7 was later reported to fix this case. Here are some possible explanations of this interesting phenomenon.

The team found that: token of `嘉祺` has high cos cimilarity of `lm_head` logits after pre-training, but switched extremely low after SFT. Then somnething must happen in SFT(post-training) !


## LM Head Drift

The output logit for token $i$ is:

$$
z_i = h^\top w_i + b_i
$$

For cross entropy, the gradient on $w_i$ is approximately:

$$
\frac{\partial L}{\partial w_i}
=
(p_i - \mathbf{1}_{i=y})h
$$

Target tokens receive strong updates. Non-target tokens receive weaker updates, especially when $p_i$ is small. **During SFT/RLHF, common instruction tokens, format tokens, punctuation, and high-frequency text are updated often. Rare characters and rare name paths receive much less direct correction.**


At the same time, the transformer body continues to move:

$$
h_{\text{pretrain}} \rightarrow h_{\text{posttrain}}
$$

The issue may be relative drift:

$$
\|\Delta w_{\text{rare}}\| \ll \|\Delta h\|
$$

The logit change can be expanded as:

$$
\Delta z_i
=
(\Delta h)^\top w_i
+
h^\top \Delta w_i
+
(\Delta h)^\top \Delta w_i
$$

For a rare token, $\Delta w_i$ can be small, so $(\Delta h)^\top w_i$ may dominate. **My short version of the mechanism is: representation drift plus sparse supervision may produce logit anomalies.**

This also explains why "all parameters are trained together" is not enough. They are in one system, but they may not be updated at the same rate.

**POST-TRAINING DATA MATTERS !**

## Fix
One possible fix is to repeat or upsample SFT examples that contain rare-token vocabulary. This may reduce LM head drift by giving those rare output directions more post-training supervision. Similarly, increasing post-training coverage for Japanese, Russian, and other low-resource-language tokens may reduce both drift and glitch-token behavior.

The key question is whether this becomes a one-off patch or a broader eval category. **If the self-evolution loop turns a user-visible failure into systematic coverage, the fix is more valuable than the single corrected name.**


**This drift also has a linguistic analogy: a word can change meaning even if its written form stays unchanged,** because the surrounding semantic system has moved. This is similar to semantic drift across historical stages of a language.

## Architecture

Tokenizer quality matters, but I would not put the whole problem there. The final behavior may depend on tokenizer segmentation, pretraining coverage, post-training distribution, LM head calibration, MoE routing, quantization, sampling, and logit processors.

For MoE models, routing adds another discrete layer. A small router or numerical difference may change the hidden state, then change the ranking among similar low-frequency tokens.

**The metric I would add here is vocabulary health.** Math, code, and agent benchmarks do not tell us whether long-tail tokens are calibrated. Exact copying, Unicode, multilingual names, file paths, and low-frequency character sequences should be measured separately.

## Takeaway

Glitch-token behavior may not be only a tokenizer corner case. It may be a mismatch between continuous representation space and discrete symbol reproduction.

For ordinary generation, semantic tolerance hides the mismatch. For names and exact strings, it becomes visible.

## References

- MiniMax, [MiniMax M2.7: Early Echoes of Self-Evolution](https://www.minimax.io/news/minimax-m27-en), 2026-03-18.
- 机器之心 / 新浪科技转载, [刚刚，MiniMax直接让龙虾学会自我进化，也认识「马嘉祺」了](https://finance.sina.cn/tech/2026-03-18/detail-inhrmihm7408457.d.html), 2026-03-18.
- LINUX DO, [MiniMax-M2.5-highspeed能稳定识别马嘉祺](https://linux.do/t/topic/1777694), 2026-03-18.
- Yuxi Li et al., [Glitch Tokens in Large Language Models: Categorization Taxonomy and Effective Detection](https://arxiv.org/abs/2404.09894), 2024.
- Sander Land and Max Bartolo, [Fishing for Magikarp: Automatically Detecting Under-trained Tokens in Large Language Models](https://arxiv.org/abs/2405.05417), 2024.
- Zhibo Zhang et al., [GlitchProber: Advancing Effective Detection and Mitigation of Glitch Tokens in Large Language Models](https://arxiv.org/abs/2408.04905), 2024.
