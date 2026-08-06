---
title: "From Heuristic Learning to Auto-Architecture Search"
description: "If a research system can accumulate reusable heuristics, can it eventually improve the model architecture itself?"
date: 2026-08-05
lang: en-US
outline: deep
---

# From Heuristic Learning to Auto-Architecture Search

![From heuristic learning and reusable rules to auto-architecture search](./heuristic-learning-auto-architecture-search-v2.png)

Over the past few days, I have been reading about Jiayi Weng's Heuristic Learning, which pushes Karpathy's autoresearch to a higher level: instead of merely completing one experiment, the system learns reusable heuristics and writes them back into the codebase. It can be understood as a form of meta-learning outside the research process—what is continually updated is not only the model parameters, but also the explicit knowledge that guides the next round of research.

Back in the era of traditional software engineering, people had already considered accomplishing tasks by continually adding heuristic rules. Under the technological constraints of the time, however, maintaining such a codebase was practically impossible. Large models and coding agents have now given us the tools to revisit this idea.

This, in turn, leads us to a longer-term question: can an external code system ever become as expressive and flexible as a neural network? Large language models have already demonstrated the capacity of neural networks to represent complex patterns. Could we transplant Heuristic Learning from an external codebase into the architecture of an LLM, so that the object being continually optimized expands from “model parameters” to an “external code system,” and eventually to the “model architecture” itself?

If this direction works, Auto-Architecture Search would mean more than selecting an architecture from a predefined set of candidates. The model would participate in proposing, evaluating, and accumulating architectural improvements. Reaching that point, however, requires solving at least four problems:

1. **How can architectural changes receive fast and accurate feedback?** Validating a new architecture usually requires training, which is expensive and slow, while the result can still be affected by data and training noise. This differs from Heuristic Learning, where standardized compilers and efficient CPUs and GPUs can provide rapid feedback. Coding agents take advantage of these tools and can therefore iterate quickly. What would the equivalent tool be for Auto-Architecture Search?
2. **Which part of the architecture should we iterate on?** The space of parameters and structures is effectively unbounded. The system must decide which modules are worth changing and at what granularity.
3. **How can the search converge on a family of effective architectures?** Many candidates are not worth a full trial. Without reliable priors, filtering, and attribution, they may obscure the directions that actually have potential.
4. **Which knowledge should remain explicit, and which should be stored in neural parameters?** Explicit rules are easier to inspect, reuse, and modify; neural parameters are better at absorbing patterns that resist clean formalization. Where we draw this boundary will shape how the system learns and evolves.

If Auto-Architecture Search can truly work, the result would be more than a model that optimizes parameters within a fixed architecture. It would be a research system capable of gradually rewriting its own learning machinery. That may bring us a little closer to genuine intelligence. SSI's recent work and Jeff Dean's departure from Google to found *Discovery Loop* are both bets on self-iteration and continual learning. Still, I find self-iteration at the level of model architecture particularly fascinating.
