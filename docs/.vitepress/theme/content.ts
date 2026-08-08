export type BlogCategory = "technical" | "reflection" | "article";

export type BlogLanguageCode = "zh-CN" | "en-US";

export type BlogLanguage = {
  code: BlogLanguageCode;
  label: string;
  link: string;
  title: string;
};

export type NoteSection = {
  title: string;
  label: string;
  link: string;
  prefix: string;
  summary: string;
  countLabel: string;
};

export type BlogPost = {
  title: string;
  link: string;
  date: string;
  category: BlogCategory;
  subtitle: string;
  image: string;
  languages?: BlogLanguage[];
};

export const noteSections: NoteSection[] = [
  {
    title: "Algorithm",
    label: "Course Notes",
    link: "/algorithm-design-and-analysis/lecture1",
    prefix: "/algorithm-design-and-analysis/",
    summary: "算法设计与分析课程笔记，覆盖分治、图算法、网络流、线性规划与复杂度基础。",
    countLabel: "19 lectures"
  },
  {
    title: "CS 336",
    label: "LLM Notes",
    link: "/CS336/lecture1",
    prefix: "/CS336/",
    summary: "Stanford CS336 课程与作业笔记，聚焦大模型训练、推理与系统资源。",
    countLabel: "6 lectures"
  },
  {
    title: "Computer Networking",
    label: "Networking",
    link: "/Computer%20Networking/Chapter1",
    prefix: "/Computer Networking/",
    summary: "自顶向下的计算机网络笔记，覆盖应用层、传输层、网络层与链路层。",
    countLabel: "6 chapters"
  }
];

// Keep one catalog entry per article. Translations belong in `languages`, never as duplicate rows.
export const blogPosts: BlogPost[] = [
  {
    title: "Sparse Linear Attention: Viewing SWA from the Perspective of Linear Attention",
    link: "/blogs/raven-swa-linear-attention/raven-swa-linear-attention-en",
    date: "2026-08-09",
    category: "technical",
    subtitle: "Understanding SWA, state-based linear attention, and Raven through memory slots, FIFO, and sparse routing.",
    image: "/blogs/raven-swa-linear-attention/raven-routing-overview.png",
    languages: [
      {
        code: "zh-CN",
        label: "中文",
        link: "/blogs/raven-swa-linear-attention/raven-swa-linear-attention",
        title: "Sparse Linear Attention：从 Linear Attention 角度看 SWA"
      },
      {
        code: "en-US",
        label: "English",
        link: "/blogs/raven-swa-linear-attention/raven-swa-linear-attention-en",
        title: "Sparse Linear Attention: Viewing SWA from the Perspective of Linear Attention"
      }
    ]
  },
  {
    title: "K3's KDA Precision Problem: Why Lower-Bounded Decay Is Necessary",
    link: "/blogs/k3-kda-decay-tensor-core/k3-kda-decay-tensor-core-en",
    date: "2026-08-06",
    category: "technical",
    subtitle: "How decay bounds and 16-token tiles jointly control split-exponent range, Neumann inversion, and Tensor Core coverage.",
    image: "",
    languages: [
      {
        code: "zh-CN",
        label: "中文",
        link: "/blogs/k3-kda-decay-tensor-core/k3-kda-decay-tensor-core",
        title: "K3 的 KDA 精度问题：为什么要有 lower-bound decay"
      },
      {
        code: "en-US",
        label: "English",
        link: "/blogs/k3-kda-decay-tensor-core/k3-kda-decay-tensor-core-en",
        title: "K3's KDA Precision Problem: Why Lower-Bounded Decay Is Necessary"
      }
    ]
  },
  {
    title: "From Heuristic Learning to Auto-Architecture Search",
    link: "/blogs/heuristic-learning-auto-architecture-search/heuristic-learning-auto-architecture-search-en",
    date: "2026-08-05",
    category: "reflection",
    subtitle: "从积累外部启发式规则，到让研究系统参与改进模型架构本身。",
    image: "/blogs/heuristic-learning-auto-architecture-search/heuristic-learning-auto-architecture-search-v2.png",
    languages: [
      {
        code: "zh-CN",
        label: "中文",
        link: "/blogs/heuristic-learning-auto-architecture-search/heuristic-learning-auto-architecture-search",
        title: "从 Heuristic Learning 到 Auto-Architecture Search"
      },
      {
        code: "en-US",
        label: "English",
        link: "/blogs/heuristic-learning-auto-architecture-search/heuristic-learning-auto-architecture-search-en",
        title: "From Heuristic Learning to Auto-Architecture Search"
      }
    ]
  },
  {
    title: "After DSA: Two Routes to a Better Indexer",
    link: "/blogs/dsa-indexer-optimization/dsa-indexer-optimization-en",
    date: "2026-07-31",
    category: "technical",
    subtitle: "从 index score 的冗余模式，到重新设计候选生成器。",
    image: "/blogs/dsa-indexer-optimization/images/route-layer-23.png",
    languages: [
      {
        code: "zh-CN",
        label: "中文",
        link: "/blogs/dsa-indexer-optimization/dsa-indexer-optimization",
        title: "DSA 之后：Indexer 优化的两条路线"
      },
      {
        code: "en-US",
        label: "English",
        link: "/blogs/dsa-indexer-optimization/dsa-indexer-optimization-en",
        title: "After DSA: Two Routes to a Better Indexer"
      }
    ]
  },
  {
    title: "DeepSeek DSA FLOPs",
    link: "/blogs/DSA-flops/DSA_flops_notes_zh",
    date: "2026-07-28",
    category: "technical",
    subtitle: "沿用 MLA 的计数口径，推导 DSA 核心稀疏注意力与 Lightning Indexer 的 prefill、training 和 decode FLOPs。",
    image: ""
  },
  {
    title: "Linear Attention: From DeltaNet to KDA",
    link: "/blogs/linear-attention-deltanet-kda/linear-attention-deltanet-kda-en",
    date: "2026-07-18",
    category: "technical",
    subtitle: "从固定状态线性注意力出发，推导 DeltaNet、Parallel DeltaNet、Gated DeltaNet、KDA 与 Gated DeltaNet-2。",
    image: "",
    languages: [
      {
        code: "zh-CN",
        label: "中文",
        link: "/blogs/linear-attention-deltanet-kda/linear-attention-deltanet-kda",
        title: "Linear Attention：From DeltaNet to KDA"
      },
      {
        code: "en-US",
        label: "English",
        link: "/blogs/linear-attention-deltanet-kda/linear-attention-deltanet-kda-en",
        title: "Linear Attention: From DeltaNet to KDA"
      }
    ]
  },
  {
    title: "Singular Values and Spectral Decomposition",
    link: "/blogs/singular-values-spectrum/singular-values-spectrum-en",
    date: "2026-07-12",
    category: "technical",
    subtitle: "从方向伸缩出发，理解奇异值谱、低秩近似、谱分解与谱范数。",
    image: "",
    languages: [
      {
        code: "zh-CN",
        label: "中文",
        link: "/blogs/singular-values-spectrum/singular-values-spectrum",
        title: "Singular Values and Spectral Decomposition"
      },
      {
        code: "en-US",
        label: "English",
        link: "/blogs/singular-values-spectrum/singular-values-spectrum-en",
        title: "Singular Values and Spectral Decomposition"
      }
    ]
  },
  {
    title: "Notes on MLA FLOPs, RoPE, Query Compression, and MTP",
    link: "/blogs/MLA-flops/MLA_flops_notes",
    date: "2026-07-06",
    category: "technical",
    subtitle: "A technical note on MLA FLOPs, matrix multiplication order, RoPE decoupling, query compression, and why MTP cannot freely reuse the main MLA cache.",
    image: "/blogs/MLA-flops/MLA_baseline.png",
    languages: [
      {
        code: "en-US",
        label: "English",
        link: "/blogs/MLA-flops/MLA_flops_notes",
        title: "Notes on MLA FLOPs, RoPE, Query Compression, and MTP"
      },
      {
        code: "zh-CN",
        label: "中文",
        link: "/blogs/MLA-flops/MLA_flops_notes_zh",
        title: "MLA FLOPs、RoPE、Query Compression 与 MTP 笔记"
      }
    ]
  },
  {
    title: "Attention Sink Survey",
    link: "/blogs/Survey-on-attention-sink/attention_sink_survey",
    date: "2026-05-14",
    category: "technical",
    subtitle: "A visual scaffold for attention sink notes.",
    image: ""
  },
  {
    title: "Glitch Token and MiniMax-M2",
    link: "/blogs/Glitch-token-minimax-M2/Glitch_token_minimax-M2",
    date: "2026-05-09",
    category: "technical",
    subtitle: "My notes on the MiniMax-M2 glitch token failure, LM head drift, and post-training mismatch.",
    image: ""
  },
  {
    title: "An Open Problem in RL: Beyond Verifiable Rewards",
    link: "/blogs/blogs-reflection/rl-open-problem",
    date: "2026-05-05",
    category: "reflection",
    subtitle: "A reflection on how RL might apply to research, agent planning, and writing when exact rewards and formal answers are unavailable.",
    image: ""
  },
  {
    title: "Training and Inference Mismatch: IcePop",
    link: "/blogs/IcePop/Training%20and%20Inference%20Mismatch%20--%20IcePop",
    date: "2026-04-29",
    category: "technical",
    subtitle: "A short technical note on IcePop, token-level mismatch masking, and why MoE RL turns fragile when rollout and training engines disagree.",
    image: "/blogs/IcePop/image.png"
  },
  {
    title: "GLM-5 Notes",
    link: "/blogs/GLM-5-notes/GLM_Notes",
    date: "2026-04-24",
    category: "technical",
    subtitle: "GLM-5 的模型结构、训练基础设施、后训练与 Agentic RL 笔记。",
    image: "/blogs/GLM-5-notes/archi.png"
  },
  {
    title: "GEMM: SIMD Implement on CPU and Triton Implemetn on GPU",
    link: "/blogs/gemm-notes/gemm-notes",
    date: "2026-04-23",
    category: "technical",
    subtitle: "从 CPU cache blocking、packing 和 SIMD micro-kernel，到 Triton GEMM 的 pointer math 与 L2 reuse。",
    image: "/blogs/gemm-notes/CPU_GEMM.png"
  }
];

export const categoryLabels: Record<BlogCategory, string> = {
  technical: "Technical",
  reflection: "Reflection",
  article: "Article"
};

export function normalizePath(path: string): string {
  if (!path) {
    return "/";
  }

  const withoutQuery = path.split("?")[0]?.split("#")[0] ?? path;
  const withoutHtml = withoutQuery.replace(/\.html$/, "");
  const normalized = withoutHtml.endsWith("/") ? withoutHtml : `${withoutHtml}/`;
  return decodeURIComponent(normalized);
}

export function getPageKind(path: string): "home" | "blog-index" | "blog-post" | "note-doc" | "other" {
  const normalized = normalizePath(path);

  if (normalized === "/") {
    return "home";
  }

  if (normalized === "/blogs/") {
    return "blog-index";
  }

  if (normalized.startsWith("/blogs/")) {
    return "blog-post";
  }

  if (noteSections.some((section) => normalized.startsWith(section.prefix))) {
    return "note-doc";
  }

  return "other";
}

export function getBlogByPath(path: string): BlogPost | undefined {
  const normalized = normalizePath(path);
  return blogPosts.find((post) => {
    if (normalizePath(post.link) === normalized) {
      return true;
    }

    return post.languages?.some((language) => normalizePath(language.link) === normalized) ?? false;
  });
}

export function getBlogLanguageByPath(path: string): BlogLanguage | undefined {
  const normalized = normalizePath(path);
  const blog = getBlogByPath(path);

  if (!blog?.languages) {
    return undefined;
  }

  return blog.languages.find((language) => normalizePath(language.link) === normalized);
}

export function getNoteSectionByPath(path: string): NoteSection | undefined {
  const normalized = normalizePath(path);
  return noteSections.find((section) => normalized.startsWith(section.prefix));
}
