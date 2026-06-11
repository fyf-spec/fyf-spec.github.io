export type BlogCategory = "technical" | "reflection" | "article";

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
};

export const noteSections: NoteSection[] = [
  {
    title: "Algorithm",
    label: "Course Notes",
    link: "/algorithm-design-and-analysis/lecture1",
    prefix: "/algorithm-design-and-analysis/",
    summary: "算法设计与分析课程笔记，覆盖分治、图算法、最短路与复杂度基础。",
    countLabel: "13 lectures"
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
    title: "CSAPP",
    label: "Systems Core",
    link: "/csapp/Chapter3",
    prefix: "/csapp/",
    summary: "程序的机器级表示、算术运算与系统基础知识回顾。",
    countLabel: "2 chapters"
  },
  {
    title: "Computer Networking",
    label: "Networking",
    link: "/Computer%20Networking/Chapter1",
    prefix: "/Computer Networking/",
    summary: "自顶向下的计算机网络笔记，覆盖应用层、传输层与网络基础。",
    countLabel: "4 chapters"
  },
  {
    title: "ICE2603",
    label: "Computer Architecture",
    link: "/ICE2603/Chapter5",
    prefix: "/ICE2603/",
    summary: "Computer architecture notes, currently focused on memory hierarchy, cache behavior, virtual memory, and dependability.",
    countLabel: "1 chapter"
  },
  {
    title: "Diary",
    label: "Personal Notes",
    link: "/diary/0424",
    prefix: "/diary/",
    summary: "Short personal review notes and follow-up queues.",
    countLabel: "1 note"
  }
];

export const blogPosts: BlogPost[] = [
  {
    title: "ARPO Notes",
    link: "/blogs/ARPO/ARPO_notes",
    date: "2026-05-14",
    category: "technical",
    subtitle: "Notes on Agentic Reinforced Policy Optimization, entropy-triggered branching, and token credit assignment.",
    image: ""
  },
  {
    title: "Interview Review",
    link: "/blogs/interview-review/review",
    date: "2026-05-14",
    category: "reflection",
    subtitle: "Post-interview review notes on RoPE, SFT/RL token efficiency, and KV cache serving tradeoffs.",
    image: ""
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
  return blogPosts.find((post) => normalizePath(post.link) === normalized);
}

export function getNoteSectionByPath(path: string): NoteSection | undefined {
  const normalized = normalizePath(path);
  return noteSections.find((section) => normalized.startsWith(section.prefix));
}
