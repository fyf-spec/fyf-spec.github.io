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
    countLabel: "10 lectures"
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
    title: "HPC",
    label: "Systems",
    link: "/hpc/memory-bandwidth",
    prefix: "/hpc/",
    summary: "高性能计算、性能分析、OpenBLAS 调试与底层工程实践。",
    countLabel: "2 notes"
  },
  {
    title: "LeetCode",
    label: "Practice",
    link: "/leetcode/169_MajorElement",
    prefix: "/leetcode/",
    summary: "题解、常见技巧与算法模板整理，偏重可复用思路。",
    countLabel: "2 problems"
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
    countLabel: "3 chapters"
  }
];

export const blogPosts: BlogPost[] = [
  {
    title: "GEMM Notes",
    link: "/blogs/gemm-notes",
    date: "2026-04-23",
    category: "technical",
    subtitle: "从 CPU cache blocking、packing 和 SIMD micro-kernel，到 Triton GEMM 的 pointer math 与 L2 reuse。",
    image: "/blogs/CPU_GEMM.png"
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
