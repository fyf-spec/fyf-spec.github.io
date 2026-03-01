import { defineConfig } from "vitepress";

export default defineConfig({
  title: "FYF Notes",
  description: "HPC 与 LLM 训练笔记",
  lang: "zh-CN",
  themeConfig: {
    nav: [
      { text: "首页", link: "/" },
      { text: "HPC", link: "/hpc/memory-bandwidth" },
      { text: "LLM Training", link: "/llm-training/parallel-methods" }
    ],
    sidebar: {
      "/hpc/": [
        {
          text: "HPC",
          items: [
            { text: "Memory Bandwidth", link: "/hpc/memory-bandwidth" },
            { text: "OpenBLAS Debug", link: "/hpc/openblas-debug" }
          ]
        }
      ],
      "/llm-training/": [
        {
          text: "LLM Training",
          items: [
            { text: "Parallel Methods", link: "/llm-training/parallel-methods" },
            { text: "ZeRO Optimizer", link: "/llm-training/zero-optimizer" }
          ]
        }
      ]
    }
  }
});
