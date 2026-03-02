import { defineConfig } from "vitepress";

export default defineConfig({
  title: "FYF Notes",
  description: "HPC 与 LLM 训练笔记",
  lang: "zh-CN",
  themeConfig: {
    nav: [
      { text: "首页", link: "/" },
      { text: "HPC", link: "/hpc/memory-bandwidth" },
      { text: "CS 336", link: "/CS336/lecture1" }
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
      "/CS336/": [
        {
          text: "CS 336 Lectures",
          items: [
            { text: "Lecture 1: Intro & Tokenization", link: "/CS336/lecture1" },
            { text: "Lecture 2: Resource Accounting", link: "/CS336/lecture2" },
            { text: "Lecture 3: Architecture & Hyperparams", link: "/CS336/lecture3" },
            { text: "Lecture 7: Parallelize Basics", link: "/CS336/lecture7" },
            { text: "Lecture 9: Scaling Laws 1", link: "/CS336/lecture9" }
          ]
        },
        {
          text: "CS 336 Assignments",
          items: [
            { text: "Assignment 1", link: "/CS336/assignment1" },
            { text: "Assignment 2", link: "/CS336/assignment2" }
          ]
        }
      ]
    }
  }
});
