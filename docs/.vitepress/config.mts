import { defineConfig } from "vitepress";

export default defineConfig({
  title: "FYF Notes",
  description: "HPC 与 LLM 训练笔记",
  lang: "zh-CN",
  markdown: {
    math: true
  },
  themeConfig: {
    nav: [
      { text: "首页", link: "/" },
      { text: "HPC", link: "/hpc/memory-bandwidth" },
      { text: "Algorithm", link: "/algorithm-design-and-analysis/lecture1" },
      { text: "CS 336", link: "/CS336/lecture1" },
      { text: "LeetCode", link: "/leetcode/169_MajorElement" }
    ],
    sidebar: {
      "/algorithm-design-and-analysis/": [
        {
          text: "Algorithm Design and Analysis",
          items: [
            { text: "Lecture 1: Turing Machine & Decidability", link: "/algorithm-design-and-analysis/lecture1" }
          ]
        }
      ],
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
            { text: "Lecture 9: Scaling Laws 1", link: "/CS336/lecture9" },
            { text: "Lecture 10: Inference", link: "/CS336/lecture10" }
          ]
        },
        {
          text: "CS 336 Assignments",
          items: [
            { text: "Assignment 1", link: "/CS336/assignment1" },
            { text: "Assignment 2", link: "/CS336/assignment2" }
          ]
        }
      ],
      "/leetcode/": [
        {
          text: "LeetCode",
          items: [
            { text: "169. 多数元素", link: "/leetcode/169_MajorElement" },
            { text: "189. 轮转数组", link: "/leetcode/189_Rotate" }
          ]
        }
      ]
    }
  }
});
