import { defineConfig } from "vitepress";

export default defineConfig({
  title: "Evan Feng",
  description: "",
  lang: "zh-CN",
  markdown: {
    math: true
  },
  themeConfig: {
    nav: [
      {
        text: "Notes",
        items: [
          { text: "首页", link: "/" },
          { text: "Algorithm", link: "/algorithm-design-and-analysis/lecture1" },
          { text: "CS 336", link: "/CS336/lecture1" },
          { text: "CSAPP", link: "/csapp/Chapter3" },
          { text: "Computer Networking", link: "/Computer Networking/Chapter1" },
          { text: "ICE2603", link: "/ICE2603/Chapter5" },
          { text: "Diary", link: "/diary/0424" }
        ]
      },
      {
        text: "Blogs",
        items: [
          {
            text: "2026-05-14 - ARPO Notes",
            link: "/blogs/ARPO/ARPO_notes"
          },
          {
            text: "2026-05-14 - Interview Review",
            link: "/blogs/interview-review/review"
          },
          {
            text: "2026-05-14 - Attention Sink Survey",
            link: "/blogs/Survey-on-attention-sink/attention_sink_survey"
          },
          { text: "全部博客", link: "/blogs/" },
          {
            text: "2026-05-09 · Glitch Token and MiniMax-M2",
            link: "/blogs/Glitch-token-minimax-M2/Glitch_token_minimax-M2"
          },
          {
            text: "2026-05-05 · An Open Problem in RL: Beyond Verifiable Rewards",
            link: "/blogs/blogs-reflection/rl-open-problem"
          },
          {
            text: "2026-04-29 · Training and Inference Mismatch: IcePop",
            link: "/blogs/IcePop/Training%20and%20Inference%20Mismatch%20--%20IcePop"
          },
          {
            text: "2026-04-24 · GLM-5 Notes",
            link: "/blogs/GLM-5-notes/GLM_Notes"
          },
          {
            text: "2026-04-23 · GEMM: SIMD Implement on CPU and Triton Implemetn on GPU",
            link: "/blogs/gemm-notes/gemm-notes"
          }
        ]
      }
    ],
    sidebar: {
      "/algorithm-design-and-analysis/": [
        {
          text: "Algorithm Design and Analysis",
          items: [
            { text: "Lecture 1: Turing Machine & Decidability", link: "/algorithm-design-and-analysis/lecture1" },
            { text: "Lecture 2: Divide and Conquer and Running Time analysis", link: "/algorithm-design-and-analysis/lecture2" },
            { text: "Lecture 3: Master Theorem", link: "/algorithm-design-and-analysis/lecture3" },
            { text: "Lecture 4: FFT", link: "/algorithm-design-and-analysis/lecture4" },
            { text: "Lecture 5: Graph & SCC", link: "/algorithm-design-and-analysis/lecture5" },
            { text: "Lecture 6: Shortest Path & Fibonacci Heap", link: "/algorithm-design-and-analysis/lecture6" },
            { text: "Lecture 7: Negative Weight Shortest Path & Bellman-Ford", link: "/algorithm-design-and-analysis/lecture7" },
            { text: "Lecture 8: Greedy & MST", link: "/algorithm-design-and-analysis/lecture8" },
            { text: "Lecture 9: More Greedy Algorithms", link: "/algorithm-design-and-analysis/lecture9" },
            { text: "Lecture 10: Dynamic Programming", link: "/algorithm-design-and-analysis/lecture10" },
            { text: "Lecture 11: Knapsack & LIS Optimization", link: "/algorithm-design-and-analysis/lecture11" },
            { text: "Lecture 12: Manufacturing Cost DP", link: "/algorithm-design-and-analysis/lecture12" },
            { text: "Lecture 13: Tree DP & Treewidth", link: "/algorithm-design-and-analysis/lecture13" }
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
      "/ICE2603/": [
        {
          text: "ICE2603",
          items: [
            { text: "Chapter 5: Memory Hierarchy", link: "/ICE2603/Chapter5" }
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
            { text: "Assignment 2", link: "/CS336/assignment2" },
            { text: "Assignment 5", link: "/CS336/assignment5" }
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
      ],
      "/csapp/": [
        {
          text: "CSAPP",
          items: [
            { text: "第03章 计算机的算术运算 (lecture3)", link: "/csapp/lecture3" },
            { text: "Chapter 3 程序的机器级表示", link: "/csapp/Chapter3" }
          ]
        }
      ],
      "/Computer Networking/": [
        {
          text: "Computer Networking",
          items: [
            { text: "第一章：计算机网络和因特网", link: "/Computer Networking/Chapter1" },
            { text: "第二章：应用层", link: "/Computer Networking/Chapter2" },
            { text: "第三章：传输层", link: "/Computer Networking/Chapter3" },
            { text: "Chapter 4: Network Layer Data Plane", link: "/Computer Networking/Chapter4" }
          ]
        }
      ],
      "/diary/": [
        {
          text: "Diary",
          items: [
            { text: "2026-04-24", link: "/diary/0424" }
          ]
        }
      ]
    }
  }
});
