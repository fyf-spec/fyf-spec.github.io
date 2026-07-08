import { defineConfig } from "vitepress";

export default defineConfig({
  title: "Evan Feng",
  description: "",
  lang: "zh-CN",
  markdown: {
    math: true
  },
  srcExclude: [
    "blogs/ARPO/**",
    "blogs/interview-review/**",
    "csapp/**",
    "diary/**",
    "ICE2603/**"
  ],
  themeConfig: {
    nav: [
      {
        text: "Notes",
        items: [
          { text: "首页", link: "/" },
          { text: "Algorithm", link: "/algorithm-design-and-analysis/lecture1" },
          { text: "CS 336", link: "/CS336/lecture1" },
          { text: "Computer Networking", link: "/Computer Networking/Chapter1" }
        ]
      },
      {
        text: "Blogs",
        link: "/blogs/",
        activeMatch: "^/blogs/"
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
            { text: "Lecture 13: Tree DP & Treewidth", link: "/algorithm-design-and-analysis/lecture13" },
            { text: "Lecture 14: Network Flow Correctness", link: "/algorithm-design-and-analysis/lecture14" },
            { text: "Lecture 15: Hall's Theorem & Flow Running Time", link: "/algorithm-design-and-analysis/lecture15" },
            { text: "Lecture 16: Linear Programming & Duality", link: "/algorithm-design-and-analysis/lecture16" },
            { text: "Lecture 17: Applications of LP Duality", link: "/algorithm-design-and-analysis/lecture17" },
            { text: "Lecture 18: P, NP & NP-Completeness", link: "/algorithm-design-and-analysis/lecture18" },
            { text: "Lecture 19: More NP-Complete Problems", link: "/algorithm-design-and-analysis/lecture19" }
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
      "/Computer Networking/": [
        {
          text: "Computer Networking",
          items: [
            { text: "第一章：计算机网络和因特网", link: "/Computer Networking/Chapter1" },
            { text: "第二章：应用层", link: "/Computer Networking/Chapter2" },
            { text: "第三章：传输层", link: "/Computer Networking/Chapter3" },
            { text: "第四章：网络层：数据平面", link: "/Computer Networking/Chapter4" },
            { text: "第五章：网络层：控制平面", link: "/Computer Networking/Chapter5" },
            { text: "第六章：链路层和局域网", link: "/Computer Networking/Chapter6" }
          ]
        }
      ],
    }
  }
});
