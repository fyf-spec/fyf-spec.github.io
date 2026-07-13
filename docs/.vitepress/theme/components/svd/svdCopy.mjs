const COPY = {
  zh: {
    explorer: {
      ariaLabel: "奇异值分解参数图",
      controls: {
        basisV: "输入基角度",
        sigma1: "第一奇异值",
        sigma2: "第二奇异值",
        basisU: "输出基角度"
      },
      stages: [
        { formula: "x", ariaLabel: "输入向量 x" },
        { formula: "Vᵀx", ariaLabel: "向量在 V 基下的坐标" },
        { formula: "ΣVᵀx", ariaLabel: "沿奇异方向伸缩后的向量" },
        { formula: "UΣVᵀx = Ax", ariaLabel: "矩阵 A 的输出向量" }
      ]
    },
    spectrum: {
      ariaLabel: "奇异值谱参数图",
      decay: "衰减 α",
      threshold: "阈值 ε",
      truncation: "截断阶数 k",
      effectiveRank: "有效秩",
      retainedEnergy: "保留能量",
      truncationError: "截断误差"
    },
    bars: {
      chartAria: "奇异值谱柱状图",
      singularValue: "奇异值"
    },
    stage: {
      currentVector: "当前向量坐标"
    }
  },
  en: {
    explorer: {
      ariaLabel: "Singular value decomposition controls and plots",
      controls: {
        basisV: "Input-basis angle",
        sigma1: "First singular value",
        sigma2: "Second singular value",
        basisU: "Output-basis angle"
      },
      stages: [
        { formula: "x", ariaLabel: "Input vector x" },
        { formula: "Vᵀx", ariaLabel: "Coordinates of the vector in the V basis" },
        { formula: "ΣVᵀx", ariaLabel: "Vector after scaling along singular directions" },
        { formula: "UΣVᵀx = Ax", ariaLabel: "Output vector of matrix A" }
      ]
    },
    spectrum: {
      ariaLabel: "Singular-value spectrum controls and chart",
      decay: "Decay α",
      threshold: "Threshold ε",
      truncation: "Truncation rank k",
      effectiveRank: "Effective rank",
      retainedEnergy: "Retained energy",
      truncationError: "Truncation error"
    },
    bars: {
      chartAria: "Singular-value spectrum bar chart",
      singularValue: "Singular value"
    },
    stage: {
      currentVector: "Current vector coordinates"
    }
  }
};

export function getSvdCopy(locale = "zh") {
  return COPY[locale] ?? COPY.zh;
}
