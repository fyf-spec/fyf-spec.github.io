<script setup lang="ts">
import { computed } from "vue";

type VisualTopic =
  | "manufacturing-dp"
  | "tree-dp"
  | "network-data-plane"
  | "memory-hierarchy"
  | "arpo"
  | "interview-review"
  | "diary-review"
  | "attention-sink";

type VisualStep = {
  label: string;
  detail: string;
};

type VisualBar = {
  label: string;
  value: string;
  width: number;
};

type VisualConfig = {
  eyebrow: string;
  title: string;
  summary: string;
  steps: VisualStep[];
  bars: VisualBar[];
  badges: string[];
};

const props = defineProps<{
  topic: VisualTopic;
}>();

const visuals: Record<VisualTopic, VisualConfig> = {
  "manufacturing-dp": {
    eyebrow: "Convex Hull Optimization",
    title: "Manufacturing DP",
    summary: "Prefix sums turn every split point into a line candidate, while the deque keeps only the useful hull.",
    steps: [
      { label: "Prefix", detail: "s(i)" },
      { label: "Query", detail: "best j" },
      { label: "Insert", detail: "new point" },
      { label: "Answer", detail: "f[n]" }
    ],
    bars: [
      { label: "Fixed cost", value: "C", width: 42 },
      { label: "Square term", value: "sum^2", width: 82 },
      { label: "Hull motion", value: "monotone", width: 68 }
    ],
    badges: ["O(n)", "deque", "lower hull"]
  },
  "tree-dp": {
    eyebrow: "Boundary State",
    title: "Tree DP",
    summary: "A local choice fixes the boundary, then child subtrees become independent DP pieces.",
    steps: [
      { label: "Root", detail: "orient tree" },
      { label: "Take", detail: "block child" },
      { label: "Skip", detail: "merge child" },
      { label: "Bag", detail: "treewidth k" }
    ],
    bars: [
      { label: "take/skip", value: "2 states", width: 38 },
      { label: "bag choices", value: "2^k", width: 74 },
      { label: "compatibility", value: "overlap", width: 58 }
    ],
    badges: ["postorder", "MIS", "FPT"]
  },
  "network-data-plane": {
    eyebrow: "Packet Path",
    title: "Network Layer",
    summary: "The data plane forwards each packet locally; the control plane decides the global routing state.",
    steps: [
      { label: "Ingress", detail: "packet in" },
      { label: "Lookup", detail: "table hit" },
      { label: "Switch", detail: "fabric" },
      { label: "Egress", detail: "link out" }
    ],
    bars: [
      { label: "best effort", value: "IP", width: 54 },
      { label: "routing state", value: "control", width: 72 },
      { label: "congestion", value: "feedback", width: 63 }
    ],
    badges: ["datagram", "router", "SDN"]
  },
  "memory-hierarchy": {
    eyebrow: "Locality Stack",
    title: "Memory Hierarchy",
    summary: "Fast storage is small and expensive; locality decides how often the processor can stay near the top.",
    steps: [
      { label: "Register", detail: "near core" },
      { label: "Cache", detail: "SRAM" },
      { label: "DRAM", detail: "main memory" },
      { label: "SSD", detail: "backing store" }
    ],
    bars: [
      { label: "latency gap", value: "wide", width: 86 },
      { label: "locality", value: "reuse", width: 70 },
      { label: "miss cost", value: "penalty", width: 78 }
    ],
    badges: ["cache line", "TLB", "write-back"]
  },
  arpo: {
    eyebrow: "Agentic RL",
    title: "ARPO Rollout",
    summary: "Tool feedback raises uncertainty; branching near high-entropy steps spends sampling where it matters.",
    steps: [
      { label: "Prompt", detail: "global path" },
      { label: "Tool", detail: "feedback" },
      { label: "Branch", detail: "entropy spike" },
      { label: "Credit", detail: "advantage" }
    ],
    bars: [
      { label: "entropy", value: "Delta H", width: 84 },
      { label: "branch budget", value: "adaptive", width: 68 },
      { label: "token credit", value: "shared", width: 57 }
    ],
    badges: ["tool use", "GRPO", "rollout"]
  },
  "interview-review": {
    eyebrow: "Question Map",
    title: "Interview Review",
    summary: "The review clusters system questions around model geometry, data efficiency, RL signals, and serving.",
    steps: [
      { label: "RoPE", detail: "position" },
      { label: "SFT", detail: "packing" },
      { label: "RL", detail: "reward" },
      { label: "KV Cache", detail: "serving" }
    ],
    bars: [
      { label: "geometry", value: "QK", width: 55 },
      { label: "data path", value: "SFT/RL", width: 73 },
      { label: "inference", value: "cache", width: 81 }
    ],
    badges: ["LLM", "systems", "post-training"]
  },
  "diary-review": {
    eyebrow: "Reflection Queue",
    title: "Daily Review",
    summary: "The note separates lessons, interview logistics, project experiments, and public writing into follow-up lanes.",
    steps: [
      { label: "Lesson", detail: "research depth" },
      { label: "Optiver", detail: "assessment" },
      { label: "Velotrace", detail: "experiments" },
      { label: "X", detail: "writing" }
    ],
    bars: [
      { label: "urgency", value: "near", width: 66 },
      { label: "depth", value: "high", width: 82 },
      { label: "follow-up", value: "queued", width: 59 }
    ],
    badges: ["review", "tasks", "research"]
  },
  "attention-sink": {
    eyebrow: "Attention Flow",
    title: "Attention Sink",
    summary: "Early tokens can absorb persistent attention mass, shaping how later queries distribute context.",
    steps: [
      { label: "BOS", detail: "anchor" },
      { label: "Sink", detail: "mass" },
      { label: "Window", detail: "local" },
      { label: "Query", detail: "retrieve" }
    ],
    bars: [
      { label: "sink weight", value: "stable", width: 80 },
      { label: "context drift", value: "tracked", width: 61 },
      { label: "mitigation", value: "design", width: 52 }
    ],
    badges: ["attention", "context", "survey"]
  }
};

const visual = computed(() => visuals[props.topic]);
const topicClass = computed(() => `note-visual-${props.topic}`);
</script>

<template>
  <section class="note-visual" :class="topicClass" aria-label="note visualization">
    <div class="note-visual-copy">
      <p class="note-visual-eyebrow">{{ visual.eyebrow }}</p>
      <h2>{{ visual.title }}</h2>
      <p>{{ visual.summary }}</p>
      <div class="note-visual-badges">
        <span v-for="badge in visual.badges" :key="badge">{{ badge }}</span>
      </div>
    </div>

    <div class="note-visual-map">
      <div class="note-visual-path" aria-hidden="true"></div>
      <div v-for="(step, index) in visual.steps" :key="step.label" class="note-visual-node">
        <span>{{ String(index + 1).padStart(2, "0") }}</span>
        <strong>{{ step.label }}</strong>
        <small>{{ step.detail }}</small>
      </div>
    </div>

    <div class="note-visual-bars">
      <div v-for="bar in visual.bars" :key="bar.label" class="note-visual-bar">
        <div class="note-visual-bar-label">
          <span>{{ bar.label }}</span>
          <strong>{{ bar.value }}</strong>
        </div>
        <div class="note-visual-bar-track">
          <i :style="{ width: `${bar.width}%` }"></i>
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.note-visual {
  --visual-a: #0f766e;
  --visual-b: #b7791f;
  --visual-c: #2563eb;
  --visual-d: #be123c;
  background:
    linear-gradient(135deg, rgba(15, 118, 110, 0.12), transparent 34%),
    linear-gradient(315deg, rgba(183, 121, 31, 0.14), transparent 38%),
    rgba(255, 255, 255, 0.24);
  border: 1px solid var(--fyf-border);
  border-radius: 8px;
  display: grid;
  gap: 1.1rem;
  grid-template-columns: minmax(0, 1.05fr) minmax(260px, 0.95fr);
  margin: 0.3rem 0 1.7rem;
  overflow: hidden;
  padding: 1.15rem;
  position: relative;
}

.dark .note-visual {
  background:
    linear-gradient(135deg, rgba(125, 211, 199, 0.14), transparent 34%),
    linear-gradient(315deg, rgba(246, 197, 111, 0.12), transparent 38%),
    rgba(255, 255, 255, 0.04);
}

.note-visual-tree-dp {
  --visual-a: #2563eb;
  --visual-b: #0f766e;
  --visual-c: #b7791f;
}

.note-visual-network-data-plane {
  --visual-a: #0891b2;
  --visual-b: #be123c;
  --visual-c: #b7791f;
}

.note-visual-memory-hierarchy {
  --visual-a: #b7791f;
  --visual-b: #0f766e;
  --visual-c: #2563eb;
}

.note-visual-arpo,
.note-visual-attention-sink {
  --visual-a: #be123c;
  --visual-b: #0f766e;
  --visual-c: #2563eb;
}

.note-visual-interview-review,
.note-visual-diary-review {
  --visual-a: #0f766e;
  --visual-b: #b7791f;
  --visual-c: #be123c;
}

.note-visual-copy {
  align-self: center;
  min-width: 0;
}

.note-visual-eyebrow {
  color: var(--visual-a);
  font-size: 0.74rem;
  font-weight: 800;
  letter-spacing: 0.1em;
  margin: 0 0 0.45rem;
  text-transform: uppercase;
}

.note-visual h2 {
  border: 0;
  font-family: var(--vp-font-family-heading);
  font-size: clamp(1.8rem, 4vw, 3rem);
  line-height: 1;
  margin: 0;
  padding: 0;
}

.note-visual-copy p:not(.note-visual-eyebrow) {
  color: var(--fyf-text-soft);
  line-height: 1.65;
  margin: 0.72rem 0 0;
  max-width: 52ch;
}

.note-visual-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  margin-top: 0.95rem;
}

.note-visual-badges span {
  background: color-mix(in srgb, var(--visual-a) 14%, transparent);
  border: 1px solid color-mix(in srgb, var(--visual-a) 30%, transparent);
  border-radius: 999px;
  color: var(--fyf-text);
  font-size: 0.78rem;
  font-weight: 700;
  padding: 0.32rem 0.55rem;
}

.note-visual-map {
  align-items: stretch;
  display: grid;
  gap: 0.65rem;
  grid-column: 2;
  grid-row: 1 / span 2;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  min-height: 260px;
  position: relative;
}

.note-visual-path {
  border: 2px dashed color-mix(in srgb, var(--visual-a) 44%, transparent);
  border-radius: 999px;
  inset: 16%;
  position: absolute;
  transform: rotate(-14deg);
}

.note-visual-node {
  align-content: center;
  background: color-mix(in srgb, var(--fyf-surface-strong) 74%, transparent);
  border: 1px solid var(--fyf-border);
  border-radius: 8px;
  display: grid;
  gap: 0.25rem;
  min-height: 118px;
  padding: 0.8rem;
  position: relative;
  z-index: 1;
}

.note-visual-node:nth-child(2),
.note-visual-node:nth-child(5) {
  transform: translateY(16px);
}

.note-visual-node span {
  color: var(--visual-b);
  font-size: 0.72rem;
  font-weight: 800;
  letter-spacing: 0.1em;
}

.note-visual-node strong {
  font-family: var(--vp-font-family-heading);
  font-size: 1.16rem;
  line-height: 1.1;
}

.note-visual-node small {
  color: var(--fyf-text-soft);
  font-size: 0.78rem;
}

.note-visual-node::after {
  animation: visualPulse 2.8s ease-in-out infinite;
  background: var(--visual-a);
  border-radius: 50%;
  content: "";
  height: 9px;
  position: absolute;
  right: 0.75rem;
  top: 0.75rem;
  width: 9px;
}

.note-visual-node:nth-child(3)::after {
  animation-delay: 0.3s;
  background: var(--visual-b);
}

.note-visual-node:nth-child(4)::after {
  animation-delay: 0.6s;
  background: var(--visual-c);
}

.note-visual-node:nth-child(5)::after {
  animation-delay: 0.9s;
  background: var(--visual-d);
}

.note-visual-bars {
  display: grid;
  gap: 0.7rem;
}

.note-visual-bar {
  display: grid;
  gap: 0.35rem;
}

.note-visual-bar-label {
  align-items: center;
  display: flex;
  gap: 0.7rem;
  justify-content: space-between;
}

.note-visual-bar-label span {
  color: var(--fyf-text-soft);
  font-size: 0.82rem;
}

.note-visual-bar-label strong {
  color: var(--fyf-text);
  font-size: 0.82rem;
}

.note-visual-bar-track {
  background: rgba(128, 128, 128, 0.16);
  border-radius: 999px;
  height: 9px;
  overflow: hidden;
}

.note-visual-bar-track i {
  background: linear-gradient(90deg, var(--visual-a), var(--visual-b), var(--visual-c));
  border-radius: inherit;
  display: block;
  height: 100%;
}

@keyframes visualPulse {
  0%,
  100% {
    opacity: 0.45;
    transform: scale(0.84);
  }

  50% {
    opacity: 1;
    transform: scale(1.18);
  }
}

@media (max-width: 820px) {
  .note-visual {
    grid-template-columns: 1fr;
  }

  .note-visual-map {
    grid-column: auto;
    grid-row: auto;
    min-height: 240px;
  }
}

@media (max-width: 520px) {
  .note-visual {
    padding: 0.9rem;
  }

  .note-visual-map {
    grid-template-columns: 1fr;
  }

  .note-visual-path {
    display: none;
  }

  .note-visual-node,
  .note-visual-node:nth-child(2),
  .note-visual-node:nth-child(5) {
    min-height: 92px;
    transform: none;
  }
}
</style>
