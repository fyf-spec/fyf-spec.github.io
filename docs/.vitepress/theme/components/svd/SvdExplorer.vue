<script setup lang="ts">
import { computed, ref, watch } from "vue";
import TransformStage from "./TransformStage.vue";
import { getSvdCopy } from "./svdCopy.mjs";
import { makeSvdState } from "./svdMath.mjs";

type Vector2 = { x: number; y: number };

const props = withDefaults(defineProps<{
  locale?: "zh" | "en";
}>(), {
  locale: "zh"
});

const copy = computed(() => getSvdCopy(props.locale).explorer);
const thetaV = ref(30);
const sigma1 = ref(2.1);
const sigma2 = ref(0.65);
const thetaU = ref(-20);
const inputVector: Vector2 = { x: 0.9, y: 0.45 };

const state = computed(() => makeSvdState({
  thetaU: thetaU.value,
  thetaV: thetaV.value,
  sigma1: sigma1.value,
  sigma2: Math.min(sigma1.value, sigma2.value),
  reflectU: false,
  reflectV: false
}));

const stages = computed(() => copy.value.stages.map((stage, index) => ({
  ...stage,
  accent: ["#64748b", "#2563eb", "#b7791f", "#7c3aed"][index]
})));

const plotScale = computed(() => Math.min(60, 94 / Math.max(1, state.value.sigma1)));

watch(sigma1, (value) => {
  if (sigma2.value > value) sigma2.value = value;
});
</script>

<template>
  <section class="svd-explorer" :aria-label="copy.ariaLabel">
    <div class="svd-control-rail">
      <label class="svd-control is-v">
        <span><b>θ<sub>V</sub></b><output>{{ thetaV }}°</output></span>
        <input v-model.number="thetaV" type="range" min="-180" max="180" step="1" />
        <small>{{ copy.controls.basisV }}</small>
      </label>

      <label class="svd-control is-sigma">
        <span><b>σ<sub>1</sub></b><output>{{ sigma1.toFixed(2) }}</output></span>
        <input v-model.number="sigma1" type="range" min="0" max="3" step="0.01" />
        <small>{{ copy.controls.sigma1 }}</small>
      </label>

      <label class="svd-control is-sigma">
        <span><b>σ<sub>2</sub></b><output>{{ sigma2.toFixed(2) }}</output></span>
        <input v-model.number="sigma2" type="range" min="0" :max="sigma1" step="0.01" />
        <small>{{ copy.controls.sigma2 }}</small>
      </label>

      <label class="svd-control is-u">
        <span><b>θ<sub>U</sub></b><output>{{ thetaU }}°</output></span>
        <input v-model.number="thetaU" type="range" min="-180" max="180" step="1" />
        <small>{{ copy.controls.basisU }}</small>
      </label>
    </div>

    <div class="svd-stage-grid">
      <TransformStage
        v-for="(stage, index) in stages"
        :key="stage.formula"
        :formula="stage.formula"
        :aria-label="stage.ariaLabel"
        :matrix="state.stages[index]"
        :input-vector="inputVector"
        :accent="stage.accent"
        :plot-scale="plotScale"
      />
    </div>
  </section>
</template>

<style scoped>
.svd-explorer {
  --svd-blue: var(--fyf-accent);
  --svd-orange: #b7791f;
  --svd-purple: #7c3aed;
  border-bottom: 1px solid var(--fyf-border);
  border-top: 1px solid var(--fyf-border);
  box-sizing: border-box;
  color: var(--fyf-text);
  margin: 2rem 50% 2.75rem;
  padding: 1.15rem 0;
  transform: translateX(-50%);
  width: min(920px, calc(100vw - 40px));
}

.svd-control-rail {
  display: grid;
  gap: clamp(1rem, 2.5vw, 1.8rem);
  grid-template-columns: repeat(4, minmax(0, 1fr));
  padding: 0 0.25rem 1rem;
}

.svd-control {
  display: grid;
  gap: 0.32rem;
  min-width: 0;
}

.svd-control > span {
  align-items: baseline;
  display: flex;
  justify-content: space-between;
}

.svd-control b {
  font-family: var(--vp-font-family-heading);
  font-size: 0.9rem;
}

.svd-control output {
  color: var(--fyf-text);
  font-family: var(--fyf-font-mono, monospace);
  font-size: 0.7rem;
  font-variant-numeric: tabular-nums;
}

.svd-control small {
  color: var(--fyf-text-soft);
  font-family: var(--fyf-font-ui, var(--vp-font-family-base));
  font-size: 0.65rem;
}

.svd-control input {
  cursor: pointer;
  width: 100%;
}

.svd-control.is-v input { accent-color: var(--svd-blue); }
.svd-control.is-sigma input { accent-color: var(--svd-orange); }
.svd-control.is-u input { accent-color: var(--svd-purple); }

.svd-stage-grid {
  display: grid;
  gap: 0.85rem;
  grid-template-columns: repeat(4, minmax(0, 1fr));
}

input:focus-visible {
  outline: 2px solid var(--fyf-accent);
  outline-offset: 3px;
}

@media (max-width: 820px) {
  .svd-control-rail,
  .svd-stage-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 520px) {
  .svd-explorer {
    width: min(100%, calc(100vw - 24px));
  }

  .svd-control-rail {
    gap: 0.9rem 1.2rem;
  }

  .svd-stage-grid {
    gap: 0.7rem;
  }
}

@media (prefers-reduced-motion: reduce) {
  .svd-explorer * {
    transition-duration: 0.01ms !important;
  }
}
</style>
