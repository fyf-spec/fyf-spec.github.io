<script setup lang="ts">
import { computed, ref } from "vue";
import SpectrumBars from "./SpectrumBars.vue";
import { getSvdCopy } from "./svdCopy.mjs";
import { makeExponentialSpectrum, spectrumMetrics } from "./svdMath.mjs";

const props = withDefaults(defineProps<{
  locale?: "zh" | "en";
}>(), {
  locale: "zh"
});

const decay = ref(0.24);
const epsilon = ref(0.08);
const k = ref(5);
const copy = computed(() => getSvdCopy(props.locale).spectrum);
const values = computed(() => makeExponentialSpectrum(decay.value, 16));
const metrics = computed(() => spectrumMetrics(values.value, epsilon.value, k.value));

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function formatValue(value: number): string {
  if (value === 0) return "0";
  return value < 0.001 ? value.toExponential(2) : value.toFixed(3);
}
</script>

<template>
  <section class="spectrum-microscope" :aria-label="copy.ariaLabel">
    <div class="spectrum-controls">
      <label class="spectrum-control">
        <span>{{ copy.decay }} <output>{{ decay.toFixed(2) }}</output></span>
        <input v-model.number="decay" type="range" min="0" max="0.6" step="0.01" />
      </label>

      <label class="spectrum-control">
        <span>{{ copy.threshold }} <output>{{ epsilon.toFixed(2) }}</output></span>
        <input v-model.number="epsilon" type="range" min="0" max="0.5" step="0.01" />
      </label>

      <label class="spectrum-control">
        <span>{{ copy.truncation }} <output>{{ k }}</output></span>
        <input v-model.number="k" type="range" min="1" max="16" step="1" />
      </label>
    </div>

    <SpectrumBars
      :values="values"
      :epsilon="epsilon"
      :k="k"
      :locale="props.locale"
    />

    <dl class="spectrum-results" aria-live="polite">
      <div>
        <dt>{{ copy.effectiveRank }}</dt>
        <dd>r<sub>ε</sub> = {{ metrics.effectiveRank }}</dd>
      </div>
      <div>
        <dt>{{ copy.retainedEnergy }}</dt>
        <dd>E<sub>{{ k }}</sub> = {{ formatPercent(metrics.retainedEnergy) }}</dd>
      </div>
      <div>
        <dt>{{ copy.truncationError }}</dt>
        <dd>σ<sub>{{ Math.min(k + 1, 17) }}</sub> = {{ formatValue(metrics.truncationError) }}</dd>
      </div>
    </dl>
  </section>
</template>

<style scoped>
.spectrum-microscope {
  border-bottom: 1px solid var(--fyf-border);
  border-top: 1px solid var(--fyf-border);
  box-sizing: border-box;
  color: var(--fyf-text);
  margin: 2rem 50% 2.75rem;
  padding: 1.15rem 0;
  transform: translateX(-50%);
  width: min(920px, calc(100vw - 40px));
}

.spectrum-controls {
  display: grid;
  gap: clamp(1rem, 3vw, 2.4rem);
  grid-template-columns: repeat(3, minmax(0, 1fr));
  padding: 0 0.25rem 1.1rem;
}

.spectrum-control {
  display: grid;
  gap: 0.35rem;
}

.spectrum-control > span {
  color: var(--fyf-text-soft);
  font-size: 0.72rem;
  font-weight: 650;
}

.spectrum-control output {
  color: var(--fyf-text);
  float: right;
  font-family: var(--fyf-font-mono, monospace);
  font-variant-numeric: tabular-nums;
}

.spectrum-control input {
  accent-color: var(--fyf-accent);
  cursor: pointer;
  width: 100%;
}

.spectrum-control input:focus-visible {
  outline: 2px solid var(--fyf-accent);
  outline-offset: 3px;
}

.spectrum-results {
  display: flex;
  flex-wrap: wrap;
  gap: 0.55rem 1.6rem;
  margin: 0;
  padding: 0.85rem 0.25rem 0;
}

.spectrum-results div {
  align-items: baseline;
  display: flex;
  gap: 0.4rem;
}

.spectrum-results dt {
  color: var(--fyf-text-soft);
  font-size: 0.68rem;
}

.spectrum-results dd {
  font-family: var(--vp-font-family-heading);
  font-size: 0.82rem;
  font-variant-numeric: tabular-nums;
  margin: 0;
}

@media (max-width: 620px) {
  .spectrum-microscope {
    width: min(100%, calc(100vw - 24px));
  }

  .spectrum-controls {
    gap: 0.9rem;
    grid-template-columns: 1fr;
  }

  .spectrum-results {
    gap: 0.45rem 1rem;
  }
}

@media (prefers-reduced-motion: reduce) {
  .spectrum-microscope * {
    transition-duration: 0.01ms !important;
  }
}
</style>
