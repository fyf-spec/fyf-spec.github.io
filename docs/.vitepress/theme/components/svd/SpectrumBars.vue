<script setup lang="ts">
import { computed } from "vue";
import { getSvdCopy } from "./svdCopy.mjs";

const props = withDefaults(defineProps<{
  values: number[];
  epsilon?: number;
  k?: number;
  locale?: "zh" | "en";
}>(), {
  epsilon: 0,
  k: Number.POSITIVE_INFINITY,
  locale: "zh"
});

const copy = computed(() => getSvdCopy(props.locale).bars);
const maxValue = computed(() => Math.max(1e-6, ...props.values));
const keptCount = computed(() => Math.max(0, Math.min(props.values.length, Math.round(props.k))));

function heightFor(value: number): number {
  return Math.max(0, Math.min(100, (value / maxValue.value) * 100));
}

const thresholdHeight = computed(() => (
  props.epsilon <= 0 ? 0 : heightFor(props.epsilon)
));

function formatValue(value: number): string {
  if (value === 0) return "0";
  if (value < 0.001) return value.toExponential(2);
  return value.toFixed(value < 0.1 ? 3 : 2);
}
</script>

<template>
  <div class="spectrum-bars">
    <div
      v-if="epsilon > 0"
      class="spectrum-threshold"
      :style="{ '--threshold-height': `${thresholdHeight}%` }"
      aria-hidden="true"
    >
      <span>ε</span>
    </div>

    <div class="spectrum-rail" role="list" :aria-label="copy.chartAria">
      <div
        v-for="(value, index) in values"
        :key="index"
        class="spectrum-bar"
        :class="{
          'is-kept': index < keptCount,
          'is-tail': index >= keptCount,
          'is-below-threshold': value < epsilon
        }"
        :style="{ '--bar-height': `${heightFor(value)}%` }"
        role="listitem"
        :title="`σ${index + 1} = ${formatValue(value)}`"
        :aria-label="`${copy.singularValue} ${index + 1}: ${formatValue(value)}`"
      >
        <span class="spectrum-bar-track" aria-hidden="true">
          <span class="spectrum-bar-fill"></span>
        </span>
        <span class="spectrum-bar-index">σ<sub>{{ index + 1 }}</sub></span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.spectrum-bars {
  --spectrum-accent: var(--fyf-accent);
  --spectrum-tail: var(--fyf-text-faint, #818b98);
  color: var(--fyf-text);
  font-variant-numeric: tabular-nums;
  position: relative;
  width: 100%;
}

.spectrum-rail {
  align-items: stretch;
  border-bottom: 1px solid var(--fyf-border);
  display: grid;
  gap: 6px;
  grid-template-columns: repeat(16, minmax(0, 1fr));
  min-height: 220px;
  padding: 0.5rem 0.25rem 0;
}

.spectrum-bar {
  color: var(--fyf-text-soft);
  display: grid;
  grid-template-rows: minmax(0, 1fr) 1.55rem;
  min-width: 0;
}

.spectrum-bar-track {
  align-items: end;
  display: flex;
  min-height: 0;
  padding: 0 1px;
}

.spectrum-bar-fill {
  background: var(--spectrum-accent);
  display: block;
  height: var(--bar-height);
  opacity: 0.88;
  transition: height 180ms ease, opacity 180ms ease;
  width: 100%;
}

.spectrum-bar.is-tail .spectrum-bar-fill {
  background: var(--spectrum-tail);
  opacity: 0.34;
}

.spectrum-bar.is-below-threshold .spectrum-bar-fill {
  opacity: 0.16;
}

.spectrum-bar-index {
  align-self: end;
  font-family: var(--vp-font-family-heading);
  font-size: 0.7rem;
  line-height: 1;
  padding-top: 0.45rem;
  text-align: center;
  white-space: nowrap;
}

.spectrum-threshold {
  border-top: 1px dashed color-mix(in srgb, var(--fyf-text-soft) 66%, transparent);
  bottom: calc(1.55rem + var(--threshold-height));
  left: 0;
  pointer-events: none;
  position: absolute;
  right: 0;
  z-index: 2;
}

.spectrum-threshold span {
  background: var(--fyf-canvas, var(--fyf-surface));
  color: var(--fyf-text-soft);
  font-size: 0.65rem;
  padding: 0 0.2rem;
  position: absolute;
  right: 0;
  top: -1.05rem;
}

@media (max-width: 640px) {
  .spectrum-rail {
    gap: 3px;
    min-height: 190px;
  }

  .spectrum-bar-index {
    font-size: 0.62rem;
  }

  .spectrum-bar:not(:first-child):not(:nth-child(4n + 1)):not(:last-child) .spectrum-bar-index {
    opacity: 0;
  }
}

@media (prefers-reduced-motion: reduce) {
  .spectrum-bar-fill {
    transition: none;
  }
}
</style>
