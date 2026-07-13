<script setup lang="ts">
import { computed, useId } from "vue";
import { apply2, sampleUnitCircle } from "./svdMath.mjs";

type Vector2 = { x: number; y: number };
type Matrix2 = [[number, number], [number, number]];

const props = withDefaults(defineProps<{
  formula: string;
  ariaLabel: string;
  matrix: Matrix2;
  inputVector: Vector2;
  accent: string;
  plotScale?: number;
}>(), {
  plotScale: 48
});

const markerId = `svd-arrow-${useId().replaceAll(":", "")}`;
const circle = sampleUnitCircle(80);

const gridLines = computed(() => {
  const lines: Array<{ start: Vector2; end: Vector2; major: boolean }> = [];
  for (let value = -2; value <= 2.001; value += 0.5) {
    const normalized = Math.round(value * 2) / 2;
    lines.push({
      start: apply2(props.matrix, { x: normalized, y: -3.4 }),
      end: apply2(props.matrix, { x: normalized, y: 3.4 }),
      major: Number.isInteger(normalized)
    });
    lines.push({
      start: apply2(props.matrix, { x: -3.4, y: normalized }),
      end: apply2(props.matrix, { x: 3.4, y: normalized }),
      major: Number.isInteger(normalized)
    });
  }
  return lines;
});

const circlePath = computed(() => circle
  .map((point, index) => {
    const transformed = apply2(props.matrix, point);
    return `${index === 0 ? "M" : "L"} ${transformed.x * props.plotScale} ${-transformed.y * props.plotScale}`;
  })
  .join(" ") + " Z");

const sampleVectors = computed(() => sampleUnitCircle(16).map((point) => apply2(props.matrix, point)));
const transformedVector = computed(() => apply2(props.matrix, props.inputVector));

function formatCoordinate(value: number): string {
  return Math.abs(value) < 0.005 ? "0.00" : value.toFixed(2);
}
</script>

<template>
  <figure class="transform-stage" :style="{ '--stage-accent': accent }">
    <figcaption><code>{{ formula }}</code></figcaption>

    <svg
      class="transform-stage-svg"
      viewBox="-140 -140 280 280"
      role="img"
      :aria-label="`${ariaLabel}: ${formatCoordinate(transformedVector.x)}, ${formatCoordinate(transformedVector.y)}`"
    >
      <defs>
        <marker
          :id="markerId"
          viewBox="0 0 10 10"
          refX="8"
          refY="5"
          markerWidth="5"
          markerHeight="5"
          orient="auto-start-reverse"
        >
          <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor" />
        </marker>
        <clipPath :id="`${markerId}-clip`">
          <rect x="-139" y="-139" width="278" height="278" rx="6" />
        </clipPath>
      </defs>

      <rect class="plot-background" x="-140" y="-140" width="280" height="280" rx="6" />
      <g :clip-path="`url(#${markerId}-clip)`">
        <line class="fixed-axis" x1="-140" y1="0" x2="140" y2="0" />
        <line class="fixed-axis" x1="0" y1="-140" x2="0" y2="140" />

        <line
          v-for="(line, index) in gridLines"
          :key="`grid-${index}`"
          class="transformed-grid"
          :class="{ 'is-major': line.major }"
          :x1="line.start.x * plotScale"
          :y1="-line.start.y * plotScale"
          :x2="line.end.x * plotScale"
          :y2="-line.end.y * plotScale"
        />

        <line
          v-for="(vector, index) in sampleVectors"
          :key="`vector-${index}`"
          class="sample-vector"
          x1="0"
          y1="0"
          :x2="vector.x * plotScale"
          :y2="-vector.y * plotScale"
        />

        <path class="transformed-circle" :d="circlePath" />
        <line
          class="input-vector"
          x1="0"
          y1="0"
          :x2="transformedVector.x * plotScale"
          :y2="-transformedVector.y * plotScale"
          :marker-end="`url(#${markerId})`"
        />
      </g>
    </svg>
  </figure>
</template>

<style scoped>
.transform-stage {
  color: var(--fyf-text);
  margin: 0;
  min-width: 0;
}

.transform-stage figcaption {
  align-items: center;
  display: flex;
  height: 1.8rem;
}

.transform-stage code {
  background: transparent;
  color: var(--stage-accent);
  font-family: var(--vp-font-family-heading);
  font-size: 0.78rem;
  font-weight: 650;
  padding: 0;
}

.transform-stage-svg {
  display: block;
  max-width: 100%;
  overflow: hidden;
  width: 100%;
}

.plot-background {
  fill: var(--fyf-surface-muted);
  stroke: var(--fyf-border);
  stroke-width: 1;
}

.fixed-axis {
  stroke: color-mix(in srgb, var(--fyf-text) 40%, transparent);
  stroke-width: 0.9;
}

.transformed-grid {
  stroke: color-mix(in srgb, var(--stage-accent) 18%, transparent);
  stroke-width: 0.7;
}

.transformed-grid.is-major {
  stroke: color-mix(in srgb, var(--stage-accent) 27%, transparent);
  stroke-width: 0.9;
}

.sample-vector {
  stroke: color-mix(in srgb, var(--stage-accent) 23%, transparent);
  stroke-width: 0.75;
}

.transformed-circle {
  fill: color-mix(in srgb, var(--stage-accent) 6%, transparent);
  stroke: var(--stage-accent);
  stroke-width: 1.8;
}

.input-vector {
  color: var(--fyf-text);
  stroke: currentColor;
  stroke-linecap: round;
  stroke-width: 2.8;
}
</style>
