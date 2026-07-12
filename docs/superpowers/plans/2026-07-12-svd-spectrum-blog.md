# Interactive SVD and Spectrum Blog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Chinese VitePress learning blog titled “矩阵的奇异值与谱分解” with a four-stage adjustable 2D SVD laboratory and fine-grained singular-spectrum bar visualizations.

**Architecture:** Keep the article in Markdown and mount two Vue experiences inside it: a 2D SVD explorer and a higher-dimensional spectrum microscope. Put all matrix and spectrum calculations in a dependency-free JavaScript module so the SVGs, displayed matrices, metrics, and Node tests consume the same results.

**Tech Stack:** VitePress 1.6, Vue 3 Composition API, TypeScript in Vue SFCs, responsive SVG, CSS custom properties, Node 22 built-in test runner.

## Global Constraints

- Preserve the user’s unrelated changes in `docs/blogs/MLA-flops.zip`, `docs/blogs/MLA-flops/MLA_flops_notes.md`, and `docs/blogs/MLA-flops/mla_flops_beginning.png`; never stage them.
- Use exact-path `git add` commands only. Do not use `git add .` or `git add -A`.
- Add no runtime packages and no test framework packages.
- Use no external images or generated image assets; this was explicitly approved in the design.
- Keep the article in Chinese and in the site’s concise technical-note voice.
- Keep `0 <= sigma2 <= sigma1 <= 3` at every UI state.
- Parameterize `U` and `V` with angles plus optional reflections; do not add direct editing of matrix `A`.
- Use the stage colors consistently: `V^T` blue `#2563eb`, `Sigma` orange `#d97706`, `U` purple `#7c3aed`.
- Support dark mode, keyboard input, pointer drag, `prefers-reduced-motion`, and SSR-safe rendering.
- Treat “奇异值谱”, the rank-one SVD expansion, and strict symmetric-matrix spectral decomposition as distinct concepts.
- Final verification must cover desktop `1440x1000` and mobile `390x844` viewports.

## File Map

**Create**

- `docs/.vitepress/theme/components/svd/svdMath.mjs` — pure 2D matrix and spectrum calculations.
- `docs/.vitepress/theme/components/svd/svdMath.test.mjs` — Node tests for mathematical invariants and boundary states.
- `docs/.vitepress/theme/components/svd/SpectrumBars.vue` — reusable accessible spectrum bar chart.
- `docs/.vitepress/theme/components/svd/SpectrumMicroscope.vue` — 16-value spectrum presets, threshold, and truncation controls.
- `docs/.vitepress/theme/components/svd/TransformStage.vue` — one reusable transformed-grid SVG stage.
- `docs/.vitepress/theme/components/svd/SvdExplorer.vue` — controls, derived matrices, four-stage composition, and current 2D spectrum.
- `docs/blogs/singular-values-spectrum/singular-values-spectrum.md` — Chinese blog article.

**Modify**

- `docs/.vitepress/theme/index.ts` — globally register `SvdExplorer` and `SpectrumMicroscope`.
- `docs/.vitepress/theme/content.ts` — add the blog index metadata.

---

### Task 1: Dependency-Free SVD Math Kernel

**Files:**

- Create: `docs/.vitepress/theme/components/svd/svdMath.test.mjs`
- Create: `docs/.vitepress/theme/components/svd/svdMath.mjs`

**Interfaces:**

- Consumes: plain numbers, `{ x: number, y: number }`, and `[[number, number], [number, number]]` matrices.
- Produces: `rotation2`, `orthogonal2`, `transpose2`, `multiply2`, `apply2`, `makeSvdState`, `sampleUnitCircle`, `makeSpectrum`, and `spectrumMetrics`.

- [ ] **Step 1: Write the failing mathematical tests**

Create `svdMath.test.mjs` with explicit tolerance helpers and these cases:

```js
import test from "node:test";
import assert from "node:assert/strict";
import {
  apply2,
  makeSpectrum,
  makeSvdState,
  multiply2,
  orthogonal2,
  spectrumMetrics,
  transpose2
} from "./svdMath.mjs";

const EPS = 1e-9;

function close(actual, expected, message = "values differ") {
  assert.ok(Math.abs(actual - expected) <= EPS, `${message}: ${actual} != ${expected}`);
}

function matrixClose(actual, expected) {
  for (let row = 0; row < 2; row += 1) {
    for (let column = 0; column < 2; column += 1) {
      close(actual[row][column], expected[row][column], `entry ${row},${column}`);
    }
  }
}

test("rotation and reflected bases stay orthogonal", () => {
  for (const reflected of [false, true]) {
    const basis = orthogonal2(37, reflected);
    matrixClose(multiply2(transpose2(basis), basis), [[1, 0], [0, 1]]);
  }
});

test("the three cumulative stages compose to A x", () => {
  const state = makeSvdState({
    thetaU: -23,
    thetaV: 41,
    sigma1: 2.4,
    sigma2: 0.6,
    reflectU: true,
    reflectV: false
  });
  const x = { x: 0.8, y: -0.35 };
  const staged = apply2(state.stages[3], x);
  const direct = apply2(state.A, x);
  close(staged.x, direct.x);
  close(staged.y, direct.y);
});

test("A transpose A has V eigenvectors and squared singular values", () => {
  const state = makeSvdState({
    thetaU: 18,
    thetaV: -31,
    sigma1: 2,
    sigma2: 0.25,
    reflectU: false,
    reflectV: true
  });
  const ata = multiply2(transpose2(state.A), state.A);
  const expected = multiply2(
    multiply2(state.V, [[4, 0], [0, 0.0625]]),
    transpose2(state.V)
  );
  matrixClose(ata, expected);
});

test("zero spectrum metrics never contain NaN", () => {
  const metrics = spectrumMetrics([0, 0], 0.01, 1);
  assert.equal(metrics.exactRank, 0);
  assert.equal(metrics.effectiveRank, 0);
  assert.equal(metrics.retainedEnergy, 0);
  assert.equal(metrics.truncationError, 0);
  assert.deepEqual(metrics.energyShares, [0, 0]);
});

test("spectrum presets are sorted and truncation error is sigma k plus one", () => {
  for (const kind of ["flat", "slow", "fast", "cutoff"]) {
    const values = makeSpectrum(kind, 1.4, 16);
    assert.equal(values.length, 16);
    for (let index = 1; index < values.length; index += 1) {
      assert.ok(values[index - 1] >= values[index]);
      assert.ok(values[index] >= 0);
    }
  }
  const metrics = spectrumMetrics([1, 0.7, 0.2, 0.05], 0.1, 2);
  close(metrics.truncationError, 0.2);
  assert.equal(metrics.effectiveRank, 3);
});
```

- [ ] **Step 2: Run the test and verify the module is missing**

Run:

```powershell
node --test docs/.vitepress/theme/components/svd/svdMath.test.mjs
```

Expected: FAIL with `ERR_MODULE_NOT_FOUND` for `svdMath.mjs`.

- [ ] **Step 3: Implement the minimal pure math module**

Create `svdMath.mjs` with the following exact public behavior:

```js
const DEG_TO_RAD = Math.PI / 180;
const ZERO_EPSILON = 1e-10;

export function rotation2(degrees) {
  const radians = degrees * DEG_TO_RAD;
  const cosine = Math.cos(radians);
  const sine = Math.sin(radians);
  return [[cosine, -sine], [sine, cosine]];
}

export function multiply2(left, right) {
  return [
    [
      left[0][0] * right[0][0] + left[0][1] * right[1][0],
      left[0][0] * right[0][1] + left[0][1] * right[1][1]
    ],
    [
      left[1][0] * right[0][0] + left[1][1] * right[1][0],
      left[1][0] * right[0][1] + left[1][1] * right[1][1]
    ]
  ];
}

export function transpose2(matrix) {
  return [[matrix[0][0], matrix[1][0]], [matrix[0][1], matrix[1][1]]];
}

export function apply2(matrix, vector) {
  return {
    x: matrix[0][0] * vector.x + matrix[0][1] * vector.y,
    y: matrix[1][0] * vector.x + matrix[1][1] * vector.y
  };
}

export function orthogonal2(degrees, reflected = false) {
  const rotation = rotation2(degrees);
  return reflected ? multiply2(rotation, [[1, 0], [0, -1]]) : rotation;
}

export function makeSvdState(parameters) {
  const sigma1 = Math.max(0, Math.min(3, parameters.sigma1));
  const sigma2 = Math.max(0, Math.min(sigma1, parameters.sigma2));
  const U = orthogonal2(parameters.thetaU, parameters.reflectU);
  const V = orthogonal2(parameters.thetaV, parameters.reflectV);
  const Vt = transpose2(V);
  const Sigma = [[sigma1, 0], [0, sigma2]];
  const SigmaVt = multiply2(Sigma, Vt);
  const A = multiply2(U, SigmaVt);
  return {
    U,
    V,
    Vt,
    Sigma,
    A,
    sigma1,
    sigma2,
    stages: [[[1, 0], [0, 1]], Vt, SigmaVt, A]
  };
}

export function sampleUnitCircle(count = 64) {
  return Array.from({ length: count }, (_, index) => {
    const angle = (index / count) * Math.PI * 2;
    return { x: Math.cos(angle), y: Math.sin(angle) };
  });
}

export function makeSpectrum(kind, decay = 1.4, count = 16) {
  const safeDecay = Math.max(0.2, Math.min(3, decay));
  return Array.from({ length: count }, (_, index) => {
    if (kind === "flat") return 1;
    if (kind === "slow") return 1 / Math.pow(index + 1, 0.2 + safeDecay * 0.28);
    if (kind === "fast") return Math.exp(-index * (0.12 + safeDecay * 0.16));
    const cutoff = Math.max(2, Math.min(count, Math.round(count / safeDecay)));
    return index < cutoff ? 1 / Math.pow(index + 1, 0.18) : 0;
  });
}

export function spectrumMetrics(values, epsilon = 0, k = values.length) {
  const squares = values.map((value) => value * value);
  const totalEnergy = squares.reduce((sum, value) => sum + value, 0);
  const kept = Math.max(0, Math.min(values.length, Math.round(k)));
  const retained = squares.slice(0, kept).reduce((sum, value) => sum + value, 0);
  return {
    exactRank: values.filter((value) => value > ZERO_EPSILON).length,
    effectiveRank: values.filter((value) => value >= epsilon && value > ZERO_EPSILON).length,
    energyShares: totalEnergy === 0 ? values.map(() => 0) : squares.map((value) => value / totalEnergy),
    retainedEnergy: totalEnergy === 0 ? 0 : retained / totalEnergy,
    truncationError: values[kept] ?? 0,
    conditionNumber:
      values.length < 2 || values[1] <= ZERO_EPSILON
        ? Number.POSITIVE_INFINITY
        : values[0] / values[1]
  };
}
```

- [ ] **Step 4: Run the focused tests**

Run:

```powershell
node --test docs/.vitepress/theme/components/svd/svdMath.test.mjs
```

Expected: 5 tests pass, 0 fail.

- [ ] **Step 5: Commit the kernel and tests**

```powershell
git add -- docs/.vitepress/theme/components/svd/svdMath.mjs docs/.vitepress/theme/components/svd/svdMath.test.mjs
git commit -m "feat: add tested SVD math kernel"
```

---

### Task 2: Reusable Spectrum Chart and High-Dimensional Microscope

**Files:**

- Create: `docs/.vitepress/theme/components/svd/SpectrumBars.vue`
- Create: `docs/.vitepress/theme/components/svd/SpectrumMicroscope.vue`
- Modify: `docs/.vitepress/theme/index.ts`

**Interfaces:**

- Consumes: `values: number[]`, `epsilon: number`, `k: number`, `scale: "linear" | "log"`, optional `compact: boolean`.
- Produces: accessible bars, selected-bar details, and the global Markdown component `<SpectrumMicroscope />`.

- [ ] **Step 1: Create the reusable bar-chart contract**

Implement `SpectrumBars.vue` with these exact props and derived values:

```ts
type ScaleMode = "linear" | "log";

const props = withDefaults(defineProps<{
  values: number[];
  epsilon?: number;
  k?: number;
  scale?: ScaleMode;
  compact?: boolean;
}>(), {
  epsilon: 0,
  k: Number.POSITIVE_INFINITY,
  scale: "linear",
  compact: false
});

const selectedIndex = ref(0);
const metrics = computed(() => spectrumMetrics(props.values, props.epsilon, props.k));
const maxValue = computed(() => Math.max(1e-6, ...props.values));
const heightFor = (value: number): number => {
  if (props.scale === "linear") return (value / maxValue.value) * 100;
  const floor = 1e-4;
  const safe = Math.max(floor, value);
  return ((Math.log10(safe) - Math.log10(floor)) /
    (Math.log10(maxValue.value) - Math.log10(floor) || 1)) * 100;
};
```

The template must use one focusable `<button>` per bar. Each button receives `aria-label="第 i 个奇异值：value"`, a CSS custom property `--bar-height`, and the classes `is-kept`, `is-tail`, and `is-below-threshold`. Show the selected bar’s index, exact value, per-mode energy, and cumulative energy below the chart. In compact mode hide x-axis labels except the first and last.

Use component-scoped CSS with an open chart rail, a `min-height` of `220px` (`118px` compact), `gap: 6px`, orange kept bars, muted tail bars, a dashed threshold line, visible `:focus-visible`, tabular numeric text, and no gradients.

- [ ] **Step 2: Create the 16-mode spectrum microscope**

Implement `SpectrumMicroscope.vue` with state:

```ts
type SpectrumKind = "flat" | "slow" | "fast" | "cutoff";

const kind = ref<SpectrumKind>("fast");
const decay = ref(1.4);
const epsilon = ref(0.08);
const k = ref(5);
const scale = ref<"linear" | "log">("linear");
const values = computed(() => makeSpectrum(kind.value, decay.value, 16));
const metrics = computed(() => spectrumMetrics(values.value, epsilon.value, k.value));
```

Use four preset buttons with the exact labels `平坦谱`, `慢衰减`, `快衰减`, and `硬截断`; sliders labeled `衰减速度 α`, `有效秩阈值 ε`, and `截断阶数 k`; and a two-button scale switch labeled `线性` and `对数`. Render `<SpectrumBars>` and the exact metrics `有效秩`, `累计能量`, and `谱范数误差`.

Layout rules: two-column control/chart grid above `900px`, one column below; open section with one thin border at the top and bottom; no nested card grid; labels at least `13px`; touch targets at least `40px` high.

- [ ] **Step 3: Register only the completed microscope component**

Modify `docs/.vitepress/theme/index.ts`:

```ts
import SpectrumMicroscope from "./components/svd/SpectrumMicroscope.vue";
```

and inside `enhanceApp`:

```ts
app.component("SpectrumMicroscope", SpectrumMicroscope);
```

- [ ] **Step 4: Compile the spectrum components**

Run:

```powershell
npm run docs:build
```

Expected: VitePress build exits 0 with no Vue template or SSR errors.

- [ ] **Step 5: Re-run math tests to protect metric behavior**

Run:

```powershell
node --test docs/.vitepress/theme/components/svd/svdMath.test.mjs
```

Expected: 5 tests pass, 0 fail.

- [ ] **Step 6: Commit the spectrum components**

```powershell
git add -- docs/.vitepress/theme/components/svd/SpectrumBars.vue docs/.vitepress/theme/components/svd/SpectrumMicroscope.vue docs/.vitepress/theme/index.ts
git commit -m "feat: add interactive singular spectrum charts"
```

---

### Task 3: Four-Stage SVD Geometry Explorer

**Files:**

- Create: `docs/.vitepress/theme/components/svd/TransformStage.vue`
- Create: `docs/.vitepress/theme/components/svd/SvdExplorer.vue`
- Modify: `docs/.vitepress/theme/index.ts`

**Interfaces:**

- Consumes: the Task 1 math API and Task 2 `<SpectrumBars>`.
- Produces: the global Markdown component `<SvdExplorer />` and `update:inputVector` drag events from each transform stage.

- [ ] **Step 1: Build a reusable transform-stage SVG**

`TransformStage.vue` must accept:

```ts
type Vector2 = { x: number; y: number };
type Matrix2 = [[number, number], [number, number]];

const props = withDefaults(defineProps<{
  title: string;
  formula: string;
  matrix: Matrix2;
  inputVector: Vector2;
  accent: string;
  interactive?: boolean;
}>(), { interactive: false });

const emit = defineEmits<{
  "update:inputVector": [value: Vector2];
}>();
```

Generate base grid lines for coordinates `-2` through `2` at `0.5` intervals and a 64-point unit circle with `sampleUnitCircle(64)`. Transform every endpoint with `apply2(props.matrix, point)`. Render fixed x/y axes, transformed grid, transformed unit-circle path, 16 radial sample vectors, and the highlighted transformed input vector in an SVG with `viewBox="-140 -140 280 280"`.

For the interactive first stage, convert pointer coordinates from the SVG client rectangle into mathematical coordinates, invert the y-axis, clamp both coordinates to `[-1.5, 1.5]`, call `setPointerCapture`, and emit the new vector. Add `role="img"` and an `aria-label` containing the stage title and current vector coordinates.

- [ ] **Step 2: Build the SVD explorer state and controls**

`SvdExplorer.vue` must initialize:

```ts
const thetaV = ref(30);
const sigma1 = ref(2.1);
const sigma2 = ref(0.65);
const thetaU = ref(-20);
const reflectV = ref(false);
const reflectU = ref(false);
const inputVector = ref({ x: 0.9, y: 0.45 });
const activeStage = ref(0);

const state = computed(() => makeSvdState({
  thetaU: thetaU.value,
  thetaV: thetaV.value,
  sigma1: sigma1.value,
  sigma2: Math.min(sigma1.value, sigma2.value),
  reflectU: reflectU.value,
  reflectV: reflectV.value
}));
```

Use a watcher so lowering `sigma1` clamps `sigma2` immediately. Define five complete presets: `恒等`, `各向异性`, `秩一坍缩`, `近奇异`, and `带反射`; every preset assigns all six SVD fields, not only a subset.

Render controls in this order: presets, `θV`, `σ1`, `σ2`, `θU`, reflection switches, input-vector numeric fields, then `U`, `Σ`, `V^T`, and derived `A`. Format matrix entries to two decimals and values with absolute magnitude below `0.005` as `0.00`.

- [ ] **Step 3: Compose the four synchronized stages and current spectrum**

Use the exact stage metadata:

```ts
const stageMeta = [
  { title: "输入空间", formula: "x", accent: "#64748b" },
  { title: "写入 V 的基", formula: "Vᵀx", accent: "#2563eb" },
  { title: "沿奇异方向伸缩", formula: "ΣVᵀx", accent: "#d97706" },
  { title: "映射到输出基", formula: "UΣVᵀx = Ax", accent: "#7c3aed" }
];
```

Map `state.stages[index]` to four `<TransformStage>` instances. On screens below `640px`, hide inactive stages and expose four labeled stage buttons; above `640px` show all stages in a two-column grid; above `1100px` show four columns.

Place a compact `<SpectrumBars :values="[state.sigma1, state.sigma2]" :epsilon="0.000000001" :k="2" compact />` beside exact rank, energy shares, and condition number. Display `∞` when the condition number is not finite and a plain-language callout when `sigma2 === 0`.

- [ ] **Step 4: Apply the approved visual system**

Component-scoped styles must define:

```css
.svd-explorer {
  --svd-blue: #2563eb;
  --svd-orange: #d97706;
  --svd-purple: #7c3aed;
  margin: 2rem 50% 2.5rem;
  transform: translateX(-50%);
  width: min(1120px, calc(100vw - 40px));
}
```

Use the site variables `--fyf-surface`, `--fyf-border`, `--fyf-text`, and `--fyf-text-soft`; use `font-variant-numeric: tabular-nums` for matrices and values. Controls use one open rail with grouped rows, `border-radius` no larger than `18px`, no gradients, and no decorative badges. Add `:focus-visible` outlines and a `@media (prefers-reduced-motion: reduce)` rule that disables transitions.

- [ ] **Step 5: Register the explorer**

Modify `docs/.vitepress/theme/index.ts`:

```ts
import SvdExplorer from "./components/svd/SvdExplorer.vue";
```

and inside `enhanceApp`:

```ts
app.component("SvdExplorer", SvdExplorer);
```

- [ ] **Step 6: Compile and test**

Run:

```powershell
node --test docs/.vitepress/theme/components/svd/svdMath.test.mjs
npm run docs:build
```

Expected: 5 tests pass; VitePress build exits 0 without hydration, TypeScript transpilation, or template errors.

- [ ] **Step 7: Commit the four-stage explorer**

```powershell
git add -- docs/.vitepress/theme/components/svd/TransformStage.vue docs/.vitepress/theme/components/svd/SvdExplorer.vue docs/.vitepress/theme/index.ts
git commit -m "feat: add four-stage SVD explorer"
```

---

### Task 4: Chinese Article and Blog Registration

**Files:**

- Create: `docs/blogs/singular-values-spectrum/singular-values-spectrum.md`
- Modify: `docs/.vitepress/theme/content.ts`

**Interfaces:**

- Consumes: global `<SvdExplorer />` and `<SpectrumMicroscope />` components.
- Produces: `/blogs/singular-values-spectrum/singular-values-spectrum` and a first-page blog catalog entry.

- [ ] **Step 1: Write the article with the approved six-section structure**

Use this exact frontmatter and visible section skeleton:

```markdown
---
title: "矩阵的奇异值与谱分解"
description: 用可调节的二维 SVD 实验理解奇异方向、奇异值谱、秩一展开与谱范数。
date: 2026-07-12
outline: deep
---

# 矩阵的奇异值与谱分解

一个矩阵究竟对空间做了什么？对一般线性变换，我更愿意把答案压缩成三步：换一组正交坐标、沿坐标轴伸缩，再换到输出空间的正交坐标。

## 一个矩阵，三步变换

$$
Ax=U\Sigma V^Tx
$$

依次解释 `V^T`、`Σ` 和 `U`，并明确 `V^Tx` 是输入在右奇异向量基中的坐标，而不是把向量随意旋转一次。

<SvdExplorer />

## 奇异值、秩与奇异值谱

解释精确秩等于非零奇异值数量；小而非零的奇异值表示方向仍被保留，但区分度弱且逆变换会放大噪声。明确“奇异值谱”与通常表示特征值集合的“谱”之间的术语差异。

<SpectrumMicroscope />

## 从 AᵀA 看见奇异值

$$
A^TA=V\Sigma^T\Sigma V^T,
\qquad
A^TAv_i=\sigma_i^2v_i
$$

给出 `sigma_i = sqrt(lambda_i)`、`u_i = Av_i / sigma_i`，说明零奇异值需要正交补齐，并注明显式形成 `A^TA` 会平方条件数，因此这里只把它当作概念推导。

## 秩一展开与谱分解

$$
A=\sum_i\sigma_i u_iv_i^T
$$

解释每一项是一个秩一矩阵；再分别写出一般矩阵的 SVD 秩一展开、对称矩阵的 `A=QΛQ^T`，以及 `A^TA`、`AA^T` 的真正谱分解。连接谱显微镜中的 `k`：

$$
A_k=\sum_{i=1}^{k}\sigma_i u_iv_i^T,
\qquad
\lVert A-A_k\rVert_2=\sigma_{k+1}.
$$

## 谱范数

$$
\lVert A\rVert_2
=\max_{\lVert x\rVert_2=1}\lVert Ax\rVert_2
=\sigma_1
$$

用单位圆的最长半轴解释谱范数，并指出反射和旋转不改变长度，因此最大伸缩只由 `sigma1` 决定。

## Takeaway

用四条短句总结：SVD 是两次正交换基加一次伸缩；秩只判断方向是否完全消失；奇异值谱量化各方向尺度；谱范数是最大奇异值。
```

Replace each instruction sentence in this skeleton with polished Chinese prose rather than leaving meta-writing language in the published article. Keep each section to short paragraphs, retain every displayed equation, and use no external references unless a source is actually added and cited.

- [ ] **Step 2: Add the blog metadata at the top of `blogPosts`**

Insert this object before the 2026-07-06 entry:

```ts
{
  title: "矩阵的奇异值与谱分解",
  link: "/blogs/singular-values-spectrum/singular-values-spectrum",
  date: "2026-07-12",
  category: "technical",
  subtitle: "用可调节的二维 SVD 实验理解奇异方向、奇异值谱、秩一展开与谱范数。",
  image: ""
},
```

Do not modify `docs/.vitepress/config.mts`; the current blog navigation is already automatic.

- [ ] **Step 3: Build the complete site**

Run:

```powershell
npm run docs:build
```

Expected: exit 0 and generated page `docs/.vitepress/dist/blogs/singular-values-spectrum/singular-values-spectrum.html` exists.

- [ ] **Step 4: Verify the generated article contains both interactive mounts**

Run:

```powershell
Select-String -LiteralPath 'docs/.vitepress/dist/blogs/singular-values-spectrum/singular-values-spectrum.html' -Pattern 'svd-explorer|spectrum-microscope'
```

Expected: both component class names are present in the rendered HTML.

- [ ] **Step 5: Commit the article and catalog entry**

```powershell
git add -- docs/blogs/singular-values-spectrum/singular-values-spectrum.md docs/.vitepress/theme/content.ts
git commit -m "docs: add singular values and spectrum blog"
```

---

### Task 5: Browser Fidelity, Interaction, and Responsive Verification

**Files:**

- Modify if required: `docs/.vitepress/theme/components/svd/SpectrumBars.vue`
- Modify if required: `docs/.vitepress/theme/components/svd/SpectrumMicroscope.vue`
- Modify if required: `docs/.vitepress/theme/components/svd/TransformStage.vue`
- Modify if required: `docs/.vitepress/theme/components/svd/SvdExplorer.vue`
- Modify if required: `docs/blogs/singular-values-spectrum/singular-values-spectrum.md`

**Interfaces:**

- Consumes: the finished article route and all interactive controls.
- Produces: verified desktop/mobile screenshots and a clean final build.

- [ ] **Step 1: Start VitePress in a hidden background process**

Run from the repository root:

```powershell
$server = Start-Process -FilePath 'npm.cmd' -ArgumentList 'run','docs:dev','--','--host','127.0.0.1' -PassThru -WindowStyle Hidden
$server.Id
```

Expected: a process ID is printed and `http://127.0.0.1:5173/blogs/singular-values-spectrum/singular-values-spectrum` responds.

- [ ] **Step 2: Verify the desktop composition at `1440x1000`**

Open the article in the browser and capture a screenshot. Check these concrete points:

1. The title and opening paragraph fit before the first section break.
2. The explorer breaks out to roughly `1120px` without horizontal page overflow.
3. All four stages are visible in one row at 1440px, with identical plot scales.
4. Stage colors match blue/orange/purple semantics in controls, formulas, and plots.
5. Matrices, vectors, current spectrum, and rank metrics update from one parameter change.
6. The 16-bar spectrum chart has readable selected-mode details and no clipped labels.

Expected: no clipped content, nested-card clutter, default browser typography, or unexplained decorative text.

- [ ] **Step 3: Verify the core interaction path**

Perform this exact sequence:

1. Select `秩一坍缩`; expect `sigma2 = 0`, the third/fourth-stage unit circle collapses to a line, rank displays `1`, and condition number displays `∞`.
2. Drag `x` in the input stage; expect all four vector arrows and coordinates to update.
3. Toggle the `V` reflection; expect the second-stage grid orientation and the displayed `V^T`/`A` matrices to change.
4. Select `快衰减`, set `k = 3`, switch to `对数`; expect exactly the first three bars highlighted and truncation error equal to the fourth singular value.
5. Raise `epsilon`; expect effective rank to decrease without changing exact rank.

- [ ] **Step 4: Verify mobile at `390x844`**

Capture the full article and inspect:

1. Only the active geometry stage is visible; the four stage tabs remain keyboard reachable.
2. Sliders, numeric inputs, reflection switches, and preset controls do not overflow.
3. Matrix rows remain legible without forcing page-level horizontal scrolling.
4. Spectrum bars remain individually focusable and the selected detail wraps cleanly.
5. Headings and formulas fit or scroll only inside their own math container.

- [ ] **Step 5: Verify accessibility and theme states**

Use keyboard-only navigation to reach every slider, preset, reflection switch, stage tab, and spectrum bar. Switch VitePress to dark mode and verify contrast. Emulate `prefers-reduced-motion: reduce` and verify the UI stays usable with transitions disabled.

- [ ] **Step 6: Fix every visible or functional mismatch and repeat Steps 2–5**

Only modify the five files listed for this task. For each fix, state the observed mismatch and the exact CSS, DOM, or state correction in the commit body. Repeat screenshots until no fixable mismatch remains.

- [ ] **Step 7: Run the final verification suite**

```powershell
node --test docs/.vitepress/theme/components/svd/svdMath.test.mjs
npm run docs:build
git diff --check -- docs/.vitepress/theme/components/svd docs/blogs/singular-values-spectrum docs/.vitepress/theme/index.ts docs/.vitepress/theme/content.ts
git status --short
```

Expected:

- 5 tests pass, 0 fail.
- VitePress build exits 0.
- `git diff --check` reports nothing.
- `git status --short` shows only the user’s pre-existing MLA changes plus any intentional unstaged QA screenshot artifacts; no SVD implementation file remains uncommitted.

- [ ] **Step 8: Commit QA fixes if any exist**

```powershell
git add -- docs/.vitepress/theme/components/svd/SpectrumBars.vue docs/.vitepress/theme/components/svd/SpectrumMicroscope.vue docs/.vitepress/theme/components/svd/TransformStage.vue docs/.vitepress/theme/components/svd/SvdExplorer.vue docs/blogs/singular-values-spectrum/singular-values-spectrum.md
git diff --cached --quiet
if ($LASTEXITCODE -ne 0) { git commit -m "fix: polish SVD blog interactions and responsive layout" }
```

- [ ] **Step 9: Stop the development server**

```powershell
Stop-Process -Id $server.Id
```

Expected: the hidden VitePress process exits and no server process is left running.
