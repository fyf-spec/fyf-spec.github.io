import test from "node:test";
import assert from "node:assert/strict";
import {
  apply2,
  makeExponentialSpectrum,
  makeSpectrum,
  makeSvdState,
  multiply2,
  orthogonal2,
  sampleUnitCircle,
  spectrumMetrics,
  transpose2
} from "./svdMath.mjs";
import { getSvdCopy } from "./svdCopy.mjs";

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

test("the cumulative stages compose to A x", () => {
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

test("the minimal spectrum control follows an exponential decay", () => {
  const values = makeExponentialSpectrum(0.25, 5);
  assert.equal(values.length, 5);
  close(values[0], 1);
  close(values[1], Math.exp(-0.25));
  close(values[4], Math.exp(-1));
});

test("unit-circle samples lie on the unit circle", () => {
  const samples = sampleUnitCircle(32);
  assert.equal(samples.length, 32);
  for (const point of samples) {
    close(Math.hypot(point.x, point.y), 1);
  }
});

test("interactive copy is complete in Chinese and English", () => {
  const zh = getSvdCopy("zh");
  const en = getSvdCopy("en");

  assert.equal(zh.explorer.ariaLabel, "奇异值分解参数图");
  assert.equal(en.explorer.ariaLabel, "Singular value decomposition controls and plots");
  assert.equal(zh.explorer.stages.length, 4);
  assert.equal(en.explorer.stages.length, 4);
  assert.equal(en.spectrum.decay, "Decay α");
  assert.equal(en.bars.singularValue, "Singular value");
  assert.equal("title" in en.explorer, false);
  assert.equal("title" in en.spectrum, false);
  assert.equal(getSvdCopy("unsupported"), zh);
});
