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

export function makeExponentialSpectrum(decay = 0.24, count = 16) {
  const safeDecay = Math.max(0, Math.min(1, decay));
  const safeCount = Math.max(1, Math.round(count));
  return Array.from({ length: safeCount }, (_, index) => Math.exp(-safeDecay * index));
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
