<script setup lang="ts">
import { useData } from "vitepress";
import { onBeforeUnmount, onMounted, ref, watch } from "vue";

type DayParticle = {
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
  alpha: number;
  phase: number;
  label: string;
};

type NightNode = {
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
  alpha: number;
  phase: number;
};

const { isDark } = useData();
const canvas = ref<HTMLCanvasElement | null>(null);

const pointer = {
  active: false,
  activity: 0,
  x: 0,
  y: 0,
  targetX: 0,
  targetY: 0,
  lastX: 0,
  lastY: 0,
  speed: 0
};

const dayParticles: DayParticle[] = [];
const nightNodes: NightNode[] = [];
const dayLabels = ["π", "λᵢ", "σ₁", "P(x)", "∇ · F = 0", "xᵀAx"];

let context: CanvasRenderingContext2D | null = null;
let width = 0;
let height = 0;
let animationFrame = 0;
let lastFrame = 0;
let reducedMotion = false;
let scrollTarget = 0;
let scrollPosition = 0;

function randomBetween(min: number, max: number): number {
  return min + Math.random() * (max - min);
}

function resizeBackingStore(): void {
  if (!canvas.value) return;

  width = window.innerWidth;
  height = window.innerHeight;
  const ratio = Math.min(window.devicePixelRatio || 1, 2);

  canvas.value.width = Math.max(1, Math.floor(width * ratio));
  canvas.value.height = Math.max(1, Math.floor(height * ratio));
  context = canvas.value.getContext("2d");
  context?.setTransform(ratio, 0, 0, ratio, 0, 0);

  seedDayScene();
  seedNightScene();
  drawScene(performance.now());
}

function seedDayScene(): void {
  dayParticles.length = 0;
  const count = Math.max(14, Math.min(26, Math.round(width / 64)));

  for (let index = 0; index < count; index += 1) {
    dayParticles.push({
      x: randomBetween(24, Math.max(25, width - 24)),
      y: randomBetween(80, Math.max(81, height - 24)),
      vx: randomBetween(-2.6, 2.6),
      vy: randomBetween(-2.1, 2.1),
      radius: randomBetween(0.65, 1.45),
      alpha: randomBetween(0.2, 0.62),
      phase: Math.random() * Math.PI * 2,
      label: index % 4 === 0 ? dayLabels[(index / 4) % dayLabels.length] ?? "" : ""
    });
  }
}

function seedNightScene(): void {
  nightNodes.length = 0;
  const count = Math.max(26, Math.min(42, Math.round(width / 38)));

  for (let index = 0; index < count; index += 1) {
    nightNodes.push({
      x: randomBetween(28, Math.max(29, width - 28)),
      y: randomBetween(76, Math.max(77, height - 28)),
      vx: randomBetween(-5.2, 5.2),
      vy: randomBetween(-4.2, 4.2),
      radius: randomBetween(1, 2.4),
      alpha: randomBetween(0.3, 0.82),
      phase: Math.random() * Math.PI * 2
    });
  }
}

function onPointerMove(event: PointerEvent): void {
  if (!pointer.active) {
    pointer.x = event.clientX;
    pointer.y = event.clientY;
    pointer.lastX = event.clientX;
    pointer.lastY = event.clientY;
  }

  pointer.speed = Math.hypot(event.clientX - pointer.lastX, event.clientY - pointer.lastY);
  pointer.lastX = event.clientX;
  pointer.lastY = event.clientY;
  pointer.targetX = event.clientX;
  pointer.targetY = event.clientY;
  pointer.active = true;
}

function onPointerLeave(): void {
  pointer.active = false;
  pointer.speed = 0;
}

function onScroll(): void {
  scrollTarget = window.scrollY;
  if (reducedMotion) {
    scrollPosition = scrollTarget;
    drawScene(0);
  }
}

function updateAmbientState(delta: number): void {
  const pointerEase = 1 - Math.exp(-delta * 12);
  const scrollEase = 1 - Math.exp(-delta * 5);

  pointer.x += (pointer.targetX - pointer.x) * pointerEase;
  pointer.y += (pointer.targetY - pointer.y) * pointerEase;
  pointer.activity += ((pointer.active ? 1 : 0) - pointer.activity) * (1 - Math.exp(-delta * 7));
  pointer.speed *= Math.pow(0.12, delta);
  scrollPosition += (scrollTarget - scrollPosition) * scrollEase;
}

function updateDay(delta: number, time: number): void {
  for (const particle of dayParticles) {
    if (pointer.active) {
      const dx = pointer.x - particle.x;
      const dy = pointer.y - particle.y;
      const distance = Math.hypot(dx, dy) || 1;
      const influence = Math.max(0, 1 - distance / 180);
      particle.vx -= (dx / distance) * influence * delta * 7;
      particle.vy -= (dy / distance) * influence * delta * 7;
    }

    particle.x += (particle.vx + Math.cos(time * 0.18 + particle.phase) * 1.2) * delta;
    particle.y += (particle.vy + Math.sin(time * 0.16 + particle.phase) * 0.9) * delta;

    if (particle.x < -36) particle.x = width + 36;
    if (particle.x > width + 36) particle.x = -36;
    if (particle.y < 62) particle.y = height + 24;
    if (particle.y > height + 36) particle.y = 62;
  }
}

function updateNight(delta: number, time: number): void {
  for (const node of nightNodes) {
    if (pointer.active) {
      const dx = pointer.x - node.x;
      const dy = pointer.y - node.y;
      const distance = Math.hypot(dx, dy) || 1;
      const influence = Math.max(0, 1 - distance / 240);
      node.vx += (dx / distance) * influence * delta * 8;
      node.vy += (dy / distance) * influence * delta * 8;
    }

    node.vx += Math.cos(node.phase + time * 0.22) * delta * 0.4;
    node.vy += Math.sin(node.phase + time * 0.19) * delta * 0.4;
    node.vx *= 0.998;
    node.vy *= 0.998;
    node.x += node.vx * delta;
    node.y += node.vy * delta;

    if (node.x < 18 || node.x > width - 18) {
      node.vx *= -1;
      node.x = Math.min(width - 18, Math.max(18, node.x));
    }
    if (node.y < 72 || node.y > height - 18) {
      node.vy *= -1;
      node.y = Math.min(height - 18, Math.max(72, node.y));
    }
  }
}

function drawContour(cx: number, cy: number, radiusX: number, radiusY: number, time: number, phase: number): void {
  if (!context) return;

  for (let ring = 0; ring < 7; ring += 1) {
    const rx = radiusX + ring * 28;
    const ry = radiusY + ring * 23;
    context.beginPath();

    for (let point = 0; point <= 96; point += 1) {
      const angle = (point / 96) * Math.PI * 2;
      const wobble = 1 + Math.sin(angle * 3 + phase + time * 0.08) * 0.032 + Math.cos(angle * 5 - time * 0.05) * 0.018;
      const x = cx + Math.cos(angle) * rx * wobble;
      const y = cy + Math.sin(angle) * ry * wobble;
      if (point === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    }

    context.closePath();
    context.strokeStyle = `rgba(31, 35, 40, ${0.032 + ring * 0.003})`;
    context.lineWidth = 1;
    context.stroke();
  }
}

function drawDayFlow(time: number): void {
  if (!context) return;

  const cursorShift = pointer.activity * (pointer.x - width / 2) * 0.012;

  context.save();
  context.lineWidth = 0.75;

  for (let line = 0; line < 5; line += 1) {
    const phase = line * 1.27;
    const baseY = height * (0.15 + line * 0.175) - scrollPosition * (0.008 + line * 0.002);
    const amplitude = 18 + line * 4;
    const drift = Math.sin(time * 0.11 + phase) * amplitude;

    context.beginPath();
    context.moveTo(-90, baseY + drift * 0.3);
    context.bezierCurveTo(
      width * 0.24 + cursorShift,
      baseY + amplitude + drift,
      width * 0.7 - cursorShift,
      baseY - amplitude * 1.2 - drift * 0.55,
      width + 90,
      baseY + Math.sin(time * 0.09 + phase) * amplitude * 0.45
    );
    context.strokeStyle = `rgba(9, 105, 218, ${0.013 + line * 0.002})`;
    context.stroke();
  }

  context.restore();
}

function drawNightOrbits(time: number): void {
  if (!context) return;

  const cursorX = pointer.activity * (pointer.x - width / 2) * 0.018;
  const cursorY = pointer.activity * (pointer.y - height / 2) * 0.014;

  context.save();
  context.lineWidth = 0.75;

  for (let orbit = 0; orbit < 4; orbit += 1) {
    const phase = orbit * 0.83;
    const centerX = width * (orbit % 2 === 0 ? 0.18 : 0.82) + cursorX * (orbit + 1);
    const centerY = height * (0.22 + orbit * 0.19) + cursorY - scrollPosition * 0.012;
    const radiusX = 150 + orbit * 74;
    const radiusY = 42 + orbit * 17;

    context.beginPath();
    context.ellipse(centerX, centerY, radiusX, radiusY, time * 0.008 + phase, 0, Math.PI * 2);
    context.strokeStyle = `rgba(88, 166, 255, ${0.018 + orbit * 0.003})`;
    context.stroke();
  }

  context.restore();
}

function drawDay(time: number): void {
  if (!context) return;

  const parallax = scrollPosition * 0.018;
  const cursorX = pointer.activity * (pointer.x - width / 2) * 0.012;
  const cursorY = pointer.activity * (pointer.y - height / 2) * 0.01;

  drawDayFlow(time);
  drawContour(-28 + cursorX, height * 0.2 + cursorY - parallax, 120, 165, time, 0.2);
  drawContour(width + 32 - cursorX, height * 0.72 - cursorY - parallax, 138, 178, time, 2.1);
  drawContour(width * 0.08 + cursorX * 0.5, height + 46 - parallax * 1.8, 112, 134, time, 4.2);

  context.save();
  context.font = 'italic 13px Georgia, "Times New Roman", serif';
  context.textBaseline = "middle";

  for (const particle of dayParticles) {
    const shimmer = 0.72 + Math.sin(time * 0.42 + particle.phase) * 0.18;
    context.fillStyle = `rgba(31, 35, 40, ${particle.alpha * shimmer * 0.34})`;
    context.beginPath();
    context.arc(particle.x, particle.y, particle.radius, 0, Math.PI * 2);
    context.fill();

    if (particle.label) {
      context.fillStyle = `rgba(31, 35, 40, ${particle.alpha * 0.23})`;
      context.fillText(particle.label, particle.x + 12, particle.y - 10);
    }
  }

  if (pointer.activity > 0.01) {
    for (const particle of dayParticles) {
      const distance = Math.hypot(pointer.x - particle.x, pointer.y - particle.y);
      if (distance > 145) continue;

      context.strokeStyle = `rgba(9, 105, 218, ${(1 - distance / 145) * 0.055 * pointer.activity})`;
      context.lineWidth = 0.7;
      context.beginPath();
      context.moveTo(pointer.x, pointer.y);
      context.lineTo(particle.x, particle.y);
      context.stroke();
    }

    context.strokeStyle = `rgba(9, 105, 218, ${0.1 * pointer.activity})`;
    context.lineWidth = 0.8;
    context.beginPath();
    context.arc(pointer.x, pointer.y, 34 + Math.min(pointer.speed, 20), 0, Math.PI * 2);
    context.stroke();

    context.strokeStyle = `rgba(9, 105, 218, ${0.035 * pointer.activity})`;
    context.beginPath();
    context.arc(pointer.x, pointer.y, 62 + Math.sin(time * 1.4) * 3, 0, Math.PI * 2);
    context.stroke();
  }
  context.restore();
}

function drawNight(time: number): void {
  if (!context) return;

  drawNightOrbits(time);

  context.save();
  for (let first = 0; first < nightNodes.length; first += 1) {
    for (let second = first + 1; second < nightNodes.length; second += 1) {
      const a = nightNodes[first];
      const b = nightNodes[second];
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const distance = Math.hypot(dx, dy);
      if (distance > 185) continue;

      const strength = (1 - distance / 185) * 0.13;
      const midpointX = (a.x + b.x) / 2;
      const midpointY = (a.y + b.y) / 2;
      const curve = Math.sin(time * 0.12 + a.phase + b.phase) * Math.min(12, distance * 0.06);
      const normalX = distance ? -dy / distance : 0;
      const normalY = distance ? dx / distance : 0;

      context.strokeStyle = `rgba(88, 166, 255, ${strength})`;
      context.lineWidth = 0.8;
      context.beginPath();
      context.moveTo(a.x, a.y);
      context.quadraticCurveTo(midpointX + normalX * curve, midpointY + normalY * curve, b.x, b.y);
      context.stroke();

      if ((first * 17 + second * 7) % 29 === 0) {
        const progress = (time * 0.075 + first * 0.11 + second * 0.037) % 1;
        const inverse = 1 - progress;
        const controlX = midpointX + normalX * curve;
        const controlY = midpointY + normalY * curve;
        const pulseX = inverse * inverse * a.x + 2 * inverse * progress * controlX + progress * progress * b.x;
        const pulseY = inverse * inverse * a.y + 2 * inverse * progress * controlY + progress * progress * b.y;

        context.fillStyle = `rgba(121, 192, 255, ${Math.min(0.5, strength * 3.8)})`;
        context.beginPath();
        context.arc(pulseX, pulseY, 1.25, 0, Math.PI * 2);
        context.fill();
      }
    }
  }

  for (const node of nightNodes) {
    const shimmer = 0.78 + Math.sin(time * 0.5 + node.phase) * 0.18;
    context.fillStyle = `rgba(201, 209, 217, ${node.alpha * shimmer})`;
    context.beginPath();
    context.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
    context.fill();

    if (node.radius > 1.9) {
      context.strokeStyle = `rgba(88, 166, 255, ${node.alpha * 0.22})`;
      context.beginPath();
      context.arc(node.x, node.y, node.radius + 5, 0, Math.PI * 2);
      context.stroke();
    }
  }

  if (pointer.activity > 0.01) {
    context.strokeStyle = `rgba(88, 166, 255, ${0.17 * pointer.activity})`;
    context.lineWidth = 0.8;
    context.beginPath();
    context.ellipse(pointer.x, pointer.y, 72, 28, time * 0.08, 0, Math.PI * 2);
    context.stroke();

    context.strokeStyle = `rgba(201, 209, 217, ${0.055 * pointer.activity})`;
    context.beginPath();
    context.ellipse(pointer.x, pointer.y, 106, 42, -time * 0.045, 0, Math.PI * 2);
    context.stroke();
  }
  context.restore();
}

function drawScene(timestamp: number): void {
  if (!context) return;
  context.clearRect(0, 0, width, height);
  const time = timestamp * 0.001;
  if (isDark.value) drawNight(time);
  else drawDay(time);
}

function animate(timestamp: number): void {
  const delta = Math.min(0.033, (timestamp - lastFrame || 16) / 1000);
  lastFrame = timestamp;

  updateAmbientState(delta);
  if (isDark.value) updateNight(delta, timestamp * 0.001);
  else updateDay(delta, timestamp * 0.001);
  drawScene(timestamp);
  animationFrame = window.requestAnimationFrame(animate);
}

function startAnimation(): void {
  window.cancelAnimationFrame(animationFrame);
  lastFrame = 0;
  if (reducedMotion) {
    drawScene(0);
    return;
  }
  animationFrame = window.requestAnimationFrame(animate);
}

function onVisibilityChange(): void {
  if (document.hidden) window.cancelAnimationFrame(animationFrame);
  else startAnimation();
}

watch(isDark, () => {
  seedDayScene();
  seedNightScene();
  startAnimation();
});

onMounted(() => {
  reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  scrollTarget = window.scrollY;
  scrollPosition = scrollTarget;
  resizeBackingStore();
  window.addEventListener("resize", resizeBackingStore);
  window.addEventListener("scroll", onScroll, { passive: true });
  window.addEventListener("pointermove", onPointerMove, { passive: true });
  window.addEventListener("pointerleave", onPointerLeave);
  document.addEventListener("visibilitychange", onVisibilityChange);
  startAnimation();
});

onBeforeUnmount(() => {
  window.cancelAnimationFrame(animationFrame);
  window.removeEventListener("resize", resizeBackingStore);
  window.removeEventListener("scroll", onScroll);
  window.removeEventListener("pointermove", onPointerMove);
  window.removeEventListener("pointerleave", onPointerLeave);
  document.removeEventListener("visibilitychange", onVisibilityChange);
});
</script>

<template>
  <div class="home-game-shell" aria-hidden="true">
    <canvas ref="canvas" class="home-game-canvas"></canvas>
  </div>
</template>
