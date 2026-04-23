<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from "vue";

type SignalNode = {
  x: number;
  y: number;
  vx: number;
  vy: number;
  energy: number;
  size: number;
  phase: number;
};

type Pulse = {
  x: number;
  y: number;
  radius: number;
  life: number;
  maxLife: number;
};

const canvas = ref<HTMLCanvasElement | null>(null);

const pointer = {
  active: false,
  x: 0,
  y: 0,
  lastX: 0,
  lastY: 0,
  speed: 0
};

const nodes: SignalNode[] = [];
const pulses: Pulse[] = [];

let context: CanvasRenderingContext2D | null = null;
let width = 0;
let height = 0;
let animationFrame = 0;
let lastFrame = 0;
let lastPulseAt = 0;

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function randomBetween(min: number, max: number): number {
  return min + Math.random() * (max - min);
}

function resetNode(node: SignalNode): void {
  const margin = 80;
  node.x = randomBetween(margin, Math.max(margin + 1, width - margin));
  node.y = randomBetween(margin, Math.max(margin + 1, height - margin));
  node.vx = randomBetween(-18, 18);
  node.vy = randomBetween(-18, 18);
  node.energy = randomBetween(0.12, 0.45);
  node.size = randomBetween(3.2, 6.8);
  node.phase = Math.random() * Math.PI * 2;
}

function seedNodes(): void {
  nodes.length = 0;

  const count = Math.max(24, Math.min(46, Math.round(width / 38)));
  for (let index = 0; index < count; index += 1) {
    const node: SignalNode = {
      x: 0,
      y: 0,
      vx: 0,
      vy: 0,
      energy: 0,
      size: 0,
      phase: 0
    };

    resetNode(node);
    nodes.push(node);
  }
}

function resizeCanvas(): void {
  if (!canvas.value) {
    return;
  }

  width = window.innerWidth;
  height = window.innerHeight;

  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.value.width = Math.max(1, Math.floor(width * ratio));
  canvas.value.height = Math.max(1, Math.floor(height * ratio));

  context = canvas.value.getContext("2d");
  if (!context) {
    return;
  }

  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  seedNodes();
}

function emitPulse(x: number, y: number, maxLife = 0.9): void {
  pulses.push({
    x,
    y,
    radius: 12,
    life: 0,
    maxLife
  });
}

function onPointerMove(event: PointerEvent): void {
  const nextX = event.clientX;
  const nextY = event.clientY;

  pointer.speed = Math.hypot(nextX - pointer.lastX, nextY - pointer.lastY);
  pointer.lastX = nextX;
  pointer.lastY = nextY;
  pointer.x = nextX;
  pointer.y = nextY;
  pointer.active = true;
}

function onPointerLeave(): void {
  pointer.active = false;
  pointer.speed = 0;
}

function updateScene(delta: number, time: number): void {
  if (pointer.active && pointer.speed > 18 && time - lastPulseAt > 0.08) {
    emitPulse(pointer.x, pointer.y, 0.72);
    lastPulseAt = time;
  }

  for (let index = pulses.length - 1; index >= 0; index -= 1) {
    const pulse = pulses[index];
    pulse.life += delta;
    pulse.radius += delta * 220;

    if (pulse.life >= pulse.maxLife) {
      pulses.splice(index, 1);
    }
  }

  const margin = 30;

  for (const node of nodes) {
    node.energy = Math.max(0.08, node.energy - delta * 0.04);

    if (pointer.active) {
      const dx = pointer.x - node.x;
      const dy = pointer.y - node.y;
      const distance = Math.hypot(dx, dy) || 1;
      const influence = Math.max(0, 1 - distance / 260);

      if (influence > 0) {
        const attraction = influence * (14 + Math.min(pointer.speed * 0.05, 22));
        node.vx += (dx / distance) * attraction * delta;
        node.vy += (dy / distance) * attraction * delta;
        node.energy = Math.min(1.2, node.energy + influence * delta * 0.8);
      }
    }

    for (const pulse of pulses) {
      const dx = node.x - pulse.x;
      const dy = node.y - pulse.y;
      const distance = Math.hypot(dx, dy) || 1;
      const edge = Math.abs(distance - pulse.radius);

      if (edge < 26) {
        const wave = 1 - edge / 26;
        node.vx += (dx / distance) * wave * delta * 32;
        node.vy += (dy / distance) * wave * delta * 32;
        node.energy = Math.min(1.25, node.energy + wave * delta * 1.8);
      }
    }

    node.vx += Math.cos(node.phase + time * 0.9) * delta * 1.8;
    node.vy += Math.sin(node.phase + time * 1.1) * delta * 1.8;
    node.vx *= 0.988;
    node.vy *= 0.988;

    node.x += node.vx * delta;
    node.y += node.vy * delta;

    if (node.x < margin || node.x > width - margin) {
      node.vx *= -1;
      node.x = clamp(node.x, margin, width - margin);
    }

    if (node.y < margin || node.y > height - margin) {
      node.vy *= -1;
      node.y = clamp(node.y, margin, height - margin);
    }
  }
}

function drawBackgroundGlow(): void {
  if (!context) {
    return;
  }

  context.save();

  const leftGlow = context.createRadialGradient(width * 0.18, height * 0.25, 0, width * 0.18, height * 0.25, width * 0.34);
  leftGlow.addColorStop(0, "rgba(33, 117, 255, 0.16)");
  leftGlow.addColorStop(1, "rgba(33, 117, 255, 0)");
  context.fillStyle = leftGlow;
  context.fillRect(0, 0, width, height);

  const rightGlow = context.createRadialGradient(width * 0.82, height * 0.68, 0, width * 0.82, height * 0.68, width * 0.3);
  rightGlow.addColorStop(0, "rgba(0, 229, 255, 0.12)");
  rightGlow.addColorStop(1, "rgba(0, 229, 255, 0)");
  context.fillStyle = rightGlow;
  context.fillRect(0, 0, width, height);

  context.restore();
}

function drawLinks(): void {
  if (!context) {
    return;
  }

  context.save();

  for (let i = 0; i < nodes.length; i += 1) {
    for (let j = i + 1; j < nodes.length; j += 1) {
      const a = nodes[i];
      const b = nodes[j];
      const distance = Math.hypot(a.x - b.x, a.y - b.y);

      if (distance > 180) {
        continue;
      }

      const intensity = (1 - distance / 180) * (0.12 + (a.energy + b.energy) * 0.16);
      context.strokeStyle = `rgba(96, 188, 255, ${intensity})`;
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(a.x, a.y);
      context.lineTo(b.x, b.y);
      context.stroke();
    }
  }

  if (pointer.active) {
    for (const node of nodes) {
      const distance = Math.hypot(pointer.x - node.x, pointer.y - node.y);
      if (distance > 160) {
        continue;
      }

      const intensity = (1 - distance / 160) * 0.22;
      context.strokeStyle = `rgba(146, 238, 255, ${intensity})`;
      context.beginPath();
      context.moveTo(pointer.x, pointer.y);
      context.lineTo(node.x, node.y);
      context.stroke();
    }
  }

  context.restore();
}

function drawPulses(): void {
  if (!context) {
    return;
  }

  context.save();

  for (const pulse of pulses) {
    const alpha = 1 - pulse.life / pulse.maxLife;
    context.strokeStyle = `rgba(110, 216, 255, ${alpha * 0.28})`;
    context.lineWidth = 1.5;
    context.beginPath();
    context.arc(pulse.x, pulse.y, pulse.radius, 0, Math.PI * 2);
    context.stroke();
  }

  context.restore();
}

function drawNodes(time: number): void {
  if (!context) {
    return;
  }

  context.save();

  for (const node of nodes) {
    const shimmer = 0.68 + Math.sin(time * 2 + node.phase) * 0.22;
    const size = node.size + node.energy * 2;

    context.fillStyle = `rgba(220, 245, 255, ${0.42 + node.energy * 0.26})`;
    context.shadowBlur = 14 + node.energy * 18;
    context.shadowColor = `rgba(63, 184, 255, ${0.2 + node.energy * 0.34})`;
    context.beginPath();
    context.arc(node.x, node.y, size, 0, Math.PI * 2);
    context.fill();

    context.strokeStyle = `rgba(124, 208, 255, ${0.12 + shimmer * 0.2})`;
    context.lineWidth = 1;
    context.beginPath();
    context.arc(node.x, node.y, size + 8 + node.energy * 4, 0, Math.PI * 2);
    context.stroke();
  }

  context.restore();
}

function drawCursor(): void {
  if (!context || !pointer.active) {
    return;
  }

  context.save();

  const gradient = context.createRadialGradient(pointer.x, pointer.y, 0, pointer.x, pointer.y, 180);
  gradient.addColorStop(0, "rgba(73, 209, 255, 0.16)");
  gradient.addColorStop(1, "rgba(73, 209, 255, 0)");
  context.fillStyle = gradient;
  context.beginPath();
  context.arc(pointer.x, pointer.y, 180, 0, Math.PI * 2);
  context.fill();

  context.strokeStyle = "rgba(166, 240, 255, 0.34)";
  context.lineWidth = 1;
  context.beginPath();
  context.arc(pointer.x, pointer.y, 42, 0, Math.PI * 2);
  context.stroke();

  context.restore();
}

function drawScene(timestamp: number): void {
  if (!context) {
    return;
  }

  const time = timestamp * 0.001;
  context.clearRect(0, 0, width, height);
  drawBackgroundGlow();
  drawLinks();
  drawPulses();
  drawNodes(time);
  drawCursor();
}

function animate(timestamp: number): void {
  const delta = Math.min(0.033, (timestamp - lastFrame || 16) / 1000);
  lastFrame = timestamp;

  updateScene(delta, timestamp * 0.001);
  drawScene(timestamp);
  animationFrame = window.requestAnimationFrame(animate);
}

onMounted(() => {
  resizeCanvas();
  window.addEventListener("resize", resizeCanvas);
  window.addEventListener("pointermove", onPointerMove, { passive: true });
  window.addEventListener("pointerleave", onPointerLeave);
  animationFrame = window.requestAnimationFrame(animate);
});

onBeforeUnmount(() => {
  window.cancelAnimationFrame(animationFrame);
  window.removeEventListener("resize", resizeCanvas);
  window.removeEventListener("pointermove", onPointerMove);
  window.removeEventListener("pointerleave", onPointerLeave);
});
</script>

<template>
  <div class="home-game-shell" aria-hidden="true">
    <canvas ref="canvas" class="home-game-canvas"></canvas>
  </div>
</template>
