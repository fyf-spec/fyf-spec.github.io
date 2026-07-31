<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from "vue";
import { blogPosts, noteSections } from "../content";
import HomeCursorGame from "./HomeCursorGame.vue";

const visible = ref(false);
const motionReady = ref(false);
const homeRoot = ref<HTMLElement | null>(null);
const recentPosts = blogPosts.slice(0, 3);
let revealObserver: IntersectionObserver | undefined;
let revealFrame = 0;

onMounted(() => {
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const revealTargets = homeRoot.value?.querySelectorAll<HTMLElement>("[data-home-reveal]") ?? [];

  if (reduceMotion || !("IntersectionObserver" in window)) {
    revealTargets.forEach((target) => target.classList.add("is-revealed"));
  } else {
    revealObserver = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          entry.target.classList.add("is-revealed");
          revealObserver?.unobserve(entry.target);
        });
      },
      { rootMargin: "0px 0px -10%", threshold: 0.12 }
    );

    revealTargets.forEach((target) => revealObserver?.observe(target));
    motionReady.value = true;
  }

  revealFrame = requestAnimationFrame(() => {
    visible.value = true;
  });
});

onBeforeUnmount(() => {
  revealObserver?.disconnect();
  cancelAnimationFrame(revealFrame);
});
</script>

<template>
  <main
    ref="homeRoot"
    class="home-page page-fade"
    :class="{ 'is-visible': visible, 'is-motion-ready': motionReady }"
  >
    <HomeCursorGame />

    <section class="home-hero home-container" aria-labelledby="home-title">
      <div class="home-intro">
        <h1 id="home-title">Hi, I'm Evan.</h1>
        <p class="home-lede">
          I'm currently learning LLM pretraining architecture and LLM infrastructure.
        </p>
        <p class="home-focus">
          Research interests: model architecture, efficient attention, and GPU kernels.
        </p>
        <nav class="home-profile-links" aria-label="Contact and profiles">
          <a href="https://github.com/fyf-spec" target="_blank" rel="noreferrer">
            GitHub
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 17 17 7M8 7h9v9" /></svg>
          </a>
          <a href="https://x.com/yffeng3920" target="_blank" rel="noreferrer">
            X
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 17 17 7M8 7h9v9" /></svg>
          </a>
          <a href="mailto:yfeng8696@gmail.com">
            Email
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 17 17 7M8 7h9v9" /></svg>
          </a>
        </nav>
        <div class="home-actions" aria-label="Primary links">
          <a href="/blogs/">
            Read my writing
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6" /></svg>
          </a>
        </div>
      </div>

      <div class="home-beliefs" aria-labelledby="beliefs-title" data-home-reveal>
        <h2 id="beliefs-title">Beliefs</h2>
        <ol>
          <li><span>taste is important</span></li>
          <li><span>the world is predictable in <em>intelligent</em> ways</span></li>
          <li><span>agency harness matters a lot</span></li>
          <li><span>write things down</span></li>
        </ol>
      </div>
    </section>

    <section class="home-section home-container" aria-labelledby="recent-writing-title" data-home-reveal>
      <h2 id="recent-writing-title">Recent writing</h2>
      <div class="home-index-list home-writing-list">
        <a v-for="post in recentPosts" :key="post.link" :href="post.link" class="home-index-row">
          <strong>{{ post.title }}</strong>
          <time :datetime="post.date">{{ post.date }}</time>
          <span>{{ post.category === "technical" ? "Technical" : post.category === "reflection" ? "Reflection" : "Article" }}</span>
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6" /></svg>
        </a>
      </div>
    </section>

    <section id="notes" class="home-section home-container" aria-labelledby="notes-title" data-home-reveal>
      <h2 id="notes-title">Notes</h2>
      <div class="home-index-list home-notes-list">
        <a v-for="section in noteSections" :key="section.link" :href="section.link" class="home-index-row">
          <strong>{{ section.title }}</strong>
          <span>{{ section.label }}</span>
          <span>{{ section.countLabel }}</span>
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6" /></svg>
        </a>
      </div>
    </section>

  </main>
</template>
