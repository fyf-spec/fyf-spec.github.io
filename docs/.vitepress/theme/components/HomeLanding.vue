<script setup lang="ts">
import { onMounted, ref } from "vue";
import HomeCursorGame from "./HomeCursorGame.vue";

const visible = ref(false);

onMounted(() => {
  requestAnimationFrame(() => {
    visible.value = true;
  });
});
</script>

<template>
  <section class="landing-shell landing-shell-about page-fade" :class="{ 'is-visible': visible }">
    <HomeCursorGame />
    <div class="about-panel about-panel-game about-copy">
      <p class="about-lede">
        hi, i'm Evan -  <a href="/blogs/">writings</a>
      </p>

      <p>
        i am currently learning LLM systems, algorithms, and machine learning. prior to this,
        i spent time writing course notes and small engineering writeups.
      </p>

      <p>
        research interests: model architecture, efficient attention, and gpu kernels
      </p>

      <p>here are some things i believe in:</p>

      <ul class="about-beliefs">
        <li>taste is important</li>
        <li>agency harness matters a lot</li>
        <li>the world is predictale in <em>intelligent</em> ways</li>
        <li>write things down</li>
      </ul>

      <p>
        i love people and write back. <a href="https://github.com/fyf-spec">My github!</a>
      </p>
    </div>
  </section>
</template>
