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
    <div class="about-panel about-panel-game">
      <div class="landing-kicker">About</div>
      <h1>About</h1>
      <div class="about-panel-body"></div>
    </div>
  </section>
</template>
