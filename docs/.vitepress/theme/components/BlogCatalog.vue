<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { blogPosts, categoryLabels } from "../content";
import type { BlogCategory } from "../content";

type BlogFilter = "all" | BlogCategory;

const visible = ref(false);
const activeFilter = ref<BlogFilter>("all");

const filters: Array<{ value: BlogFilter; label: string }> = [
  { value: "all", label: "All" },
  { value: "technical", label: categoryLabels.technical },
  { value: "reflection", label: categoryLabels.reflection },
  { value: "article", label: categoryLabels.article }
];

const filteredPosts = computed(() => {
  if (activeFilter.value === "all") return blogPosts;
  return blogPosts.filter((post) => post.category === activeFilter.value);
});

onMounted(() => {
  requestAnimationFrame(() => {
    visible.value = true;
  });
});
</script>

<template>
  <main class="blog-page page-fade" :class="{ 'is-visible': visible }">
    <header class="blog-header">
      <h1>Writing</h1>
      <p>Technical notes, course reflections, and engineering writeups.</p>
    </header>

    <nav class="blog-filters" aria-label="Filter writing by category">
      <button
        v-for="filter in filters"
        :key="filter.value"
        type="button"
        :class="{ 'is-active': activeFilter === filter.value }"
        :aria-pressed="activeFilter === filter.value"
        @click="activeFilter = filter.value"
      >
        {{ filter.label }}
      </button>
    </nav>

    <div id="blog-list" class="blog-listing" aria-live="polite">
      <a v-for="post in filteredPosts" :key="post.link" class="blog-entry" :href="post.link">
        <time :datetime="post.date">{{ post.date }}</time>
        <span class="blog-entry-title">{{ post.title }}</span>
        <span class="blog-entry-category">{{ categoryLabels[post.category] }}</span>
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6" /></svg>
      </a>
    </div>
  </main>
</template>
