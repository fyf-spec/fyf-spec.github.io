<script setup lang="ts">
import { onMounted, ref } from "vue";
import { blogPosts, categoryLabels } from "../content";

const visible = ref(false);

function formatDate(date: string): string {
  const [year, month, day] = date.split("-");
  if (!year || !month || !day) {
    return date;
  }

  return `${month}-${day}-${year}`;
}

onMounted(() => {
  requestAnimationFrame(() => {
    visible.value = true;
  });
});
</script>

<template>
  <section class="blog-page page-fade" :class="{ 'is-visible': visible }">
    <div class="blog-layout-simple">
      <aside class="blog-legend">
        <div class="blog-legend-item technical">
          <span class="blog-legend-color"></span>
          <span>{{ categoryLabels.technical }}</span>
        </div>
        <div class="blog-legend-item reflection">
          <span class="blog-legend-color"></span>
          <span>{{ categoryLabels.reflection }}</span>
        </div>
        <div class="blog-legend-item article">
          <span class="blog-legend-color"></span>
          <span>{{ categoryLabels.article }}</span>
        </div>
      </aside>

      <div class="blog-listing blog-listing-simple" id="blog-list">
        <a
          v-for="(post, index) in blogPosts"
          :key="post.link"
          class="blog-entry blog-entry-simple"
          :class="post.category"
          :href="post.link"
          :style="{ transitionDelay: `${100 + index * 35}ms` }"
        >
          <span class="blog-entry-date">{{ formatDate(post.date) }}</span>
          <span class="blog-entry-title-wrap">
            <span class="blog-entry-title">{{ post.title }}</span>
          </span>
        </a>
      </div>
    </div>
  </section>
</template>
