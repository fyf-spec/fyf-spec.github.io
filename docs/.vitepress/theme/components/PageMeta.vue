<script setup lang="ts">
import { computed } from "vue";
import { useRoute } from "vitepress";
import { categoryLabels, getBlogByPath, getNoteSectionByPath, getPageKind } from "../content";

const route = useRoute();

const pageKind = computed(() => getPageKind(route.path));
const blog = computed(() => getBlogByPath(route.path));
const noteSection = computed(() => getNoteSectionByPath(route.path));
</script>

<template>
  <div v-if="pageKind === 'blog-post' && blog" class="page-meta-block page-meta-blog">
    <div class="page-meta-row">
      <span class="page-pill" :class="blog.category">{{ categoryLabels[blog.category] }}</span>
      <span class="page-meta-date">{{ blog.date }}</span>
    </div>
    <p class="page-meta-summary">{{ blog.subtitle }}</p>
  </div>

  <div v-else-if="pageKind === 'note-doc' && noteSection" class="page-meta-block page-meta-note">
    <div class="page-meta-row">
      <span class="page-pill note">Notes</span>
      <span class="page-meta-collection">{{ noteSection.title }}</span>
    </div>
    <p class="page-meta-summary">{{ noteSection.summary }}</p>
  </div>
</template>
