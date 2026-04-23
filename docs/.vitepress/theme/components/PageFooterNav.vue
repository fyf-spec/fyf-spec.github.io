<script setup lang="ts">
import { computed } from "vue";
import { useRoute } from "vitepress";
import { getBlogByPath, getNoteSectionByPath, getPageKind } from "../content";

const route = useRoute();

const pageKind = computed(() => getPageKind(route.path));
const blog = computed(() => getBlogByPath(route.path));
const noteSection = computed(() => getNoteSectionByPath(route.path));
</script>

<template>
  <div v-if="pageKind === 'blog-post' && blog" class="page-tail-nav">
    <a class="page-tail-link" href="/blogs/">All Blogs</a>
    <a class="page-tail-link" href="/">Home</a>
  </div>

  <div v-else-if="pageKind === 'note-doc' && noteSection" class="page-tail-nav">
    <a class="page-tail-link" :href="noteSection.link">{{ noteSection.title }}</a>
    <a class="page-tail-link" href="/">Home</a>
  </div>
</template>
