<script setup lang="ts">
import { computed } from "vue";
import { useRoute } from "vitepress";
import { getNoteSectionByPath, getPageKind } from "../content";

const route = useRoute();

const pageKind = computed(() => getPageKind(route.path));
const noteSection = computed(() => getNoteSectionByPath(route.path));
</script>

<template>
  <div v-if="pageKind === 'note-doc' && noteSection" class="page-meta-block page-meta-note">
    <div class="page-meta-row">
      <span class="page-pill note">Notes</span>
      <span class="page-meta-collection">{{ noteSection.title }}</span>
    </div>
    <p class="page-meta-summary">{{ noteSection.summary }}</p>
  </div>
</template>
