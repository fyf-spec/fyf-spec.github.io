<script setup lang="ts">
import { computed } from "vue";
import { useRoute } from "vitepress";
import { getBlogByPath, getBlogLanguageByPath } from "../content";

const route = useRoute();

const blog = computed(() => getBlogByPath(route.path));
const activeLanguage = computed(() => getBlogLanguageByPath(route.path));
const languages = computed(() => blog.value?.languages ?? []);
</script>

<template>
  <nav v-if="languages.length > 1" class="article-language-switch" aria-label="Article language">
    <span class="article-language-label">Language</span>
    <div class="article-language-options">
      <a
        v-for="language in languages"
        :key="language.code"
        :href="language.link"
        :lang="language.code"
        :hreflang="language.code"
        :aria-current="activeLanguage?.code === language.code ? 'page' : undefined"
        :title="language.title"
      >
        {{ language.label }}
      </a>
    </div>
  </nav>
</template>
