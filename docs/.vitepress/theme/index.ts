import DefaultTheme from "vitepress/theme";
import { useData, useRoute } from "vitepress";
import { h, nextTick, onMounted, onUnmounted, watch } from "vue";
import ArticleLanguageSwitch from "./components/ArticleLanguageSwitch.vue";
import BlogCatalog from "./components/BlogCatalog.vue";
import HomeLanding from "./components/HomeLanding.vue";
import NoteVisual from "./components/NoteVisual.vue";
import PageFooterNav from "./components/PageFooterNav.vue";
import PageMeta from "./components/PageMeta.vue";
import SpectrumMicroscope from "./components/svd/SpectrumMicroscope.vue";
import SvdExplorer from "./components/svd/SvdExplorer.vue";
import { getBlogByPath, getPageKind } from "./content";
import "./style.css";

function escapeHtml(input: string): string {
  return input
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

async function renderMermaidDiagrams(isDark: boolean): Promise<void> {
  if (typeof window === "undefined") {
    return;
  }

  const mermaid = (await import("mermaid")).default;
  mermaid.initialize({
    startOnLoad: false,
    securityLevel: "loose",
    theme: isDark ? "dark" : "default"
  });

  const containers = document.querySelectorAll<HTMLElement>(".language-mermaid");

  for (const container of containers) {
    let source = "";

    if (container.dataset.mermaidSource) {
      source = decodeURIComponent(container.dataset.mermaidSource);
    } else {
      const codeEl = container.querySelector("code");
      source = codeEl?.textContent?.trim() ?? "";
      if (!source) {
        continue;
      }
      container.dataset.mermaidSource = encodeURIComponent(source);
    }

    const renderedKey = `${isDark ? "dark" : "light"}:${source}`;
    if (container.dataset.mermaidRenderedKey === renderedKey) {
      continue;
    }

    try {
      const id = `vp-mermaid-${Math.random().toString(36).slice(2, 10)}`;
      const { svg } = await mermaid.render(id, source);
      container.innerHTML = `<div class="vp-mermaid">${svg}</div>`;
      container.dataset.mermaidRenderedKey = renderedKey;
    } catch {
      container.innerHTML = `<pre class="vp-mermaid-error">${escapeHtml(source)}</pre>`;
      container.dataset.mermaidRenderedKey = renderedKey;
    }
  }
}

export default {
  extends: DefaultTheme,
  Layout: () => {
    return h(DefaultTheme.Layout, null, {
      "doc-before": () => [h(PageMeta), h(ArticleLanguageSwitch)],
      "doc-after": () => h(PageFooterNav)
    });
  },
  enhanceApp({ app }) {
    app.component("BlogCatalog", BlogCatalog);
    app.component("HomeLanding", HomeLanding);
    app.component("NoteVisual", NoteVisual);
    app.component("SpectrumMicroscope", SpectrumMicroscope);
    app.component("SvdExplorer", SvdExplorer);
  },
  setup() {
    const route = useRoute();
    const { isDark } = useData();
    let lastScrollY = 0;
    let scrollFrame: number | undefined;

    const syncPageKind = (): void => {
      if (typeof document === "undefined") {
        return;
      }

      const root = document.documentElement;
      root.dataset.pageKind = getPageKind(route.path);
      root.classList.remove("nav-scrolled-away");
      root.classList.remove("page-ready");
      lastScrollY = typeof window === "undefined" ? 0 : window.scrollY;
      requestAnimationFrame(() => {
        root.classList.add("page-ready");
      });
    };

    const syncNavScrollState = (): void => {
      if (typeof window === "undefined" || typeof document === "undefined") {
        return;
      }

      const currentScrollY = Math.max(window.scrollY, 0);
      const root = document.documentElement;

      if (currentScrollY <= 32 || currentScrollY < lastScrollY - 6) {
        root.classList.remove("nav-scrolled-away");
      } else if (currentScrollY > 96 && currentScrollY > lastScrollY + 6) {
        root.classList.add("nav-scrolled-away");
      }

      lastScrollY = currentScrollY;
      scrollFrame = undefined;
    };

    const onScroll = (): void => {
      if (scrollFrame !== undefined || typeof window === "undefined") {
        return;
      }

      scrollFrame = window.requestAnimationFrame(syncNavScrollState);
    };

    const syncBlogTitleDate = (): void => {
      if (typeof document === "undefined") {
        return;
      }

      const existingDates = document.querySelectorAll(".blog-title-date");
      const blog = getBlogByPath(route.path);

      if (getPageKind(route.path) !== "blog-post" || !blog) {
        existingDates.forEach((element) => element.remove());
        return;
      }

      const title = document.querySelector<HTMLElement>(
        ".VPDoc > .container > .content > .content-container .vp-doc h1"
      );

      if (!title) {
        existingDates.forEach((element) => element.remove());
        return;
      }

      existingDates.forEach((element) => {
        if (element.previousElementSibling !== title) {
          element.remove();
        }
      });

      let dateElement = title.nextElementSibling as HTMLElement | null;
      if (!dateElement?.classList.contains("blog-title-date")) {
        dateElement = document.createElement("div");
        dateElement.className = "blog-title-date";
        title.insertAdjacentElement("afterend", dateElement);
      }

      dateElement.textContent = blog.date;
    };

    const queueBlogTitleDateSync = (): void => {
      if (typeof window === "undefined") {
        return;
      }

      window.requestAnimationFrame(() => {
        syncBlogTitleDate();
        window.requestAnimationFrame(syncBlogTitleDate);
      });
    };

    const rerender = (): void => {
      void nextTick(() => {
        syncPageKind();
        queueBlogTitleDateSync();
        void renderMermaidDiagrams(isDark.value);
      });
    };

    watch(() => route.path, rerender, { immediate: true });
    watch(isDark, rerender);

    onMounted(() => {
      lastScrollY = window.scrollY;
      window.addEventListener("scroll", onScroll, { passive: true });
      queueBlogTitleDateSync();
    });

    onUnmounted(() => {
      window.removeEventListener("scroll", onScroll);
      if (scrollFrame !== undefined) {
        window.cancelAnimationFrame(scrollFrame);
      }
    });
  }
};
