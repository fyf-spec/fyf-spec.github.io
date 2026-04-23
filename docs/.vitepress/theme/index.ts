import DefaultTheme from "vitepress/theme";
import { useData, useRoute } from "vitepress";
import { h, nextTick, watch } from "vue";
import BlogCatalog from "./components/BlogCatalog.vue";
import HomeLanding from "./components/HomeLanding.vue";
import PageFooterNav from "./components/PageFooterNav.vue";
import PageMeta from "./components/PageMeta.vue";
import { getPageKind } from "./content";
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
      "doc-before": () => h(PageMeta),
      "doc-after": () => h(PageFooterNav)
    });
  },
  enhanceApp({ app }) {
    app.component("BlogCatalog", BlogCatalog);
    app.component("HomeLanding", HomeLanding);
  },
  setup() {
    const route = useRoute();
    const { isDark } = useData();

    const syncPageKind = (): void => {
      if (typeof document === "undefined") {
        return;
      }

      const root = document.documentElement;
      root.dataset.pageKind = getPageKind(route.path);
      root.classList.remove("page-ready");
      requestAnimationFrame(() => {
        root.classList.add("page-ready");
      });
    };

    const rerender = (): void => {
      void nextTick(() => {
        syncPageKind();
        void renderMermaidDiagrams(isDark.value);
      });
    };

    watch(() => route.path, rerender, { immediate: true });
    watch(isDark, rerender);
  }
};
