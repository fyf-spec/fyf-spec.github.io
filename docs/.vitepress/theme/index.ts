import DefaultTheme from "vitepress/theme";
import { useData, useRoute } from "vitepress";
import { nextTick, watch } from "vue";
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
  setup() {
    const route = useRoute();
    const { isDark } = useData();

    const rerender = (): void => {
      void nextTick(() => renderMermaidDiagrams(isDark.value));
    };

    watch(() => route.path, rerender, { immediate: true });
    watch(isDark, rerender);
  }
};
