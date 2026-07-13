# Evan Feng — Editorial Research System

## Direction

The site uses GitHub Primer as its structural reference: accessible Day/Night tokens, quiet one-pixel borders, readable technical navigation, explicit focus states, and responsive list-first layouts. It borrows only the editorial restraint and true-white whitespace of Thinking Machines Lab. Interactive lessons use the document-plus-lab hierarchy seen in Stanford's Probability for Computer Science demos, without copying branding, assets, or source code.

Primer is the single primary design language. The implementation recreates the relevant principles in the existing VitePress/Vue stack instead of importing a component library.

## Shortlist

| System | Fit | Tradeoff | License / reuse decision |
| --- | --- | --- | --- |
| GitHub Primer | Best match for technical writing, code, docs, navigation, and accessible Day/Night themes | Product UI defaults need an editorial type layer | Primer CSS is MIT; principles were recreated locally |
| Vercel Geist | Strong developer typography and compact controls | Reads as a product/deployment console more than a research notebook | Font is OFL; not imported to avoid a new asset dependency |
| IBM Carbon | Excellent accessibility and data-heavy controls, including Vue options | Too dense and enterprise-oriented for a personal knowledge site | Open source, but no Carbon code or assets were copied |

## Tokens

| Role | Day | Night |
| --- | --- | --- |
| Canvas | `#ffffff` | `#0d1117` |
| Raised surface | `#ffffff` | `#161b22` |
| Muted surface | `#f6f8fa` | `#161b22` |
| Primary text | `#1f2328` | `#f0f6fc` |
| Secondary text | `#59636e` | `#8b949e` |
| Border | `#d0d7de` | `#30363d` |
| Accent / focus | `#0969da` | `#4493f8` |
| Small / medium / large radius | `6px / 8px / 12px` | same |

Typography uses an editorial serif stack for reading and headings, the system sans stack for navigation and controls, and a native monospace stack for matrices, code, and numeric output. This is an intentional brand adaptation of Primer.

## Components

- Navigation: one quiet top rail, brand, essential links, and the existing accessible theme control.
- Homepage: open typographic hero, profile/contact links directly beneath the introduction, ruled beliefs, notes index, and recent-writing index.
- Writing index: text filters and table-like rows; no colored category tiles or card grid.
- Articles and notes: narrow reading measure, open page canvas, stable outline navigation, restrained code/table/callout surfaces. A translated article is always one catalog entity; language variants are exposed through an understated in-article underline switch, never duplicated as separate writing rows.
- Interactive labs: one thin worksheet boundary, text tabs, square inputs, visible focus, 2×2 transform plots on desktop, one plot at a time on small screens.
- Background motion: Day uses drifting mathematical particles, slow contour lines, pointer-linked threads, and low-contrast flow lines. Night uses sparse nodes, curved orbital connections, traveling signal pulses, and quiet orbital traces. The scenes use smoothed pointer depth and scroll parallax, remain visually distinct, pause when hidden, and both respect reduced motion.

## Motion rules

- Motion must reveal hierarchy or make the mathematical background feel spatial; it must not introduce decorative cards, glows, badges, or new color families.
- Hero elements enter once with short staggered timing. Downstream sections reveal once at the viewport edge. Hover motion is limited to underlines and 2–3 px directional icon travel.
- Day and Night retain the existing true-white and near-black canvases. Background lines remain below text contrast and never intercept pointer events.
- `prefers-reduced-motion: reduce` removes entrance, reveal, underline, and icon transitions and renders each canvas as a static scene.

## Concept references

- `design/concepts/homepage-day.png`
- `design/concepts/homepage-night.png`
- `design/concepts/homepage-sections-day.png`
- `design/concepts/writing-index-day.png`
- `design/concepts/svd-article-day.png`
- `design/concepts/svd-article-night.png`

The concepts are layout and token references. All visible UI is implemented as native Vue, HTML, CSS, canvas, and SVG.
