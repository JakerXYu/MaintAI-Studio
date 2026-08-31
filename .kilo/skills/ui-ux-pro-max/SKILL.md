---
name: ui-ux-pro-max
description: UI/UX design intelligence for building, reviewing, or fixing interfaces, including Streamlit pages, accessibility, responsive layout, typography, color, charts, forms, navigation, and interaction feedback. Use automatically for any task that changes how an interface looks, feels, or is used.
license: MIT
metadata:
  source: https://github.com/nextlevelbuilder/ui-ux-pro-max-skill
  adapted-for: Kilo project skills and Streamlit
---

# UI/UX Pro Max

Apply this skill to visual or interaction work. Skip it for pure backend,
database, infrastructure, or non-visual scripts.

## Priority Order

1. Accessibility: 4.5:1 text contrast, visible focus, labels, keyboard access,
   and meaning that does not depend on color alone.
2. Interaction: clear loading/success/error feedback and usable touch targets.
3. Responsive layout: no horizontal overflow; content must reflow on mobile,
   zoom, and long identifiers.
4. Product fit: prefer a coherent industrial control-room visual language over
   generic gradients, decorative glass, emoji icons, or marketing-page chrome.
5. Typography and color: readable base size and line height, restrained type
   scale, semantic tokens, and consistent status colors.
6. Forms: visible labels, helper text, validation near the field, and safe
   defaults for consequential actions.
7. Navigation: predictable workflow order, obvious current location, and no
   overloaded sidebars.
8. Data displays: units, legends, evidence, thresholds, and accessible colors.
9. Motion: only when it communicates state; avoid gratuitous animation.
10. Performance: prefer native framework components and CSS over new packages.

## Industrial Dashboard Direction

- Use a calm, high-density operations console rather than a consumer SaaS page.
- Keep evidence and operational status above decoration.
- Use neutral surfaces, one restrained ABB-red accent, and semantic amber/green.
- Separate workflow stages with spacing and hierarchy, not excessive cards.
- Pair every warning or recommendation with evidence and a disclaimer.
- Make pending approvals and destructive actions visually explicit.

## Streamlit Implementation

- Reuse existing `st.*` primitives before custom HTML.
- Keep custom CSS centralized and narrowly scoped.
- Use columns only where they collapse safely on narrow screens.
- Do not use HTML for controls when a native Streamlit widget exists.
- Preserve the HTTP-only boundary: UI calls the API and never imports DB or ML
  internals.
- Test rendering helpers separately from network calls where practical.

## Pre-delivery Check

- Keyboard focus and form labels remain visible.
- Text contrast is readable and status is not color-only.
- Long IDs, tables, chips, and headings wrap without clipping.
- Desktop and mobile layouts have no horizontal page scroll.
- Loading, empty, success, and error states are understandable.
- Charts show labels, units, thresholds, and caveats.
- No new UI dependency was added when Streamlit or CSS already covered the need.

This is a Kilo/Streamlit adapter derived from UI UX Pro Max. Upstream's full
searchable catalog and CLI are optional and are not required at runtime.
