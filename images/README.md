# Screenshots

Drop PNG/JPG files here that you want to reference from the main `README.md`
or `README.tech.md`.

## Naming convention

Kebab-case, descriptive, no spaces:

- `ui-overview.png` — full app interface
- `ui-progress-gpu.png` — progress bar + GPU panel during inference
- `ui-history-strip.png` — history thumbnails
- `before-after-hairdryer.jpg` — side-by-side before/after of an edit
- `architecture.svg` — pipeline diagrams (SVG renders cleaner than PNG)
- `vram-layout.png` — VRAM breakdown chart

## File size

GitHub renders inline up to a few MB without slowdown. Keep screenshots:

- **Width ≤ 1600 px** — wider is wasted, GitHub clips to ~960 px on the
  README page anyway.
- **PNG** for UI screenshots (lossless, sharp text).
- **JPG quality 80** for photo content (input/output images).
- Run through a compressor (e.g. `pngquant`, `oxipng`, `mozjpeg`) before
  committing — typical screenshot drops from ~1 MB to ~100 KB without
  visible loss.

## Reference from README

Relative path from repo root:

```markdown
![UI overview](images/ui-overview.png)
```

For side-by-side comparison use an HTML table — markdown image side-by-side
doesn't render reliably:

```markdown
| before | after |
|---|---|
| ![](images/before.jpg) | ![](images/after.jpg) |
```

## Privacy reminder

This is a public repo. Anything committed here is forever, indexed by
search engines, and impossible to take back. Strip EXIF, anonymize faces
if needed, and double-check before `git add` on photos of yourself or
identifiable locations.
