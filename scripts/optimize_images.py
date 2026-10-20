#!/usr/bin/env python3
"""optimize_images.py — keep the 精读版 pages (reads/<id>.html) light.

  python3 scripts/optimize_images.py          # all pages
  python3 scripts/optimize_images.py --check  # report only, exit 1 if anything would change

What it does (idempotent; safe to re-run after adding a new Dispatch):
  1. every raster in reads/img/ larger than MIN_BYTES gets a .webp sibling (long side capped
     at MAX_SIDE, animated GIFs stay animated); kept only when it saves >= MIN_SAVING.
     The original file is left in place.
  2. references to those originals in reads/*.html and docs/blog/*.md are pointed at the .webp.
  3. every <img> in reads/*.html gets loading="lazy" decoding="async" and width/height
     (intrinsic size, so the page does not jump while images arrive).

Needs Pillow with WebP support (python3 -m pip install pillow).
"""
import argparse
import os
import re
import sys

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, "reads", "img")
MIN_BYTES = 150 * 1024
MAX_SIDE = 1800
MIN_SAVING = 0.30
RASTER = (".png", ".jpg", ".jpeg", ".gif")


def to_webp(src, dst):
    im = Image.open(src)
    animated = getattr(im, "is_animated", False)
    if animated:
        frames, durations = [], []
        for i in range(im.n_frames):
            im.seek(i)
            f = im.convert("RGBA")
            f.thumbnail((MAX_SIDE, MAX_SIDE))
            frames.append(f)
            durations.append(im.info.get("duration", 100))
        frames[0].save(dst, "WEBP", save_all=True, append_images=frames[1:], duration=durations,
                       loop=im.info.get("loop", 0), quality=75, method=6)
        return
    has_alpha = im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info)
    im = im.convert("RGBA" if has_alpha else "RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    im.save(dst, "WEBP", quality=82, method=6)


def convert_all(check):
    """returns {relative original path under reads/: relative webp path}"""
    mapping = {}
    for dirpath, _, files in os.walk(IMG_DIR):
        for name in sorted(files):
            if not name.lower().endswith(RASTER):
                continue
            src = os.path.join(dirpath, name)
            dst = os.path.splitext(src)[0] + ".webp"
            rel = os.path.relpath(src, os.path.join(ROOT, "reads")).replace(os.sep, "/")
            if os.path.exists(dst):
                if os.path.getsize(dst) <= os.path.getsize(src) * (1 - MIN_SAVING):
                    mapping[rel] = os.path.splitext(rel)[0] + ".webp"
                continue
            if os.path.getsize(src) < MIN_BYTES:
                continue
            out = dst + ".tmp" if check else dst
            to_webp(src, out)
            before, after = os.path.getsize(src), os.path.getsize(out)
            if check or after > before * (1 - MIN_SAVING):
                os.remove(out)
            if after > before * (1 - MIN_SAVING):
                if not check:
                    print(f"skip    {rel} (webp saves only {1 - after / before:.0%})")
                continue
            if check:
                print(f"would convert {rel}")
                mapping[rel] = None
                continue
            mapping[rel] = os.path.splitext(rel)[0] + ".webp"
            print(f"webp    {rel}  {before // 1024} KB -> {after // 1024} KB")
    return mapping


def image_size(path):
    if path.endswith(".svg"):
        with open(path, encoding="utf8") as f:
            head = f.read(2000)
        m = re.search(r'viewBox="[-\d.]+[ ,]+[-\d.]+[ ,]+([\d.]+)[ ,]+([\d.]+)"', head)
        return (round(float(m.group(1))), round(float(m.group(2)))) if m else None
    with Image.open(path) as im:
        return im.size


def fix_img_tag(tag, page_dir):
    src = re.search(r'src="([^"]+)"', tag)
    if not src or src.group(1).startswith(("http:", "https:", "data:")):
        return tag
    add = []
    if "loading=" not in tag:
        add.append('loading="lazy"')
    if "decoding=" not in tag:
        add.append('decoding="async"')
    if "width=" not in tag and "height=" not in tag:
        path = os.path.join(page_dir, src.group(1))
        size = image_size(path) if os.path.exists(path) else None
        if size:
            add.append(f'width="{size[0]}" height="{size[1]}"')
    if not add:
        return tag
    end = "/>" if tag.endswith("/>") else ">"
    return tag[: -len(end)].rstrip() + " " + " ".join(add) + end


def rewrite(path, mapping, page_dir, fix_imgs):
    with open(path, encoding="utf8") as f:
        text = f.read()
    new = text
    for orig, webp in mapping.items():
        if webp:
            # html uses img/<id>/x.png, markdown uses reads/img/<id>/x.png
            new = re.sub(r'(?<=["(/])' + re.escape(orig) + r'(?=[")\s])', webp, new)
    if fix_imgs:
        new = re.sub(r"<img\b[^>]*>", lambda m: fix_img_tag(m.group(0), page_dir), new)
    return text, new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    mapping = convert_all(args.check)
    targets = []
    reads = os.path.join(ROOT, "reads")
    for name in sorted(os.listdir(reads)):
        if name.endswith(".html"):
            targets.append((os.path.join(reads, name), reads, True))
    blog = os.path.join(ROOT, "docs", "blog")
    for name in sorted(os.listdir(blog)):
        if name.endswith(".md"):
            targets.append((os.path.join(blog, name), blog, False))
    changed = sum(1 for v in mapping.values() if v is None)
    for path, page_dir, fix_imgs in targets:
        old, new = rewrite(path, mapping, page_dir, fix_imgs)
        if old != new:
            changed += 1
            print(("would update " if args.check else "updated ") + os.path.relpath(path, ROOT))
            if not args.check:
                with open(path, "w", encoding="utf8") as f:
                    f.write(new)
    if args.check and changed:
        sys.exit(1)


if __name__ == "__main__":
    main()
