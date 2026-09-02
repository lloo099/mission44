/**
 * check-reads.mjs — structural lint for the 精读版 pages in reads/.
 *   node scripts/check-reads.mjs            # every reads/<id>.html
 *   node scripts/check-reads.mjs <post-id>  # one page
 * Exit 1 on any error. Warnings do not fail the run.
 */
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const blog = JSON.parse(fs.readFileSync(path.join(ROOT, "data/blog.json"), "utf8"));
const ids = new Set(blog.posts.map((p) => p.id));
const only = process.argv[2];
const files = fs.readdirSync(path.join(ROOT, "reads")).filter((f) => f.endsWith(".html") && (!only || f === only + ".html"));
let errors = 0, warnings = 0;

for (const f of files) {
  const id = f.replace(/\.html$/, "");
  const html = fs.readFileSync(path.join(ROOT, "reads", f), "utf8");
  const err = (m) => { errors++; console.log(`✗ ${id}: ${m}`); };
  const warn = (m) => { warnings++; console.log(`! ${id}: ${m}`); };

  if (!ids.has(id)) err("file name is not a post id in data/blog.json");
  if (!/^<!DOCTYPE html>/i.test(html.trim())) err("missing <!DOCTYPE html>");
  if (!/<link rel="stylesheet" href="read.css">/.test(html)) err("must link read.css");
  if (/<style[\s>]/i.test(html)) err("inline <style> not allowed");
  if (/\{\{[A-Za-z_0-9]+\}\}/.test(html)) err("template placeholder left in page");
  if (/<(img|script|link)[^>]+(src|href)="https?:\/\/(?!lloo099\.github\.io)/i.test(html)) err("external resource (img/script/link) not allowed");
  if (!/<div class="readbar">/.test(html)) err("missing .readbar");
  if (!/<header class="masthead">/.test(html)) err("missing .masthead");
  if (!/<nav class="toc">/.test(html)) err("missing nav.toc");
  if (!/<footer>/.test(html)) err("missing <footer>");
  if (!/id="theme-toggle"/.test(html)) err("missing theme toggle");
  if (!new RegExp(`href="\\.\\./index\\.html#blog/${id}"`).test(html)) err("readbar must link back to ../index.html#blog/<id>");

  const h2ids = [...html.matchAll(/<h2 id="([^"]+)"/g)].map((m) => m[1]);
  const tocids = [...html.matchAll(/<nav class="toc">[\s\S]*?<\/nav>/g)].flatMap((m) => [...m[0].matchAll(/href="#([^"]+)"/g)].map((x) => x[1]));
  if (!h2ids.length) err("no <h2 id=…> sections");
  for (const t of tocids) if (!h2ids.includes(t)) err(`toc links to missing section #${t}`);
  for (const h of h2ids) if (!tocids.includes(h)) warn(`section #${h} not in toc`);

  const imgs = [...html.matchAll(/<img[^>]+src="([^"]+)"/g)].map((m) => m[1]);
  if (imgs.length < 3) warn(`only ${imgs.length} figure(s) (guideline: at least 3)`);
  for (const src of imgs) {
    if (!src.startsWith(`img/${id}/`)) err(`image outside img/${id}/: ${src}`);
    else if (!fs.existsSync(path.join(ROOT, "reads", src))) err(`missing image file ${src}`);
  }
  const figs = (html.match(/<figure>/g) || []).length;
  const readouts = (html.match(/class="readout"/g) || []).length;
  if (figs && readouts < figs) warn(`${figs} figures but ${readouts} readouts`);
  if (!/class="vs"/.test(html)) warn("no .vs contrast blocks");
  if (!/class="delta"/.test(html)) warn("no .delta verdict blocks");
  if (!/class="tbl"/.test(html)) warn("no summary table");
  for (const w of ["崩法", "洼地", "占坑", "黑箱", "动物园", "押注", "期房", "刺眼"]) if (html.includes(w)) warn(`colloquial term: ${w}`);
  // unbalanced key tags
  for (const t of ["div", "figure", "section", "main", "table"]) {
    const open = (html.match(new RegExp(`<${t}[\\s>]`, "g")) || []).length, close = (html.match(new RegExp(`</${t}>`, "g")) || []).length;
    if (open !== close) err(`unbalanced <${t}>: ${open} open / ${close} close`);
  }
}
console.log(`checked ${files.length} page(s): ${errors} error(s), ${warnings} warning(s)`);
process.exit(errors ? 1 : 0);
