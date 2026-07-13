// Usage: node tools/validate_mermaid.mjs <file.md>  (needs: npm i mermaid jsdom)
import fs from 'fs';
import { JSDOM } from 'jsdom';
const dom = new JSDOM('<!DOCTYPE html><body></body>', { pretendToBeVisual: true, url: 'http://localhost' });
globalThis.window = dom.window; globalThis.document = dom.window.document;
const mermaid = (await import('mermaid')).default;
mermaid.initialize({ startOnLoad: false, securityLevel: 'loose' });
const md = fs.readFileSync(process.argv[2], 'utf8');
const re = /```mermaid\n([\s\S]*?)```/g;
let m, idx = 0, bad = 0;
while ((m = re.exec(md))) { idx++;
  try { await mermaid.parse(m[1]); console.log('diagram', idx, 'OK'); }
  catch (e) { bad++; console.log('diagram', idx, 'ERROR:', String(e.message||e).split('\n')[0]); }
}
console.log(`RESULT: ${idx} diagrams, ${bad} errors`);
process.exit(bad ? 1 : 0);
