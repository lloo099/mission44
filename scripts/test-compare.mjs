import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import vm from 'node:vm';

const source = readFileSync(new URL('../js/app.js', import.meta.url), 'utf8');
const data = JSON.parse(readFileSync(new URL('../data/compare.json', import.meta.url)));
const start = source.indexOf('/* ---------- model comparison:');
const end = source.indexOf('/* ---------- training curves', start);
assert(start >= 0 && end > start);
const escapeStart = source.indexOf('function escapeHtml(');
const escapeEnd = source.indexOf('// re-render all card panels', escapeStart);
const context = vm.createContext({});
vm.runInContext(source.slice(start, end) + source.slice(escapeStart, escapeEnd), context);

test('independent rows sort only within one group without mutating source', () => {
  const metric = structuredClone(data.metrics[1]);
  const original = JSON.stringify(metric);
  const rows = context.comparisonItems(metric);
  assert.equal(rows[0].score, 65.2);
  assert.equal(JSON.stringify(metric), original);
  assert.equal(rows.filter(r => r.model === 'Claude Fable 5.2').length, 2);
});

test('vendor and mixed-harness rows retain report order, with no bars', () => {
  for (const metric of data.metrics.filter(m => m.evidence === 'vendor')) {
    assert.equal(JSON.stringify(context.comparisonItems(metric)), JSON.stringify(metric.items));
    const html = context.renderComparisonMetric(metric);
    assert(!html.includes('cmp-score-track'));
    assert(html.includes('非独立复测'));
    assert(html.includes('不作统一排名'));
    for (const row of metric.items) if (row.harness) assert(html.includes(row.harness));
  }
});

test('missing scores stay missing and sort last; zero remains a real score', () => {
  const metric = structuredClone(data.metrics[0]);
  metric.items = [
    { model: 'Missing', setting: 'Unknown', score: null, missingReason: 'No run', sourceUrl: metric.sourceUrl },
    { model: 'Zero', setting: 'low', score: 0, sourceUrl: metric.sourceUrl }
  ];
  assert.equal(context.comparisonItems(metric)[1].model, 'Missing');
  assert.equal(context.fmtScore(metric, null), '未收录');
  assert.equal(context.fmtScore(metric, 0), '0');
  const html = context.renderComparisonMetric(metric);
  assert(html.includes('No run'));
  assert(!html.includes('NaN'));
  assert.equal((html.match(/cmp-score-track/g) || []).length, 1);
});

test('index is not rendered as percent; dates and provenance are visible', () => {
  const html = context.renderComparisonMetric(data.metrics[0]);
  assert(html.includes('v5.0.1'));
  assert(html.includes('成绩（指数）'));
  assert(!html.includes('65%'));
  assert(html.includes('来源未披露逐模型实测日期'));
  assert(html.includes('Default Fallback'));
  assert(html.includes('来源未注明'));
});

test('untrusted labels are escaped and unsafe source URLs rejected', () => {
  const metric = structuredClone(data.metrics[0]);
  metric.items[0].model = '<img src=x onerror=alert(1)>';
  const html = context.renderComparisonMetric(metric);
  assert(!html.includes('<img'));
  assert(html.includes('&lt;img'));
  assert.equal(context.comparisonLink('javascript:alert(1)', 'source'), '来源未记录');
});

test('historical data never gets a ranking or current evidence badge', () => {
  const archive = JSON.parse(readFileSync(new URL('../' + data.archive.url, import.meta.url)));
  for (const metric of archive.metrics.list) {
    const html = context.renderHistoricalMetric(metric);
    assert(html.includes('不参与当前排名'));
    assert(!html.includes('cmp-score-track'));
    assert(!html.includes('cmp-provenance'));
  }
});
