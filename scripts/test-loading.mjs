import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import vm from 'node:vm';

const source = readFileSync(new URL('../js/app.js', import.meta.url), 'utf8');
const prefix = source.slice(0, source.indexOf('/* copy a per-post share link'));
const makeContext = fetch => {
  const context = vm.createContext({
    fetch, AbortController, setTimeout, clearTimeout,
    console: { warn() {} }, window: {}, document: { title: 'Test', querySelector: () => null },
    renderPanel() {}, refreshEvents: []
  });
  vm.runInContext(prefix, context);
  return context;
};

test('fetch failure retries, then returns data', async () => {
  let calls = 0;
  const ctx = makeContext(async url => {
    assert(url.includes('v=20261113-loadfix'));
    if (++calls === 1) throw Error('temporary failure');
    return { ok: true, json: async () => ({ items: [1] }) };
  });
  assert.deepEqual(await ctx.loadJSON('data/rl.json'), { items: [1] });
  assert.equal(calls, 2);
});

test('stalled requests are aborted and return a bounded failure', async () => {
  const signals = [];
  const ctx = makeContext((_, options) => {
    signals.push(options.signal);
    return new Promise(() => {});
  });
  assert.equal(await ctx.loadJSON('data/feed.json', { timeoutMs: 5, attempts: 2 }), null);
  assert.equal(signals.length, 2);
  assert(signals.every(s => s.aborted));
});

test('body parsing also has a timeout', async () => {
  const ctx = makeContext(async () => ({ ok: true, json: () => new Promise(() => {}) }));
  assert.equal(await ctx.loadJSON('data/rl.json', { timeoutMs: 5, attempts: 1 }), null);
});

test('HTTP and malformed JSON failures do not escape the loader', async () => {
  for (const response of [{ ok: false, status: 503 }, { ok: true, json: async () => { throw Error('invalid JSON'); } }]) {
    const ctx = makeContext(async () => response);
    assert.equal(await ctx.loadJSON('data/rl.json'), null);
  }
});

test('healthy domains render while another domain is still pending; retry preserves them', async () => {
  const ctx = makeContext(() => {});
  vm.runInContext(`
    Object.keys(DATA_SOURCES).forEach(key => { store[key] = []; dataStatus[key] = 'loading'; });
    renderDataStatus = () => {};
    refreshDataViews = () => refreshEvents.push(JSON.parse(JSON.stringify({ dataStatus, store })));
    let releaseFeed;
    loadJSON = url => url.includes('feed') ? new Promise(resolve => { releaseFeed = resolve; })
      : Promise.resolve({ updated: '2026-11-13', items: [{ title: 'Available' }] });
  `, ctx);
  const delayed = ctx.loadDomainData('live');
  await ctx.loadDomainData('rl');
  let snapshot = ctx.refreshEvents.at(-1);
  assert.equal(snapshot.dataStatus.rl, 'ready');
  assert.equal(snapshot.dataStatus.live, 'loading');
  assert.equal(snapshot.store.rl.length, 1);
  vm.runInContext('releaseFeed(null)', ctx);
  await delayed;
  assert.equal(ctx.refreshEvents.at(-1).dataStatus.live, 'error');
  vm.runInContext('loadJSON = async () => ({ items: [{ title: "Recovered" }] })', ctx);
  await ctx.loadDomainData('live');
  snapshot = ctx.refreshEvents.at(-1);
  assert.equal(snapshot.dataStatus.live, 'ready');
  assert.equal(snapshot.store.rl[0].title, 'Available');
});

test('invalid data is not presented as an empty successful dataset', async () => {
  const ctx = makeContext(async () => ({ ok: true, json: async () => ({ items: {} }) }));
  vm.runInContext(`
    renderDataStatus = () => {};
    refreshDataViews = () => refreshEvents.push(dataStatus.rl);
  `, ctx);
  await ctx.loadDomainData('rl');
  assert.equal(ctx.refreshEvents.at(-1), 'error');
});

test('core controls initialize before slow requests complete', async () => {
  const events = [];
  const ctx = makeContext(() => new Promise(() => {}));
  for (const name of ['wireTheme', 'wireTabs', 'wireSearch', 'wireLive', 'wireCurves', 'wireCompare', 'wireBlog', 'wireMermaidLightbox', 'wireTabsScrollHint', 'wireShareButtons']) {
    ctx[name] = () => events.push(name);
  }
  ctx.document.getElementById = () => ({ addEventListener() {} });
  vm.runInContext('refreshDataViews = () => {}; loadDomainData = () => new Promise(() => {});', ctx);
  ctx.init();
  assert(events.includes('wireTabs'));
  assert(events.includes('wireBlog'));
  assert(events.includes('wireCompare'));
  assert.equal(ctx.window.__dashboardStarted, true);
});

test('optional scripts cannot block the main application; static loading state exists', () => {
  const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
  assert.match(html, /<script async src="js\/app\.js/);
  assert.match(html, /<script async src="js\/bg\.js/);
  assert.match(html, /<script async src="js\/arch-demo\.js[^>]+onload=/);
  assert.match(html, /id="data-load-message"/);
  assert.match(html, /window\.__dashboardLoadError/);
});

test('existing feed refresh controls also clear a previous loading failure', async () => {
  const handlers = {};
  const ctx = makeContext(async () => ({ ok: true, json: async () => ({ items: [{ title: 'Recovered' }] }) }));
  ctx.document.getElementById = id => ({ addEventListener: (_, fn) => { handlers[id] = fn; } });
  ctx.renderFeedStatus = () => {};
  ctx.buildFilterbars = () => {};
  ctx.tryArxivLive = async () => [{ title: 'Fresh' }];
  vm.runInContext('renderDataStatus = () => {}; dataStatus.live = "error";', ctx);
  const start = source.indexOf('function wireLive()');
  vm.runInContext(source.slice(start, source.indexOf('async function tryArxivLive()', start)), ctx);
  ctx.wireLive();
  await handlers['live-reload']();
  assert.equal(vm.runInContext('dataStatus.live', ctx), 'ready');
  vm.runInContext('dataStatus.live = "error";', ctx);
  await handlers['live-refresh']();
  assert.equal(vm.runInContext('dataStatus.live', ctx), 'ready');
});
