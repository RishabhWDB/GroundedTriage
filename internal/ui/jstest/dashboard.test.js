// node --test internal/ui/jstest  (run by `go test ./internal/ui` when node is installed)
"use strict";
const test = require("node:test");
const assert = require("node:assert");
const { dashboardModel } = require("../static/dashboard.js");

const resp = {
  summary: {
    tickets: 21, by_source: { csv: 20, email: 1 }, by_category: { Billing: 4, "Technical Issue": 9, unclassified: 8 },
    by_urgency: { low: 2, critical: 1, high: 5 }, by_tier: { fast: 9, deep: 3, spam_filter: 1 }, by_action: { reply: 6, mute: 2 },
    open_routes: 11, sla_breached_open: 2, escalated_routes: 3, drafts_by_status: { grounded: 2, abstained: 4 },
    abstain_rate: 0.6667, avg_faithfulness: 0.95, top_senders: [{ sender: "a@x.com", tickets: 4 }, { sender: "b@y.com", tickets: 2 }],
    generated_at: "2026-10-17T10:00:00Z",
  },
  models: { ml_service: "ok", fast_tier_loaded: true, deep_tier: false, urgency: false },
};

test("tiles cover the spec's dashboard numbers", () => {
  const m = dashboardModel(resp);
  const tile = Object.fromEntries(m.tiles.map((t) => [t.key, t]));
  assert.strictEqual(tile["Tickets"].value, 21);
  assert.strictEqual(tile["SLA breached (open)"].value, 2);
  assert.strictEqual(tile["SLA breached (open)"].alert, true);
  assert.strictEqual(tile["Deep-tier share"].value, "23%"); // 3 of 13
  assert.strictEqual(tile["Abstain rate"].value, "67%");
  assert.strictEqual(tile["Avg faithfulness"].value, "95%");
  assert.strictEqual(tile["Models loaded"].value, "fast_tier");
});

test("charts: volume, urgency order, tiers, top senders", () => {
  const m = dashboardModel(resp);
  const chart = Object.fromEntries(m.charts.map((c) => [c.title, c.bars]));
  assert.deepStrictEqual(chart["Volume by category"].map((b) => b.label), ["Technical Issue", "unclassified", "Billing"]);
  assert.deepStrictEqual(chart["Urgency"].map((b) => b.label), ["critical", "high", "low"]);
  assert.deepStrictEqual(chart["Tier (fast vs deep vs spam filter)"].map((b) => b.label), ["fast", "deep", "spam_filter"]);
  assert.strictEqual(chart["Volume by category"][0].share, 1);
  assert.deepStrictEqual(chart["Top senders"].map((b) => [b.label, b.share]), [["a@x.com", 1], ["b@y.com", 0.5]]);
});

test("empty summary and unreachable ml-service", () => {
  const m = dashboardModel({ summary: { tickets: 0, abstain_rate: null }, models: { ml_service: "unreachable" } });
  const tile = Object.fromEntries(m.tiles.map((t) => [t.key, t.value]));
  assert.strictEqual(tile["Abstain rate"], "—");
  assert.strictEqual(tile["Deep-tier share"], "—");
  assert.strictEqual(tile["Models loaded"], "ml-service unreachable");
  assert.ok(m.charts.every((c) => Array.isArray(c.bars)));
});

const { monitoringModel } = require("../static/dashboard.js");

test("monitoring: rows per day, newest first, PSI alert and stage columns", () => {
  const m = monitoringModel({ days: 14, metrics: [
    { day: "2026-10-16T00:00:00Z", metric: "category_psi", value: null, detail: { classified: 3, min_tickets: 30, alert: false } },
    { day: "2026-10-17T00:00:00Z", metric: "category_psi", value: 0.3141, detail: { alert: true } },
    { day: "2026-10-17T00:00:00Z", metric: "abstain_rate", value: 0.25, detail: {} },
    { day: "2026-10-17T00:00:00Z", metric: "sla_breach_rate", value: null, detail: {} },
    { day: "2026-10-17T00:00:00Z", metric: "latency_p95_ms.verify", value: 1499.6, detail: { n: 3 } },
    { day: "2026-10-16T00:00:00Z", metric: "latency_p95_ms.classify", value: 80, detail: { n: 40 } },
  ] });
  assert.deepStrictEqual(m.columns.slice(4), ["p95 classify (ms)", "p95 verify (ms)"]);
  assert.strictEqual(m.rows[0].cells[0], "2026-10-17");
  assert.deepStrictEqual(m.rows[0].cells.slice(1), ["0.314", "25%", "—", "—", "1500 (n=3)"]);
  assert.strictEqual(m.rows[0].alert, true);
  assert.strictEqual(m.latestAlert, true);
  assert.strictEqual(m.rows[1].cells[1], "— (3 < 30 tickets)");
  assert.strictEqual(m.rows[1].cells[4], "80 (n=40)");
  assert.strictEqual(monitoringModel({ metrics: [] }).latestAlert, false);
});
