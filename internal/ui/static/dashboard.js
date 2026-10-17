// Dashboard model (P6.4): turns GET /metrics/summary into tiles and bar
// charts. Pure (no DOM), so it runs in the browser and in Node tests.
"use strict";

(function (root) {
  const pct = (x) => (x === null || x === undefined ? "—" : `${Math.round(x * 100)}%`);
  const ORDER = {
    by_urgency: ["critical", "high", "medium", "low"],
    by_tier: ["fast", "deep", "spam_filter"],
  };

  function bars(map, order) {
    const entries = Object.entries(map || {});
    const keys = order ? [...order.filter((k) => k in (map || {})), ...entries.map(([k]) => k).filter((k) => !order.includes(k))]
      : entries.sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).map(([k]) => k);
    const max = Math.max(1, ...entries.map(([, v]) => v));
    return keys.map((k) => ({ label: k, value: map[k], share: map[k] / max }));
  }

  function dashboardModel(resp) {
    const s = (resp && resp.summary) || {};
    const models = (resp && resp.models) || {};
    const loaded = models.ml_service === "ok"
      ? ["fast_tier_loaded", "deep_tier", "urgency"].filter((k) => models[k]).map((k) => k.replace("_loaded", "")).join(", ") || "none"
      : "ml-service unreachable";
    const tiers = s.by_tier || {};
    const tierTotal = Object.values(tiers).reduce((a, b) => a + b, 0);
    const tiles = [
      { key: "Tickets", value: s.tickets ?? 0 },
      { key: "Open routes", value: s.open_routes ?? 0 },
      { key: "SLA breached (open)", value: s.sla_breached_open ?? 0, alert: (s.sla_breached_open || 0) > 0 },
      { key: "Escalated", value: s.escalated_routes ?? 0 },
      { key: "Deep-tier share", value: tierTotal ? pct((tiers.deep || 0) / tierTotal) : "—" },
      { key: "Abstain rate", value: pct(s.abstain_rate) },
      { key: "Avg faithfulness", value: pct(s.avg_faithfulness) },
      { key: "Models loaded", value: loaded },
    ];
    const charts = [
      { title: "Volume by source", bars: bars(s.by_source) },
      { title: "Volume by category", bars: bars(s.by_category) },
      { title: "Urgency", bars: bars(s.by_urgency, ORDER.by_urgency) },
      { title: "Tier (fast vs deep vs spam filter)", bars: bars(s.by_tier, ORDER.by_tier) },
      { title: "Actions", bars: bars(s.by_action) },
      { title: "Drafts by status", bars: bars(s.drafts_by_status) },
      { title: "Top senders", bars: (s.top_senders || []).map((t) => ({ label: t.sender, value: t.tickets })) },
    ];
    const maxSender = Math.max(1, ...charts[6].bars.map((b) => b.value));
    charts[6].bars.forEach((b) => (b.share = b.value / maxSender));
    return { tiles, charts, generatedAt: s.generated_at };
  }

  if (typeof module !== "undefined" && module.exports) module.exports = { dashboardModel };
  else root.dashboardModel = dashboardModel;
})(typeof window !== "undefined" ? window : globalThis);
