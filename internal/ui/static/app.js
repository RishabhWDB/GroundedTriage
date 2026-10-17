// GroundedTriage agent review UI (P6.2). No framework, no build step.
// Ticket and draft text is untrusted: it is only ever inserted with
// textContent (see el()), never innerHTML.
"use strict";

const CATEGORIES = ["Technical Issue", "Billing", "Feature Request", "General Inquiry", "Spam", "Meeting Scheduling"];
const PAGE = 50;
const state = { offset: 0, total: 0, filters: { status: "open" } };

// ---------- small helpers ----------

function el(tag, attrs = {}, ...children) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") n.className = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v === true ? "" : String(v));
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    n.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return n;
}

const $ = (sel) => document.querySelector(sel);
const fmtTime = (t) => (t ? new Date(t).toLocaleString() : "—");
const pct = (x) => (x === null || x === undefined ? "—" : `${Math.round(x * 100)}%`);
// Demo-only signal (no auto-send exists): grounded, unedited, faithfulness 1.0.
// A grounded draft's retrieval was above the gate's floor (below it, drafts abstain).
const autoSendEligible = (d) => d.status === "grounded" && !d.edited && d.faithfulness_score === 1 &&
  d.retrieval_max_score !== null && d.retrieval_max_score !== undefined;
const badge = (text, cls) => el("span", { class: `badge ${cls || String(text).replace(/\s+/g, "_")}` }, text);

function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (t.hidden = true), 3500);
}

function apiKey() {
  try { return localStorage.getItem("gt_api_key") || ""; } catch { return ""; }
}

async function api(path, opts = {}) {
  const headers = { "X-API-Key": apiKey(), ...(opts.headers || {}) };
  if (opts.json !== undefined) {
    headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(opts.json);
  }
  const res = await fetch(path, { ...opts, headers });
  let body = null;
  try { body = await res.json(); } catch { /* empty */ }
  if (res.status === 401) {
    showLogin("The API key was rejected.");
    throw new Error("unauthorized");
  }
  if (!res.ok) throw new Error((body && body.error && body.error.message) || `HTTP ${res.status}`);
  return body;
}

// ---------- login / reviewer ----------

function showLogin(err) {
  $("#app").hidden = true;
  $("#login").hidden = false;
  const e = $("#login-error");
  e.hidden = !err;
  e.textContent = err || "";
}

$("#login-form").addEventListener("submit", (ev) => {
  ev.preventDefault();
  try { localStorage.setItem("gt_api_key", $("#api-key").value.trim()); } catch { /* private mode */ }
  $("#login").hidden = true;
  route();
});

$("#logout").addEventListener("click", () => {
  try { localStorage.removeItem("gt_api_key"); } catch { /* ignore */ }
  showLogin();
});

const reviewerInput = $("#reviewer");
try { reviewerInput.value = localStorage.getItem("gt_reviewer") || ""; } catch { /* ignore */ }
reviewerInput.addEventListener("change", () => {
  try { localStorage.setItem("gt_reviewer", reviewerInput.value.trim()); } catch { /* ignore */ }
});

function reviewer() {
  const r = reviewerInput.value.trim();
  if (!r) {
    toast("Enter your name in Reviewer first.");
    reviewerInput.focus();
  }
  return r;
}

// ---------- routing ----------

function setView(name) {
  for (const v of ["inbox", "ticket", "dashboard"]) $(`#${v}-view`).hidden = v !== name;
  $("#nav-inbox").classList.toggle("active", name !== "dashboard");
  $("#nav-dashboard").classList.toggle("active", name === "dashboard");
}

function route() {
  if (!apiKey()) return showLogin();
  $("#app").hidden = false;
  const h = location.hash || "#/inbox";
  const m = h.match(/^#\/tickets\/([0-9a-f-]{36})$/i);
  if (m) {
    setView("ticket");
    loadTicket(m[1]).catch((e) => toast(e.message));
  } else if (h === "#/dashboard") {
    setView("dashboard");
    loadDashboard().catch((e) => toast(e.message));
  } else {
    setView("inbox");
    loadInbox().catch((e) => toast(e.message));
  }
}
window.addEventListener("hashchange", route);

// ---------- inbox ----------

const catSelect = $('#filters select[name="category"]');
for (const c of CATEGORIES) catSelect.append(el("option", { value: c }, c));

$("#filters").addEventListener("submit", (ev) => {
  ev.preventDefault();
  const fd = new FormData(ev.target);
  state.filters = {};
  for (const [k, v] of fd.entries()) if (String(v).trim()) state.filters[k] = String(v).trim();
  state.offset = 0;
  loadInbox().catch((e) => toast(e.message));
});
$("#prev").addEventListener("click", () => { state.offset = Math.max(0, state.offset - PAGE); loadInbox(); });
$("#next").addEventListener("click", () => { state.offset += PAGE; loadInbox(); });

async function loadInbox() {
  const q = new URLSearchParams({ ...state.filters, limit: PAGE, offset: state.offset });
  const data = await api(`/tickets?${q}`);
  state.total = data.total;
  const tbody = $("#inbox tbody");
  tbody.replaceChildren();
  const now = Date.now();
  for (const t of data.items) {
    const breached = t.sla_due_at && t.route_status === "open" && new Date(t.sla_due_at).getTime() < now;
    const row = el("tr", { onclick: () => (location.hash = `#/tickets/${t.id}`) },
      el("td", { class: t.priority === 1 ? "p1" : "" }, t.priority ?? "—"),
      el("td", { class: breached ? "breached" : "" }, t.sla_due_at ? fmtTime(t.sla_due_at) : "—", breached ? " (breached)" : ""),
      el("td", {}, t.subject || "(no subject)", t.ingest_status === "skipped" ? [" ", badge("skipped", "abstained")] : null),
      el("td", {}, t.sender),
      el("td", {}, t.category || "—", t.possible_spam ? [" ", badge("possible spam", "possible_spam")] : null),
      el("td", {}, t.tier ? badge(t.tier) : "—", t.low_confidence ? [" ", badge("low conf.", "low_confidence")] : null),
      el("td", {}, t.urgency ? badge(t.urgency) : "—"),
      el("td", {}, t.queue || "—"),
      el("td", {}, t.draft_status ? badge(t.draft_status) : "—",
        t.draft_status === "grounded" && t.faithfulness_score === 1 ? [" ", badge("auto-send eligible", "eligible")] : null),
      el("td", {}, pct(t.faithfulness_score)));
    tbody.append(row);
  }
  if (!data.items.length) tbody.append(el("tr", {}, el("td", { colspan: 10, class: "note" }, "No tickets match.")));
  const end = Math.min(state.offset + PAGE, data.total);
  $("#page-info").textContent = data.total ? `${state.offset + 1}–${end} of ${data.total}` : "0";
  $("#prev").disabled = state.offset === 0;
  $("#next").disabled = end >= data.total;
}

$("#csv-file").addEventListener("change", async (ev) => {
  const file = ev.target.files[0];
  if (!file) return;
  const fd = new FormData();
  fd.append("file", file);
  const out = $("#upload-result");
  try {
    const r = await api("/ingest/csv", { method: "POST", body: fd });
    out.textContent = `${file.name}: ${r.fetched} rows read, ${r.inserted} new, ${r.inserted_skipped} stored but skipped, ` +
      `${r.duplicate_external_id + r.duplicate_content} duplicates, ${r.dropped} dropped. The worker classifies new tickets on its next poll.`;
    out.hidden = false;
    loadInbox();
  } catch (e) {
    toast(`Upload failed: ${e.message}`);
  } finally {
    ev.target.value = "";
  }
});

// ---------- ticket view ----------

function kv(pairs) {
  const dl = el("dl");
  for (const [k, v] of pairs) if (v !== null && v !== undefined && v !== "") dl.append(el("dt", {}, k), el("dd", {}, v));
  return dl;
}

async function loadTicket(id) {
  const d = await api(`/tickets/${id}`);
  const t = d.ticket;
  $("#ticket-card").replaceChildren(
    el("h2", {}, t.subject || "(no subject)"),
    kv([["From", t.sender], ["Received", fmtTime(t.received_at)], ["Source", t.source],
      ["Ingest", t.ingest_status === "skipped" ? `skipped (${t.skip_reason})` : "accepted"]]),
    el("pre", { class: "body" }, t.body));

  const c = d.classification;
  $("#classification-card").replaceChildren(el("h3", {}, "Classification"), c
    ? kv([
      ["Category", [c.category, ` (${pct(c.category_confidence)})`]],
      ["Tier", [badge(c.tier), c.low_confidence ? [" ", badge("low confidence", "low_confidence")] : null]],
      ["Urgency", [badge(c.urgency), " from ", badge(c.urgency_source), ` (${pct(c.urgency_confidence)})`]],
      ["Spam score", c.spam_score === null || c.spam_score === undefined ? "no spam filter" :
        [c.spam_score.toFixed(3), c.possible_spam ? [" ", badge("possible spam", "possible_spam")] : null]],
      ["Models", Object.entries(c.model_versions || {}).map(([k, v]) => `${k}: ${v}`).join(" · ")],
    ])
    : el("p", { class: "note" }, t.ingest_status === "skipped" ? "Not classified (skipped at ingestion)." : "Not classified yet."));

  const r = d.route;
  $("#route-card").replaceChildren(el("h3", {}, "Route"), r
    ? kv([["Action", r.action], ["Queue", r.queue], ["Priority", r.priority], ["Status", r.status],
      ["SLA due", fmtTime(r.sla_due_at)], ["Escalations", r.escalations || "0"], ["Reason", r.reason]])
    : el("p", { class: "note" }, "Not routed yet."));

  const m = d.meeting;
  $("#meeting-card").hidden = !m;
  if (m) {
    $("#meeting-card").replaceChildren(el("h3", {}, "Meeting"), kv([["Outcome", badge(m.status, m.status === "booked" ? "approved" : "abstained")],
      ["Requested", fmtTime(m.requested_start)], ["Slot", m.slot_start ? `${fmtTime(m.slot_start)} – ${fmtTime(m.slot_end)}` : "—"],
      ["Venue", m.venue], ["Event", m.event_id], ["Reason", m.reason]]));
  }

  $("#chunk-panel").hidden = true;
  if (d.draft) await loadDraft(id);
  else $("#draft-card").replaceChildren(el("h3", {}, "Draft"), el("p", { class: "note" }, "No draft (only `reply` routes get one)."));
}

// Splits text into strings and citation buttons for [n] / [n, m].
function withCitations(text, chunksByNumber) {
  const parts = [];
  const re = /\[(\d+(?:\s*,\s*\d+)*)\]/g;
  let last = 0;
  let mt;
  while ((mt = re.exec(text)) !== null) {
    parts.push(text.slice(last, mt.index));
    for (const n of mt[1].split(/\s*,\s*/).map(Number)) {
      const ch = chunksByNumber.get(n);
      parts.push(el("button", { class: "cite", type: "button", title: ch ? ch.title : "unknown source",
        onclick: (ev) => showChunk(ch, n, ev.currentTarget) }, `[${n}]`));
    }
    last = re.lastIndex;
  }
  parts.push(text.slice(last));
  return parts;
}

function showChunk(ch, n, btn) {
  document.querySelectorAll(".cite.selected").forEach((b) => b.classList.remove("selected"));
  btn.classList.add("selected");
  const p = $("#chunk-panel");
  p.hidden = false;
  p.replaceChildren(el("h3", {}, `Source [${n}]`), ch
    ? [el("p", {}, el("strong", {}, ch.title), ch.heading ? ` · ${ch.heading}` : ""),
      ch.url && /^https?:\/\//.test(ch.url) ? el("p", {}, el("a", { href: ch.url, target: "_blank", rel: "noopener noreferrer" }, ch.url)) : null,
      el("pre", { class: "body" }, ch.text), el("p", { class: "note" }, `chunk ${ch.id}`)]
    : el("p", { class: "error" }, "This citation points at no retrieved source."));
}

async function loadDraft(ticketId) {
  const { draft: d, chunks } = await api(`/tickets/${ticketId}/draft`);
  const byNumber = new Map(chunks.map((c) => [c.number, c]));
  const card = $("#draft-card");
  const claims = Array.isArray(d.claims) ? d.claims : [];
  const unsupported = Array.isArray(d.unsupported_claims) ? d.unsupported_claims : [];
  card.replaceChildren(
    el("h3", {}, "Draft reply ", badge(d.status), d.edited ? [" ", badge("edited", "abstained")] : null,
      autoSendEligible(d) ? [" ", el("span", { class: "badge eligible", title:
        "Grounded, faithfulness 1.0, retrieval above the floor. Shown for the demo only: every reply still needs human approval." },
      "eligible for auto-send")] : null),
    kv([["Faithfulness", d.faithfulness_score === null ? "not judged" : pct(d.faithfulness_score)],
      ["Retrieval", d.retrieval_max_score === null ? "—" : `best ${d.retrieval_max_score.toFixed(3)} (${d.retrieval_mode})`],
      ["Abstained", d.abstain_reason], ["Reviewed", d.reviewer ? `${d.reviewer}, ${fmtTime(d.reviewed_at)}` : ""],
      ["Note", d.review_note], ["Sent", d.sent_at ? fmtTime(d.sent_at) : ""], ["Send error", d.send_error]]),
    d.body ? [el("p", {}, el("strong", {}, d.subject)), el("pre", { class: "body" }, withCitations(d.body, byNumber))]
      : el("p", { class: "note" }, "No draft text: the gate abstained before drafting. Edit to write a reply yourself."),
    unsupported.length ? [el("h3", {}, "Unsupported claims"), el("ul", { class: "claims" },
      unsupported.map((u) => el("li", { class: "unsupported" }, u)))] : null,
    claims.length ? el("details", {}, el("summary", {}, `All judged claims (${claims.length})`), el("ul", { class: "claims" },
      claims.map((c) => el("li", { class: c.supported ? "supported" : "unsupported" },
        c.claim, ` [${(c.cited || []).join(", ")}] `, el("span", { class: "note" }, c.reason || ""))))) : null,
    chunks.length ? el("details", {}, el("summary", {}, `Retrieved sources (${chunks.length})`), el("ul", {},
      chunks.map((c) => el("li", {}, el("button", { class: "cite", type: "button", onclick: (ev) => showChunk(c, c.number, ev.currentTarget) },
        `[${c.number}]`), ` ${c.title}`, c.cited ? " (cited)" : "")))) : null,
    draftActions(d));
}

function draftActions(d) {
  const box = el("div", { class: "actions" });
  const editable = d.status === "grounded" || d.status === "abstained";
  const approvable = (d.status === "grounded" || (d.edited && d.status === "abstained")) && d.body.trim();
  const rejectable = ["grounded", "abstained", "approved"].includes(d.status) && !d.sent_at;
  const act = async (verb, extra = {}) => {
    const who = reviewer();
    if (!who) return;
    try {
      await api(`/drafts/${d.id}/${verb}`, { method: "POST", json: { reviewer: who, ...extra } });
      toast(`Draft ${verb === "edit" ? "saved" : verb + "d"}.`);
      loadTicket(d.ticket_id);
    } catch (e) {
      toast(`${verb} failed: ${e.message}`);
    }
  };
  box.append(
    el("button", { type: "button", disabled: !approvable, onclick: () => act("approve", { note: prompt("Note (optional)") || "" }) }, "Approve"),
    el("button", { type: "button", class: "secondary", disabled: !editable, onclick: () => openEditor(d, act) }, "Edit"),
    el("button", { type: "button", class: "danger", disabled: !rejectable, onclick: () => {
      const why = prompt("Why reject?");
      if (why !== null) act("reject", { note: why });
    } }, "Reject"));
  if (d.status === "approved") box.append(el("span", { class: "note" }, "Approved: the worker sends it via the ticket's source."));
  return box;
}

function openEditor(d, act) {
  const subject = el("input", {});
  subject.value = d.subject || "";
  const body = el("textarea", {});
  body.value = d.body || "";
  const card = $("#draft-card");
  card.replaceChildren(el("h3", {}, "Edit draft"), el("p", { class: "note" },
    "Edited text is your reply: its faithfulness score no longer applies."),
    el("label", {}, "Subject ", subject), body,
    el("div", { class: "actions" },
      el("button", { type: "button", onclick: () => act("edit", { subject: subject.value, body: body.value }) }, "Save"),
      el("button", { type: "button", class: "secondary", onclick: () => loadTicket(d.ticket_id) }, "Cancel")));
}

// ---------- dashboard (P6.4) ----------

async function loadDashboard() {
  const [summary, daily] = await Promise.all([api("/metrics/summary"),
    api("/metrics/daily?days=14").catch((e) => ({ error: e.message }))]);
  const m = window.dashboardModel(summary);
  $("#tiles").replaceChildren(...m.tiles.map((t) =>
    el("div", { class: "tile" }, el("div", { class: `v${t.alert ? " breached" : ""}` }, t.value), el("div", { class: "k" }, t.key))));
  $("#charts").replaceChildren(...m.charts.map((c) => el("div", { class: "card" }, el("h3", {}, c.title),
    c.bars.length ? c.bars.map((b) => {
      const fill = el("div", { class: "fill" });
      fill.style.width = `${Math.round(b.share * 100)}%`; // CSSOM, allowed by the CSP
      return el("div", { class: "bar" }, el("span", { title: b.label }, b.label), el("div", { class: "track" }, fill), el("span", {}, b.value));
    }) : el("p", { class: "note" }, "No data yet."))),
    el("p", { class: "note" }, `Generated ${fmtTime(m.generatedAt)}. Refresh the page to update.`));
  renderMonitoring(daily);
}

// ---------- monitoring (P7.3) ----------

function renderMonitoring(daily) {
  const box = $("#monitoring");
  if (daily.error) {
    box.replaceChildren(el("h3", {}, "Monitoring"), el("p", { class: "error" }, daily.error));
    return;
  }
  const mon = window.monitoringModel(daily);
  box.replaceChildren(
    el("h3", {}, "Monitoring (last 14 days)"),
    mon.latestAlert ? el("p", { class: "breached" }, "Category drift alert: today's PSI is above the threshold.") : null,
    mon.rows.length ? el("div", { class: "scroll" }, el("table", {},
      el("thead", {}, el("tr", {}, mon.columns.map((c) => el("th", {}, c)))),
      el("tbody", { class: "static" }, mon.rows.map((r) => el("tr", {}, r.cells.map((c, i) =>
        el("td", { class: i === 1 && r.alert ? "breached" : undefined }, c))))))) :
      el("p", { class: "note" }, "No monitoring rows yet (the worker writes them after each poll)."),
    el("p", { class: "note" }, "PSI compares each day's categories with the reference in config/monitoring.yaml; " +
      "flagged above psi_alert (0.2). SLA breach rate covers routes created that day."));
}

route();
