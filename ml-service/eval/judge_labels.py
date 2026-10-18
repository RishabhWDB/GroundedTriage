"""Judge validation (P7.2, Phase 0 review): hand-label claims, then score judges.

    python -m eval.judge_labels sample [--n 80]   # == make judge-sample
    python -m eval.judge_labels label             # == make judge-label (interactive)
    python -m eval.judge_labels score [--with-llm]  # == make judge-score

`sample` draws claims from the P7.2 run (datasets/raw/rag_eval/progress.jsonl),
stratified by the gate judge's verdict (half supported, half not, when there
are enough) and by system (grounded/ungrounded). Each row shows the claim
and the source text it must be checked against (its cited chunks, or all
retrieved chunks when it cites none). The judge's verdicts are kept in a
separate file and never shown while labelling.

`label` asks yes / no / unsure per unlabelled row and saves after every answer
(quit any time; it resumes). `score` compares the human labels (unsure rows
excluded) with:
- the gate judge's own verdicts (qwen3.8-27b, from the run), and, with
  --with-llm, a per-claim verdict from qwen and from gpt-oss-120b on the same
  claim + sources (cached; ~1 call per claim per model).
It reports accuracy, Cohen's kappa and the confusion counts. Below
MIN_LABELS labelled claims the numbers are marked PRELIMINARY. The decision
rule (Phase 5 review): kappa < 0.6 for the gate judge -> revisit the judge.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

from app.fileio import atomic_path, atomic_write_text
from app.settings import REPO_ROOT

LABEL_DIR = REPO_ROOT / "datasets" / "judge_labels"
LABELS = LABEL_DIR / "claims.csv"
VERDICTS = LABEL_DIR / "judge_verdicts.json"
PROGRESS = REPO_ROOT / "datasets" / "raw" / "rag_eval" / "progress.jsonl"
OUT = REPO_ROOT / "results" / "rag"
FIELDS = ["claim_id", "item_id", "system", "claim", "sources", "human_supported", "notes"]
ANSWERS = {"y": "yes", "n": "no", "u": "unsure"}
MIN_LABELS = 60
SEED = 11

CLAIM_SYSTEM = (
    "You check one factual claim from a customer-support reply against numbered "
    "sources. The claim is supported only if the sources state it or directly imply "
    'it. Reply with JSON only: {"supported": true|false, "reason": "<short>"}'
)
CLAIM_USER = "Sources:\n{sources}\n\nClaim: {claim}"


def claim_id(item_id: str, system: str, idx: int) -> str:
    return hashlib.sha256(f"{item_id}|{system}|{idx}".encode()).hexdigest()[:12]


def load_rows(path: Path = PROGRESS) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fetch_chunks(ids: list[str]) -> dict[str, dict]:
    from app.settings import get_settings
    from rag.kb import connect

    conn = connect(get_settings().database_url or None)
    with conn.cursor() as cur:
        cur.execute("select id, title, text from kb_chunks where id = any(%s)", (ids,))
        out = {r[0]: {"title": r[1], "text": r[2]} for r in cur.fetchall()}
    conn.close()
    return out


def sources_text(retrieved: list[str], cited: list[int], chunks: dict[str, dict]) -> str:
    nums = [n for n in cited if 1 <= n <= len(retrieved)] or list(range(1, len(retrieved) + 1))
    parts = []
    for n in nums:
        c = chunks.get(retrieved[n - 1], {"title": "", "text": "(missing chunk)"})
        parts.append(f"[{n}] {c['title']}\n{c['text']}".strip())
    return "\n\n".join(parts)


def sample(rows: list[dict], n: int, fetch=fetch_chunks, seed: int = SEED):
    """-> (label rows, {claim_id: judge_supported})."""
    pool = []
    for r in rows:
        for i, c in enumerate(r.get("claims") or []):
            pool.append((r, i, c))
    rng = random.Random(seed)
    rng.shuffle(pool)
    # stratify: half judged-unsupported / half supported (the rest from the other
    # side when one is short), alternating systems within each verdict
    buckets: dict[bool, dict[str, list]] = {False: {}, True: {}}
    for p in pool:
        buckets[bool(p[2]["supported"])].setdefault(p[0]["system"], []).append(p)
    avail = {v: sum(map(len, b.values())) for v, b in buckets.items()}
    n_unsup = min(avail[False], n // 2)
    n_sup = min(avail[True], n - n_unsup)
    n_unsup = min(avail[False], n - n_sup)
    picked = []
    for verdict, k in ((False, n_unsup), (True, n_sup)):
        lists = [buckets[verdict].get(sys_, []) for sys_ in ("grounded", "ungrounded")]
        got = 0
        while got < k:
            for lst in lists:
                if lst and got < k:
                    picked.append(lst.pop())
                    got += 1
    chunks = fetch(sorted({cid for r, _, _ in picked for cid in r["retrieved"]}))
    out, verdicts = [], {}
    for r, i, c in picked:
        cid = claim_id(r["item_id"], r["system"], i)
        out.append(
            {
                "claim_id": cid,
                "item_id": r["item_id"],
                "system": r["system"],
                "claim": c["claim"],
                "sources": sources_text(r["retrieved"], c.get("cited", []), chunks),
                "human_supported": "",
                "notes": "",
            }
        )
        verdicts[cid] = bool(c["supported"])
    return out, verdicts


def read_labels(path: Path = LABELS) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_labels(rows: list[dict], path: Path = LABELS) -> None:
    with atomic_path(path) as tmp, open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def label(path: Path = LABELS, ask=input, show=print) -> int:
    rows = read_labels(path)
    todo = [r for r in rows if not r["human_supported"]]
    show(
        f"{len(rows) - len(todo)} of {len(rows)} labelled. "
        "y = supported, n = not supported, u = unsure, q = quit."
    )
    for k, r in enumerate(todo, start=1):
        show(
            f"\n--- {k}/{len(todo)}  ({r['claim_id']})\nSOURCES:\n{r['sources']}"
            f"\n\nCLAIM: {r['claim']}"
        )
        while True:
            a = ask("supported? [y/n/u/q] ").strip().lower()
            if a == "q":
                return 0
            if a in ANSWERS:
                r["human_supported"] = ANSWERS[a]
                write_labels(rows, path)
                break
    show("all claims labelled; run `make judge-score`.")
    return 0


def kappa(a: list[bool], b: list[bool]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def agreement(human: list[bool], judge: list[bool]) -> dict:
    pairs = list(zip(human, judge, strict=True))
    k = kappa(human, judge)
    return {
        "n": len(pairs),
        "accuracy": round(sum(h == j for h, j in pairs) / len(pairs), 4) if pairs else None,
        "kappa": round(k, 4) if k is not None else None,
        "human_yes_judge_yes": sum(h and j for h, j in pairs),
        "human_yes_judge_no": sum(h and not j for h, j in pairs),
        "human_no_judge_yes": sum(not h and j for h, j in pairs),
        "human_no_judge_no": sum(not h and not j for h, j in pairs),
    }


def claim_verdict(llm, model: str, claim: str, sources: str, cache) -> bool | None:
    messages = [
        {"role": "system", "content": CLAIM_SYSTEM},
        {"role": "user", "content": CLAIM_USER.format(sources=sources, claim=claim)},
    ]
    key = cache.key(model, messages)
    text = cache.get(key)
    if text is None:
        text = llm.complete(messages, temperature=0.0, json_mode=True)
    try:
        v = json.loads(text)["supported"]
    except (ValueError, KeyError, TypeError):
        return None
    cache.put(key, text)
    return bool(v)


def score(
    rows: list[dict], verdicts: dict[str, bool], extra: dict[str, dict[str, bool]] | None = None
) -> dict:
    labelled = [r for r in rows if r["human_supported"] in ("yes", "no")]
    out = {
        "labelled": len(labelled),
        "unsure": sum(r["human_supported"] == "unsure" for r in rows),
        "total": len(rows),
        "preliminary": len(labelled) < MIN_LABELS,
        "judges": {},
    }
    judges = {"gate (qwen, whole-draft)": verdicts, **(extra or {})}
    for name, v in judges.items():
        both = [r for r in labelled if v.get(r["claim_id"]) is not None]
        out["judges"][name] = agreement(
            [r["human_supported"] == "yes" for r in both], [v[r["claim_id"]] for r in both]
        )
    return out


def report_md(s: dict) -> str:
    tag = f" — PRELIMINARY (fewer than {MIN_LABELS} labels)" if s["preliminary"] else ""
    lines = [
        f"# Judge validation{tag}",
        "",
        f"Human labels: {s['labelled']} yes/no, {s['unsure']} unsure, {s['total']} sampled.",
        "",
        "| judge | n | accuracy | Cohen's kappa | H yes / J yes | H yes / J no "
        "| H no / J yes | H no / J no |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, a in s["judges"].items():
        lines.append(
            f"| {name} | {a['n']} | {a['accuracy']} | {a['kappa']} | {a['human_yes_judge_yes']} | "
            f"{a['human_yes_judge_no']} | {a['human_no_judge_yes']} | {a['human_no_judge_no']} |"
        )
    lines += ["", "Rule (Phase 5 review): gate-judge kappa < 0.6 -> revisit the judge.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("sample")
    sp.add_argument("--n", type=int, default=80)
    sp.add_argument("--force", action="store_true", help="overwrite a file that already has labels")
    sub.add_parser("label")
    sc = sub.add_parser("score")
    sc.add_argument(
        "--with-llm", action="store_true", help="also ask qwen and gpt-oss per claim (cached)"
    )
    args = ap.parse_args(argv)

    if args.cmd == "sample":
        if LABELS.exists() and any(r["human_supported"] for r in read_labels()) and not args.force:
            print(
                f"{LABELS} already has labels; refusing to overwrite (use --force)", file=sys.stderr
            )
            return 1
        rows, verdicts = sample(load_rows(), args.n)
        if not rows:
            print("no judged claims yet; run `make rag-eval` first", file=sys.stderr)
            return 1
        write_labels(rows)
        atomic_write_text(VERDICTS, json.dumps(verdicts, indent=2, sort_keys=True))
        print(f"wrote {len(rows)} claims to {LABELS} (judge supported: {sum(verdicts.values())})")
        return 0
    if args.cmd == "label":
        return label()

    rows = read_labels()
    verdicts = json.loads(VERDICTS.read_text())
    extra = {}
    if args.with_llm:
        from app.llm import build_llm
        from app.settings import get_settings
        from rag.faithfulness import JudgeCache

        s, cache = get_settings(), JudgeCache()
        for role, model in (("judge", s.judge_model), ("draft", s.groq_model)):
            llm = build_llm(s, role)
            extra[f"{model} (per claim)"] = {
                r["claim_id"]: claim_verdict(llm, model, r["claim"], r["sources"], cache)
                for r in rows
                if r["human_supported"] in ("yes", "no")
            }
    result = score(rows, verdicts, extra)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "judge_validation.json").write_text(json.dumps(result, indent=2))
    (OUT / "judge_validation.md").write_text(report_md(result))
    print(report_md(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
