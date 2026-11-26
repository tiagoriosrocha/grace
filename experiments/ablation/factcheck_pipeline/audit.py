"""JSONL and human-readable HTML audit trail."""
from __future__ import annotations

import html
import json
from pathlib import Path


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def write_html(path: Path, rows: list[dict]) -> None:
    blocks = []
    for row in rows:
        conditions = []
        for condition in row.get("conditions", []):
            decision = condition["decision"]
            colour = {"SUPPORTED": "#d9f2df", "CONTRADICTED": "#ffdada", "INSUFFICIENT": "#fff3c4"}.get(decision, "#eee")
            candidates = []
            for candidate in condition.get("candidates", []):
                candidates.append("<li><b>rank %s</b> %s — support %.3f, contradict %.3f, irrelevant %.3f<br><code>%s</code></li>" % (
                    candidate.get("candidate_rank", "?"), candidate.get("decision", ""),
                    candidate.get("p_support", 0), candidate.get("p_contradict", 0), candidate.get("p_irrelevant", 0),
                    html.escape(json.dumps(candidate.get("evidence_triple", {}), ensure_ascii=False))))
            conditions.append("<section style='background:%s'><h3>%s — %s</h3><p><code>%s</code></p><p>support=%.3f; contradict=%.3f</p><ol>%s</ol></section>" % (
                colour, html.escape(condition["condition_id"]), html.escape(decision),
                html.escape(json.dumps(condition.get("claim_triple", {}), ensure_ascii=False)),
                condition.get("support_score", 0), condition.get("contradict_score", 0), "".join(candidates)))
        blocks.append("<article><h2>%s — %s</h2><p><b>Claim:</b> %s</p><p>gold=%s | prediction=%s</p>%s</article>" % (
            html.escape(str(row.get("id"))), html.escape(str(row.get("global_verdict"))),
            html.escape(str(row.get("claim", ""))), html.escape(str(row.get("gold", ""))),
            html.escape(str(row.get("prediction", ""))), "".join(conditions)))
    content = "<!doctype html><meta charset='utf-8'><title>Local pipeline audit</title><style>body{font:15px system-ui;max-width:1200px;margin:30px auto;padding:0 18px;color:#24313b}article{border:1px solid #ccd6dc;padding:15px;margin:15px 0}section{padding:10px;margin:10px 0;border-radius:6px}code{white-space:pre-wrap}</style><h1>Local fact-checking audit</h1>" + "".join(blocks)
    path.write_text(content, encoding="utf-8")
