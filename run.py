"""run.py - one command pipeline.
python run.py --config configs/company_a.json --name yourname [--data-dir X --suffix A2 --hide-labels]
python run.py --all --name yourname"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.ingest import load_config, ingest_all, load_taxonomy, load_labels
from src.matching import match_systems
from src.exclusions import status_reasons, apply_exclusions
from src.entities import supplier_keys, item_keys, pairwise_scores
from src.classify import classify

COLS = ["record_id", "include", "exclusion_reason", "supplier_key", "item_key",
        "spend", "l1", "l2", "l3", "l4", "confidence"]


def run(cfg_path, name, data_dir=None, suffix=None, hide_labels=False, out_dir="outputs"):
    t0 = time.time()
    cfg = load_config(cfg_path)
    recs, ddir = ingest_all(cfg, data_dir)
    tax, lc = load_taxonomy(cfg, ddir)
    labels = None if hide_labels else load_labels(cfg, ddir)
    gold = load_labels(cfg, ddir) if hide_labels else None
    print(f"[run] ingested {recs.shape}")

    st = status_reasons(recs, cfg)
    sec = recs["role"] == "secondary"
    status_ok = pd.Series((st[sec] == "").values, index=recs.loc[sec, "record_id"].values)
    match = match_systems(recs, cfg, status_ok)
    ex = apply_exclusions(recs, cfg, match, labels)
    reason = ex["reason"].fillna("").astype(str)
    inc = reason == ""
    print("[run] exclusions:", reason.value_counts().to_dict())

    sk = supplier_keys(recs, match)
    ik = item_keys(recs, match, sk)
    cl = classify(recs, inc, tax, lc, cfg, labels, sk, ik)

    out = pd.DataFrame({"record_id": recs["record_id"].values}, index=recs.index)
    out["include"] = inc.astype(int).values
    out["exclusion_reason"] = reason.values
    out["supplier_key"] = sk.reindex(recs.index).fillna("").astype(str).values
    out["item_key"] = ik.reindex(recs.index).fillna("").astype(str).values
    sp = pd.to_numeric(recs["spend"], errors="coerce").fillna(0.0)
    if "amount_bad" in recs.columns:
        sp = sp.where(~recs["amount_bad"].fillna(False).astype(bool), 0.0)
    out["spend"] = np.where(inc, sp, 0.0).round(2)
    for c in ["l1", "l2", "l3", "l4"]:
        out[c] = ""
        out.loc[cl.index, c] = cl[c].values
    out["confidence"] = 0.0
    out.loc[cl.index, "confidence"] = cl["confidence"].values
    out = out[COLS]

    suffix = suffix or cfg.get("output_suffix", cfg.get("company_id", "X"))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}__{suffix}.csv")
    out.to_csv(path, index=False)
    print(f"[run] wrote {path} rows={len(out)} included={int(inc.sum())} spend={out.spend.sum():,.0f} time={time.time()-t0:.0f}s")

    if hide_labels and gold is not None:
        evaluate(out, recs, gold, cfg)
    return path


def evaluate(out, recs, gold, cfg):
    idc = next((c for c in ("record_id", "line_id") if c in gold.columns), None)
    g = gold.drop_duplicates(idc).set_index(idc)
    gcols = [c for c in g.columns if c.startswith("category_l")]
    o = out.set_index("record_id").join(g[gcols], how="inner")
    o = o[o.include == 1]
    if len(o) == 0:
        print("[eval] no overlap"); return
    w = o.spend.abs().clip(lower=1e-9)
    for i, gc in enumerate(gcols, 1):
        ok = (o[f"l{i}"].astype(str) == o[gc].fillna("").astype(str))
        print(f"[eval] L{i} acc count={ok.mean():.3f} spend={(ok*w).sum()/w.sum():.3f}")
    for k in ("supplier_key", "item_key"):
        if k in g.columns:
            m = out.set_index("record_id").join(g[[k]].rename(columns={k: "gold"}), how="inner")
            m = m[m.gold.notna()]
            if len(m):
                print(f"[eval] {k} pairwise P/R:", pairwise_scores(m[k], m["gold"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--name", required=True)
    ap.add_argument("--data-dir")
    ap.add_argument("--suffix")
    ap.add_argument("--hide-labels", action="store_true")
    ap.add_argument("--out-dir", default="outputs")
    a = ap.parse_args()
    if a.all:
        for c in ("configs/company_a.json", "configs/company_b.json"):
            run(c, a.name, out_dir=a.out_dir)
    else:
        run(a.config, a.name, a.data_dir, a.suffix, a.hide_labels, a.out_dir)