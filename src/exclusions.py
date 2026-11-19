"""Decide what is NOT spend and give a reason code. Precedence: status > DUPLICATE > NON_PROCUREMENT."""
import json
import os
import re
import warnings
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", message="This pattern is interpreted as a regular expression")
CANCEL_WORDS = ("cancel", "void", "revers")
DEFAULT_MODEL_COLS = ["account_code", "account_desc", "supplier_type", "doc_type", "po_type", "erp_system",
                      "region", "cost_centre", "transport_flag", "plant"]


def status_reasons(recs, cfg):
    """CANCELLED / NOT_APPROVED from the status column of sources that have one. Blank status = kept."""
    reason = pd.Series("", index=recs.index)
    for src in cfg["sources"]:
        if not src["columns"].get("status"):
            continue
        keep = [k.lower().strip() for k in src.get("status_include", ["approved", "paid"])]
        m = recs.source == src["name"]
        st = recs.loc[m, "status"].astype("string").str.lower().str.strip()
        known = st.notna() & (st != "")
        is_cancel = known & st.str.contains("|".join(CANCEL_WORDS), regex=True, na=False)
        not_ok = known & ~st.isin(keep) & ~is_cancel
        reason.loc[st.index[is_cancel]] = "CANCELLED"
        reason.loc[st.index[not_ok]] = "NOT_APPROVED"
    return reason


def _lexicon(cfg):
    p = os.path.join(cfg["_cfgdir"], "default_lexicon.json")
    lex = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    npc = cfg.get("non_procurement", {})
    return {
        "strong": [re.compile(x, re.I) for x in lex.get("strong_text_patterns", []) + npc.get("text_patterns", [])],
        "short": [re.compile(x, re.I) for x in lex.get("short_text_patterns", [])],
        "short_n": lex.get("short_text_max_words", 6),
        "acct": [re.compile(x, re.I) for x in lex.get("account_patterns", []) + npc.get("gl_description_patterns", [])],
    }


def rule_non_procurement(recs, cfg):
    """Regex lexicon on description / account description / supplier name. Returns bool Series."""
    lex = _lexicon(cfg)
    desc = recs["desc"].fillna("").astype(str)
    acct = recs["account_desc"].fillna("").astype(str) if "account_desc" in recs else pd.Series("", index=recs.index)
    sup = recs["supplier"].fillna("").astype(str)
    nwords = desc.str.split().str.len().fillna(0)
    hit = pd.Series(False, index=recs.index)
    for rx in lex["strong"]:
        hit |= desc.str.contains(rx) | sup.str.contains(rx) | acct.str.contains(rx)
    for rx in lex["short"]:
        hit |= desc.str.contains(rx) & (nwords <= lex["short_n"])
    for rx in lex["acct"]:
        hit |= acct.str.contains(rx)
    return hit


def _features(recs, cols, vec=None, enc=None):
    X = recs[cols].fillna("NA").astype(str)
    txt = recs["desc"].fillna("").astype(str)
    if enc is None:
        enc = OneHotEncoder(handle_unknown="ignore", min_frequency=2).fit(X)
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, lowercase=True).fit(txt)
    flags = np.c_[(txt.str.strip() == "").astype(float), recs["spend"].fillna(0).abs().add(1).pipe(np.log10)]
    return hstack([enc.transform(X), vec.transform(txt), flags]).tocsr(), vec, enc


def learned_non_procurement(recs, cfg, labels, seed=0):
    """Optional: train on analyst labels (include column) -> P(NON_PROCUREMENT). None if no labels."""
    lab = cfg.get("labels") or {}
    if labels is None or not lab.get("include_col"):
        return None
    cols = [c for c in cfg.get("non_procurement", {}).get("model_columns", DEFAULT_MODEL_COLS) if c in recs.columns]
    lab = labels.set_index(cfg["labels"]["record_id_col"])
    y = lab[cfg["labels"]["include_col"]].astype(str).str.strip().isin(["0", "False", "false"]).astype(int)
    idx = recs.index[recs.record_id.isin(y.index)]
    if len(idx) < 50 or y.sum() < 5:
        return None
    X, vec, enc = _features(recs, cols)
    yy = y.reindex(recs.loc[idx, "record_id"]).values
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    rf.fit(X[recs.index.get_indexer(idx)], yy)
    p = pd.Series(rf.predict_proba(X)[:, 1], index=recs.index)
    p.loc[idx] = yy.astype(float)          # analyst labels are the truth where we have them
    return p


def apply_exclusions(recs, cfg, match_df, labels=None):
    """Returns DataFrame(reason, np_source) aligned with recs.index. reason '' = included."""
    reason = status_reasons(recs, cfg)
    dup = recs.record_id.map(match_df["dup_kind"]) if len(match_df) else pd.Series(np.nan, index=recs.index)
    reason[(reason == "") & dup.notna()] = "DUPLICATE"
    rule = rule_non_procurement(recs, cfg)
    # learned model is OFF by default: seeds are biased (ERP-B6/B8 seeds are all intercompany) so it learns shortcuts
    use_model = cfg.get("non_procurement", {}).get("use_model", False)
    prob = learned_non_procurement(recs, cfg, labels) if use_model else None
    model = (prob >= 0.5) if prob is not None else pd.Series(False, index=recs.index)
    np_mask = rule | model
    reason[(reason == "") & np_mask] = "NON_PROCUREMENT"
    src = pd.Series("", index=recs.index)
    src[(reason == "NON_PROCUREMENT") & rule] = "rule"
    src[(reason == "NON_PROCUREMENT") & ~rule & model] = "model"
    return pd.DataFrame({"reason": reason, "np_source": src})