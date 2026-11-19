"""Cross-system matching (secondary records vs primary) and duplicate detection inside the secondary system."""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from .utils import norm_text, norm_supplier, po_key


def _nospace(s):
    return norm_text(s).replace(" ", "")


def match_systems(recs, cfg, status_ok=None):
    """Returns DataFrame (index = record_id of duplicate secondary rows) with dup_of, dup_kind, score."""
    m = cfg.get("matching", {})
    tol = m.get("amount_abs_tol", 0.02)
    minlen = m.get("min_po_len", 6)
    prim = recs[recs.role == "primary"]
    sec = recs[recs.role == "secondary"]
    if len(sec) == 0 or len(prim) == 0:
        return pd.DataFrame(columns=["dup_of", "dup_kind", "score"]).rename_axis("record_id")

    p = pd.DataFrame({"rid": prim.record_id.values, "spend": prim.spend.values, "qty": prim.quantity.values,
                      "po": [po_key(x, minlen) for x in prim.po_raw], "txt": [_nospace(x) for x in prim.desc],
                      "sup": [norm_supplier(x) for x in prim.supplier]}).dropna(subset=["spend"])
    p = p.sort_values("spend").reset_index(drop=True)
    vals = p.spend.values
    s = pd.DataFrame({"rid": sec.record_id.values, "spend": sec.spend.values, "qty": sec.quantity.values,
                      "po": [po_key(x, minlen) for x in sec.po_raw], "txt": [_nospace(x) for x in sec.desc],
                      "sup": [norm_supplier(x) for x in sec.supplier], "inv": sec.invoice_number.values,
                      "amt": sec.amount_raw.values})
    out = {}
    # 1) match to the primary system: amount + PO / quantity / text / supplier evidence
    for r in s.itertuples(index=False):
        if pd.isna(r.spend):
            continue
        t = max(tol, 1e-7 * abs(r.spend))
        lo, hi = np.searchsorted(vals, r.spend - t, "left"), np.searchsorted(vals, r.spend + t, "right")
        best, best_sc = None, 0.0
        for j in range(lo, hi):
            c = p.iloc[j]
            po_eq = bool(r.po) and bool(c.po) and r.po == c.po
            po_diff = bool(r.po) and bool(c.po) and r.po != c.po
            qty_eq = pd.notna(r.qty) and pd.notna(c.qty) and abs(r.qty - c.qty) <= 0.011
            ts = fuzz.ratio(r.txt, c.txt) / 100.0 if r.txt and c.txt else 0.0
            ss = fuzz.token_set_ratio(r.sup, c.sup) / 100.0 if r.sup and c.sup else 0.0
            if po_eq:
                ok = True
            elif po_diff:                           # both POs present but different = different purchase
                ok = False
            else:                                   # PO missing on one side
                ok = qty_eq and (ts >= 0.6 or ss >= 0.7)
            if ok:
                sc = 3 * po_eq + qty_eq + ts + 0.5 * ss
                if sc > best_sc:
                    best, best_sc = c.rid, sc
        if best is not None:
            out[r.rid] = (best, "CROSS", round(best_sc, 3))
    # 2) exported twice inside the secondary system (invoice digits + text + qty + amount)
    ok_mask = status_ok if status_ok is not None else pd.Series(True, index=sec.record_id.values)
    seen = {}
    for r in s.itertuples(index=False):
        if r.rid in out or not bool(ok_mask.get(r.rid, True)) or pd.isna(r.amt):
            continue
        inv = po_key(r.inv, 4)
        key = (inv, r.txt, None if pd.isna(r.qty) else round(r.qty, 3), round(r.amt, 2))
        if not inv:
            continue
        if key in seen:
            out[r.rid] = (seen[key], "INTERNAL", 1.0)
        else:
            seen[key] = r.rid
    return pd.DataFrame.from_dict(out, orient="index", columns=["dup_of", "dup_kind", "score"]).rename_axis("record_id")
