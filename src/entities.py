"""Supplier and item de-duplication: one key per real supplier / real item."""
import re
from collections import Counter
import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz
from .utils import norm_text, norm_supplier, UnionFind


def _union_similar(strings, uf, cutoff, prefix):
    """Union strings whose rapidfuzz ratio >= cutoff (chunked so memory stays small)."""
    u = list(dict.fromkeys(strings))
    n = len(u)
    if n < 2:
        return
    step = 1500
    for s0 in range(0, n, step):
        sc = process.cdist(u[s0:s0 + step], u, scorer=fuzz.ratio, score_cutoff=cutoff, dtype=np.uint8, workers=-1)
        ii, jj = np.nonzero(sc)
        for i, j in zip(ii, jj):
            if s0 + i < j:
                uf.union(prefix + u[s0 + i], prefix + u[j])


def _label(groups, root_of):
    """Readable key per cluster: most common member text; made unique if two clusters share a label."""
    best = {}
    for root, cnt in groups.items():
        best[root] = cnt.most_common(1)[0][0]
    seen, out = Counter(), {}
    for root in sorted(best, key=lambda r: best[r]):
        lab = best[root]
        seen[lab] += 1
        out[root] = lab if seen[lab] == 1 else f"{lab}~{seen[lab]}"
    return out


def supplier_keys(recs, match_df, cutoff=90):
    """Names are fuzzy-clustered; rows sharing a supplier id are joined; matched tool rows join their ERP supplier."""
    uf = UnionFind()
    name = recs.supplier.map(lambda s: norm_supplier(s) if pd.notna(s) else "")
    sf = name.str.replace(" ", "", regex=False)
    sid = recs.supplier_id.fillna("").astype(str).str.strip().str.upper()
    _union_similar([x for x in sf.unique() if x], uf, cutoff, "n:")
    node = []
    for a, b in zip(sf.values, sid.values):
        na, nb = ("n:" + a if a else None), ("i:" + b if b and b != "NAN" else None)
        if na and nb:
            uf.union(na, nb)
        node.append(na or nb)
    node = pd.Series(node, index=recs.index)
    if len(match_df):
        by_id = pd.Series(node.values, index=recs.record_id.values)
        for tool_id, erp_id in match_df["dup_of"].items():
            a, b = by_id.get(tool_id), by_id.get(erp_id)
            if a and b:
                uf.union(a, b)
    root = node.map(lambda x: uf.find(x) if x else None)
    groups = {}
    for r, nm, i in zip(root.values, name.values, sid.values):
        if r is not None:
            groups.setdefault(r, Counter())[nm or i] += 1
    lab = _label(groups, root)
    return root.map(lambda r: lab[r] if r is not None else "unknown-supplier")


_SANDWICH = re.compile(r"(?<=[a-z])\d(?=[a-z])")      # OCR noise: 'Ba1last', 'C0UPLING'


def _item_sig(desc):
    t = _SANDWICH.sub("", norm_text(desc))
    nums = tuple(sorted(re.findall(r"\d+(?:\.\d+)?", t)))
    alpha = re.sub(r"[^a-z]", "", t)
    return nums, alpha, t


def item_keys(recs, match_df, sup_key, cutoff=90):
    """Same numbers (sizes, part numbers) + near-identical letters = same item; item code is the primary identity;
    matched tool rows take the item of their ERP line."""
    uf = UnionFind()
    sig = [_item_sig(d) for d in recs.desc.fillna("")]
    blocks = {}
    for nums, alpha, _ in sig:
        if alpha:
            blocks.setdefault(nums, set()).add(alpha)
    for nums, al in blocks.items():
        _union_similar(sorted(al), uf, cutoff, "|".join(nums) + "#")
    code = recs.item_code.fillna("").astype(str).str.strip().str.upper()
    tn_list = ["|".join(n) + "#" + al if al else None for n, al, _ in sig]
    cn_list = ["c:" + c if c and c != "NAN" else None for c in code.values]
    # a text cluster that always carries ONE item code takes that code; the code is the primary identity
    tcodes = {}
    for tn, cn in zip(tn_list, cn_list):
        if tn and cn:
            tcodes.setdefault(uf.find(tn), set()).add(cn)
    node = []
    for tn, cn, sk in zip(tn_list, cn_list, sup_key.values):
        if cn:
            node.append(cn)
        elif tn and len(tcodes.get(uf.find(tn), ())) == 1:
            node.append(next(iter(tcodes[uf.find(tn)])))
        else:
            node.append(tn or "nodesc:" + sk)
    node = pd.Series(node, index=recs.index)
    if len(match_df):
        by_id = pd.Series(node.values, index=recs.record_id.values)
        for tool_id, erp_id in match_df["dup_of"].items():
            a, b = by_id.get(tool_id), by_id.get(erp_id)
            if a and b and not str(a).startswith("nodesc:") and not str(b).startswith("nodesc:"):
                uf.union(a, b)
    root = node.map(uf.find)
    groups = {}
    for r, (_, _, t) in zip(root.values, sig):
        groups.setdefault(r, Counter())[t[:60] or r] += 1
    lab = _label(groups, root)
    return root.map(lambda r: lab[r])


def pairwise_scores(pred, gold):
    """Pairwise precision / recall of a clustering against golden keys (rows with gold only)."""
    m = gold.notna()
    ct = pd.crosstab(pred[m].values, gold[m].values).values.astype(float)
    c2 = lambda x: (x * (x - 1) / 2).sum()
    tp = c2(ct)
    p, r = tp / max(c2(ct.sum(1)), 1), tp / max(c2(ct.sum(0)), 1)
    return round(p, 4), round(r, 4)