"""Supplier and item de-duplication (FAST version): one key per real supplier / real item.
Supplier: normalised name (+ light fuzzy merge when the name pool is small) + same supplier id.
Item: item code first; else exact signature of the description (numbers + letters, OCR-noise removed).
Matched tool rows take the supplier / item of their ERP line."""
import re
from collections import Counter
import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz
from .utils import norm_text, norm_supplier

MAX_FUZZY_POOL = 3000   # fuzzy merge only if the unique-name pool is small enough to be fast


class UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, x):
        p = self.p
        if x not in p:
            p[x] = x
            return x
        r = x
        while p[r] != r:
            r = p[r]
        while p[x] != r:
            p[x], x = r, p[x]
        return r

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _union_similar(strings, uf, cutoff, prefix):
    """Union strings with rapidfuzz ratio >= cutoff; skipped when the pool is too big."""
    u = list(dict.fromkeys(strings))
    n = len(u)
    if n < 2 or n > MAX_FUZZY_POOL:
        return
    for s0 in range(0, n, 1000):
        sc = process.cdist(u[s0:s0 + 1000], u, scorer=fuzz.ratio, score_cutoff=cutoff,
                           dtype=np.uint8, workers=-1)
        ii, jj = np.nonzero(sc)
        for i, j in zip(ii, jj):
            if s0 + i < j:
                uf.union(prefix + u[s0 + i], prefix + u[j])


def _label(groups):
    """Readable key per cluster = most common member text; made unique if two clusters share a label."""
    best = {root: cnt.most_common(1)[0][0] for root, cnt in groups.items()}
    seen, out = Counter(), {}
    for root in sorted(best, key=lambda r: (best[r], str(r))):
        lab = best[root]
        seen[lab] += 1
        out[root] = lab if seen[lab] == 1 else f"{lab}~{seen[lab]}"
    return out


def supplier_keys(recs, match_df, cutoff=90):
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
    node = pd.Series(node, index=recs.index, dtype=object)
    if len(match_df):
        by_id = dict(zip(recs.record_id.values, node.values))
        for tool_id, erp_id in match_df["dup_of"].items():
            a, b = by_id.get(tool_id), by_id.get(erp_id)
            if a and b:
                uf.union(a, b)
    root = [uf.find(x) if isinstance(x, str) and x else None for x in node.values]
    groups = {}
    for r, nm, i in zip(root, name.values, sid.values):
        if r is not None:
            groups.setdefault(r, Counter())[nm or i] += 1
    lab = _label(groups)
    return pd.Series([lab[r] if r is not None else "unknown-supplier" for r in root], index=recs.index)


_SANDWICH = re.compile(r"(?<=[a-z])\d(?=[a-z])")      # OCR noise: 'Ba1last', 'C0UPLING'


def _item_sig(desc):
    t = _SANDWICH.sub("", norm_text(desc))
    nums = tuple(sorted(re.findall(r"\d+(?:\.\d+)?", t)))
    alpha = re.sub(r"[^a-z]", "", t)
    return nums, alpha, t


def item_keys(recs, match_df, sup_key, cutoff=90):
    """Item code is the primary identity; else exact description signature (no fuzzy => fast)."""
    uf = UnionFind()
    descs = recs.desc.fillna("")
    cache = {}
    sig = []
    for d in descs.values:
        s = cache.get(d)
        if s is None:
            s = cache[d] = _item_sig(d)
        sig.append(s)
    code = recs.item_code.fillna("").astype(str).str.strip().str.upper()
    tn_list = ["t:" + "|".join(n) + "#" + al[:60] if al else None for n, al, _ in sig]
    cn_list = ["c:" + c if c and c != "NAN" else None for c in code.values]
    tcodes = {}                      # a text signature that always carries ONE item code takes that code
    for tn, cn in zip(tn_list, cn_list):
        if tn and cn:
            tcodes.setdefault(tn, set()).add(cn)
    node = []
    for tn, cn, sk in zip(tn_list, cn_list, sup_key.values):
        if cn:
            node.append(cn)
        elif tn and len(tcodes.get(tn, ())) == 1:
            node.append(next(iter(tcodes[tn])))
        else:
            node.append(tn or "nodesc:" + str(sk))
    node = pd.Series(node, index=recs.index, dtype=object)
    if len(match_df):
        by_id = dict(zip(recs.record_id.values, node.values))
        for tool_id, erp_id in match_df["dup_of"].items():
            a, b = by_id.get(tool_id), by_id.get(erp_id)
            if a and b and not str(a).startswith("nodesc:") and not str(b).startswith("nodesc:"):
                uf.union(a, b)
    root = node.map(uf.find)
    groups = {}
    for r, (_, _, t) in zip(root.values, sig):
        groups.setdefault(r, Counter())[t[:60] or r] += 1
    lab = _label(groups)
    return root.map(lambda r: lab[r])


def pairwise_scores(pred, gold):
    """Pairwise precision / recall of a clustering against golden keys (rows with gold only)."""
    m = gold.notna()
    ct = pd.crosstab(pred[m].values, gold[m].values).values.astype(float)
    c2 = lambda x: (x * (x - 1) / 2).sum()
    tp = c2(ct)
    p, r = tp / max(c2(ct.sum(1)), 1), tp / max(c2(ct.sum(0)), 1)
    return round(p, 4), round(r, 4)