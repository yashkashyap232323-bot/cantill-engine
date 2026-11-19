"""classify.py - taxonomy-agnostic classifier.
Labels -> supervised (TF-IDF + supplier + categorical cols -> LinearSVC).
No labels -> tree-text similarity -> pseudo-labels -> self-train -> item/supplier smoothing."""
import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack, vstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC


def _norm(s):
    return s.fillna("").astype(str).str.lower().str.replace(r"[\W_]+", " ", regex=True).str.strip()


def _softmax(z, t=3.0):
    z = (z - z.max(1, keepdims=True)) * t
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _vec(v, s):
    try:
        return v.fit_transform(s).tocsr()
    except ValueError:
        return csr_matrix((len(s), 1))


def _align(s, recs):
    s = pd.Series(s)
    if s.index.equals(recs.index):
        return s
    if len(s) == len(recs):
        return pd.Series(s.values, index=recs.index)
    return s.reindex(recs.index)


def _tax_classes(tax_df, level_cols):
    cols = list(level_cols)[:4]
    t = tax_df[cols].fillna("").astype(str).apply(lambda c: c.str.strip()).drop_duplicates()
    return [tuple(r) for r in t.values.tolist()]


def _label_rows(recs, labels, K):
    empty = pd.Series([np.nan] * len(recs), index=recs.index, dtype=object)
    if labels is None or len(labels) == 0:
        return empty
    l = labels.copy()
    idc = next((c for c in ("record_id", "line_id") if c in l.columns), None)
    cc = sorted([c for c in l.columns if re.fullmatch(r"(category_)?l\d", str(c).lower())],
                key=lambda c: int(str(c)[-1]))
    if idc is None or not cc:
        return empty
    if "include" in l.columns:
        l = l[pd.to_numeric(l["include"], errors="coerce") == 1]
    l = l.drop_duplicates(idc)
    vals = l[cc[:K]].fillna("").astype(str).apply(lambda c: c.str.strip())
    tup = [tuple(r) + ("",) * (K - len(r)) for r in vals.values.tolist()]
    lab = pd.Series(tup, index=l[idc].astype(str).values, dtype=object)
    return recs["record_id"].astype(str).map(lab).set_axis(recs.index)


def _cat_tokens(recs, cols):
    out = pd.Series("", index=recs.index)
    for c in cols:
        v = recs[c].fillna("").astype(str).str.strip().str.lower()
        v = v.where(~v.isin(["nan", "none"]), "")
        whole = pd.Series(np.where(v == "", "", c + "=" + v.str.replace(r"\s+", "_", regex=True)), index=recs.index)
        words = v.map(lambda s: " ".join(f"{c}:{w}" for w in re.split(r"\W+", s) if w) if len(s.split()) > 1 else "")
        out = out + " " + whole + " " + words
    return out.str.replace(r"\s+", " ", regex=True).str.strip()


def _place(model, X, L):
    dec = model.decision_function(X)
    if dec.ndim == 1:
        dec = np.c_[-dec, dec]
    P = np.zeros((X.shape[0], L))
    P[:, model.classes_] = _softmax(dec, 3.0)
    return P


def _smooth(P, key, w):
    k = pd.Series(np.asarray(key, dtype=object)).astype(str)
    bad = k.isin(["", "nan", "None"])
    k = k.where(~bad, "u" + pd.Series(k.index.astype(str), index=k.index))
    gm = pd.DataFrame(P).groupby(k.values).transform("mean").values
    return (1 - w) * P + w * gm


def _embed(texts, tree, mode):
    if str(mode).lower() in ("off", "false", "0", "none"):
        return None
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        t = texts.str[:200]
        uniq = pd.unique(t)
        enc = m.encode(list(uniq) + list(tree), batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        pos = pd.Series(range(len(uniq)), index=uniq).reindex(t.values).values
        return enc[:len(uniq)][pos], enc[len(uniq):]
    except Exception as e:
        print("[classify] embeddings unavailable, TF-IDF only:", type(e).__name__)
        return None


def _zero_label(ztext, sup, classes, mode):
    L = len(classes)
    paths = [[w for w in reversed(c) if w] for c in classes]
    tree = _norm(pd.Series([(p[0] + " " + p[0] + " " if p else "") + " ".join(dict.fromkeys(p)) for p in paths]))
    allt = pd.concat([tree, ztext])
    vw = TfidfVectorizer(sublinear_tf=True).fit(allt)
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, max_features=300000).fit(allt)
    sim = 0.5 * (vw.transform(ztext) @ vw.transform(tree).T).toarray() \
        + 0.5 * (vc.transform(ztext) @ vc.transform(tree).T).toarray()
    emb = _embed(ztext, tree, mode)
    E = None
    if emb is not None:
        E, TE = emb
        es = E @ TE.T
        es = es - es.mean(0)
        sim = 0.6 * es + 0.4 * sim
    empty = (ztext == "").values
    P_sim = _softmax(sim, 15.0)
    ok = ~empty
    if ok.sum() < 10:
        return P_sim, empty
    srt = np.sort(sim, 1)
    margin = srt[:, -1] - srt[:, -2]
    top = sim.argmax(1)
    ps = ok & (margin >= np.quantile(margin[ok], 0.5))
    if len(set(top[ps])) < 2:
        return P_sim, empty
    parts = [_vec(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=100000), ztext),
             _vec(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=3, sublinear_tf=True, max_features=100000), ztext),
             _vec(CountVectorizer(token_pattern=r"\S+", binary=True, lowercase=False), sup)]
    if E is not None:
        parts.append(csr_matrix(E))
    X = hstack(parts).tocsr()
    svc = LinearSVC(C=0.5, tol=1e-3, max_iter=300).fit(X[ps], top[ps])
    return 0.6 * _place(svc, X, L) + 0.4 * P_sim, empty


def classify(recs, include_mask, tax_df, level_cols, cfg, labels, sup_key, item_key):
    """-> DataFrame(l1..l4, confidence) indexed by recs.index[include_mask]."""
    cc = (cfg.get("classifier") or {}) if isinstance(cfg, dict) else {}
    cat_cols = [c for c in cc.get("feature_columns", []) if c in recs.columns]
    classes = _tax_classes(tax_df, level_cols)
    K = len(classes[0])
    cidx = {c: i for i, c in enumerate(classes)}
    include_mask = include_mask.reindex(recs.index).fillna(False).astype(bool)

    lab_row = _label_rows(recs, labels if cc.get("use_labels", True) else None, K)
    lab_mask = lab_row.notna()
    for t in set(lab_row[lab_mask]):
        if t not in cidx:
            cidx[t] = len(classes)
            classes.append(t)
    L = len(classes)
    y_all = lab_row[lab_mask]
    supervised = lab_mask.sum() >= 50 and len(set(y_all)) >= 2

    use = (include_mask | lab_mask) if supervised else include_mask
    sub = recs.loc[use]
    sk = _align(sup_key, recs).loc[sub.index]
    ik = _align(item_key, recs).loc[sub.index]
    base = sub["desc"].fillna("").astype(str) if "desc" in sub.columns else pd.Series("", index=sub.index)
    text = _norm(base)
    cat = _cat_tokens(sub, cat_cols)
    sup = ("sup_" + sk.fillna("").astype(str).str.replace(r"\s+", "_", regex=True)).where(sk.notna(), "")
    inc_pos = include_mask.loc[sub.index].values
    print(f"[classify] mode={'supervised' if supervised else 'zero-label'} classes={L} rows={inc_pos.sum()} cat_cols={cat_cols}", flush=True)

    if supervised:
        tp = np.where(lab_mask.loc[sub.index].values)[0]
        y = np.array([cidx[t] for t in lab_row.loc[sub.index].values[tp]])
        W = _vec(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=60000), text)
        C = _vec(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=5, sublinear_tf=True, max_features=60000), text)
        S = _vec(CountVectorizer(token_pattern=r"\S+", binary=True, lowercase=False), sup)
        Xt = hstack([W, C, S]).tocsr()
        Xc = _vec(CountVectorizer(token_pattern=r"\S+", binary=True, lowercase=False), cat)
        full = hstack([Xt, Xc]).tocsr()
        tonly = hstack([Xt, csr_matrix(Xc.shape)]).tocsr()
        m = np.random.RandomState(0).rand(len(tp)) < 0.3
        svc = LinearSVC(C=0.3, tol=1e-3, max_iter=300).fit(vstack([full[tp], tonly[tp[m]]]), np.r_[y, y[m]])
        P_all = _place(svc, full, L)
        empty = np.zeros(len(sub), dtype=bool)
    else:
        extra = pd.Series("", index=sub.index)
        for c in cat_cols:
            extra = extra + " " + sub[c].fillna("").astype(str)
        ztext = _norm(base + " " + extra)
        P_all, empty = _zero_label(ztext, sup, classes, cc.get("embeddings", "auto"))
        if empty.any() and (~empty).any():
            keys = sk.fillna("").astype(str).values
            dfp = pd.DataFrame(P_all[~empty])
            gm = dfp.groupby(keys[~empty]).mean()
            gd = dict(zip(gm.index, gm.values))
            glob = P_all[~empty].mean(0)
            for i in np.where(empty)[0]:
                P_all[i] = gd.get(keys[i], glob)

    P = P_all[inc_pos]
    P = _smooth(P, ik.values[inc_pos], 0.3 if supervised else 0.5)
    if not supervised:
        P = _smooth(P, sk.values[inc_pos], 0.15)
    pred = P.argmax(1)
    conf = np.clip(P.max(1), 0.05, 0.99 if supervised else 0.90)
    if not supervised:
        conf[empty[inc_pos]] = 0.10

    arr = np.array([list(c) + [""] * (4 - len(c)) for c in classes], dtype=object)
    out = pd.DataFrame({f"l{j+1}": arr[pred, j] for j in range(4)}, index=sub.index[inc_pos])
    out["confidence"] = conf.round(4)
    return out