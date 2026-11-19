"""Config-driven ingest: read files, clean amounts (net of tax, reporting currency) and dates."""
import json
import os
import numpy as np
import pandas as pd
from .utils import parse_number, parse_dates

CORE = ["supplier", "supplier_id", "item_code", "po", "quantity", "invoice_number", "date", "status", "vat_rate"]


def load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_root"] = os.path.dirname(os.path.dirname(os.path.abspath(path)))   # folder that holds configs/
    cfg["_cfgdir"] = os.path.dirname(os.path.abspath(path))
    return cfg


def data_dir(cfg, override=None):
    d = override or cfg.get("data_dir", ".")
    if os.path.isabs(d) and os.path.isdir(d):
        return d
    for base in (cfg["_root"], os.getcwd(), cfg["_cfgdir"]):
        p = os.path.normpath(os.path.join(base, d))
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(f"data_dir '{d}' not found (tried relative to project root, cwd, config folder)")


def read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=True, encoding_errors="replace")


def load_fx(cfg, ddir):
    fx = cfg.get("fx")
    if not fx:
        return None
    df = read_csv(os.path.join(ddir, fx["file"]))
    df["_rate"] = parse_number(df[fx["units_per_usd_col"]])
    df["_m"] = pd.to_datetime(df[fx["month_col"]], errors="coerce").dt.strftime("%Y-%m")
    return {(c, m): r for c, m, r in zip(df[fx["currency_col"]].str.strip().str.upper(), df["_m"], df["_rate"]) if pd.notna(r)}


def _fx_lookup(fxd, cur, months):
    """units per USD for currency in the first month (of the candidates) the table knows; else nearest month."""
    if cur == "USD":
        return 1.0, "ok"
    for m in months:
        if m and (cur, m) in fxd:
            return fxd[(cur, m)], "ok"
    have = sorted(m for (c, m) in fxd if c == cur)
    if not have:
        return np.nan, "fx_missing"
    ref = next((m for m in months if m), None)
    if ref is None:
        return fxd[(cur, have[-1])], "fx_nearest"
    nearest = min(have, key=lambda m: abs((int(m[:4]) * 12 + int(m[5:])) - (int(ref[:4]) * 12 + int(ref[5:]))))
    return fxd[(cur, nearest)], "fx_nearest"


def ingest_source(cfg, src, ddir, fxd):
    df = read_csv(os.path.join(ddir, src["file"]))
    cols = src["columns"]
    out = pd.DataFrame(index=df.index)
    out["record_id"] = src.get("record_id_prefix", "") + df[src["record_id_col"]].astype(str).str.strip()
    out["source"] = src["name"]
    out["role"] = src.get("role", "primary")

    def col(key):
        c = cols.get(key)
        return df[c] if c and c in df.columns else pd.Series(np.nan, index=df.index, dtype=object)

    # text-like fields
    for k in ["supplier", "supplier_id", "item_code", "invoice_number", "status"]:
        out[k] = col(k).astype("string").str.strip()
    out["po_raw"] = col("po")
    out["quantity"] = parse_number(col("quantity"))
    desc_cols = cols.get("description", [])
    desc_cols = [desc_cols] if isinstance(desc_cols, str) else desc_cols
    parts = [df[c].fillna("").astype(str).str.strip() for c in desc_cols if c in df.columns]
    out["desc"] = (pd.concat(parts, axis=1).apply(lambda r: " ".join(x for x in r if x and x.lower() != "nan"), axis=1)
                   if parts else "")
    # every other mapped column is kept under its own key (account_desc, cost_centre, ...)
    skip = set(CORE) | {"description", "amount", "currency", "fx_rate", "po", "posting_date", "date"}
    for k, c in cols.items():
        if k not in skip and isinstance(c, str) and c in df.columns:
            out[k] = df[c].astype("string").str.strip()

    # dates
    fmts = src.get("date_formats", [])
    out["date"], out["date_flag"] = parse_dates(col("date"), fmts)
    if cols.get("posting_date"):
        out["posting_date"], _ = parse_dates(col("posting_date"), fmts)

    # amounts: parse -> net of tax -> reporting currency
    amt = parse_number(col("amount"))
    out["amount_bad"] = amt.isna()
    out["amount_raw"] = amt
    basis = src.get("amount_basis", "net")
    if basis == "gross":
        v = parse_number(col("vat_rate").astype("string").str.replace("%", "", regex=False))
        v = np.where(v > 1, v / 100.0, v)
        v = pd.Series(v, index=df.index).fillna(src.get("default_vat", 0.15))
        out["vat_used"] = v
        net = amt / (1.0 + v)
    else:
        out["vat_used"] = 0.0
        net = amt
    out["net_local"] = net

    rep = cfg.get("reporting_currency", "USD").upper()
    out["fx_flag"] = "ok"
    if src.get("amount_currency_is_reporting", False) or not cols.get("currency"):
        out["currency"] = rep
        out["spend"] = net
    else:
        cur = col("currency").astype("string").str.strip().str.upper().fillna(rep)
        rate = parse_number(col("fx_rate")) if cols.get("fx_rate") else pd.Series(np.nan, index=df.index)
        fxcols = src.get("fx_date_columns", ["date", "posting_date"])
        months = [out[c].dt.strftime("%Y-%m") if c in out else pd.Series(None, index=df.index, dtype=object) for c in fxcols]
        spend = np.full(len(df), np.nan)
        flags = np.array(["ok"] * len(df), dtype=object)
        for i, (c, r, n) in enumerate(zip(cur.values, rate.values, net.values)):
            if pd.isna(n):
                continue
            if c == rep:
                spend[i] = n
                continue
            if pd.isna(r) or r <= 0:
                if fxd is None:
                    flags[i] = "fx_missing"
                    continue
                mm = [m.iloc[i] for m in months]
                r, fl = _fx_lookup(fxd, c, mm)
                flags[i] = fl
                if pd.isna(r):
                    continue
            usd = n / r
            if rep != "USD":                               # cross to a non-USD reporting currency
                rr, _ = _fx_lookup(fxd, rep, [m.iloc[i] for m in months]) if fxd else (np.nan, "")
                usd = usd * rr
            spend[i] = usd
        out["currency"] = cur
        out["spend"] = spend
        out["fx_flag"] = flags
    return out


def load_taxonomy(cfg, ddir):
    t = cfg["taxonomy"]
    df = read_csv(os.path.join(ddir, t["file"]))
    lv = t["level_columns"]
    df = df[lv].apply(lambda s: s.str.strip()).replace("", np.nan).dropna(how="all").drop_duplicates().reset_index(drop=True)
    return df, lv


def load_labels(cfg, ddir):
    lab = cfg.get("labels")
    if not lab:
        return None
    p = os.path.join(ddir, lab["file"])
    return read_csv(p) if os.path.exists(p) else None


def ingest_all(cfg, ddir_override=None):
    ddir = data_dir(cfg, ddir_override)
    fxd = load_fx(cfg, ddir)
    frames = [ingest_source(cfg, s, ddir, fxd) for s in cfg["sources"]]
    recs = pd.concat(frames, ignore_index=True)
    return recs, ddir