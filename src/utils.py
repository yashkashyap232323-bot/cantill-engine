"""Helpers: number/date parsing and text normalisation."""
import re
import unicodedata
import numpy as np
import pandas as pd

_JUNK = re.compile(r"[^\d,.\-()]")


def parse_number_str(v):
    """'1,234.56' '1.234,56' '(12.30)' '45-' 'R 1 200,50' -> float (NaN if impossible)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    s = str(v).strip()
    if s == "" or s.lower() in ("nan", "none", "null", "-", "n/a"):
        return np.nan
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg, s = True, s[1:-1]
    if s.endswith("-") and not s.startswith("-"):
        neg, s = True, s[:-1]
    s = _JUNK.sub("", s).replace("(", "").replace(")", "")
    if s.startswith("-"):
        neg, s = True, s[1:]
    if s == "":
        return np.nan
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):          # 1.234,56
            s = s.replace(".", "").replace(",", ".")
        else:                                    # 1,234.56
            s = s.replace(",", "")
    elif "," in s:
        parts = s.split(",")
        if len(parts) > 1 and all(len(p) == 3 for p in parts[1:]) and 1 <= len(parts[0]) <= 3:
            s = s.replace(",", "")               # 1,234,567
        else:
            s = s.replace(",", ".")              # 12,5
    elif s.count(".") > 1:
        s = s.replace(".", "")
    try:
        x = float(s)
    except ValueError:
        return np.nan
    return -x if neg else x


def parse_number(series):
    m = {u: parse_number_str(u) for u in series.dropna().unique()}
    return series.map(m).astype(float)


def parse_dates(series, formats=None, min_year=2000, max_year=2035):
    """Parse many date formats. Returns (dates, flag); flag = ok/blank/unparseable/implausible."""
    s = series.astype("string").str.strip()
    blank = s.isna() | (s == "") | s.str.lower().isin(["nan", "none", "null", "nat"])
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    todo = ~blank
    for fmt in (formats or []):
        if not todo.any():
            break
        p = pd.to_datetime(s[todo], format=fmt, errors="coerce")
        hit = p.notna()
        out.loc[p.index[hit]] = p[hit]
        todo = todo & out.isna()
    if todo.any():
        p = pd.to_datetime(s[todo], errors="coerce", dayfirst=True, format="mixed")
        hit = p.notna()
        out.loc[p.index[hit]] = p[hit]
    flag = pd.Series("ok", index=series.index)
    flag[blank] = "blank"
    flag[~blank & out.isna()] = "unparseable"
    bad = out.notna() & ((out.dt.year < min_year) | (out.dt.year > max_year))
    out[bad] = pd.NaT
    flag[bad] = "implausible"
    return out, flag


_DEC_COMMA = re.compile(r"(?<=\d),(?=\d)")


def norm_text(s):
    """lower-case, no accents, decimal comma -> dot, punctuation -> space."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c)).lower()
    s = _DEC_COMMA.sub(".", s)
    s = re.sub(r"[^a-z0-9.\s]", " ", s)
    s = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", s)
    return re.sub(r"\s+", " ", s).strip()


_LEGAL = re.compile(r"\b(pty|ltd|limited|llc|inc|corp|corporation|gmbh|ag|sa|bv|nv|sas|sarl|spa|srl|co|cc|plc|lp|llp|pvt|private|the|company|group|holdings)\b")


def norm_supplier(s):
    return re.sub(r"\s+", " ", _LEGAL.sub(" ", norm_text(s))).strip()


def po_key(v, min_len=6):
    """Longest digit run (>= min_len): 'PO#4510055507' / '4510085877/10.0' -> '4510085877'."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    runs = [r for r in re.findall(r"\d+", str(v)) if len(r) >= min_len]
    return max(runs, key=len).lstrip("0") if runs else ""


class UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        r = x
        while self.p[r] != r:
            r = self.p[r]
        while self.p[x] != r:
            self.p[x], x = r, self.p[x]
        return r

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra