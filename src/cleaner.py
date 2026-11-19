"""
cleaner.py  --  STAGE 3
Kaam: har source table ko saaf karna.
  - amount text -> number   ("25,704.47" -> 25704.47)
  - date text  -> date      (config ke formats + fallback), na parse ho to flag
  - VAT rate   -> fraction  ("15%", "15", "0.15" -> 0.15; blank -> default 15%)
  - gross -> net            (net = gross / (1 + vat))
  - PO number  -> 10 digit saaf PO + item
  - status     -> approved / cancelled / not_approved
  - text       -> lowercase, bina accent/symbol (matching ke liye)
Sab rules config se aate hain.
"""
import re
import unicodedata

import numpy as np
import pandas as pd
from dateutil import parser as dateparser


# ---------------------------------------------------------------- amount
def parse_amount(s):
    """Text amount -> float. Na ban paye to NaN. Comma/decimal dono tarah handle."""
    if s is None:
        return np.nan
    s = str(s).strip()
    if s == "":
        return np.nan
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg, s = True, s[1:-1]
    if s.endswith("-"):
        neg, s = True, s[:-1]
    s = re.sub(r"[^\d,.\-]", "", s)          # currency symbol, space hatao
    if s.startswith("-"):
        neg, s = (not neg) if False else True, s[1:]
    if s == "":
        return np.nan
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):       # 1.234,56  (European)
            s = s.replace(".", "").replace(",", ".")
        else:                                 # 1,234.56
            s = s.replace(",", "")
    elif "," in s:
        if re.fullmatch(r"\d{1,3}(,\d{3})+", s):   # 1,234 / 1,234,567 -> thousands
            s = s.replace(",", "")
        else:                                       # 1234,56 -> decimal comma
            s = s.replace(",", ".")
    try:
        v = float(s)
    except ValueError:
        return np.nan
    return -v if neg else v


# ---------------------------------------------------------------- date
def parse_date(s, formats):
    """Text date -> Timestamp, ya NaT. Pehle config ke formats, phir dayfirst fallback."""
    s = (s or "").strip()
    if s == "":
        return pd.NaT
    for fmt in formats:
        try:
            return pd.Timestamp(pd.to_datetime(s, format=fmt))
        except (ValueError, TypeError):
            continue
    try:  # fallback: day-first (Indian/European/South African style)
        return pd.Timestamp(dateparser.parse(s, dayfirst=True))
    except (ValueError, OverflowError, TypeError):
        return pd.NaT


def parse_dates_series(series, formats):
    """Unique values ko ek baar parse karo (tez)."""
    uniq = {v: parse_date(v, formats) for v in series.unique()}
    parsed = series.map(uniq)
    flag = np.where(series.str.strip() == "", "blank",
                    np.where(parsed.isna(), "unparseable", "ok"))
    return pd.to_datetime(parsed), pd.Series(flag, index=series.index)


# ---------------------------------------------------------------- vat
def parse_vat(s, default):
    """'15%', '15', '0.15' -> 0.15 ; blank -> default ; '0' -> 0.0 (explicit zero-rated)."""
    s = (s or "").strip().replace("%", "")
    if s == "":
        return default
    try:
        v = float(s.replace(",", "."))
    except ValueError:
        return default
    return v / 100.0 if v > 1 else v


# ---------------------------------------------------------------- po
def extract_po(s, regex):
    """'PO#4510059209', '4510085877/10.0' -> ('4510059209', '' ) / ('4510085877', '10')."""
    s = (s or "").strip()
    m = re.search(regex, s)
    if not m:
        return "", ""
    po = m.group(1)
    rest = s[m.end():]
    mi = re.match(r"\s*[/\-]\s*(\d+)(?:\.0+)?", rest)
    return po, (str(int(mi.group(1))) if mi else "")


# ---------------------------------------------------------------- text
def normalize_text(s):
    """Lowercase, accents hatao, symbols -> space, extra space hatao."""
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9]+", " ", s.lower())
    return s.strip()


# ---------------------------------------------------------------- status
def classify_status(status_series, status_cfg):
    """approved / cancelled / not_approved. Config me status nahi => sab approved."""
    if not status_cfg:
        return pd.Series("approved", index=status_series.index)
    low = status_series.str.strip().str.lower()
    approved = {v.lower() for v in status_cfg.get("approved_values", [])}
    cancelled = {v.lower() for v in status_cfg.get("cancelled_values", [])}
    out = np.where(low.isin(approved), "approved",
                   np.where(low.isin(cancelled), "cancelled", "not_approved"))
    return pd.Series(out, index=status_series.index)


# ---------------------------------------------------------------- main
def clean_source(df, src_cfg, cfg):
    """Ek source table saaf karo. Naye columns jodta hai, purane nahi badalta."""
    df = df.copy()
    n = len(df)

    # amount
    df["amount_num"] = df["amount"].map(parse_amount)
    df["amount_bad"] = df["amount_num"].isna()

    # date
    fmts = src_cfg.get("date_formats", [])
    if "date" in df.columns:
        df["date_parsed"], df["date_flag"] = parse_dates_series(df["date"], fmts)
    else:
        df["date_parsed"], df["date_flag"] = pd.NaT, "blank"

    # vat + net
    if src_cfg.get("amount_basis", "net") == "gross":
        default_vat = float(src_cfg.get("default_vat_rate", 0.0))
        raw_vat = df["vat_rate"] if "vat_rate" in df.columns else pd.Series("", index=df.index)
        df["vat_used"] = raw_vat.map(lambda x: parse_vat(x, default_vat))
        df["amount_net"] = (df["amount_num"] / (1.0 + df["vat_used"])).round(2)
    else:
        df["vat_used"] = 0.0
        df["amount_net"] = df["amount_num"].round(2)

    # po
    if "po_number" in df.columns:
        regex = cfg.get("matching", {}).get("po_digits_regex", r"(\d{10})")
        pairs = df["po_number"].map(lambda x: extract_po(x, regex))
        df["po_clean"] = pairs.map(lambda p: p[0])
        po_item_raw = df["po_item"] if "po_item" in df.columns else None
        df["po_item_clean"] = pairs.map(lambda p: p[1])
        if po_item_raw is not None:   # ERP: "10.0" -> "10"
            df["po_item_clean"] = po_item_raw.map(
                lambda x: str(int(float(x))) if re.fullmatch(r"\d+(\.0+)?", (x or "").strip()) else "")
    else:
        df["po_clean"], df["po_item_clean"] = "", ""

    # status
    status_col = df["status"] if "status" in df.columns else pd.Series("", index=df.index)
    df["status_class"] = classify_status(status_col, src_cfg.get("status"))

    # text for matching
    df["desc_norm"] = df["description"].map(normalize_text) if "description" in df.columns else ""
    df["supplier_norm"] = df["supplier_name"].map(normalize_text) if "supplier_name" in df.columns else ""
    if "currency" in df.columns:
        df["currency"] = df["currency"].str.upper()
    return df


def clean_company(company):
    """load_company() ka output lo, saare sources saaf karke wapas do."""
    cfg = company["config"]
    cfg_by_name = {s["name"]: s for s in cfg["sources"]}
    company["clean"] = {name: clean_source(df, cfg_by_name[name], cfg)
                        for name, df in company["sources"].items()}
    return company