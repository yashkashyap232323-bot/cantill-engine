"""
data_loader.py  --  STAGE 2
Kaam: config file padhna aur config ke hisaab se CSV files load karna.
Yahan koi company-specific naam hardcoded nahi hai; sab config se aata hai.

Har source (ERP, tool, invoice_lines...) ko ek "standard" table me badalta hai:
  record_id, source, <config ke columns ke standard naam: po_number, date, amount ...>
Saari values text (string) me rehti hain; safai (cleaning) agle stage me hoti hai.
"""
import json
from pathlib import Path

import pandas as pd

ENGINE_DIR = Path(__file__).resolve().parent.parent  # .../cantill_engine


# ---------------------------------------------------------------- config
def load_config(config_path):
    """JSON ya YAML config padho."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    text = config_path.read_text(encoding="utf-8-sig")
    if config_path.suffix.lower() in (".yaml", ".yml"):
        import yaml
        cfg = yaml.safe_load(text)
    else:
        cfg = json.loads(text)
    cfg["_config_path"] = str(config_path.resolve())
    return cfg


def resolve_data_dir(cfg, data_dir_override=None):
    """Data folder ka path. Relative path cantill_engine folder se maana jata hai."""
    d = Path(data_dir_override or cfg.get("data_dir", "."))
    if not d.is_absolute():
        d = (ENGINE_DIR / d).resolve()
    if not d.exists():
        raise FileNotFoundError(f"Data folder not found: {d}")
    return d


# ---------------------------------------------------------------- csv
def read_csv_text(path):
    """CSV ko hamesha text (string) ki tarah padho; blank ko '' rakho (NaN nahi)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    df.columns = [c.strip() for c in df.columns]
    return df


def _join_text(parts):
    """Description ke kai columns ko jodo; khali aur repeat hataao."""
    seen, out = set(), []
    for p in parts:
        p = (p or "").strip()
        if p and p.lower() not in seen:
            seen.add(p.lower())
            out.append(p)
    return " | ".join(out)


# ---------------------------------------------------------------- source
def load_source(cfg, src, data_dir):
    """Ek source file ko standard table me badlo."""
    raw = read_csv_text(data_dir / src["file"])
    id_col = src["record_id_col"]
    if id_col not in raw.columns:
        raise KeyError(f"[{src['name']}] record id column '{id_col}' not in file. "
                       f"Columns: {list(raw.columns)}")

    out = pd.DataFrame(index=raw.index)
    out["record_id"] = src.get("record_id_prefix", "") + raw[id_col].str.strip()
    out["source"] = src["name"]

    for std_name, col in src["columns"].items():
        cols = col if isinstance(col, list) else [col]
        missing = [c for c in cols if c not in raw.columns]
        if missing:
            # Config me likha column file me nahi hai: saaf error, chup-chaap skip nahi.
            raise KeyError(f"[{src['name']}] column(s) {missing} (for '{std_name}') "
                           f"not found in {src['file']}")
        if isinstance(col, list):
            out[std_name] = [_join_text(r) for r in raw[cols].itertuples(index=False, name=None)]
        else:
            out[std_name] = raw[col].str.strip()
    return out


def load_taxonomy(cfg, data_dir):
    """Taxonomy padho. Levels ki ginti config se (3, 4 ya jitne bhi). Output columns: l1..lN."""
    tcfg = cfg["taxonomy"]
    raw = read_csv_text(data_dir / tcfg["file"])
    level_cols = tcfg["level_columns"]
    missing = [c for c in level_cols if c not in raw.columns]
    if missing:
        raise KeyError(f"Taxonomy columns {missing} not found. Columns: {list(raw.columns)}")
    tax = raw[level_cols].copy()
    tax.columns = [f"l{i + 1}" for i in range(len(level_cols))]
    for c in tax.columns:
        tax[c] = tax[c].str.strip()
    tax = tax[(tax != "").any(axis=1)].drop_duplicates().reset_index(drop=True)
    return tax


def load_labels(cfg, data_dir):
    """Labels (agar config me hain aur file mili) -- warna None (zero-label mode)."""
    lcfg = cfg.get("labels")
    if not lcfg:
        return None
    path = data_dir / lcfg["file"]
    if not path.exists():
        return None
    raw = read_csv_text(path)
    out = pd.DataFrame({"record_id": raw[lcfg["record_id_col"]].str.strip()})
    for i, c in enumerate(lcfg["level_columns"]):
        out[f"l{i + 1}"] = raw[c].str.strip()
    for key in ("supplier_key", "item_key"):
        col = lcfg.get(f"{key}_col")
        out[key] = raw[col].str.strip() if col and col in raw.columns else ""
    return out


# ---------------------------------------------------------------- all
def load_company(config_path, data_dir_override=None):
    """Poori company load karo -> dict with config, sources, taxonomy, labels."""
    cfg = load_config(config_path)
    data_dir = resolve_data_dir(cfg, data_dir_override)
    sources = {s["name"]: load_source(cfg, s, data_dir) for s in cfg["sources"]}
    return {
        "config": cfg,
        "data_dir": data_dir,
        "sources": sources,
        "taxonomy": load_taxonomy(cfg, data_dir),
        "labels": load_labels(cfg, data_dir),
    }