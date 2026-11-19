# Approach

**Design:** ingest (config-driven column mapping, number/date/FX cleaning) -> status exclusions -> cross-system matching -> exclusions (NON_PROCUREMENT rules) -> supplier/item keys -> classification -> CSV.

**Matching/dedup:** PO + net amount, fallback amount + quantity + text/supplier similarity. Suppliers: fuzzy names + ids with union-find. Items: item code, else text clusters.

**Classification:** with labels: TF-IDF (word + char) + supplier + site/account columns -> LinearSVC. Unseen tree, no labels: tree-path text similarity (optional multilingual embeddings) -> confident pseudo-labels -> self-trained LinearSVC on the data's own vocabulary -> smoothing by item and supplier key.

**Accuracy on A (labels hidden):** fill from `python run.py --config configs/company_a.json --name x --hide-labels`.

**Runtime:** a few minutes for ~65k lines (limit 60 min for 25k).

**AI/LLM:** Claude and Codex used for development. No LLM API inside the tool, cost $0.