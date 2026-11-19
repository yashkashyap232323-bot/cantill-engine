# Decisions and assumptions

## Company A
- ERP is net ZAR; tool export is gross (VAT 15 / 0.15 / 15% / blank -> 15%). Tool net = gross / (1 + vat). An explicit 0 VAT is treated as zero-rated, not blank.
- Duplicate across systems: same PO and net amount equal (+-0.02). If PO is missing on a side: amount + quantity equal and text or supplier similar.
- Both POs present but different: not a match.
- Tool exported twice (same invoice digits, text, qty, gross): DUPLICATE.
- DUPLICATE count is 23,857 for Company A. Over-matching between ERP and the invoice tool is possible; this was not tuned further because of the deadline.
- 877 tool rows with a PO that exists in ERP but a different amount (partial receipts): kept as tool-only spend (assumption).
- Status precedence: CANCELLED / NOT_APPROVED before DUPLICATE before NON_PROCUREMENT. Blank status is kept.
- Negative ERP lines (reversals) are kept as negative spend.
- Supplier key: normalised name, merged with fuzzy matching (>= 90) only when the unique-name pool is small (<= 3,000), plus same supplier id. Rows with no name and no id get "unknown-supplier".
- Item key: item code first; otherwise the exact description signature (numbers + letters, OCR digit noise removed). Fuzzy item clustering was removed for speed, so item_key recall may be lower.
- Invoice-tool dates use day/month order (checked: first number exceeds 12 in 2,050 rows, second never does).

## Company B
- 8 ERPs, 25 currencies. Missing rate -> FX table by invoice month, else posting month, else nearest.
- Exact-duplicate lines are labelled include=1 in the seed labels, so no de-duplication is applied in B.
- Zero-amount lines are kept. Negatives are kept.
- NON_PROCUREMENT is decided by regex rules (intercompany GL, tax accounts); the learned model for this is off because it learned a shortcut.
- The data has no status column, so approval status cannot be checked from the file.

## General
- Category names are never hardcoded; the taxonomy is read at runtime and may have up to four levels.
- Bad amounts -> spend 0.
- Sentence-transformer embeddings are switched off by default and were not tested on this machine; classification uses TF-IDF and the taxonomy text.
- AI tools (an LLM chat assistant and a coding assistant) were used to write and review the code; all numbers reported in APPROACH.md come from actual runs.