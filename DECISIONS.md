# Decisions and assumptions

## Company A
- ERP is net ZAR; tool export is gross (VAT 15/0.15/15%/blank -> 15%). Tool net = gross/(1+vat).
- Duplicate across systems: same PO and net amount equal (+-0.02). If PO missing on a side: amount + quantity equal and text or supplier similar.
- Both POs present but different: not a match.
- Tool exported twice (same invoice digits, text, qty, gross): DUPLICATE.
- 877 tool rows with PO in ERP but different amount (partial receipts): kept as tool-only spend (assumption).
- Status precedence: CANCELLED / NOT_APPROVED before DUPLICATE before NON_PROCUREMENT. Blank status kept.
- Negative ERP lines (reversals) kept as negative spend.
- Supplier key: fuzzy normalised name (>=90) + same supplier id. Item key: item code first, else text clusters.

## Company B
- 8 ERPs, 25 currencies. Missing rate -> FX table by invoice month, else posting month, else nearest.
- Exact-duplicate lines are labelled include=1 in seeds, so no dedup in B.
- Zero-amount lines kept. Negatives kept.
- NON_PROCUREMENT by regex rules (intercompany GL, tax accounts); learned model off (it learned a shortcut).

## General
- Category names never hardcoded; taxonomy read at runtime.
- Bad amounts -> spend 0.