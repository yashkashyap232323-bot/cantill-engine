# Approach

## Pipeline (run.py, one command per company)

python run.py --config configs/company_a.json --name YOURNAME
python run.py --config configs/company_b.json --name YOURNAME

Every company is described by a JSON config plus a taxonomy file. No category name is hardcoded in Python.
A new company needs a new config and taxonomy only.

Stages, in order:
1. Ingest (src/ingest.py): read every source file named in the config, map columns to a standard schema, parse amounts and dates, convert to the reporting currency, remove VAT where amounts are gross.
2. Status (src/exclusions.py): CANCELLED and NOT_APPROVED from the status field, when the company has one.
3. System matching (src/matching.py): find the same purchase in two systems (Company A: ERP vs invoice tool) and find invoices exported more than once. The ERP record is kept, the other record is marked DUPLICATE.
4. Exclusions (src/exclusions.py): NON_PROCUREMENT by configurable rules (for example intercompany and tax accounts); precedence CANCELLED / NOT_APPROVED, then DUPLICATE, then NON_PROCUREMENT.
5. Entities (src/entities.py): one supplier_key per real supplier and one item_key per real item.
   - Supplier: normalised name (legal suffixes removed), fuzzy merge (rapidfuzz, cutoff 90) when the unique-name pool is small (<= 3,000), plus the same supplier id. Matched duplicate rows join the supplier of their ERP line. Rows with no name and no id get "unknown-supplier".
   - Item: item code first; otherwise an exact description signature (numbers and letters, OCR digit noise removed). Fuzzy item clustering was removed for speed.
6. Classification (src/classify.py): every included row is assigned a path in the company taxonomy, with a confidence between 0 and 1.
   - Supervised mode when labels are available for the company.
   - Zero-label mode when there are no labels. The taxonomy file is read at runtime; the number of levels comes from the config (up to four).
   - Text features are built from the row description, supplier and account fields; embeddings are switched off by default and were not tested.
7. Output (run.py): exactly one row per input record, columns record_id, include, exclusion_reason, supplier_key, item_key, spend, l1, l2, l3, l4, confidence.
   Excluded rows have include=0, a reason code and spend=0. Included rows have include=1 and a blank reason.

The output file name is YOURNAME__SUFFIX.csv, where the name comes from --name and the suffix from the config or --suffix (A, B, A2, B2, C).

## Output checks (Company A, submitted file)
- 65,549 rows, equal to the number of input records; record_id is unique.
- 41,184 included; excluded: 23,857 DUPLICATE, 315 NOT_APPROVED, 188 CANCELLED, 5 NON_PROCUREMENT.
- Excluded rows with non-zero spend: 0. Included rows with an exclusion reason: 0. Included rows with a blank l1: 0.

## Measured results (from actual runs)

Runtime (laptop, Windows, Python 3.14, no GPU, no LLM API calls):
- Company A (65,549 rows): 97 s end to end (supervised run); 54 s for the zero-label run.
- Company B (41,028 rows): 15 s end to end.
- LLM API cost per 1,000 lines: 0 (no LLM API is used).

Zero-label evaluation, Company B (seed labels hidden from the model, scored against seed_labels.csv, included rows only, n = 38,518):
- L1 accuracy: 0.558 by count, 0.597 by spend.
- L2 accuracy: 0.541 by count, 0.594 by spend.
- L3 accuracy: 0.375 by count, 0.490 by spend.
- Caveat: seed labels are known to be imperfect, so these numbers are measured against noisy references.

Zero-label evaluation, Company A (labels hidden from the model, scored against labels_A.csv, included rows only, n = 41,184):
- L1 accuracy: 0.437 by count, 0.605 by spend.
- L2 accuracy: 0.200 by count, 0.333 by spend.
- L3 accuracy: 0.137 by count, 0.239 by spend.
- supplier_key pairwise precision / recall: 0.729 / 0.968 (some distinct suppliers are over-merged).
- item_key pairwise precision / recall: 0.810 / 0.549 (item recall is lower because fuzzy item clustering was removed for speed).

The submitted gunnu__A.csv was produced in supervised mode (trained on labels_A.csv).
A held-out supervised accuracy was NOT measured; no figure is claimed for it.

## Limits and honest notes
- Company A duplicate detection removes 23,857 tool records; over-matching is possible and was not tuned further.
- Zero-label accuracy on Company A is weak at the deeper levels (L2 0.200, L3 0.137), so a new company with no labels should be expected to score low at the lower taxonomy levels.
- Item de-duplication is exact-signature based, so item_key recall is lower than a fuzzy approach would give.
- Company B exact-duplicate lines are labelled include=1 in the seed labels, so no de-duplication is applied there.

## AI usage
An LLM chat assistant and a coding assistant were used to write and review the code.
The tool itself does not call any LLM API and needs no API key.s