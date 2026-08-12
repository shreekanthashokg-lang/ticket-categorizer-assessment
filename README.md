# TICKET CATEGORIZER

AUTO EMAIL / TICKET CATEGORIES — AI/ML INTERN ASSESSMENT (FOBES SKILL ITECH PVT LTD)

A MODEL THAT READS SUPPORT TICKET TEXT AND PREDICTS ITS CATEGORY : **BILLLING, TECHNICAL, HR, OR General**. ALSO TAGS priority (Urgent/Normal) and flags low-confidence predictions for HUMAN REVIEW INSTEAD OF AUTO-
AN ASSIGNING TIME.
## FILES

| File | WHAT IT IS |
|---|---|
| `ticket_categorizer.py` | MAIN SCRIPT — RUN THIS DIRECTLY |
| `ticket_categorizer.ipynb` | SAME CODE, split into NOTEBOOK CELLS FOR Colab |
| `full_output.txt` | CONSOLE OUTPUT FROM AN actual run (ACCURACY, REPORT, predictions) |
| `ticket_classifier.pkl` | TRAINED MODEL, SAVED JOBLIB |

## HOW IT WORKS

1. **Clean the text** — LOWERCASE, STRIP URLs, REMOVE PUNCTUATION
2. **Vectorize** — TF-IDF with 1-2 grams, English stopwords removed
3. **Classify** — Multinomial Naive Bayes (works well on small text datasets, fast to train)
4. **Score confidence** — if the model's top prediction is below 60%, the ticket goes to `NEEDS HUMAN REVIEW` instead of getting auto-assigned
5. **Tag priority** — KEYWORD MATCH against words like "urgent", "down", "crashed", "error 500" → `URGENT`, else `NORMAL`

## DATASET

OWN DUMMY DATASET — 80 TICKETS, 20 per CATEGORY, written by hand (not the sample DATA given in the brief).

## RESULTS

- ACCURACY: **75%** on a stratified 20-ticket TEST SPLIT
- FULL CLASSIFICATION report and confusion matrix in `full_output.txt`
- CONFUSION IS MOSTLY Billing ↔ HR, WHICH MAKES SENSE — BOTH HAVE similar request-style phrasing ("need help with...", "requesting...")

## EDGE CASES

EMPTY STRINGS AND `None` values don't crash the script — they get tagged `GENERAL` with a `NEEDS HUMAN REVIEW (empty ticket)` flag.

## RUN IT

```bash
pip install scikit-learn pandas numpy joblib
python ticket_categorizer.py
```

Or open `ticket_categorizer.ipynb` IN GOOGLE COLLAB and run all cells.

## WHAT I'd IMPROVE WITH MORE TIME

- TRY LOGISTIC Regression / LinearSVC alongside Naive Bayes for comparison
- BIGGER DATASET — 80 tickets is enough to prove the pipeline works, not enough for production-level accuracy
- Wrap it in a FastAPI endpoint with logging for a real deployment
