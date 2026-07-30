# TICKET Categorizer

Auto Email / Ticket Categorizer — AI/ML Intern Assessment (Fobes Skill Itech Pvt Ltd)

A model that reads support ticket text and predicts its category: **Billing, Technical, HR, or General**. Also tags priority (Urgent/Normal) and flags low-confidence predictions for human review instead of auto-assigning them.

## FILES

| File | What it is |
|---|---|
| `ticket_categorizer.py` | Main script — run this directly |
| `ticket_categorizer.ipynb` | Same code, split into notebook cells for Colab |
| `full_output.txt` | Console output from an actual run (accuracy, report, predictions) |
| `ticket_classifier.pkl` | Trained model, saved with joblib |

## HOW IT WORKS

1. **Clean the text** — lowercase, strip URLs, remove punctuation
2. **Vectorize** — TF-IDF with 1-2 grams, English stopwords removed
3. **Classify** — Multinomial Naive Bayes (works well on small text datasets, fast to train)
4. **Score confidence** — if the model's top prediction is below 60%, the ticket goes to `NEEDS HUMAN REVIEW` instead of getting auto-assigned
5. **Tag priority** — keyword match against words like "urgent", "down", "crashed", "error 500" → `URGENT`, else `NORMAL`

## Dataset

Own dummy dataset — 80 tickets, 20 per category, written by hand (not the sample data given in the brief).

## RESULTS

- Accuracy: **75%** on a stratified 20-ticket test split
- Full classification report and confusion matrix in `full_output.txt`
- Confusion is mostly Billing ↔ HR, which makes sense — both have similar request-style phrasing ("need help with...", "requesting...")

## EDGE CASES

Empty strings and `None` values don't crash the script — they get tagged `GENERAL` with a `NEEDS HUMAN REVIEW (empty ticket)` flag.

## RUN IT

```bash
pip install scikit-learn pandas numpy joblib
python ticket_categorizer.py
```

Or open `ticket_categorizer.ipynb` in Google Colab and run all cells.

## What I'd improve with more time

- Try Logistic Regression / LinearSVC alongside Naive Bayes for comparison
- Bigger dataset — 80 tickets is enough to prove the pipeline works, not enough for production-level accuracy
- Wrap it in a FastAPI endpoint with logging for a real deployment
