"""
Auto Email / Ticket Categorizer
Fobes Skill Itech - AI/ML Intern Assessment
Author: Shreekanth A GUTTEDAR
"""

import re
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.exceptions import NotFittedError
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 1. TEXT PREPROCESSING (with error handling)
# ============================================================
def clean_text(text):
    """Clean raw ticket text safely."""
    try:
        if text is None or (isinstance(text, float) and pd.isna(text)):
            return ""
        text = str(text).lower().strip()
        text = re.sub(r'http\S+|www\S+', '', text)      # remove urls
        text = re.sub(r'[^a-z0-9\s]', ' ', text)         # keep only letters/numbers
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception:
        return ""


# ============================================================
# 2. PRIORITY TAGGING (BONUS)
# ============================================================
URGENT_KEYWORDS = [
    "urgent", "asap", "immediately", "critical", "down",
    "not working", "crash", "crashed", "outage", "emergency",
    "blocked", "cannot access", "failed", "error 500"
]

def get_priority(text: str) -> str:
    """Simple keyword-based priority tag."""
    text_lower = str(text).lower()
    for kw in URGENT_KEYWORDS:
        if kw in text_lower:
            return "URGENT"
    return "NORMAL"


# ============================================================
# 3. MODEL CREATION
# ============================================================
def create_model():
    """TF-IDF + Multinomial Naive Bayes (lightweight & perfect for text)."""
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=1,
            sublinear_tf=True
        )),
        ('clf', MultinomialNB(alpha=0.3))
    ])


# ============================================================
# 4. TRAINING + EVALUATION
# ============================================================
def train_and_evaluate(df, text_col='ticket_text', label_col='category'):
    """Train model and print full evaluation metrics."""
    try:
        if df is None or df.empty:
            raise ValueError("Dataset is empty")
        required = [text_col, label_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        df = df.copy()
        df['clean_text'] = df[text_col].apply(clean_text)
        df = df[df['clean_text'].str.len() > 0]

        if df[label_col].nunique() < 2:
            raise ValueError("Need at least 2 categories")

        X = df['clean_text']
        y = df[label_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        model = create_model()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print("=" * 60)
        print(f"ACCURACY: {acc:.4f}")
        print("=" * 60)
        print("\nCLASSIFICATION REPORT:")
        print(classification_report(y_test, y_pred))
        print("\nCONFUSION MATRIX:")
        print(confusion_matrix(y_test, y_pred))
        print("=" * 60)

        return model, acc
    except Exception as e:
        print(f"[ERROR] Training failed → {e}")
        raise


# ============================================================
# 5. PREDICTION WITH CONFIDENCE + HUMAN REVIEW (BONUS)
# ============================================================
def predict_ticket(model, ticket_text, confidence_threshold=0.60):
    """
    Returns:
        category, confidence%, priority, action
    """
    try:
        if model is None:
            raise ValueError("Model not trained")

        cleaned = clean_text(ticket_text)
        priority = get_priority(ticket_text)

        if not cleaned:
            return {
                "category": "GENERAL",
                "confidence": 0.0,
                "priority": priority,
                "action": "NEEDS HUMAN REVIEW (empty ticket)"
            }

        # Get probabilities
        proba = model.predict_proba([cleaned])[0]
        classes = model.classes_
        best_idx = np.argmax(proba)
        confidence = float(proba[best_idx])
        category = str(classes[best_idx]).upper()

        # Human review threshold (BONUS)
        if confidence < confidence_threshold:
            action = "NEEDS HUMAN REVIEW"
        else:
            action = "AUTO-ASSIGN"

        return {
            "category": category,
            "confidence": round(confidence * 100, 2),
            "priority": priority,
            "action": action
        }
    except NotFittedError:
        raise RuntimeError("Model has not been fitted yet")
    except Exception as e:
        print(f"[ERROR] Prediction failed → {e}")
        return {
            "category": "GENERAL",
            "confidence": 0.0,
            "priority": "NORMAL",
            "action": "NEEDS HUMAN REVIEW (error)"
        }


# ============================================================
# 6. SAVE / LOAD
# ============================================================
def save_model(model, path="ticket_classifier.pkl"):
    try:
        joblib.dump(model, path)
        print(f"Model saved → {path}")
    except Exception as e:
        print(f"Save failed: {e}")


def load_model(path="ticket_classifier.pkl"):
    try:
        return joblib.load(path)
    except Exception as e:
        raise RuntimeError(f"Could not load model: {e}")


# ============================================================
# 7. MINI LIVE DEMO (CLI) - BONUS
# ============================================================
def live_demo(model):
    print("\n" + "=" * 60)
    print("LIVE TICKET CATEGORIZER (type 'quit' to exit)")
    print("=" * 60)
    while True:
        try:
            ticket = input("\nEnter ticket text: ").strip()
            if ticket.lower() in ['quit', 'exit', 'q']:
                break
            result = predict_ticket(model, ticket)
            print(f"\n→ Category   : {result['category']}")
            print(f"→ Confidence : {result['confidence']}%")
            print(f"→ Priority   : {result['priority']}")
            print(f"→ Action     : {result['action']}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")


# ============================================================
# 8. OWN DUMMY DATASET
# ============================================================
def build_dummy_dataset():
    """
    Custom dummy dataset (80 tickets, 4 balanced categories):
    BILLING, TECHNICAL, HR, GENERAL
    """
    data = {
        "ticket_text": [
            # ---- BILLING (20) ----
            "Invoice for March has not been received yet",
            "Payment failed while renewing the subscription",
            "When will the billing cycle reset?",
            "I was charged twice for the same order",
            "Need a refund for the cancelled plan",
            "Credit card declined during checkout",
            "Can I get a copy of last month's receipt?",
            "Subscription auto-renewed without my consent",
            "GST number is missing on my invoice",
            "Upgrade plan billing not reflecting correctly",
            "Why is my invoice amount different from the quote?",
            "Requesting proforma invoice for the annual plan",
            "Late payment fee was charged incorrectly",
            "How do I update my payment method?",
            "Discount coupon not applied at checkout",
            "Please cancel my auto-debit for next month",
            "Billing address needs to be corrected",
            "Refund not credited even after 10 days",
            "Downgrade my plan and adjust the billing",
            "Error 500 while processing the payment gateway",

            # ---- TECHNICAL (20) ----
            "App crashes every time I try to login",
            "System is extremely slow after the latest update",
            "Server is down and no one can access the portal",
            "Unable to reset password - getting error",
            "Urgent: production database is not responding",
            "Mobile app freezes on the dashboard screen",
            "API integration is returning error 500",
            "Cannot upload files, upload button not working",
            "Getting a blank screen after login",
            "Two factor authentication code is not arriving",
            "Website is completely down right now",
            "Search feature returns no results even for valid queries",
            "Notifications are not working on the mobile app",
            "Data sync between devices has stopped working",
            "Getting a 404 error on the reports page",
            "App crashed and lost my unsaved work",
            "Dashboard charts are not loading",
            "Login page keeps redirecting in a loop",
            "Export to PDF feature is broken",
            "Critical: outage affecting all users since morning",

            # ---- HR (20) ----
            "How do I apply for maternity leave?",
            "My salary slip is missing for last month",
            "Request for work from home approval",
            "I cannot see my leave balance for this month",
            "When is the next appraisal cycle starting?",
            "How to apply for sick leave",
            "PF withdrawal process is not clear to me",
            "Need help updating my bank details for salary",
            "What is the notice period for resignation?",
            "Reimbursement claim has not been processed",
            "Onboarding documents are pending approval",
            "How many casual leaves do I have left?",
            "Requesting a relieving letter after resignation",
            "Health insurance card has not arrived yet",
            "Query about the annual bonus policy",
            "How do I update my emergency contact details?",
            "Requesting an experience certificate",
            "My attendance is not marked correctly in the system",
            "What is the process for internal job transfer?",
            "Leave application was rejected without a reason",

            # ---- GENERAL (20) ----
            "How do I apply for maternity leave?" if False else "Need help understanding the new tax policy",
            "General question about company holidays",
            "Feature request: dark mode in the app",
            "Can someone explain the new remote work policy?",
            "What are the office timings on Saturdays?",
            "Is there a parking facility at the new office?",
            "Requesting information about the referral program",
            "How can I provide feedback about the product?",
            "What is the process to raise a general complaint?",
            "Looking for the company's CSR initiatives details",
            "Need the contact details of the support team",
            "Is remote work allowed for the entire team?",
            "General inquiry about upcoming company events",
            "Where can I find the employee handbook?",
            "What is the dress code policy for the office?",
            "Requesting details about the wellness program",
            "How do I subscribe to the company newsletter?",
            "General question about the office relocation",
            "Is there a cafeteria available in the new building?",
            "Looking for details on the annual company retreat",
        ],
        "category": (
            ["BILLING"] * 20 +
            ["TECHNICAL"] * 20 +
            ["HR"] * 20 +
            ["GENERAL"] * 20
        )
    }
    return pd.DataFrame(data)


# ============================================================
# 9. MAIN EXECUTION
# ============================================================
if __name__ == "__main__":

    # -------------------------------------------------------
    # OWN DUMMY DATASET (80 tickets, 4 categories)
    # -------------------------------------------------------
    df = build_dummy_dataset()
    print(f"Dataset loaded: {len(df)} tickets across {df['category'].nunique()} categories")
    print(df['category'].value_counts(), "\n")

    # Train
    model, accuracy = train_and_evaluate(df)

    # Save
    save_model(model)

    # -------------------------------------------------------
    # PREDICT AT LEAST 5 NEW SAMPLE TICKETS (Required)
    # -------------------------------------------------------
    new_tickets = [
        "the app keeps crashing on the checkout screen",
        "when is my next salary increment due",
        "I was billed twice this month, please refund",
        "what time does the office open on weekends",
        "the server has been down since this morning, urgent",
        "how to apply for annual leave next week",
        "",      # edge case: empty string
        None     # edge case: None value
    ]

    print("\n" + "=" * 60)
    print("PREDICTIONS ON NEW UNSEEN TICKETS")
    print("=" * 60)
    for i, ticket in enumerate(new_tickets, 1):
        result = predict_ticket(model, ticket)
        print(f"\n[{i}] Ticket: {ticket!r}")
        print(f"    → {result['category']} | Confidence: {result['confidence']}% | "
              f"Priority: {result['priority']} | Action: {result['action']}")

    # Optional: Start live demo
    # live_demo(model)

    # -------------------------------------------------------
    # REFLECTION NOTE (BONUS - 3-5 lines)
    # -------------------------------------------------------
    print("\n" + "=" * 60)
    print("REFLECTION NOTE")
    print("=" * 60)
    print("""
With more data I would add class balancing techniques and try Logistic Regression
or LinearSVC for comparison, since Naive Bayes assumes feature independence which
rarely holds for real ticket text. I would also introduce a proper stratified
validation set and track confidence calibration over time. For production I would
wrap this in a FastAPI endpoint with structured logging, and route low-confidence
tickets to a human-review dashboard instead of just flagging them in the console.
    """)
