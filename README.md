# 🎫 TICKET CATEGORIZER

## Auto Email / Support Ticket Classification System

### AI/ML INTERN Assessment — FOBES Skill Itech Pvt Ltd

---

## 📌 PROJECT OVERVIEW

**Ticket Categorizer** is an NLP-based Machine Learning system designed to automatically analyze incoming support-ticket or email text and determine the most appropriate support category.

The system classifies each ticket into one of four categories:

* **BILLING**
* **TECHNICAL**
* **HR**
* **GENERAL**

IN ADDITION  to category prediction, the system performs two additional tasks:

1. **Priority Detection** — determines whether the ticket is `URGENT` or `NORMAL`.
2. **Confidence-Based Human Review** — identifies predictions where the ML model is not sufficiently confident and sends them for `NEEDS HUMAN REVIEW` instead of automatically assigning them.

The project demonstrates a complete basic NLP/ML pipeline:

```text
Raw Ticket
    ↓
Text Cleaning
    ↓
TF-IDF Feature Extraction
    ↓
Multinomial Naive Bayes
    ↓
Category Prediction
    ↓
Confidence Evaluation
    ↓
Human Review / Auto Assignment
    ↓
Priority Detection
    ↓
Final Ticket Result
```

---

# 🎯 PROBLEM STATEMENT

Organizations receive support requests through:

* Email
* Contact forms
* Customer-support portals
* Internal employee systems
* Help-desk platforms

Manually reading every incoming ticket and forwarding it to the correct department is repetitive and time-consuming.

For example:

```text
"My card was charged twice for the same subscription."
```

should be routed to:

```text
BILLING
```

while:

```text
"The application is showing error 500 and users cannot log in."
```

should be routed to:

```text
TECHNICAL
```

The purpose of this project is to automate the **first-level ticket triage process** using Natural Language Processing and Machine Learning.

---

# 🎯 PROJECT OBJECTIVES

The project has the following objectives:

### PRIMARY OBJECTIVES 

* Read and process support-ticket text.
* Automatically classify tickets into four categories.
* Generate a confidence score for every prediction.
* Prevent uncertain predictions from being automatically assigned.
* Identify potentially urgent tickets.
* Handle missing and empty ticket text safely.
* Evaluate model performance using standard ML metrics.
* Save the trained model for future predictions.

### SECONDARY OBJECTIVES 

* Demonstrate practical NLP preprocessing.
* Demonstrate TF-IDF feature engineering.
* Implement a multiclass classification model.
* Generate a classification report.
* Generate a confusion matrix.
* Create a reusable trained model.
* Provide a foundation for future API deployment.

---

# 🏷️ SUPPORTED TICKET CATEGORIES

## 1. 💳 BILLING

The **Billing** category contains tickets related to financial and subscription-related issues.

Typical examples include:

* Payment problems
* Duplicate charges
* Refund requests
* Subscription issues
* Invoice problems
* Incorrect charges
* Payment failures
* Billing information

Example:

```text
"I was charged twice for my monthly subscription."
```

Expected result:

```text
Category: BILLING
```

---

## 2. 🛠️ TECHNICAL

The **Technical** category contains tickets related to software, systems, applications, servers, and technical failures.

Examples include:

* Application errors
* Server failures
* Login problems
* API errors
* Website failures
* System crashes
* Database problems
* Connectivity issues
* Error messages

Example:

```text
"The website is showing error 500 whenever I try to log in."
```

EXPECTED RESULT :

```text
Category: TECHNICAL
```

---

## 3. 👥 HR

The **HR** category contains employee-related requests.

EXAMPLES INCLUDES :

* Leave requests
* Attendance issues
* Payroll-related questions
* Employee records
* Benefits
* Onboarding
* Employee information
* Internal HR requests

Example:

```text
"I need help updating my employee information."
```

Expected result:

```text
Category: HR
```

---

## 4. 📩 GENERAL

The **General** category is used for tickets that do not clearly belong to Billing, Technical, or HR.

Examples include:

* General questions
* Product information requests
* Feedback
* Miscellaneous inquiries
* Service information

Example:

```text
"Can you provide more information about your services?"
```

Expected result:

```text
Category: GENERAL
```

---

# 📊 DATASET

The project uses a **custom synthetic dataset containing 80 manually written support tickets**.

The dataset was created specifically for this assessment and is **not the sample dataset provided in the project brief**.

Each category contains exactly 20 tickets.

| Category  | Tickets |
| --------- | ------: |
| BILLING   |      20 |
| TECHNICAL |      20 |
| HR        |      20 |
| GENERAL   |      20 |
| **TOTAL** |  **80** |

The balanced distribution helps prevent one category from dominating the training process.

---

# 🧪 DATASET CREATION

The 80 tickets were manually written to represent realistic support requests.

The examples were designed to contain category-specific vocabulary and common customer-support phrasing.

Examples of relevant terminology include:

### BILLING VOCABULARY

```text
payment
invoice
refund
charged
subscription
transaction
price
billing
```

### Technical vocabulary

```text
server
error
crash
login
API
database
website
application
```

### HR vocabulary

```text
employee
leave
payroll
attendance
salary
HR
joining
benefits
```

### GENERAL VOCABULARY 

```text
information
service
question
feedback
details
help
request
```

Because the dataset is synthetic and relatively small, the model's performance should be interpreted as a demonstration of the pipeline rather than as evidence of production-level performance.

---

# 📂 PROJECT FILES

| File                       | Description                     |
| -------------------------- | ------------------------------- |
| `ticket_categorizer.py`    | Main Python implementation      |
| `ticket_categorizer.ipynb` | Jupyter/Google Colab version    |
| `full_output.txt`          | Output from an actual model run |
| `ticket_classifier.pkl`    | Saved trained model             |
| `README.md`                | Project documentation           |

---

# 🐍 `ticket_categorizer.py`

This is the main executable file.

It contains the complete pipeline for:

```text
Data
 ↓
Preprocessing
 ↓
Vectorization
 ↓
Training
 ↓
Evaluation
 ↓
Prediction
 ↓
Confidence Checking
 ↓
Priority Detection
```

It can be executed directly from the command line.

---

# 📓 `ticket_categorizer.ipynb`

The notebook contains the same core implementation as the Python script but separates the workflow into individual cells.

It is useful for:

* Demonstrating the ML workflow
* Experimentation
* Debugging
* Visualization
* Google Colab execution
* Explaining the project during an interview

---

# 📄 `full_output.txt`

This file contains console output from an actual run.

It includes information such as:

* Model accuracy
* Classification report
* Confusion matrix
* Example predictions
* Confidence scores
* Priority results
* Edge-case results

This provides evidence that the project was actually executed.

---

# 💾 `ticket_classifier.pkl`

The trained model is serialized using **Joblib**.

Instead of retraining the model every time, the saved model can later be loaded and used for inference.

Conceptually:

```text
Training
   ↓
Trained Pipeline
   ↓
ticket_classifier.pkl
   ↓
Load Model
   ↓
Predict New Ticket
```

---

# ⚙️ HOW THE SYSTEM WORKS

The system consists of five major stages.

---

# 1️⃣ TEXT PREPROCESSING

Raw support tickets often contain:

* Uppercase letters
* URLs
* Punctuation
* Unnecessary formatting
* Inconsistent capitalization

The project performs basic text cleaning.

### Operations

1. Convert text to lowercase.
2. Remove URLs.
3. Remove punctuation.
4. Normalize the resulting text.

Example:

### Raw Input

```text
URGENT!!! My server is DOWN.
Please check https://example.com
```

### Cleaned Text

```text
urgent my server is down please check
```

This gives the model a more consistent textual representation.

---

# 2️⃣ TF-IDF FEATURE EXTRACTION

Machine-learning algorithms cannot directly process sentences.

Therefore, the cleaned ticket text is transformed into numerical features using:

**TF-IDF — Term Frequency-Inverse Document Frequency**

The implementation uses:

* Unigrams
* Bigrams
* English stopword removal

---

## 🔹 Unigrams

Individual words are treated as features.

Example:

```text
payment failed
```

becomes features such as:

```text
payment
failed
```

---

## 🔹 Bigrams

Pairs of consecutive words are also considered.

Example:

```text
payment failed
```

can generate:

```text
payment failed
```

Similarly:

```text
error 500
server down
payment issue
```

can become meaningful features.

Using 1–2 grams helps the model capture both individual keywords and short phrases.

---

# 3️⃣ CLASSIFICATION MODEL

The project uses:

## Multinomial Naive Bayes

Multinomial Naive Bayes is a common baseline algorithm for text classification.

It is particularly useful for this type of project because it is:

* Fast
* Lightweight
* Simple to implement
* Suitable for sparse text features
* Effective for small text datasets
* Easy to train and deploy

The model learns the relationship between TF-IDF features and ticket categories.

Conceptually:

```text
Ticket Text
     ↓
Clean Text
     ↓
TF-IDF Vector
     ↓
Naive Bayes
     ↓
Category Probabilities
     ↓
Highest Probability
     ↓
Predicted Category
```

---

# 4️⃣ CONFIDENCE SCORING

The system does not blindly accept every prediction.

After classification, the model calculates the probability associated with each category.

For example:

```text
BILLING      0.78
TECHNICAL    0.12
HR           0.06
GENERAL      0.04
```

The highest probability is:

```text
BILLING = 78%
```

Since:

```text
78% >= 60%
```

the system considers the prediction sufficiently confident for automatic assignment.

---

# 👤 HUMAN REVIEW MECHANISM

The project uses a confidence threshold of:

## 60%

The decision rule is:

```text
IF confidence >= 60%
        ↓
AUTO ASSIGN

IF confidence < 60%
        ↓
NEEDS HUMAN REVIEW
```

For example:

```text
BILLING      48%
TECHNICAL    44%
HR            5%
GENERAL       3%
```

Although Billing is technically the highest prediction, the model is uncertain.

Therefore:

```text
Category: BILLING
Confidence: 48%
Status: NEEDS HUMAN REVIEW
```

This prevents the system from making potentially unreliable automatic routing decisions.

---

# 🧠 WHY HUMAN-IN-THE-LOOP?

A real ticket-routing system should not assume that every ML prediction is correct.

Some tickets can be ambiguous.

For example:

```text
"I need help with my account."
```

This could potentially relate to:

* Billing
* Technical support
* HR
* General support

For such cases, human review provides an additional safety layer.

The approach is:

```text
High Confidence
       ↓
Automation

Low Confidence
       ↓
Human Review
```

This is commonly referred to as a **human-in-the-loop approach**.

---

# 5️⃣ PRIORITY DETECTION

Priority is handled independently from category classification.

The system uses keyword-based rules to identify potentially urgent tickets.

Example urgent keywords include:

```text
urgent
critical
down
crashed
failure
failed
error 500
not working
outage
production down
```

If an urgent indicator is found:

```text
Priority = URGENT
```

Otherwise:

```text
Priority = NORMAL
```

---

# 🚨 EXAMPLE — URGENT TICKET

Input:

```text
"The production server is down and customers cannot access the application."
```

Possible result:

```text
Category: TECHNICAL
Priority: URGENT
```

---

# 🟢 EXAMPLE — NORMAL TICKET

Input:

```text
"Can you send me information about the subscription plans?"
```

Possible result:

```text
Category: BILLING
Priority: NORMAL
```

---

# 🔄 COMPLETE DECISION PIPELINE

The final decision process is:

```text
                 SUPPORT TICKET
                       │
                       ▼
                TEXT CLEANING
                       │
                       ▼
                 TF-IDF FEATURES
                       │
                       ▼
              NAIVE BAYES MODEL
                       │
                       ▼
              CATEGORY PREDICTION
                       │
                       ▼
              CONFIDENCE CHECK
                  /           \
                 /             \
          >= 60%               < 60%
             │                    │
             ▼                    ▼
       AUTO ASSIGN         HUMAN REVIEW
             │
             └──────────┬─────────┘
                        ▼
                PRIORITY CHECK
                  /          \
                 /            \
             URGENT          NORMAL
                 \            /
                  \          /
                   ▼        ▼
                  FINAL RESULT
```

---

# 📤 FINAL PREDICTION FORMAT

A prediction can conceptually contain:

```text
Ticket:
"The server crashed and customers cannot access the website."

Category:
TECHNICAL

Confidence:
92%

Priority:
URGENT

Action:
AUTO-ASSIGNED
```

For an uncertain ticket:

```text
Ticket:
"I need help with my account."

Category:
BILLING

Confidence:
52%

Priority:
NORMAL

Action:
NEEDS HUMAN REVIEW
```

---

# 🧪 TRAIN/TEST SPLIT

The project evaluates the model using a **stratified 20-ticket test split**.

Total dataset:

```text
80 tickets
```

Test set:

```text
20 tickets
```

Training set:

```text
60 tickets
```

The split is stratified so that the four categories remain represented in the evaluation data.

This is important because the dataset contains four equally represented categories.

---

# 📈 MODEL PERFORMANCE

The current implementation achieved:

## **75% Accuracy**

on the 20-ticket test set.

This corresponds approximately to:

```text
15 correctly classified
5 incorrectly classified
```

out of 20 test examples.

However, because the test set is very small, a single prediction can significantly change the reported percentage.

Therefore, the 75% result should be interpreted as a **prototype/assessment result**, not as a production performance guarantee.

---

# 📋 CLASSIFICATION REPORT

The project generates a classification report containing:

* Precision
* Recall
* F1-score
* Support

These metrics provide a more detailed understanding of performance than accuracy alone.

### Precision

Measures how many predictions for a category were actually correct.

### Recall

Measures how many actual tickets belonging to a category were correctly identified.

### F1-score

Combines precision and recall into one metric.

### Support

Shows the number of test examples belonging to each class.

---

# 🔀 CONFUSION MATRIX

The project also produces a confusion matrix.

The main observed confusion is between:

```text
BILLING ↔ HR
```

This is understandable because both categories can contain similar request-oriented language.

Examples:

```text
"need help with..."
"requesting..."
"please assist..."
"I need assistance..."
```

A short ticket containing generic language may not provide enough category-specific information.

For example:

```text
"I need help with my request."
```

contains very little information that distinguishes Billing from HR.

---

# ⚠️ EDGE CASE HANDLING

The system also handles invalid or incomplete inputs.

## Empty String

Input:

```python
""
```

Result:

```text
Category: GENERAL
Status: NEEDS HUMAN REVIEW
Reason: empty ticket
```

## None Value

Input:

```python
None
```

The system safely handles the value instead of crashing.

This is important because real systems may receive:

* Empty emails
* Missing ticket descriptions
* Null values
* Incomplete forms
* Invalid user submissions

---

# 🛡️ SAFE AUTOMATION DESIGN

The project follows a simple safety principle:

> **Do not automatically assign a ticket when the model is uncertain.**

Instead:

```text
Confidence >= 60%
        ↓
Automatic assignment allowed

Confidence < 60%
        ↓
Human review required
```

This reduces the risk of blindly routing ambiguous tickets.

---

# 🧰 TECHNOLOGIES USED

| Technology              | Purpose                   |
| ----------------------- | ------------------------- |
| Python                  | Core programming language |
| Pandas                  | Data manipulation         |
| NumPy                   | Numerical operations      |
| Scikit-learn            | ML/NLP implementation     |
| TF-IDF                  | Text feature extraction   |
| Multinomial Naive Bayes | Classification            |
| Joblib                  | Model serialization       |
| Jupyter Notebook        | Experimentation           |
| Google Colab            | Cloud notebook execution  |

---

# 📦 INSTALLATION

Install all dependencies using:

```bash
pip install scikit-learn pandas numpy joblib
```

Or:

```bash
pip install -r requirements.txt
```

---

# ▶️ RUNNING THE PROJECT

## Method 1 — Python Script

Navigate to the project directory:

```bash
cd ticket-categorizer
```

Install dependencies:

```bash
pip install scikit-learn pandas numpy joblib
```

Run:

```bash
python ticket_categorizer.py
```

The script will execute the complete workflow.

---

# 📓 METHOD 2 — GOOGLE COLAB

Open:

```text
ticket_categorizer.ipynb
```

using Google Colab.

Then:

1. Upload/open the notebook.
2. Install required packages if necessary.
3. Run the cells sequentially.
4. Review model evaluation.
5. Review sample predictions.
6. Test edge cases.
7. Save/export the trained model.

---

# 📁 RECOMMENDED PROJECT STRUCTURE

```text
ticket-categorizer/
│
├── ticket_categorizer.py
├── ticket_categorizer.ipynb
├── full_output.txt
├── ticket_classifier.pkl
├── requirements.txt
├── README.md
│
└── data/
    └── tickets.csv
```

If the dataset is currently embedded directly inside the Python script, it can later be moved into:

```text
data/tickets.csv
```

to make the project structure cleaner and more scalable.

---

# 💾 MODEL PERSISTENCE

The trained classifier is stored as:

```text
ticket_classifier.pkl
```

This makes it possible to reuse the trained model later.

The future inference workflow can become:

```text
Load ticket_classifier.pkl
        ↓
Receive new ticket
        ↓
Preprocess ticket
        ↓
Transform using trained TF-IDF
        ↓
Predict category
        ↓
Calculate confidence
        ↓
Check priority
        ↓
Return result
```

---

# ⚠️ CURRENT LIMITATIONS

## 1. Small Dataset

The dataset contains only 80 tickets.

This is sufficient to demonstrate the pipeline but is not sufficient for production deployment.

A real system would require substantially more examples.

---

## 2. Synthetic Data

The dataset was manually created.

Real support tickets can contain:

* Typos
* Slang
* Abbreviations
* Multiple issues
* Long conversations
* Mixed languages
* Missing context
* Informal writing

Therefore, real-world performance may differ.

---

## 3. Simple Priority Rules

Priority detection currently depends on keywords.

This can create false positives.

For example:

```text
"The server was down last month."
```

contains the word `down`, but the issue may no longer be urgent.

A future ML-based priority classifier could understand context more effectively.

---

## 4. Only One Main Classifier

The current implementation uses Multinomial Naive Bayes.

Other algorithms have not yet been systematically compared.

---

## 5. Fixed Confidence Threshold

The threshold is currently set to:

```text
60%
```

This threshold is a project design choice.

A production implementation should determine the threshold using validation data and business requirements.

---

# 🚀 FUTURE IMPROVEMENTS

## 1. Increase Dataset Size

Expand the dataset from:

```text
80 tickets
```

to at least:

```text
1,000+ tickets
```

and ideally much larger real-world datasets.

---

# 2. Compare Multiple Models

Experiment with:

```text
Multinomial Naive Bayes
Logistic Regression
LinearSVC
```

and compare:

```text
Accuracy
Precision
Recall
F1-score
Training time
Prediction time
```

---

# 3. Hyperparameter Tuning

Potential parameters to optimize include:

```text
TF-IDF:
- ngram_range
- min_df
- max_df
- sublinear_tf

Naive Bayes:
- alpha
```

This may improve generalization.

---

# 4. Cross-Validation

Instead of relying on one 20-ticket test split, use techniques such as:

```text
Stratified K-Fold Cross-Validation
```

This would provide a more reliable estimate of model performance.

---

# 5. Better Priority Model

Replace the keyword-based priority detector with a dedicated classifier:

```text
Ticket Text
     ↓
Priority Model
     ↓
URGENT / NORMAL
```

This would allow the model to understand context rather than relying only on keyword matching.

---

# 6. Confidence Calibration

The 60% threshold could be optimized using validation data.

For example:

```text
50% → More automatic assignments
60% → Current threshold
70% → More human reviews
80% → Very conservative automation
```

The ideal threshold depends on how costly incorrect ticket routing is.

---

# 7. FastAPI Deployment

The trained model could be converted into a REST API.

Example endpoint:

```text
POST /predict
```

Input:

```json
{
  "ticket": "The production server is down"
}
```

Possible response:

```json
{
  "category": "TECHNICAL",
  "confidence": 0.91,
  "priority": "URGENT",
  "status": "AUTO-ASSIGNED"
}
```

---

# 8. Logging

A production system should record:

```text
Ticket ID
Timestamp
Predicted category
Confidence
Priority
Human-review status
Final human decision
```

This would make it possible to monitor model performance over time.

---

# 9. Feedback Loop

Human decisions can be collected and used as new training data.

For example:

```text
ML Prediction
      ↓
Human Review
      ↓
Corrected Category
      ↓
Stored as Training Data
      ↓
Future Model Retraining
```

This can create a continuous improvement loop.

---

# 10. Web Dashboard

A future version could include a dashboard displaying:

```text
Total Tickets
        │
        ├── Billing
        ├── Technical
        ├── HR
        └── General

Urgent Tickets
Human Review Queue
Average Confidence
Category Distribution
Model Accuracy
```

Potential technologies:

```text
Streamlit
FastAPI
React
```

---

# 🏭 FUTURE PRODUCTION ARCHITECTURE

A production-oriented implementation could follow:

```text
             CUSTOMER EMAIL / TICKET
                       │
                       ▼
                 API / Ingestion
                       │
                       ▼
               Text Preprocessing
                       │
                       ▼
                NLP Feature Layer
                       │
                       ▼
               Category Classifier
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
       Confidence             Priority
         Scoring              Detection
             │                   │
             └─────────┬─────────┘
                       ▼
                 Decision Engine
                       │
              ┌────────┴────────┐
              ▼                 ▼
        High Confidence     Low Confidence
              │                 │
              ▼                 ▼
       Auto Assignment      Human Review
              │                 │
              └────────┬────────┘
                       ▼
                Ticket Database
                       │
                       ▼
                 Analytics
                       │
                       ▼
              Model Monitoring
                       │
                       ▼
                 Retraining
```

---

# 📌 KEY MACHINE LEARNING CONCEPTS DEMONSTRATED

This project demonstrates practical knowledge of:

* Natural Language Processing
* Text preprocessing
* Feature extraction
* TF-IDF
* Unigrams
* Bigrams
* Stopword removal
* Multiclass classification
* Multinomial Naive Bayes
* Train/test splitting
* Stratified sampling
* Classification accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* Prediction probabilities
* Confidence thresholds
* Human-in-the-loop systems
* Rule-based classification
* Edge-case handling
* Model serialization
* Basic deployment architecture

---

# 💼 BUSINESS VALUE

The proposed system can help organizations reduce the amount of manual work required for first-level ticket triage.

Potential benefits include:

### Faster Routing

Tickets can be automatically directed to the appropriate team.

### Reduced Manual Work

Support staff do not need to manually categorize every incoming ticket.

### Priority Identification

Potentially urgent issues can be highlighted.

### Human Oversight

Uncertain predictions can be reviewed manually.

### Scalability

The same pipeline can process a larger number of tickets after deployment.

---

# 🎓 AI/ML INTERNSHIP LEARNING VALUE

This project demonstrates an end-to-end approach rather than only showing a trained model.

The workflow covers:

```text
Business Problem
       ↓
Dataset Creation
       ↓
Data Preprocessing
       ↓
Feature Engineering
       ↓
Machine Learning
       ↓
Model Evaluation
       ↓
Confidence Analysis
       ↓
Business Rules
       ↓
Human-in-the-Loop
       ↓
Model Serialization
       ↓
Deployment Planning
```

This makes the project suitable for demonstrating foundational **AI/ML + NLP + practical automation** skills during an internship assessment.

---

# 📊 CURRENT PROJECT SUMMARY

| Component            | Current Implementation        |
| -------------------- | ----------------------------- |
| Problem              | Support ticket categorization |
| Dataset              | Custom synthetic dataset      |
| Total tickets        | 80                            |
| Categories           | 4                             |
| Tickets/category     | 20                            |
| Feature extraction   | TF-IDF                        |
| N-grams              | 1–2 grams                     |
| Stopwords            | English stopwords removed     |
| Classifier           | Multinomial Naive Bayes       |
| Test set             | 20 tickets                    |
| Split                | Stratified                    |
| Accuracy             | **75%**                       |
| Confidence threshold | **60%**                       |
| Priority             | Keyword-based                 |
| Priority classes     | URGENT / NORMAL               |
| Low confidence       | Human review                  |
| Empty input          | Handled safely                |
| Model format         | Joblib `.pkl`                 |
| Deployment           | Future FastAPI implementation |

---

# 🏁 CONCLUSION

The **Ticket Categorizer** project demonstrates a complete NLP-based ticket classification pipeline capable of automatically categorizing support requests into **Billing, Technical, HR, and General** categories.

The system combines:

```text
TF-IDF
+
Multinomial Naive Bayes
+
Confidence Scoring
+
Priority Rules
+
Human Review
```

The current prototype achieves **75% accuracy on a stratified 20-ticket test set**.

The most important design feature is that the system does not blindly automate uncertain decisions. Predictions below the **60% confidence threshold** are flagged for human review.

The project is intentionally positioned as a **working AI/ML prototype**, not a production-ready support system. Its next development stages would include a larger real-world dataset, model comparison, cross-validation, improved priority classification, confidence calibration, logging, FastAPI deployment, and a monitoring/feedback pipeline.

The final goal is to evolve the prototype into a complete intelligent ticket-triage platform capable of:

```text
READ → UNDERSTAND → CLASSIFY → PRIORITIZE
→ CONFIDENCE CHECK → AUTO-ASSIGN / HUMAN REVIEW
```

---

## 👨‍💻 PROJECT TYPE

**AI/ML Internship Assessment Project**

### Domain

```text
Natural Language Processing
Machine Learning
Text Classification
Support Automation
Human-in-the-Loop AI
```

### Core Model

```text
TF-IDF + Multinomial Naive Bayes
```

### Current Result

```text
75% Test Accuracy
```

### Safety Mechanism

```text
< 60% Confidence → NEEDS HUMAN REVIEW
```
