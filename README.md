# Credit Card Fraud Detection: Model Comparison Study

This repository contains a comprehensive data science project focused on identifying fraudulent credit card transactions. The project follows the complete machine learning lifecycle—from exploratory data analysis (EDA) and rigorous preprocessing to the implementation and comparison of multiple classification algorithms.

## Project Overview

Fraud detection is a critical challenge for financial institutions. This project analyzes a dataset of 8,000 transactions to build a predictive model that can distinguish between legitimate and fraudulent activities. We evaluate three different machine learning approaches to determine which is most effective at minimizing financial risk.

##  Data Science Workflow

### 1. Exploratory Data Analysis (EDA)
- **Data Inspection:** Initial look at the schema (8,000 rows, 20 columns) using `.head()`, `.info()`, and `.describe()`.
- **Target Distribution:** Visualized the `Fraud Flag or Label` distribution, finding a nearly balanced dataset (~49.8% fraud).
- **Feature Identification:** Categorized columns into numerical, categorical, and "features to drop" (such as PII like Cardholder Name or high-cardinality IDs).

### 2. Data Preprocessing & Cleaning
- **Handling Missing Values:** Imputed missing entries in the `Previous Transactions` column using the mode.
- **Categorical Encoding:** Applied **One-Hot Encoding** to transform categorical variables (Card Type, Device Info, Transaction Source, etc.) into a machine-readable format.
- **Feature Scaling:** Used `StandardScaler` to normalize the `Transaction Amount`, ensuring that the varied range of dollar values doesn't bias the models.
- **Data Splitting:** Partitioned the data into **Training (70%)** and **Testing (30%)** sets using stratified sampling to maintain class balance.

### 3. Machine Learning Models
We implemented and compared three distinct algorithms:
* **Logistic Regression:** Serves as the baseline linear classifier.
* **Random Forest:** An ensemble tree-based model used to capture non-linear relationships.
* **XGBoost:** A high-performance Gradient Boosting framework optimized for speed and accuracy.

### 4. Evaluation & Comparison
The models were evaluated using a comprehensive suite of metrics:
- **Accuracy:** Overall correctness.
- **Precision:** Ability to avoid flagging legitimate transactions as fraud.
- **Recall (Sensitivity):** Ability to find all fraudulent transactions (critical for this use case).
- **F1-Score:** The harmonic mean of Precision and Recall.
- **Confusion Matrix:** Visualizing the True Positives, True Negatives, False Positives, and False Negatives.

## Key Findings & Results

The final comparison table from the study:

| Metric | Logistic Regression | Random Forest | XGBoost |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 0.4942 | 0.4954 | **0.4967** |
| **Precision** | 0.4927 | 0.4935 | **0.4955** |
| **Recall** | 0.4787 | 0.4444 | **0.5004** |
| **F1-Score** | 0.4856 | 0.4677 | **0.4979** |

**Conclusion:** For fraud detection, **Recall** is the most critical metric because missing a fraudulent transaction (False Negative) is more costly than a false alarm. In this study, **XGBoost** emerged as the most suitable model, achieving the highest Recall (0.5004).

## Technologies Used

- **Python**
- **Pandas/NumPy:** Data manipulation and analysis.
- **Scikit-Learn:** Preprocessing, Logistic Regression, Random Forest, and metrics.
- **XGBoost:** Gradient Boosting implementation.
- **Matplotlib/Seaborn:** Data visualization and feature importance plotting.

---

### Repository Structure
- `FinTech.ipynb`: The main Jupyter Notebook containing all code, visualizations, and analysis.
- `credit_card_fraud.csv`: The raw transaction dataset.
- `README.md`: Project summary and documentation.
