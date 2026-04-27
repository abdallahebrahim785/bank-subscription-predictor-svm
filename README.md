# 🏦 Customer Subscription Prediction — Bank Marketing Campaign

> **Alexandria University · Faculty of Computers and Data Science**  
> Data Computation — Spring 2026 | Final Project

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-SVM-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Dataset](https://img.shields.io/badge/Dataset-UCI%20Bank%20Marketing-blue?style=flat)](https://archive.ics.uci.edu/dataset/222/bank+marketing)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)
[![Live Demo](https://bank-subscription-predictor-svm.streamlit.app/)](#)

---

## 📌 Project Overview

This project builds a complete end-to-end **SVM classification pipeline** to predict whether a bank client will subscribe to a term deposit, based on data from direct phone marketing campaigns. The project covers the full ML lifecycle — from exploratory data analysis and data cleaning, through dimensionality reduction, to model training, hyperparameter tuning, evaluation, and a live web deployment.

**Target Question:** *Will this customer subscribe to a term deposit? (Yes / No)*

---

## 🌐 Live Demo

> 🔗 **Deployed App:** [https://bank-subscription-predictor-svm.streamlit.app/]  
> *(Replace this line with your Streamlit Cloud / Hugging Face / Render URL once deployed)*

[![Deploy to Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](#)

| Tab | Description |
|-----|-------------|
| 📊 Exploratory Data Analysis | Interactive charts with sidebar filters |
| 🎯 Subscription Predictor   | Real-time SVM prediction for new customers |

---

## 📁 Project Structure

```
Customer-Subscription-Prediction/
│
├── 📓 eda_model.ipynb              ← Full ML pipeline notebook
├── 🚀 app.py                       ← Streamlit deployment app
├── 📊 bank-additional-full.csv     ← Dataset (UCI, semicolon-separated)
├── 🤖 best_model.pkl               ← Trained & serialized SVM pipeline
├── 🖼️ image.jpg                    ← Sidebar logo
├── 📋 requirements.txt             ← Python dependencies
└── 📖 README.md                    ← This file
```

---

## 📦 Dataset

| Property | Value |
|---|---|
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing) |
| **File** | `bank-additional-full.csv` |
| **Records** | 41,188 |
| **Features** | 20 input features + 1 target |
| **Separator** | Semicolon (`;`) |
| **Domain** | Bank phone marketing campaigns (May 2008 – Nov 2010) |
| **Task** | Binary Classification (subscribed: yes / no) |
| **Class Balance** | ~88.7% No · ~11.3% Yes (imbalanced) |

### Feature Summary

**Numerical (10)**

| Feature | Description |
|---|---|
| `age` | Client age in years |
| `duration` | Last call duration in seconds |
| `campaign` | Number of contacts during this campaign |
| `pdays` | Days since last contact (999 = never contacted) |
| `previous` | Number of contacts before this campaign |
| `emp.var.rate` | Employment variation rate (quarterly) |
| `cons.price.idx` | Consumer price index (monthly) |
| `cons.conf.idx` | Consumer confidence index (monthly) |
| `euribor3m` | Euribor 3-month rate (daily) |
| `nr.employed` | Number of employees (quarterly) |

**Categorical (10)**

| Feature | Description |
|---|---|
| `job` | Type of job (admin, blue-collar, technician, ...) |
| `marital` | Marital status (married, single, divorced) |
| `education` | Education level (basic.4y → university.degree) |
| `default` | Has credit in default? (yes/no/unknown) |
| `housing` | Has housing loan? (yes/no/unknown) |
| `loan` | Has personal loan? (yes/no/unknown) |
| `contact` | Contact communication type (cellular/telephone) |
| `month` | Last contact month of the year |
| `day_of_week` | Last contact day of the week |
| `poutcome` | Outcome of the previous marketing campaign |

---

## 🔬 ML Pipeline

The full pipeline was implemented using `sklearn.pipeline.Pipeline` to prevent any data leakage.

```
Raw Data
│
▼
① Train / Test Split (80% / 20%) — FIRST STEP, before any preprocessing
│
▼
② Data Cleaning (fit on train only)
├── Replace "unknown" with column mode
└── IQR-based outlier capping
│
▼
③ Feature Engineering
├── OneHotEncoder  → Categorical columns
└── StandardScaler → Numerical columns
│                    (47 total features after encoding)
▼
④ Dimensionality Reduction
└── PCA (n_components=0.95) → 47 features → 34 components (95.2% variance retained)
│
▼
⑤ SVM Classifier
└── SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced', probability=True)
│
▼
⑥ Prediction → Yes / No
```

> ⚠️ **No Data Leakage:** All transformers (scaler, encoder, PCA) were fitted **exclusively on the training set** and only `transform()` was applied to the test set.

---

## ⚙️ Hyperparameter Tuning

`GridSearchCV` with 3-fold cross-validation was used, scoring on **F1** (preferred over accuracy for imbalanced classes).

| Parameter | Values Tried | Best |
|---|---|---|
| `kernel` | rbf, linear | **rbf** |
| `C` | 0.1, 1, 10 | **10** |
| `gamma` | scale, auto | **scale** |
| CV folds | — | 3 |
| Total fits | — | 36 |
| Scoring | — | F1-score |

---

## 📊 Model Evaluation

| Metric | Score |
|---|---|
| **Accuracy** | **98.00%** |
| **Precision** | **96.00%** |
| **Recall** ⭐ | **92.00%** |
| **F1-Score** | **94.00%** |

### Classification Report

|  | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **No (0)** | 0.99 | 0.99 | 0.99 | 2,578 |
| **Yes (1)** | **0.96** | **0.92** | **0.94** | 422 |
| **Macro Avg** | 0.97 | 0.96 | 0.97 | 3,000 |
| **Weighted Avg** | 0.98 | 0.98 | 0.98 | 3,000 |

### Why Recall is the Key Metric

> In a bank marketing campaign, **missing a real subscriber (false negative)** is far more costly than calling someone who won't subscribe (false positive). Our model achieves **92.00% recall** — catching 9 out of 10 actual subscribers. The high precision (96%) ensures minimal wasted calls, making this model exceptionally balanced and production-ready.

---

## 📉 Dimensionality Reduction (PCA)

| | Value |
|---|---|
| Features before PCA | 47 (after one-hot encoding) |
| Features after PCA | 34 components |
| Variance retained | 95.2% |
| Threshold used | `n_components=0.95` |

---

## 🖥️ Streamlit App Features

### 📊 Tab 1 — Exploratory Data Analysis
- **KPI cards**: Total contacts, subscribers, conversion rate, avg call duration
- **Subscription Distribution** — Donut chart
- **Contact Method Analysis** — Horizontal bar chart
- **Age Distribution by Subscription** — Overlapping histograms
- **Subscription Rate by Job** — Horizontal bar chart
- **Call Duration Distribution** — Overlapping histograms
- **Monthly Subscription Trends** — Spline line chart
- **Subscription Rate by Education & Marital Status** — Bar charts
- **Feature Correlation Matrix** — Interactive heatmap
- **Key Business Insights** — Summary panel

### 🎯 Tab 2 — Subscription Predictor
- Full 20-feature input form (Customer Profile + Campaign Info + Economic Indicators)
- Real-time SVM prediction via loaded `best_model.pkl`
- **✅ Will Subscribe** / **❌ Will Not Subscribe** result banner
- Actionable business recommendations per prediction result
- Model information expander

### 🔎 Sidebar Filters (EDA Tab)
- Filter by **Job Type** (multiselect)
- Filter by **Campaign Month** (multiselect)
- Dynamic KPI cards update based on active filters

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/customer-subscription-prediction.git
cd customer-subscription-prediction
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Required Files

Make sure these files are in the project root:

```
bank-additional-full.csv    ← Download from UCI (link below)
best_model.pkl              ← Generated by running the notebook
image.jpg                   ← Your sidebar logo image
```

Dataset download: https://archive.ics.uci.edu/dataset/222/bank+marketing  
→ Extract the zip → use `bank-additional-full.csv`

### 5. Run the Notebook (to generate best_model.pkl)

```bash
jupyter notebook eda_model.ipynb
```

Run all cells top to bottom. The final cell saves `best_model.pkl`.

### 6. Launch the App

```bash
streamlit run app.py
```

Open your browser at http://localhost:8501

---

## 📋 Requirements

```
streamlit
pandas
numpy
scikit-learn
plotly
joblib
matplotlib
seaborn
jupyter
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## 🔑 Key Findings from EDA

| Finding | Detail |
|---|---|
| 📞 **Call Duration** | Strongest predictor — longer calls → much higher subscription rate (r = +0.40) |
| ✅ **Previous Success** | Past successful campaign → 66% subscription rate vs 11% average |
| 📱 **Contact Type** | Cellular outperforms telephone (~15% vs ~5% conversion) |
| 📅 **Best Months** | March, September, October, December have highest conversion rates |
| 💼 **Best Jobs** | Students and retired customers are most responsive |
| 📉 **Economic Indicators** | Lower euribor3m and emp.var.rate correlate with more subscriptions |
| ⚠️ **Class Imbalance** | 88.7% No / 11.3% Yes — handled via `class_weight='balanced'` |
| 🧹 **Unknown Values** | Found in 6 columns — imputed with mode (computed on train set only) |

---

## 📄 License

This project is submitted as an academic final project for **Data Computation — Spring 2026** at **Alexandria University**. For educational use only.

---

*Built with ❤️ using Python · Scikit-learn · Streamlit · Plotly*
