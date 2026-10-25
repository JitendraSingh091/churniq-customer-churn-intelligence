# 📊 ChurnIQ — Customer Churn Prediction & Retention Intelligence

**An end-to-end Machine Learning and Data Science application** that predicts customer churn, analyzes customer behavior, and provides actionable insights to support data-driven customer retention strategies.

Built using **Python, Scikit-learn, Streamlit, Plotly, FastAPI, and SQL**, ChurnIQ transforms customer information into churn probability estimates and retention intelligence.

🌐 **Live Application:** [Launch ChurnIQ](https://churniq-intelligence.streamlit.app/)

💻 **GitHub Repository:** [ChurnIQ — Customer Churn Intelligence](https://github.com/JitendraSingh091/churniq-customer-churn-intelligence)

---

## 🎯 About the Project

**ChurnIQ** is a machine learning-powered customer churn prediction and retention intelligence platform designed to help businesses understand customer behavior and identify customers who may discontinue their services.

Customer churn is a significant challenge for subscription-based businesses, particularly telecommunications companies.

ChurnIQ addresses this challenge by combining **predictive machine learning, exploratory data analysis, interactive visualization, and customer risk assessment** into a unified application.

Users can enter customer information, estimate churn probability, explore customer behavior patterns, and review suggested retention actions.

### 🎯 Project Goals

- Predict customer churn using supervised machine learning.
- Identify potential high-risk customers.
- Analyze factors associated with customer churn.
- Compare classification algorithms.
- Optimize classification thresholds to improve churn detection.
- Visualize customer behavior through interactive dashboards.
- Provide estimated risk levels and retention recommendations.
- Expose machine learning predictions through REST API endpoints.

---

## 🚀 Live Demo

Experience ChurnIQ directly through the deployed Streamlit application.

### 🌐 [Open ChurnIQ — Live Application](https://churniq-intelligence.streamlit.app/)

**Application URL:**

```text
https://churniq-intelligence.streamlit.app/
```

The application includes:

- 📊 Customer Overview Dashboard
- 🤖 Customer Churn Prediction
- 📈 Customer Insights and Visualizations
- 🎯 Churn Probability and Risk Assessment
- 💡 Retention Recommendations

---

## ⚡ Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/JitendraSingh091/churniq-customer-churn-intelligence.git
cd churniq-customer-churn-intelligence
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit Application

```bash
streamlit run app.py
```

Open your browser:

```text
http://localhost:8501
```

### 6. Run the FastAPI Backend (Optional)

```bash
uvicorn src.api:app --reload
```

**API Documentation:**

```text
http://127.0.0.1:8000/docs
```

---

## ✨ Features

- 🤖 **Machine Learning Prediction** – Predict customer churn using a trained Logistic Regression pipeline.
- 📊 **Interactive Dashboard** – Explore customer churn statistics and key business indicators.
- 🎯 **Churn Probability Estimation** – Estimate the probability of a customer leaving.
- ⚠️ **Customer Risk Assessment** – Display illustrative risk-priority categories.
- 📈 **Customer Insights** – Analyze customer behavior using interactive Plotly charts.
- 🧠 **Model Comparison** – Compare Logistic Regression and Random Forest.
- ⚙️ **Threshold Optimization** – Adjust classification sensitivity based on validation results.
- 💡 **Retention Recommendations** – Display suggested retention actions based on customer risk.
- 🗄️ **SQL Analytics** – Analyze churn trends using SQL queries.
- 🌐 **REST API** – Provide model predictions through FastAPI.
- ☁️ **Cloud Deployment** – Streamlit Community Cloud deployment.
- 📱 **Interactive Interface** – Customer inputs and prediction results through a web interface.

---

## 🌐 How It Works

```text
Customer enters information
            ↓
Streamlit application
            ↓
Input validation
            ↓
Feature preprocessing
            ↓
Trained Logistic Regression model
            ↓
Churn probability estimation
            ↓
Classification threshold (0.30)
            ↓
Churn / No Churn prediction
            ↓
Risk assessment
            ↓
Retention recommendations
```

The application uses a saved machine learning pipeline to preprocess customer information and generate churn probability estimates.

The selected **0.30 classification threshold** prioritizes identifying potential churners, accepting more false-positive predictions in exchange for higher recall.

---

## 📊 Dataset

The project uses the **IBM Telco Customer Churn Dataset**, which contains customer demographic information, subscription details, service usage, billing information, and churn outcomes.

### Dataset Overview

| Metric | Value |
|---|---:|
| Total Customers | 7,043 |
| Total Features in Original Dataset | 21 |
| Churned Customers | 1,869 |
| Retained Customers | 5,174 |
| Churn Rate | 26.54% |
| Retention Rate | 73.46% |

### Key Dataset Features

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies
- Contract
- Paperless Billing
- Payment Method
- Monthly Charges
- Total Charges
- Churn

**Target Variable:** `Churn`

---

## 🧠 Machine Learning

ChurnIQ uses supervised classification algorithms to estimate the likelihood of customer churn.

### Algorithms Implemented

- Logistic Regression
- Random Forest Classifier

### Machine Learning Pipeline

```text
Raw Customer Dataset
          ↓
Data Cleaning
          ↓
Exploratory Data Analysis
          ↓
Feature Preprocessing
          ↓
Train / Test Split
          ↓
Model Training
          ↓
Model Evaluation
          ↓
Threshold Optimization
          ↓
Saved ML Pipeline
          ↓
Streamlit + FastAPI
```

### 📈 Model Performance

| Evaluation Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 80.55% | 76.37% |
| ROC-AUC | 84.19% | 82.30% |
| F1-Score at 0.50 Threshold | 60.40% | 58.32% |

**Selected Model:** Logistic Regression

Logistic Regression was selected for deployment based on its evaluation performance.

### 🎯 Classification Threshold Optimization

Instead of relying only on the default threshold of 0.50, ChurnIQ evaluates alternative classification thresholds to improve churn detection.

The threshold was selected using validation data and subsequently evaluated on the held-out test set.

| Metric | Threshold 0.30 | Threshold 0.50 |
|---|---:|---:|
| Precision | 51.93% | 65.72% |
| Recall | 75.40% | 55.88% |
| F1-Score | 61.50% | 60.40% |

**Final Classification Threshold: 0.30**

This threshold improves recall, helping identify more customers who may churn, while increasing the number of false-positive predictions.

---

## 🔍 Key Customer Insights

Exploratory data analysis identified several patterns associated with customer churn.

### 📑 Contract Type vs Churn

| Contract Type | Churn Rate |
|---|---:|
| Month-to-Month | 42.71% |
| One Year | 11.27% |
| Two Year | 2.83% |

### 💳 Payment Method

Customers using **Electronic Check** had a churn rate of approximately **45.29%**.

### 📅 Customer Tenure

- Customers with tenure between 0–12 months: **47.44% churn**
- Customers with tenure between 49–72 months: **9.51% churn**

### 💰 Monthly Charges

- Average monthly charges for churned customers: **$74.44**
- Average monthly charges for retained customers: **$61.27**

### 📡 Internet Service

Month-to-month customers using fiber-optic internet had an observed churn rate of approximately **54.61%**.

These are descriptive associations and should not be interpreted as proof that individual features cause churn.

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| **Programming** | Python 🐍, SQL |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn |
| **ML Algorithms** | Logistic Regression, Random Forest |
| **Model Serialization** | Joblib |
| **Web Application** | Streamlit |
| **Visualization** | Plotly |
| **Backend API** | FastAPI |
| **API Server** | Uvicorn |
| **Database** | SQLite |
| **Notebook Environment** | Jupyter Notebook |
| **Version Control** | Git & GitHub |
| **Deployment** | Streamlit Community Cloud |

---

## 📁 Repository Structure

```text
churniq-customer-churn-intelligence/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── telco_churn_cleaned.csv
│
├── models/
│   └── logistic_regression_pipeline.joblib
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── reports/
│
├── sql/
│   ├── 01_churn_analysis.sql
│   ├── load_data.py
│   └── run_queries.py
│
└── src/
    ├── api.py
    ├── test_saved_model.py
    ├── threshold_analysis.py
    ├── train_model.py
    └── train_random_forest.py
```

---

## 🔄 Application Architecture

```text
┌─────────────────────────┐
│          User           │
│   Customer Information  │
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│    Streamlit Frontend   │
│         app.py          │
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│   Feature Preprocessing │
│    Saved ML Pipeline    │
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│  Logistic Regression    │
│   Churn Probability     │
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│  Threshold Evaluation   │
│         0.30            │
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│    Prediction Result    │
│   Risk + Retention Tips │
└─────────────────────────┘
```

**Additional Backend:** FastAPI provides an independent prediction interface using the saved model.

---

## 🌐 REST API

ChurnIQ includes a FastAPI backend for programmatic access to churn predictions.

### Available Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | API health check |
| POST | `/predict` | Predict customer churn |

### API Documentation

Once the backend is running locally, visit:

```text
http://127.0.0.1:8000/docs
```

The Streamlit deployment and FastAPI backend are separate components. The Streamlit live URL does not automatically host the FastAPI endpoints.

---

## 💡 Retention Intelligence

ChurnIQ provides suggested retention actions based on estimated customer churn risk.

Examples include:

- Reviewing customer satisfaction for potentially high-risk customers.
- Considering personalized retention offers.
- Reviewing service quality concerns.
- Encouraging suitable long-term subscription plans.
- Identifying customers who may benefit from proactive support.

These recommendations are illustrative decision-support suggestions and are not experimentally validated retention interventions.

---

## 📌 Use Cases

ChurnIQ demonstrates techniques applicable to:

- 📡 Telecommunications customer retention
- 💳 Subscription-based services
- 🛒 E-commerce customer analytics
- 📊 Customer behavior analysis
- 🎯 Retention campaign prioritization
- 📈 Business intelligence dashboards
- 🤖 Machine learning education
- 🔍 Predictive analytics portfolios

Models trained on the Telco dataset would require retraining and validation before use in other industries.

---

## 🚀 Future Improvements

- 🧠 SHAP-based model explainability
- 📦 Batch customer churn prediction
- 📊 Customer retention campaign simulation
- 🔄 Automated model retraining
- 📈 Data drift and model monitoring
- ⚙️ Hyperparameter optimization
- 🔐 Authentication and role-based access
- 🗄️ PostgreSQL integration
- ☁️ Independent cloud deployment of FastAPI
- 📉 Business cost-sensitive threshold optimization

---

## 🎓 Learning Outcomes

Through this project, the following concepts were explored:

- End-to-end Data Science workflow
- Data cleaning and preprocessing
- Exploratory Data Analysis
- SQL-based customer analytics
- Supervised Machine Learning
- Logistic Regression
- Random Forest
- Model evaluation and comparison
- Precision, Recall, F1-Score and ROC-AUC
- Classification threshold optimization
- Scikit-learn pipelines
- Model serialization using Joblib
- Interactive Streamlit dashboards
- Plotly data visualization
- REST API development with FastAPI
- Git and GitHub version control
- Machine learning application deployment

---

## ⚠️ Disclaimer

ChurnIQ is an educational and portfolio-oriented machine learning project.

Predicted churn probabilities are statistical estimates, not guarantees of future customer behavior.

The application’s risk-priority categories and retention recommendations are illustrative. They have not been validated as optimal business decisions or causal interventions.

The trained model should be independently validated before production use with real customer data.

---

## 🏷️ Suggested GitHub Topics

```text
customer-churn
churn-prediction
machine-learning
data-science
predictive-analytics
customer-retention
python
streamlit
scikit-learn
logistic-regression
random-forest
fastapi
sql
plotly
classification
business-intelligence
data-analytics
ml-project
```

---

## 👨‍💻 Author

**Jitendra Kumar Singh**

B.Tech — Computer Science & Engineering

Data Science | Machine Learning | AI

🔗 **GitHub:**  
https://github.com/JitendraSingh091

🔗 **LinkedIn:**  
https://www.linkedin.com/in/jitendra-kumar-singh-a1b115323

📧 **Email:**  
[jitendrakumar808410@gmail.com](mailto:jitendrakumar808410@gmail.com)

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.

Feedback, suggestions, and contributions are welcome!

**Made with Python, Machine Learning & Data Science. 🚀**
