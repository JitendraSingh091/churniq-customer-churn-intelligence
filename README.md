\# ChurnIQ — Customer Churn Prediction \& Retention Intelligence



\### Machine Learning | Data Science | Predictive Analytics | Streamlit | FastAPI



ChurnIQ is an end-to-end machine learning project designed to predict customer churn, analyze customer behavior, and support data-driven retention decisions.



\## Features



\- Interactive customer churn prediction dashboard

\- Customer risk assessment using machine learning

\- Exploratory data analysis and customer insights

\- Logistic Regression and Random Forest model comparison

\- Classification threshold optimization

\- FastAPI prediction endpoints

\- SQL-based customer churn analysis

\- Retention recommendations based on estimated risk



\## Tech Stack



\*\*Programming:\*\* Python, SQL



\*\*Data Analysis:\*\* Pandas, NumPy



\*\*Machine Learning:\*\* Scikit-learn, Logistic Regression, Random Forest



\*\*Visualization:\*\* Streamlit, Plotly



\*\*Backend:\*\* FastAPI



\*\*Database:\*\* SQLite



\## Dataset



IBM Telco Customer Churn Dataset



\- Total customers: 7,043

\- Customers who churned: 1,869

\- Customers retained: 5,174

\- Churn rate: 26.54%



\## Machine Learning Performance



The Logistic Regression model was selected for deployment after comparison with Random Forest.



| Metric | Logistic Regression | Random Forest |

|---|---:|---:|

| Accuracy | 80.55% | 76.37% |

| ROC-AUC | 84.19% | 82.30% |

| F1-score at 0.50 threshold | 60.40% | 58.32% |



\### Threshold Optimization



The classification threshold was optimized using validation data, with 0.30 selected to improve churn detection.



| Metric | Threshold 0.30 | Threshold 0.50 |

|---|---:|---:|

| Precision | 51.93% | 65.72% |

| Recall | 75.40% | 55.88% |

| F1-score | 61.50% | 60.40% |



The selected threshold improves recall, helping identify more potential churners while increasing false-positive predictions.



\## Run Locally



\### 1. Clone the repository



```bash

git clone https://github.com/YOUR\_USERNAME/YOUR\_REPOSITORY.git

cd YOUR\_REPOSITORY

```



\### 2. Create a virtual environment



```bash

python -m venv .venv

```



\### 3. Activate the environment



Windows PowerShell:



```powershell

.\\.venv\\Scripts\\Activate.ps1

```



\### 4. Install dependencies



```bash

pip install -r requirements.txt

```



\### 5. Start the Streamlit application



```bash

streamlit run app.py

```



\### 6. Start the FastAPI backend (optional)



```bash

uvicorn src.api:app --reload

```



API documentation will be available at `http://127.0.0.1:8000/docs`.



\## Project Highlights



\- End-to-end data science workflow

\- Data preprocessing and feature engineering

\- Supervised classification model development

\- Model evaluation and comparison

\- Validation-based classification threshold selection

\- Interactive prediction dashboard

\- REST API integration

\- Customer segmentation and retention insights



\## Disclaimer



Churn probabilities are model estimates, not guarantees. Dashboard risk-priority bands are illustrative and have not been validated as optimal business decision thresholds.



\## Future Improvements



\- Model explainability with SHAP

\- Batch customer predictions

\- Customer retention campaign simulation

\- Model monitoring and drift detection

\- Automated model retraining

\- Cloud deployment



\## Author



Jitendra Kumar Singh



GitHub: https://github.com/JitendraSingh091



LinkedIn: https://www.linkedin.com/in/jitendra-kumar-singh-a1b115323



