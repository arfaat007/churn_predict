# AI Customer Churn Prediction Platform

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-green)
![Plotly](https://img.shields.io/badge/Plotly-Visualization-purple)
![Status](https://img.shields.io/badge/Status-Deployed-success)

## Live Demo

[Click here to view the live app](https://aichurnpredict.streamlit.app/)

---

## Project Overview

**AI Customer Churn Prediction Platform** is a machine learning web application that predicts whether a customer is likely to churn based on customer behavior, service usage, contract details, and payment information.

The application is built using **Python, Streamlit, Scikit-learn, XGBoost, Pandas, NumPy, and Plotly**. It provides an interactive interface where users can enter customer details and receive churn prediction results along with churn probability, risk level, and business recommendations.

This project demonstrates an end-to-end machine learning workflow including data preprocessing, model training, model comparison, prediction, visualization, and deployment.

---

## Problem Statement

Customer churn is a major business challenge for subscription-based companies such as telecom, SaaS, banking, insurance, and streaming platforms.

When customers leave a service, companies lose revenue and need to spend more money to acquire new customers. By predicting churn early, businesses can identify high-risk customers and take proactive retention actions.

This project helps businesses answer:

- Which customers are likely to churn?
- What is the probability of churn?
- Which customers are high-risk?
- What retention action should be taken?
- How do different customer attributes affect churn behavior?

---

## Features

- Interactive Streamlit web application
- Customer churn prediction form
- Churn probability score
- Risk level classification
- Business retention recommendation
- Customer analytics dashboard
- Visual data insights using Plotly
- Model comparison between multiple ML algorithms
- Uses Logistic Regression, Random Forest, and XGBoost
- Automatically selects the best-performing model
- Simple demo login flow
- Cloud deployment using Streamlit Cloud

---

## Tech Stack

| Area | Technology |
|---|---|
| Programming Language | Python |
| Web Framework | Streamlit |
| Machine Learning | Scikit-learn, XGBoost |
| Data Handling | Pandas, NumPy |
| Visualization | Plotly |
| Model Type | Classification |
| Deployment | Streamlit Cloud |
| Version Control | Git, GitHub |

---

## Machine Learning Models Used

The project compares multiple classification models:

1. **Logistic Regression**
2. **Random Forest Classifier**
3. **XGBoost Classifier**

The best model is selected based on the **F1 Score**, because churn prediction is a classification problem where both false positives and false negatives matter.

---

## Model Workflow

```text
Customer Data
     ↓
Data Cleaning and Preprocessing
     ↓
Feature Encoding
     ↓
Train-Test Split
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Best Model Selection
     ↓
Churn Prediction
     ↓
Business Recommendation
```

---

## Application Workflow

```text
User opens the Streamlit app
        ↓
User enters customer details
        ↓
The trained model predicts churn probability
        ↓
App displays churn result and risk level
        ↓
App provides business recommendation
        ↓
User can explore customer analytics dashboard
```

---

## Input Features

The prediction system considers customer-related features such as:

- Customer tenure
- Monthly charges
- Total charges
- Contract type
- Payment method
- Internet service
- Online security
- Tech support
- Senior citizen status
- Partner status
- Dependents status
- Paperless billing
- Streaming services
- Customer service usage behavior

---

## Output

The application provides:

- Churn Prediction: **Churn / Not Churn**
- Churn Probability Score
- Risk Level: **Low Risk / Medium Risk / High Risk**
- Customer Retention Recommendation
- Dashboard-based customer insights

Example output:

```text
Prediction: Churn
Churn Probability: 78%
Risk Level: High Risk
Recommendation: Offer retention discount or priority customer support.
```

---

## Screenshots


### Home / Login Page

![Home Page](screenshots/home_page.png)

### Churn Prediction Page

![Prediction Page](screenshots/prediction_page.png)

### Analytics Dashboard

![Analytics Dashboard](screenshots/analytics_dashboard.png)

---

## Folder Structure

```text
churn_predict/
│
├── churn_app.py
├── requirements.txt
├── README.md
└── screenshots/
    ├── home_page.png
    ├── prediction_page.png
    └── analytics_dashboard.png
```

---

## How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/arfaat007/churn_predict.git
```

### 2. Move into the Project Folder

```bash
cd churn_predict
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

For macOS/Linux:

```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit App

```bash
streamlit run churn_app.py

---

## Deployment

The project is deployed using **Streamlit Cloud**.

Live App:

```text
https://aichurnpredict.streamlit.app/
```

---

## Business Use Case

This application can be useful for companies that depend on recurring customers, such as:

- Telecom companies
- SaaS businesses
- Banking and financial services
- Insurance companies
- OTT and streaming platforms
- Subscription-based products

By identifying customers who are likely to churn, businesses can:

- Reduce customer loss
- Improve customer retention
- Offer personalized discounts
- Prioritize support for high-risk customers
- Increase customer lifetime value
- Improve revenue stability

---

## Key Business Insights

This project helps analyze:

- How contract type affects churn
- Whether high monthly charges increase churn risk
- How tenure impacts customer loyalty
- Whether customers with tech support churn less
- Which payment methods are linked with higher churn
- How service combinations influence customer retention

---

## Future Improvements

- Use a real-world Telco Customer Churn dataset
- Add saved model file using Pickle or Joblib
- Add batch prediction using CSV upload
- Add SHAP explainability for feature impact
- Add downloadable churn analysis report
- Add customer segmentation
- Add database integration
- Add user authentication
- Add model retraining pipeline
- Improve UI design and responsiveness


## Author

**Mohammed Shoaib Arfaat Nayyer**

- GitHub: [@arfaat007](https://github.com/arfaat007)
- Live App: [AI Customer Churn Prediction Platform](https://aichurnpredict.streamlit.app/)
