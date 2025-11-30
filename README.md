# Customer Churn Prediction

## Project Overview
This project predicts whether a bank customer will leave the bank (churn) or stay, using machine learning models. It demonstrates a complete ML pipeline including data cleaning, encoding, modeling, and evaluation.

**Why this project is important:**  
- Real-world application used by companies to reduce churn.  
- Shows ability to handle a full ML workflow.  
- Includes data preprocessing, feature encoding, model training, evaluation, and feature importance analysis.  

## Dataset
- Dataset: Bank Customer Churn  
- Source: [Kaggle](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset)  
- Rows: 10,000 customers  
- Columns (12): customer_id, credit_score, country, gender, age, tenure, balance, products_number, credit_card, active_member, estimated_salary, churn

## Tools & Libraries
- Python  
- Pandas, NumPy → Data handling and analysis  
- Matplotlib, Seaborn → Visualization  
- Scikit-learn → Machine learning models, preprocessing, evaluation  
- Imbalanced-learn → Handling imbalanced dataset (SMOTE)  
- Joblib → Saving and loading models  


## Steps to Run the Project
1. **Install Required Libraries**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib
