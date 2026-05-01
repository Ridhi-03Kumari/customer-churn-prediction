#  Customer Churn Prediction Dashboard

An ML-powered interactive dashboard to predict telecom customer churn in real-time.

🔗 **Live Demo:** [Click here to view](https://your-app-name.streamlit.app)

---

##  Features
- Interactive Plotly charts with hover effects
- Churn distribution, contract analysis, tenure vs charges scatter plot
- Box plots and internet service breakdown
- Random Forest vs Logistic Regression comparison
- Radar chart for model metrics (Accuracy, Precision, Recall, F1)
- Live churn prediction with gauge chart
- Glassmorphism dark UI with green/blue theme

## Models Used
- **Random Forest** — 100 estimators, ensemble learning (~80% accuracy)
- **Logistic Regression** — Linear probabilistic classifier (~80% accuracy)

## Tech Stack
- Python, Streamlit, scikit-learn
- Plotly, pandas, NumPy
- Dataset: [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Run Locally
```bash
git clone https://github.com/Ridhi-03Kumari/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure
```
customer-churn-prediction/
├── DATA/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── app.py           # Streamlit dashboard
├── churn_model.py   # ML training script
├── requirements.txt
└── README.md
```
