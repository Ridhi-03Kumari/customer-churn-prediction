import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
print("✅ Dataset loaded:", df.shape)
print(df.head(3))

# ─────────────────────────────────────────────
# 2. CLEAN DATA
# ─────────────────────────────────────────────
# Fix TotalCharges — it has spaces instead of NaN
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

# Drop customerID — not useful
df.drop(columns=["customerID"], inplace=True)

# Encode target column
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

print("\n✅ After cleaning:", df.shape)
print("Churn distribution:\n", df["Churn"].value_counts())

# ─────────────────────────────────────────────
# 3. ENCODE CATEGORICAL COLUMNS
# ─────────────────────────────────────────────
le = LabelEncoder()
cat_cols = df.select_dtypes(include="object").columns

for col in cat_cols:
    df[col] = le.fit_transform(df[col])

print("\n✅ Encoding done. Sample:\n", df.head(2))

# ─────────────────────────────────────────────
# 4. SPLIT FEATURES AND TARGET
# ─────────────────────────────────────────────
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"\n✅ Train: {X_train.shape}, Test: {X_test.shape}")

# ─────────────────────────────────────────────
# 5. TRAIN MODELS
# ─────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = {"model": model, "preds": preds, "accuracy": acc}
    print(f"\n{'='*40}")
    print(f"📊 {name} — Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))

# ─────────────────────────────────────────────
# 6. VISUALIZATIONS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Customer Churn Prediction — Model Analysis", fontsize=14)

# Plot 1: Churn Distribution
churn_counts = pd.Series(y).value_counts()
axes[0].bar(["No Churn", "Churn"], churn_counts.values, color=["steelblue", "tomato"])
axes[0].set_title("Churn Distribution")
axes[0].set_ylabel("Count")
for i, v in enumerate(churn_counts.values):
    axes[0].text(i, v + 30, str(v), ha="center", fontweight="bold")

# Plot 2: Confusion Matrix (Random Forest)
rf_preds = results["Random Forest"]["preds"]
cm = confusion_matrix(y_test, rf_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
disp.plot(ax=axes[1], colorbar=False, cmap="Blues")
axes[1].set_title("Confusion Matrix — Random Forest")

# Plot 3: Feature Importance (Random Forest)
rf_model = results["Random Forest"]["model"]
feat_importance = pd.Series(
    rf_model.feature_importances_,
    index=df.drop("Churn", axis=1).columns
).sort_values(ascending=False).head(10)

feat_importance.plot(kind="bar", ax=axes[2], color="steelblue")
axes[2].set_title("Top 10 Feature Importances")
axes[2].set_ylabel("Importance")
axes[2].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("churn_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n✅ Plot saved as churn_analysis.png")

# ─────────────────────────────────────────────
# 7. MODEL COMPARISON SUMMARY
# ─────────────────────────────────────────────
print("\n" + "="*40)
print("📈 MODEL ACCURACY COMPARISON")
print("="*40)
for name, res in results.items():
    print(f"  {name}: {res['accuracy']*100:.2f}%")

best = max(results, key=lambda k: results[k]["accuracy"])
print(f"\n🏆 Best Model: {best} ({results[best]['accuracy']*100:.2f}%)")