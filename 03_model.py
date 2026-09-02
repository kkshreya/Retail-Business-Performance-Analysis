"""Build Logistic Regression & Decision Tree classifiers to predict attrition."""
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

df = pd.read_csv("/home/claude/hr_attrition/data/hr_attrition_enriched.csv")

TARGET = "Attrition"
# EducationField dropped from modeling features: it is not a genuine attrition
# driver and its smallest category (Human Resources, n~31) was injecting noisy
# coefficients purely from small-sample imbalance relative to the dummy baseline.
drop_cols = ["EmployeeID", "SalaryBand", "PromotedRecently", "TenureBand", "EducationField"]
model_df = df.drop(columns=drop_cols).copy()

y = (model_df[TARGET] == "Yes").astype(int)
X = model_df.drop(columns=[TARGET])

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

# One-hot encode categoricals (keep it simple + interpretable for SHAP)
X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
feature_names = X_encoded.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = {}

# ---------------- Logistic Regression ----------------
log_reg = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42, C=0.25)
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)
y_proba_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

results["Logistic Regression"] = dict(
    accuracy=accuracy_score(y_test, y_pred_lr),
    precision=precision_score(y_test, y_pred_lr),
    recall=recall_score(y_test, y_pred_lr),
    f1=f1_score(y_test, y_pred_lr),
    roc_auc=roc_auc_score(y_test, y_proba_lr),
    confusion_matrix=confusion_matrix(y_test, y_pred_lr).tolist(),
    report=classification_report(y_test, y_pred_lr, target_names=["No", "Yes"]),
)

# ---------------- Decision Tree ----------------
tree = DecisionTreeClassifier(
    max_depth=5, min_samples_leaf=20, class_weight="balanced", random_state=42
)
tree.fit(X_train, y_train)  # trees don't need scaling
y_pred_dt = tree.predict(X_test)
y_proba_dt = tree.predict_proba(X_test)[:, 1]

results["Decision Tree"] = dict(
    accuracy=accuracy_score(y_test, y_pred_dt),
    precision=precision_score(y_test, y_pred_dt),
    recall=recall_score(y_test, y_pred_dt),
    f1=f1_score(y_test, y_pred_dt),
    roc_auc=roc_auc_score(y_test, y_proba_dt),
    confusion_matrix=confusion_matrix(y_test, y_pred_dt).tolist(),
    report=classification_report(y_test, y_pred_dt, target_names=["No", "Yes"]),
)

for name, r in results.items():
    print(f"\n===== {name} =====")
    print(f"Accuracy : {r['accuracy']:.3f}")
    print(f"Precision: {r['precision']:.3f}")
    print(f"Recall   : {r['recall']:.3f}")
    print(f"F1       : {r['f1']:.3f}")
    print(f"ROC AUC  : {r['roc_auc']:.3f}")
    print(r["report"])

# ---- Confusion matrix plots ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (name, r) in zip(axes, results.items()):
    cm = np.array(r["confusion_matrix"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No", "Yes"], yticklabels=["No", "Yes"], cbar=False, annot_kws={"size": 14})
    ax.set_title(f"{name}\nAccuracy = {r['accuracy']:.1%}", fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("/home/claude/hr_attrition/charts/confusion_matrices.png", dpi=150)
plt.close()

# ---- ROC curves ----
plt.figure(figsize=(7, 6))
for name, y_proba in [("Logistic Regression", y_proba_lr), ("Decision Tree", y_proba_dt)]:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = results[name]["roc_auc"]
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})", linewidth=2)
plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Attrition Prediction Models", fontweight="bold")
plt.legend()
plt.tight_layout()
plt.savefig("/home/claude/hr_attrition/charts/roc_curve.png", dpi=150)
plt.close()

# ---- Logistic Regression coefficients (feature impact direction) ----
coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": log_reg.coef_[0]
}).sort_values("coefficient", key=abs, ascending=False).head(15)

plt.figure(figsize=(9, 7))
colors = ["#E4572E" if c > 0 else "#3B82F6" for c in coef_df["coefficient"]]
plt.barh(coef_df["feature"][::-1], coef_df["coefficient"][::-1], color=colors[::-1])
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Logistic Regression — Top 15 Feature Coefficients\n(Red = increases attrition risk, Blue = decreases)",
          fontweight="bold")
plt.xlabel("Coefficient (standardized)")
plt.tight_layout()
plt.savefig("/home/claude/hr_attrition/charts/logreg_coefficients.png", dpi=150)
plt.close()

# Save everything needed downstream
import pickle
with open("/home/claude/hr_attrition/data/model_artifacts.pkl", "wb") as f:
    pickle.dump({
        "log_reg": log_reg, "tree": tree, "scaler": scaler,
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "X_train_scaled": X_train_scaled, "X_test_scaled": X_test_scaled,
        "feature_names": feature_names, "results": results,
    }, f)

with open("/home/claude/hr_attrition/outputs/model_results.json", "w") as f:
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "report"} for k, v in results.items()}, f, indent=2)

print("\nSaved model artifacts, confusion matrices, ROC curve, coefficient chart.")
