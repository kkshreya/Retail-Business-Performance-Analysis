"""SHAP value analysis to explain model predictions."""
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

with open("/home/claude/hr_attrition/data/model_artifacts.pkl", "rb") as f:
    art = pickle.load(f)

log_reg = art["log_reg"]
X_train_scaled = art["X_train_scaled"]
X_test_scaled = art["X_test_scaled"]
feature_names = art["feature_names"]
X_test = art["X_test"]

X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)

# Linear explainer is exact + fast for logistic regression
explainer = shap.LinearExplainer(log_reg, X_train_scaled_df)
shap_values = explainer(X_test_scaled_df)

# ---- Summary (beeswarm) plot ----
plt.figure()
shap.summary_plot(shap_values, X_test_scaled_df, show=False, max_display=15)
plt.title("SHAP Summary — Feature Impact on Attrition Prediction", fontweight="bold")
plt.tight_layout()
plt.savefig("/home/claude/hr_attrition/charts/shap_summary_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()

# ---- Bar plot (mean |SHAP|) ----
plt.figure()
shap.summary_plot(shap_values, X_test_scaled_df, plot_type="bar", show=False, max_display=15)
plt.title("SHAP Feature Importance (Mean |SHAP value|)", fontweight="bold")
plt.tight_layout()
plt.savefig("/home/claude/hr_attrition/charts/shap_importance_bar.png", dpi=150, bbox_inches="tight")
plt.close()

# ---- Waterfall for one at-risk employee (highest predicted prob) ----
probs = log_reg.predict_proba(X_test_scaled)[:, 1]
idx = int(np.argmax(probs))
plt.figure()
shap.plots.waterfall(shap_values[idx], show=False, max_display=12)
plt.title(f"SHAP Waterfall — Highest-Risk Employee (predicted attrition prob = {probs[idx]:.0%})",
          fontweight="bold", fontsize=10)
plt.tight_layout()
plt.savefig("/home/claude/hr_attrition/charts/shap_waterfall_highest_risk.png", dpi=150, bbox_inches="tight")
plt.close()

# Save mean abs shap ranking to a CSV for the report
mean_abs_shap = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": np.abs(shap_values.values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)
mean_abs_shap.to_csv("/home/claude/hr_attrition/outputs/shap_feature_importance.csv", index=False)
print(mean_abs_shap.head(15))
print("\nSaved SHAP plots.")
