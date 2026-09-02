"""
Prepare a clean, Power-BI-ready dataset:
- All original + engineered HR fields
- Model-predicted attrition risk score (probability) for every employee
- Risk tier bucket for easy slicer/visual use in Power BI
"""
import pickle
import pandas as pd
import numpy as np

df = pd.read_csv("/home/claude/hr_attrition/data/hr_attrition_enriched.csv")

with open("/home/claude/hr_attrition/data/model_artifacts.pkl", "rb") as f:
    art = pickle.load(f)

log_reg = art["log_reg"]
scaler = art["scaler"]
feature_names = art["feature_names"]

drop_cols = ["EmployeeID", "SalaryBand", "PromotedRecently", "TenureBand", "EducationField", "Attrition"]
X = df.drop(columns=drop_cols)
cat_cols = X.select_dtypes(include="object").columns.tolist()
X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
# align columns with training feature set
X_encoded = X_encoded.reindex(columns=feature_names, fill_value=0)
X_scaled = scaler.transform(X_encoded)

df["PredictedAttritionRisk"] = log_reg.predict_proba(X_scaled)[:, 1].round(4)
df["RiskTier"] = pd.cut(
    df["PredictedAttritionRisk"],
    bins=[-0.01, 0.25, 0.5, 0.75, 1.01],
    labels=["Low", "Moderate", "High", "Critical"]
)

# Friendly column order for Power BI
bi_cols = [
    "EmployeeID", "Age", "Gender", "MaritalStatus", "Department", "JobRole",
    "EducationField", "Education", "JobLevel", "MonthlyIncome", "SalaryBand",
    "PercentSalaryHike", "StockOptionLevel", "BusinessTravel", "DistanceFromHome",
    "OverTime", "TotalWorkingYears", "NumCompaniesWorked", "YearsAtCompany",
    "TenureBand", "YearsInCurrentRole", "YearsSinceLastPromotion",
    "YearsWithCurrManager", "PromotedRecently", "TrainingTimesLastYear",
    "EnvironmentSatisfaction", "JobSatisfaction", "RelationshipSatisfaction",
    "JobInvolvement", "WorkLifeBalance", "PerformanceRating",
    "Attrition", "PredictedAttritionRisk", "RiskTier",
]
bi_df = df[bi_cols]
bi_df.to_csv("/home/claude/hr_attrition/outputs/HR_Attrition_PowerBI_Dataset.csv", index=False)
print("Saved Power BI dataset:", bi_df.shape)
print(bi_df["RiskTier"].value_counts())

# A small summary/measures table Power BI can use for KPI cards
summary = pd.DataFrame({
    "Metric": [
        "Total Employees", "Attrition Count", "Attrition Rate",
        "Avg Monthly Income (Attrited)", "Avg Monthly Income (Retained)",
        "Avg Job Satisfaction (Attrited)", "Avg Job Satisfaction (Retained)",
        "OverTime Attrition Rate", "No-OverTime Attrition Rate",
    ],
    "Value": [
        len(df),
        (df["Attrition"] == "Yes").sum(),
        round((df["Attrition"] == "Yes").mean(), 4),
        round(df.loc[df.Attrition == "Yes", "MonthlyIncome"].mean(), 0),
        round(df.loc[df.Attrition == "No", "MonthlyIncome"].mean(), 0),
        round(df.loc[df.Attrition == "Yes", "JobSatisfaction"].mean(), 2),
        round(df.loc[df.Attrition == "No", "JobSatisfaction"].mean(), 2),
        round(df.loc[df.OverTime == "Yes", "Attrition"].eq("Yes").mean(), 4),
        round(df.loc[df.OverTime == "No", "Attrition"].eq("Yes").mean(), 4),
    ]
})
summary.to_csv("/home/claude/hr_attrition/outputs/HR_Attrition_KPI_Summary.csv", index=False)
print("\n", summary)
