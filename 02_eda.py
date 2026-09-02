"""EDA: department-wise attrition, salary bands, promotions, and other key drivers."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.0)
PALETTE = {"Yes": "#E4572E", "No": "#3B82F6"}

df = pd.read_csv("/home/claude/hr_attrition/data/hr_attrition_raw.csv")

# Feature engineering for EDA
df["SalaryBand"] = pd.cut(
    df["MonthlyIncome"],
    bins=[0, 3000, 6000, 10000, 15000, 100000],
    labels=["<3K (Low)", "3K-6K", "6K-10K", "10K-15K", "15K+ (High)"]
)
df["PromotedRecently"] = np.where(df["YearsSinceLastPromotion"] <= 1, "Promoted <=1yr", "Not Recently Promoted")
df["TenureBand"] = pd.cut(
    df["YearsAtCompany"], bins=[-1, 1, 3, 6, 10, 100],
    labels=["0-1 yr", "2-3 yr", "4-6 yr", "7-10 yr", "10+ yr"]
)

overall_rate = (df["Attrition"] == "Yes").mean()
print(f"Overall attrition rate: {overall_rate:.1%}")

fig, axes = plt.subplots(2, 3, figsize=(19, 11))
fig.suptitle("HR Attrition — Exploratory Data Analysis", fontsize=18, fontweight="bold")

# 1. Department-wise attrition
dept_rate = df.groupby("Department")["Attrition"].apply(lambda s: (s == "Yes").mean()).sort_values(ascending=False)
ax = axes[0, 0]
sns.barplot(x=dept_rate.values * 100, y=dept_rate.index, ax=ax, hue=dept_rate.index, palette="Reds_r", legend=False)
ax.set_title("Attrition Rate by Department", fontweight="bold")
ax.set_xlabel("Attrition Rate (%)")
ax.set_ylabel("")
for i, v in enumerate(dept_rate.values * 100):
    ax.text(v + 0.5, i, f"{v:.1f}%", va="center")

# 2. Salary band vs attrition
ax = axes[0, 1]
salary_ct = pd.crosstab(df["SalaryBand"], df["Attrition"], normalize="index") * 100
salary_ct.plot(kind="bar", stacked=True, ax=ax, color=[PALETTE["No"], PALETTE["Yes"]])
ax.set_title("Attrition % by Salary Band", fontweight="bold")
ax.set_xlabel("Monthly Income Band")
ax.set_ylabel("Percent")
ax.tick_params(axis="x", rotation=30)
ax.legend(title="Attrition")

# 3. Promotion recency vs attrition
ax = axes[0, 2]
promo_rate = df.groupby("PromotedRecently")["Attrition"].apply(lambda s: (s == "Yes").mean()) * 100
sns.barplot(x=promo_rate.index, y=promo_rate.values, ax=ax, hue=promo_rate.index,
            palette=["#2A9D8F", "#E4572E"], legend=False)
ax.set_title("Attrition Rate by Promotion Recency", fontweight="bold")
ax.set_ylabel("Attrition Rate (%)")
ax.set_xlabel("")
for i, v in enumerate(promo_rate.values):
    ax.text(i, v + 0.5, f"{v:.1f}%", ha="center")

# 4. OverTime vs attrition
ax = axes[1, 0]
ot_rate = df.groupby("OverTime")["Attrition"].apply(lambda s: (s == "Yes").mean()) * 100
sns.barplot(x=ot_rate.index, y=ot_rate.values, ax=ax, hue=ot_rate.index,
            palette=["#E4572E", "#2A9D8F"], legend=False)
ax.set_title("Attrition Rate: OverTime vs No OverTime", fontweight="bold")
ax.set_ylabel("Attrition Rate (%)")
for i, v in enumerate(ot_rate.values):
    ax.text(i, v + 0.5, f"{v:.1f}%", ha="center")

# 5. Job satisfaction vs attrition
ax = axes[1, 1]
sat_rate = df.groupby("JobSatisfaction")["Attrition"].apply(lambda s: (s == "Yes").mean()) * 100
sns.barplot(x=sat_rate.index, y=sat_rate.values, ax=ax, hue=sat_rate.index, palette="Reds_r", legend=False)
ax.set_title("Attrition Rate by Job Satisfaction (1=Low, 4=High)", fontweight="bold")
ax.set_ylabel("Attrition Rate (%)")
ax.set_xlabel("Job Satisfaction Level")

# 6. Tenure band vs attrition
ax = axes[1, 2]
tenure_rate = df.groupby("TenureBand", observed=True)["Attrition"].apply(lambda s: (s == "Yes").mean()) * 100
sns.barplot(x=tenure_rate.index, y=tenure_rate.values, ax=ax, hue=tenure_rate.index, palette="Reds_r", legend=False)
ax.set_title("Attrition Rate by Tenure at Company", fontweight="bold")
ax.set_ylabel("Attrition Rate (%)")
ax.set_xlabel("Years at Company")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("/home/claude/hr_attrition/charts/eda_overview.png", dpi=150)
plt.close()

# Correlation heatmap for numeric features
numeric_cols = df.select_dtypes(include=[np.number]).columns.drop("EmployeeID")
corr = df[numeric_cols].copy()
corr["Attrition"] = (df["Attrition"] == "Yes").astype(int)
plt.figure(figsize=(14, 11))
sns.heatmap(corr.corr(), cmap="RdBu_r", center=0, annot=False, square=True, cbar_kws={"shrink": 0.7})
plt.title("Correlation Heatmap — Numeric HR Features", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("/home/claude/hr_attrition/charts/correlation_heatmap.png", dpi=150)
plt.close()

# Save enriched dataset for BI + modeling
df.to_csv("/home/claude/hr_attrition/data/hr_attrition_enriched.csv", index=False)

# Print top correlations with attrition for reference
attr_corr = corr.corr()["Attrition"].drop("Attrition").sort_values(key=abs, ascending=False)
print("\nTop numeric correlations with Attrition:")
print(attr_corr.head(12))

print("\nDept attrition rates:\n", dept_rate)
print("\nSaved charts and enriched CSV.")
