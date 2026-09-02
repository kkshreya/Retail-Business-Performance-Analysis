"""
Generate a realistic synthetic HR dataset for attrition analysis.
Schema is modeled on the well-known IBM HR Analytics Employee Attrition
dataset structure, but all rows are synthetically generated here with
built-in causal relationships (low satisfaction, low salary hike, long
commute, no promotion, overtime -> higher attrition) so that EDA/ML/SHAP
results are meaningful and reproducible.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 1470  # same size as the classic IBM dataset

departments = ["Sales", "Research & Development", "Human Resources"]
dept_p = [0.31, 0.65, 0.04]

job_roles = {
    "Sales": ["Sales Executive", "Sales Representative", "Manager"],
    "Research & Development": ["Research Scientist", "Laboratory Technician",
                                "Manufacturing Director", "Healthcare Representative",
                                "Research Director", "Manager"],
    "Human Resources": ["Human Resources", "Manager"],
}

education_field_map = {
    "Sales": ["Marketing", "Life Sciences", "Other"],
    "Research & Development": ["Life Sciences", "Medical", "Technical Degree", "Other"],
    "Human Resources": ["Human Resources", "Other"],
}

rows = []
for i in range(N):
    dept = np.random.choice(departments, p=dept_p)
    role = np.random.choice(job_roles[dept])
    edu_field = np.random.choice(education_field_map[dept])

    age = int(np.clip(np.random.normal(37, 9), 18, 60))
    gender = np.random.choice(["Male", "Female"], p=[0.6, 0.4])
    marital = np.random.choice(["Single", "Married", "Divorced"], p=[0.32, 0.46, 0.22])
    education = np.random.choice([1, 2, 3, 4, 5], p=[0.11, 0.19, 0.39, 0.27, 0.04])  # 1=Below College..5=Doctor

    total_working_years = int(np.clip(age - 20 - np.random.poisson(2), 0, 40))
    years_at_company = int(np.clip(np.random.exponential(5), 0, total_working_years if total_working_years > 0 else 1))
    years_at_company = min(years_at_company, total_working_years)
    years_in_current_role = int(np.clip(np.random.exponential(2.5), 0, years_at_company))
    years_since_promotion = int(np.clip(np.random.exponential(2), 0, years_at_company))
    years_with_manager = int(np.clip(np.random.exponential(2.5), 0, years_at_company))
    num_companies_worked = int(np.clip(np.random.poisson(2.5), 0, 9))

    # Job level correlates with total working years and education
    job_level = int(np.clip(1 + total_working_years // 6 + (education >= 4) * np.random.randint(0, 2), 1, 5))

    # Monthly income depends on job level + department + role, with noise
    base_income = {1: 2800, 2: 4800, 3: 7200, 4: 11500, 5: 17500}[job_level]
    income_noise = np.random.normal(1, 0.18)
    monthly_income = int(np.clip(base_income * income_noise, 1009, 20000))

    percent_salary_hike = int(np.clip(np.random.normal(15, 3.5), 11, 25))
    stock_option_level = np.random.choice([0, 1, 2, 3], p=[0.42, 0.32, 0.19, 0.07])

    distance_from_home = int(np.clip(np.random.exponential(8), 1, 29))
    business_travel = np.random.choice(
        ["Non-Travel", "Travel_Rarely", "Travel_Frequently"], p=[0.1, 0.71, 0.19]
    )
    overtime = np.random.choice(["Yes", "No"], p=[0.28, 0.72])

    env_satisfaction = np.random.choice([1, 2, 3, 4], p=[0.18, 0.20, 0.30, 0.32])
    job_satisfaction = np.random.choice([1, 2, 3, 4], p=[0.18, 0.20, 0.30, 0.32])
    relationship_satisfaction = np.random.choice([1, 2, 3, 4], p=[0.16, 0.21, 0.31, 0.32])
    job_involvement = np.random.choice([1, 2, 3, 4], p=[0.09, 0.26, 0.52, 0.13])
    work_life_balance = np.random.choice([1, 2, 3, 4], p=[0.06, 0.24, 0.55, 0.15])
    performance_rating = np.random.choice([3, 4], p=[0.85, 0.15])

    training_times_last_year = int(np.clip(np.random.poisson(2.8), 0, 6))
    daily_rate = int(np.random.uniform(102, 1499))
    hourly_rate = int(np.random.uniform(30, 100))
    monthly_rate = int(np.random.uniform(2094, 26999))

    # ---- Attrition probability model (ground truth signal) ----
    logit = -2.1
    logit += 1.15 if overtime == "Yes" else 0
    logit += (4 - job_satisfaction) * 0.32
    logit += (4 - env_satisfaction) * 0.22
    logit += (4 - work_life_balance) * 0.28
    logit += 0.35 if marital == "Single" else (0.0 if marital == "Married" else 0.12)
    logit += -0.05 * years_at_company
    logit += 0.02 * distance_from_home
    logit += 0.25 if business_travel == "Travel_Frequently" else (0.05 if business_travel == "Travel_Rarely" else -0.1)
    logit += -0.00009 * monthly_income
    logit += 0.10 if years_since_promotion >= 5 else 0
    logit += -0.10 * stock_option_level
    logit += 0.15 if num_companies_worked >= 5 else 0
    logit += -0.06 * (age - 37) / 9.0
    logit += np.random.normal(0, 0.55)  # noise

    prob = 1 / (1 + np.exp(-logit))
    attrition = "Yes" if np.random.rand() < prob else "No"

    rows.append(dict(
        EmployeeID=1000 + i,
        Age=age,
        Gender=gender,
        MaritalStatus=marital,
        Department=dept,
        JobRole=role,
        EducationField=edu_field,
        Education=education,
        JobLevel=job_level,
        MonthlyIncome=monthly_income,
        DailyRate=daily_rate,
        HourlyRate=hourly_rate,
        MonthlyRate=monthly_rate,
        PercentSalaryHike=percent_salary_hike,
        StockOptionLevel=stock_option_level,
        BusinessTravel=business_travel,
        DistanceFromHome=distance_from_home,
        OverTime=overtime,
        TotalWorkingYears=total_working_years,
        NumCompaniesWorked=num_companies_worked,
        YearsAtCompany=years_at_company,
        YearsInCurrentRole=years_in_current_role,
        YearsSinceLastPromotion=years_since_promotion,
        YearsWithCurrManager=years_with_manager,
        TrainingTimesLastYear=training_times_last_year,
        EnvironmentSatisfaction=env_satisfaction,
        JobSatisfaction=job_satisfaction,
        RelationshipSatisfaction=relationship_satisfaction,
        JobInvolvement=job_involvement,
        WorkLifeBalance=work_life_balance,
        PerformanceRating=performance_rating,
        Attrition=attrition,
    ))

df = pd.DataFrame(rows)
df.to_csv("/home/claude/hr_attrition/data/hr_attrition_raw.csv", index=False)
print(df.shape)
print(df["Attrition"].value_counts(normalize=True))
print(df.head(3).T)
