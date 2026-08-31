import pandas as pd
from pathlib import Path

# ==============================
# 1. LOAD DATA
# ==============================

project_folder = Path(__file__).resolve().parent.parent

file_path = project_folder / "data" / "sample_-_superstore (2).xls"

df = pd.read_excel(file_path)

print("Original dataset shape:", df.shape)


# ==============================
# 2. REMOVE DUPLICATES
# ==============================

df = df.drop_duplicates()


# ==============================
# 3. HANDLE MISSING VALUES
# ==============================

# Remove rows where important business fields are missing
important_columns = [
    "Order Date",
    "Region",
    "Category",
    "Sub-Category",
    "Product Name",
    "Sales",
    "Quantity",
    "Discount",
    "Profit"
]

df = df.dropna(subset=important_columns)


# ==============================
# 4. CONVERT DATA TYPES
# ==============================

df["Order Date"] = pd.to_datetime(df["Order Date"])

df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["Discount"] = pd.to_numeric(df["Discount"], errors="coerce")
df["Profit"] = pd.to_numeric(df["Profit"], errors="coerce")


# ==============================
# 5. CREATE NEW ANALYSIS COLUMNS
# ==============================

# Profit Margin
df["Profit Margin"] = (df["Profit"] / df["Sales"]) * 100

# Year
df["Year"] = df["Order Date"].dt.year

# Month
df["Month"] = df["Order Date"].dt.month_name()

# Month Number
df["Month Number"] = df["Order Date"].dt.month

# Quarter
df["Quarter"] = df["Order Date"].dt.quarter

# Season
def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

df["Season"] = df["Order Date"].dt.month.map(get_season)


# ==============================
# 6. REMOVE INVALID VALUES
# ==============================

df = df[df["Sales"] >= 0]
df = df[df["Quantity"] > 0]


# ==============================
# 7. SAVE CLEANED DATA
# ==============================

output_folder = project_folder / "data" / "cleaned"
output_folder.mkdir(exist_ok=True)

output_file = output_folder / "superstore_cleaned.csv"

df.to_csv(output_file, index=False)


# ==============================
# 8. FINAL SUMMARY
# ==============================

print("\nCleaning completed successfully!")

print("Final dataset shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nCleaned file saved at:")
print(output_file)