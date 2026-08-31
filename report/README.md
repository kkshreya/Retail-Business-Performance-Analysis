# Retail Business Performance & Profitability Analysis

## Project Overview

This project analyzes retail transaction data to understand sales performance, profitability, product performance, regional trends, seasonal patterns, and the impact of discounts.

The project uses SQL, Python and Tableau to transform retail data into meaningful business insights and actionable recommendations.

## Objectives

- Identify the most and least profitable categories and sub-categories
- Identify profit-draining products
- Identify slow-moving product candidates
- Compare regional sales and profitability
- Analyze monthly and seasonal sales patterns
- Analyze the relationship between discount and profit
- Build an interactive Tableau dashboard
- Provide business recommendations

## Dataset

**Dataset:** Superstore Sales Dataset

**Records:** 10,194

The dataset contains retail transaction information such as:

- Order Date
- Customer information
- Region
- Product information
- Category
- Sub-Category
- Sales
- Quantity
- Discount
- Profit

## Tools Used

- **Python** — Data cleaning and analysis
- **Pandas** — Data manipulation
- **Matplotlib & Seaborn** — Data visualization
- **MySQL / SQL** — Business analysis
- **Tableau** — Interactive dashboard
- **GitHub** — Project documentation and version control

## Project Workflow

Superstore Dataset
↓
Data Cleaning using Python
↓
Cleaned Dataset
↓
MySQL Database
↓
SQL Analysis
↓
Python Exploratory Data Analysis
↓
Tableau Dashboard
↓
Business Insights
↓
Recommendations

## Data Cleaning

The dataset was cleaned and prepared using Python and Pandas.

The main steps included:

- Checking the dataset structure
- Checking missing values
- Checking duplicate records
- Standardizing data
- Converting date fields
- Checking numerical data types
- Preparing analytical fields

## SQL Analysis

SQL was used to analyze:

- Total Sales
- Total Profit
- Profit Margin
- Category profitability
- Sub-category profitability
- Top profitable products
- Profit-draining products
- Regional performance
- Monthly performance
- Seasonal performance
- Discount versus profit
- Negative-profit products

The SQL queries are available in:

`sql/retail_analysis.sql`

## Python Analysis

Python and Pandas were used for:

- Data inspection
- Business metric calculation
- Category analysis
- Sub-category analysis
- Regional analysis
- Monthly trend analysis
- Discount versus profit analysis
- Product profitability analysis

Notebook:

`python/retail_analysis.ipynb`

## Tableau Dashboard

The Tableau dashboard presents the main business performance indicators and analysis.

### Dashboard KPIs

- Total Sales
- Total Profit
- Total Quantity
- Profit Margin

### Dashboard Charts

- Profit by Category
- Profit by Sub-Category
- Monthly Sales Trend
- Profit by Region

### Dashboard Filters

- Region
- Category
- Sub-Category
- Year
- Season

## Tableau Dashboard Preview

![Retail Business Performance Dashboard](images/dashboard.png)

## Key Insights

The analysis focuses on identifying:

- High-performing categories and sub-categories
- Loss-making products
- Strong and weak regions
- Monthly and seasonal sales patterns
- The relationship between discounts and profitability
- Potential slow-moving products

## Business Recommendations

1. Focus on highly profitable products and categories.
2. Review products that consistently generate negative profit.
3. Monitor high-discount transactions to protect profit margins.
4. Focus sales strategies on strong-performing regions.
5. Review slow-moving product candidates before increasing inventory.
6. Use seasonal sales patterns for better promotion and planning.

## Inventory Analysis Note

The standard Superstore dataset does not contain actual stock quantity or inventory-days information.

Therefore, this project identifies slow-moving product candidates using sales quantity and order activity rather than claiming confirmed overstock.

## Project Structure

Retail-Business-Performance-Analysis/
├── data/
│   ├── raw/
│   └── cleaned/
│       └── superstore_cleaned.csv
├── sql/
│   └── retail_analysis.sql
├── python/
│   └── retail_analysis.ipynb
├── tableau/
│   └── Retail_Profitability_Dashboard.twbx
├── report/
│   └── Retail_Business_Performance_Report.pdf
├── images/
│   └── dashboard.png
└── README.md

## Project Outcome

This project demonstrates an end-to-end data analytics workflow:

Data Cleaning → SQL Analysis → Python EDA → Tableau Dashboard → Business Recommendations

## Conclusion

The project demonstrates how SQL, Python and Tableau can be used together to analyze retail business performance and support data-driven decisions related to profitability, products, discounts, regions, seasonal trends and inventory planning.

## Author

**Shreya KK**

Data Analyst Internship Project