USE retail_business_analysis;

-- 1. Overall Business Performance
SELECT
    ROUND(SUM(Sales), 2) AS Total_Sales,
    ROUND(SUM(Profit), 2) AS Total_Profit,
    SUM(Quantity) AS Total_Quantity,
    ROUND(SUM(Profit) / SUM(Sales) * 100, 2) AS Profit_Margin_Percent
FROM superstore_cleaned;


-- 2. Profitability by Category
SELECT
    Category,
    ROUND(SUM(Sales), 2) AS Total_Sales,
    ROUND(SUM(Profit), 2) AS Total_Profit,
    ROUND(SUM(Profit) / SUM(Sales) * 100, 2) AS Profit_Margin_Percent
FROM superstore_cleaned
GROUP BY Category
ORDER BY Total_Profit DESC;


-- 3. Profitability by Sub-Category
SELECT
    `Sub-Category`,
    ROUND(SUM(Sales), 2) AS Total_Sales,
    ROUND(SUM(Profit), 2) AS Total_Profit,
    ROUND(SUM(Profit) / SUM(Sales) * 100, 2) AS Profit_Margin_Percent
FROM superstore_cleaned
GROUP BY `Sub-Category`
ORDER BY Total_Profit DESC;


-- 4. Top 10 Most Profitable Products
SELECT
    `Product Name`,
    ROUND(SUM(Sales), 2) AS Total_Sales,
    SUM(Quantity) AS Total_Quantity,
    ROUND(SUM(Profit), 2) AS Total_Profit
FROM superstore_cleaned
GROUP BY `Product Name`
ORDER BY Total_Profit DESC
LIMIT 10;


-- 5. Bottom 10 Profit-Draining Products
SELECT
    `Product Name`,
    ROUND(SUM(Sales), 2) AS Total_Sales,
    SUM(Quantity) AS Total_Quantity,
    ROUND(SUM(Profit), 2) AS Total_Profit
FROM superstore_cleaned
GROUP BY `Product Name`
ORDER BY Total_Profit ASC
LIMIT 10;


-- 6. Regional Performance
SELECT
    Region,
    ROUND(SUM(Sales), 2) AS Total_Sales,
    ROUND(SUM(Profit), 2) AS Total_Profit,
    ROUND(SUM(Profit) / SUM(Sales) * 100, 2) AS Profit_Margin_Percent
FROM superstore_cleaned
GROUP BY Region
ORDER BY Total_Profit DESC;


-- 7. Monthly Sales and Profit
SELECT
    Year,
    `Month Number`,
    Month,
    ROUND(SUM(Sales), 2) AS Total_Sales,
    ROUND(SUM(Profit), 2) AS Total_Profit
FROM superstore_cleaned
GROUP BY Year, `Month Number`, Month
ORDER BY Year, `Month Number`;


-- 8. Seasonal Performance
SELECT
    Season,
    ROUND(SUM(Sales), 2) AS Total_Sales,
    ROUND(SUM(Profit), 2) AS Total_Profit,
    SUM(Quantity) AS Total_Quantity
FROM superstore_cleaned
GROUP BY Season
ORDER BY Total_Profit DESC;


-- 9. Discount vs Profit
SELECT
    Discount,
    ROUND(SUM(Sales), 2) AS Total_Sales,
    ROUND(SUM(Profit), 2) AS Total_Profit,
    SUM(Quantity) AS Total_Quantity
FROM superstore_cleaned
GROUP BY Discount
ORDER BY Discount;


-- 10. Products with Negative Profit
SELECT
    `Product Name`,
    Category,
    `Sub-Category`,
    ROUND(SUM(Sales), 2) AS Total_Sales,
    SUM(Quantity) AS Total_Quantity,
    ROUND(SUM(Profit), 2) AS Total_Profit
FROM superstore_cleaned
GROUP BY `Product Name`, Category, `Sub-Category`
HAVING SUM(Profit) < 0
ORDER BY Total_Profit ASC;