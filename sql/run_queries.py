
import sqlite3
import pandas as pd

connection = sqlite3.connect("data/churn.db")

query = """
SELECT
    Contract,
    InternetService,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END)
        AS churned_customers,
    ROUND(
        100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END)
        / COUNT(*),
        2
    ) AS churn_percentage
FROM customers
GROUP BY Contract, InternetService
HAVING COUNT(*) >= 30
ORDER BY churn_percentage DESC;
"""

result = pd.read_sql_query(query, connection)

print("Churn Analysis by Contract and Internet Service")
print(result.to_string(index=False))

connection.close()
