
import pandas as pd
import sqlite3

df = pd.read_csv("data/telco_churn_cleaned.csv")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

connection = sqlite3.connect("data/churn.db")

df.to_sql(
    "customers",
    connection,
    if_exists="replace",
    index=False
)

print("Database created successfully!")
print("Total customers:", pd.read_sql(
    "SELECT COUNT(*) AS total FROM customers",
    connection
))

connection.close()
