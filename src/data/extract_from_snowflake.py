import snowflake.connector
import pandas as pd
import os

conn = snowflake.connector.connect(
    user="TIWARI010",
    password="Skates@Champion121",
    account="SBPLOBQ-TY14646",
    warehouse="COMPUTE_WH",
    database="HOUSE_PRICE_DB",
    schema="RAW"
)
query = """
SELECT *
FROM HOUSE_DATA
"""
df = pd.read_sql(query, conn)
output_dir = os.path.join("data", "raw")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(
output_dir,
"house_data.csv"
)
df.to_csv(
output_file,
index=False
)
print(f"Rows extracted: {len(df)}")
print(f"Data saved to: {output_file}")
print("Data extracted successfully")
conn.close()
