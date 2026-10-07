import snowflake.connector
import pandas as pd
import os


conn = snowflake.connector.connect(
    user=os.getenv("SF_user"),
    password=os.getenv("SF_password"),
    account=os.getenv("SF_account"),
    warehouse="COMPUTE_WH",
    database="HOUSE_PRICE_DB",
    schema="RAW"
)
query = """
SELECT *
FROM HOUSE_DATA
"""
df = pd.read_sql(query, conn)
# Convert Snowflake column names to lowercase
df.columns = [col.lower().strip() for col in df.columns]
print("Columns after conversion:")
print(df.columns.tolist())
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
