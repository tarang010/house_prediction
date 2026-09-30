import snowflake.connector
import pandas as pd

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

df.to_csv(
    "../../data/raw/house_data.csv",
    index=False
)

print("Data extracted successfully")
