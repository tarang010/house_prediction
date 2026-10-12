USE ROLE ACCOUNTADMIN;
USE DATABASE HOUSE_PRICE_MLOPS;
USE SCHEMA ML;

CREATE NOTIFICATION INTEGRATION IF NOT EXISTS
    HOUSE_PRICE_DRIFT_EMAIL
    TYPE = EMAIL
    ENABLED = TRUE;

-- Test email delivery. Snowflake requires the recipient to be verified.
CALL SYSTEM$SEND_EMAIL(
    'HOUSE_PRICE_DRIFT_EMAIL',
    'tarang.termoid@gmail.com',
    'Snowflake House Price Drift - Test',
    'Test email from the HOUSE_PRICE_MLOPS drift workflow. If you received this, email delivery is working.'
);