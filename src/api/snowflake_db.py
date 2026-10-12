
import os
import snowflake.connector


def get_snowflake_connection():
    return snowflake.connector.connect(
        user=os.getenv("SF_user"),
        password=os.getenv("SF_password"),
        account=os.getenv("SF_account"),
        warehouse="ML_WH",
        database="HOUSE_PRICE_MLOPS",
        schema="ML",
    )


def store_prediction(request_data: dict, predicted_price: float):
    query = """
        INSERT INTO HOUSE_PRICE_MLOPS.ML.PRODUCTION_DATA (
            SQFT,
            BEDROOMS,
            BATHROOMS,
            LOCATION,
            YEAR_BUILT,
            CONDITION,
            PREDICTED_PRICE
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """

    conn = get_snowflake_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                query,
                (
                    request_data["sqft"],
                    request_data["bedrooms"],
                    request_data["bathrooms"],
                    request_data["location"],
                    request_data["year_built"],
                    request_data["condition"],
                    predicted_price,
                ),
            )

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()
