
import os
import uuid
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


def store_prediction(
    request_data: dict,
    predicted_price: float,
    model_version: str,
):
    prediction_id = str(uuid.uuid4())

    query = """
        INSERT INTO HOUSE_PRICE_MLOPS.ML.PRODUCTION_DATA (
            PREDICTION_ID,
            SQFT,
            BEDROOMS,
            BATHROOMS,
            LOCATION,
            YEAR_BUILT,
            CONDITION,
            PREDICTED_PRICE,
            PREDICTION_TIME,
            MODEL_VERSION
        )
        VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s,
            CURRENT_TIMESTAMP()::TIMESTAMP_NTZ,
            %s
        )
    """

    conn = get_snowflake_connection()

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                query,
                (
                    prediction_id,
                    request_data["sqft"],
                    request_data["bedrooms"],
                    request_data["bathrooms"],
                    request_data["location"],
                    request_data["year_built"],
                    request_data["condition"],
                    predicted_price,
                    model_version,
                ),
            )

        conn.commit()
        return prediction_id

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()
