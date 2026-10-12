USE DATABASE HOUSE_PRICE_MLOPS;
USE SCHEMA ML;

CREATE OR REPLACE PROCEDURE HOUSE_PRICE_MLOPS.ML.CHECK_HOUSE_PRICE_DRIFT()
RETURNS VARCHAR
LANGUAGE SQL
EXECUTE AS OWNER
AS
$$
DECLARE
    v_since TIMESTAMP_NTZ;
    v_now TIMESTAMP_NTZ;
    v_ref_rows NUMBER DEFAULT 0;
    v_prod_rows NUMBER DEFAULT 0;
    v_raw_avg FLOAT;
    v_prod_avg FLOAT;
    v_new_locations NUMBER DEFAULT 0;
    v_drift BOOLEAN DEFAULT FALSE;
    v_features VARCHAR DEFAULT '';
    v_message VARCHAR;
    v_alert_status VARCHAR DEFAULT 'NO_ALERT';
BEGIN
    v_now := CURRENT_TIMESTAMP()::TIMESTAMP_NTZ;

    -- First run checks the previous hour. Subsequent runs check rows added
    -- after the most recent successfully recorded check.
    SELECT COALESCE(
        MAX(CHECK_TIME),
        DATEADD(hour, -1, :v_now)
    )
      INTO :v_since
      FROM HOUSE_PRICE_MLOPS.ML.DRIFT_RESULTS;

    SELECT COUNT(*), AVG(SQFT)
      INTO :v_ref_rows, :v_raw_avg
      FROM HOUSE_PRICE_MLOPS.ML.RAW_DATA;

    SELECT COUNT(*), AVG(SQFT)
      INTO :v_prod_rows, :v_prod_avg
      FROM HOUSE_PRICE_MLOPS.ML.PRODUCTION_DATA
     WHERE PREDICTION_TIME > :v_since
       AND PREDICTION_TIME <= :v_now;

    SELECT COUNT(DISTINCT UPPER(TRIM(p.LOCATION)))
      INTO :v_new_locations
      FROM HOUSE_PRICE_MLOPS.ML.PRODUCTION_DATA p
     WHERE p.PREDICTION_TIME > :v_since
       AND p.PREDICTION_TIME <= :v_now
       AND p.LOCATION IS NOT NULL
       AND NOT EXISTS (
           SELECT 1
             FROM HOUSE_PRICE_MLOPS.ML.RAW_DATA r
            WHERE UPPER(TRIM(r.LOCATION)) = UPPER(TRIM(p.LOCATION))
       );

    -- Initial, transparent demo rules. Calibrate thresholds before production use.
    IF (v_prod_rows >= 5
        AND v_raw_avg IS NOT NULL
        AND v_raw_avg > 0
        AND v_prod_avg IS NOT NULL
        AND ABS(v_prod_avg - v_raw_avg) / v_raw_avg > 0.30) THEN
        v_features := 'SQFT_MEAN_SHIFT_GT_30_PERCENT';
    END IF;

    IF (v_new_locations > 0) THEN
        IF (LENGTH(v_features) > 0) THEN
            v_features := v_features || ', ';
        END IF;
        v_features := v_features || 'UNSEEN_LOCATION';
    END IF;

    v_drift := LENGTH(v_features) > 0;

    -- Send alert BEFORE staging records. If email sending errors, the procedure
    -- stops here and does not record a completed check or stage records.
    IF (v_drift) THEN
        v_message :=
            'House-price feature drift detected. ' ||
            'Features: ' || v_features ||
            '. New production rows in window: ' || v_prod_rows ||
            '. Reference rows: ' || v_ref_rows ||
            '. Check time: ' || TO_VARCHAR(v_now) ||
            '. Review HOUSE_PRICE_MLOPS.ML.DRIFT_RESULTS.';

        CALL SYSTEM$SEND_EMAIL(
            'HOUSE_PRICE_DRIFT_EMAIL',
            'tarang.termoid@gmail.com',
            'ALERT: House Price Data Drift Detected',
            :v_message
        );

        v_alert_status := 'ALERT_SENT';
    END IF;

    -- Stage recent production observations once, using PREDICTION_ID as lineage.
    INSERT INTO HOUSE_PRICE_MLOPS.ML.RETRAINING_CANDIDATES (
        SOURCE_PREDICTION_ID,
        SQFT,
        BEDROOMS,
        BATHROOMS,
        LOCATION,
        YEAR_BUILT,
        CONDITION,
        PREDICTED_PRICE,
        ACTUAL_PRICE,
        CAPTURED_AT,
        CANDIDATE_STATUS
    )
    SELECT
        p.PREDICTION_ID,
        p.SQFT,
        p.BEDROOMS,
        p.BATHROOMS,
        p.LOCATION,
        p.YEAR_BUILT,
        p.CONDITION,
        p.PREDICTED_PRICE,
        p.ACTUAL_PRICE,
        p.PREDICTION_TIME,
        IFF(p.ACTUAL_PRICE IS NULL, 'AWAITING_LABEL', 'READY')
    FROM HOUSE_PRICE_MLOPS.ML.PRODUCTION_DATA p
    WHERE p.PREDICTION_TIME > :v_since
      AND p.PREDICTION_TIME <= :v_now
      AND p.PREDICTION_ID IS NOT NULL
      AND NOT EXISTS (
          SELECT 1
          FROM HOUSE_PRICE_MLOPS.ML.RETRAINING_CANDIDATES c
          WHERE c.SOURCE_PREDICTION_ID = p.PREDICTION_ID
      );

    -- If an actual price was added after staging, promote the candidate to READY.
    UPDATE HOUSE_PRICE_MLOPS.ML.RETRAINING_CANDIDATES c
       SET ACTUAL_PRICE = p.ACTUAL_PRICE,
           CANDIDATE_STATUS = 'READY'
      FROM HOUSE_PRICE_MLOPS.ML.PRODUCTION_DATA p
     WHERE c.SOURCE_PREDICTION_ID = p.PREDICTION_ID
       AND p.ACTUAL_PRICE IS NOT NULL
       AND c.ACTUAL_PRICE IS NULL
       AND c.CANDIDATE_STATUS = 'AWAITING_LABEL';

    -- Record the completed check only after alert delivery and staging succeed.
    INSERT INTO HOUSE_PRICE_MLOPS.ML.DRIFT_RESULTS (
        CHECK_TIME,
        REFERENCE_ROWS,
        PRODUCTION_ROWS,
        DRIFTED_FEATURES,
        DRIFT_DETECTED,
        ALERT_STATUS
    )
    VALUES (
        :v_now,
        :v_ref_rows,
        :v_prod_rows,
        IFF(:v_drift, :v_features, 'NONE'),
        :v_drift,
        :v_alert_status
    );

    RETURN 'Drift check completed. Window rows=' || v_prod_rows ||
           '; drift=' || v_drift ||
           '; features=' || IFF(LENGTH(v_features) = 0, 'NONE', v_features) ||
           '; alert=' || v_alert_status;
END;
$$;