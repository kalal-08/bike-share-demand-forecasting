-- ============================================================
-- Citi Bike station-hour demand aggregation
--
-- Top stations are selected using TRAINING data only:
-- 2026-04-01 through 2026-05-31.
--
-- Final hourly table covers:
-- 2026-04-01 through 2026-06-30.
-- ============================================================


DROP TABLE IF EXISTS station_hourly_demand;
DROP TABLE IF EXISTS top_stations_train;


-- ------------------------------------------------------------
-- 1. Select the 20 most active stations using TRAIN data only.
--
-- Activity = departures + arrivals.
-- ------------------------------------------------------------

CREATE TABLE top_stations_train AS
WITH station_activity AS (

    SELECT
        start_station_id AS station_id,
        COUNT(*) AS activity
    FROM trips
    WHERE
        started_at >= TIMESTAMP '2026-04-01 00:00:00'
        AND started_at < TIMESTAMP '2026-06-01 00:00:00'
        AND start_station_id IS NOT NULL
    GROUP BY start_station_id

    UNION ALL

    SELECT
        end_station_id AS station_id,
        COUNT(*) AS activity
    FROM trips
    WHERE
        ended_at >= TIMESTAMP '2026-04-01 00:00:00'
        AND ended_at < TIMESTAMP '2026-06-01 00:00:00'
        AND end_station_id IS NOT NULL
    GROUP BY end_station_id
),

combined_activity AS (
    SELECT
        station_id,
        SUM(activity) AS total_activity
    FROM station_activity
    GROUP BY station_id
)

SELECT
    station_id,
    total_activity
FROM combined_activity
ORDER BY total_activity DESC
LIMIT 20;


ALTER TABLE top_stations_train
ADD PRIMARY KEY (station_id);


-- ------------------------------------------------------------
-- 2. Aggregate departures by station and hour.
-- ------------------------------------------------------------

WITH departures AS (
    SELECT
        start_station_id AS station_id,
        DATE_TRUNC('hour', started_at) AS hour,
        COUNT(*) AS departures
    FROM trips
    WHERE
        started_at >= TIMESTAMP '2026-04-01 00:00:00'
        AND started_at < TIMESTAMP '2026-07-01 00:00:00'
        AND start_station_id IN (
            SELECT station_id
            FROM top_stations_train
        )
    GROUP BY
        start_station_id,
        DATE_TRUNC('hour', started_at)
),

-- ------------------------------------------------------------
-- 3. Aggregate arrivals by station and hour.
-- ------------------------------------------------------------

arrivals AS (
    SELECT
        end_station_id AS station_id,
        DATE_TRUNC('hour', ended_at) AS hour,
        COUNT(*) AS arrivals
    FROM trips
    WHERE
        ended_at >= TIMESTAMP '2026-04-01 00:00:00'
        AND ended_at < TIMESTAMP '2026-07-01 00:00:00'
        AND end_station_id IN (
            SELECT station_id
            FROM top_stations_train
        )
    GROUP BY
        end_station_id,
        DATE_TRUNC('hour', ended_at)
),

-- ------------------------------------------------------------
-- 4. Generate every hour in the analysis period.
--
-- This is important because zero-demand hours must also appear
-- in the ML dataset.
-- ------------------------------------------------------------

hours AS (
    SELECT generate_series(
        TIMESTAMP '2026-04-01 00:00:00',
        TIMESTAMP '2026-06-30 23:00:00',
        INTERVAL '1 hour'
    ) AS hour
),

-- Every selected station × every hour.
station_hours AS (
    SELECT
        s.station_id,
        h.hour
    FROM top_stations_train s
    CROSS JOIN hours h
)


-- ------------------------------------------------------------
-- 5. Build complete station-hour table.
--
-- Missing departure/arrival counts become zero.
-- ------------------------------------------------------------

SELECT
    sh.station_id,
    sh.hour,

    COALESCE(d.departures, 0)::INTEGER AS departures,
    COALESCE(a.arrivals, 0)::INTEGER AS arrivals,

    (
        COALESCE(a.arrivals, 0)
        - COALESCE(d.departures, 0)
    )::INTEGER AS net_flow

INTO station_hourly_demand

FROM station_hours sh

LEFT JOIN departures d
    ON sh.station_id = d.station_id
    AND sh.hour = d.hour

LEFT JOIN arrivals a
    ON sh.station_id = a.station_id
    AND sh.hour = a.hour

ORDER BY
    sh.station_id,
    sh.hour;


ALTER TABLE station_hourly_demand
ADD PRIMARY KEY (station_id, hour);


CREATE INDEX idx_station_hourly_hour
ON station_hourly_demand (hour);


ANALYZE top_stations_train;
ANALYZE station_hourly_demand;
