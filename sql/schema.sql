CREATE TABLE IF NOT EXISTS trips (
    ride_id TEXT PRIMARY KEY,
    rideable_type TEXT,

    started_at TIMESTAMP NOT NULL,
    ended_at TIMESTAMP NOT NULL,

    start_station_name TEXT,
    start_station_id TEXT,

    end_station_name TEXT,
    end_station_id TEXT,

    start_lat DOUBLE PRECISION,
    start_lng DOUBLE PRECISION,
    end_lat DOUBLE PRECISION,
    end_lng DOUBLE PRECISION,

    member_casual TEXT
);

CREATE INDEX IF NOT EXISTS idx_trips_started_at
    ON trips (started_at);

CREATE INDEX IF NOT EXISTS idx_trips_ended_at
    ON trips (ended_at);

CREATE INDEX IF NOT EXISTS idx_trips_start_station
    ON trips (start_station_id);

CREATE INDEX IF NOT EXISTS idx_trips_end_station
    ON trips (end_station_id);
