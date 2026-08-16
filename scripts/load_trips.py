import argparse
import io
import zipfile
from pathlib import Path

import pandas as pd

from src.data import get_database_engine


CHUNK_SIZE = 250_000

RAW_ZIP_FILES = [
    Path("data/raw/202604-citibike-tripdata.zip"),
    Path("data/raw/202605-citibike-tripdata.zip"),
    Path("data/raw/202606-citibike-tripdata.zip"),
]

COLUMNS = [
    "ride_id",
    "rideable_type",
    "started_at",
    "ended_at",
    "start_station_name",
    "start_station_id",
    "end_station_name",
    "end_station_id",
    "start_lat",
    "start_lng",
    "end_lat",
    "end_lng",
    "member_casual",
]


COPY_SQL = """
COPY trip_stage (
    ride_id,
    rideable_type,
    started_at,
    ended_at,
    start_station_name,
    start_station_id,
    end_station_name,
    end_station_id,
    start_lat,
    start_lng,
    end_lat,
    end_lng,
    member_casual
)
FROM STDIN
WITH (
    FORMAT CSV,
    NULL '\\N'
);
"""


# Duplicate ride IDs are ignored at the database boundary so repeated
# source records cannot inflate demand counts.
INSERT_SQL = """
INSERT INTO trips (
    ride_id,
    rideable_type,
    started_at,
    ended_at,
    start_station_name,
    start_station_id,
    end_station_name,
    end_station_id,
    start_lat,
    start_lng,
    end_lat,
    end_lng,
    member_casual
)
SELECT
    ride_id,
    rideable_type,
    started_at,
    ended_at,
    start_station_name,
    start_station_id,
    end_station_name,
    end_station_id,
    start_lat,
    start_lng,
    end_lat,
    end_lng,
    member_casual
FROM trip_stage
ON CONFLICT (ride_id) DO NOTHING;
"""


DROP_SECONDARY_INDEXES_SQL = """
DROP INDEX IF EXISTS idx_trips_started_at;
DROP INDEX IF EXISTS idx_trips_ended_at;
DROP INDEX IF EXISTS idx_trips_start_station;
DROP INDEX IF EXISTS idx_trips_end_station;
"""


CREATE_SECONDARY_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_trips_started_at
    ON trips (started_at);

CREATE INDEX IF NOT EXISTS idx_trips_ended_at
    ON trips (ended_at);

CREATE INDEX IF NOT EXISTS idx_trips_start_station
    ON trips (start_station_id);

CREATE INDEX IF NOT EXISTS idx_trips_end_station
    ON trips (end_station_id);
"""


def clean_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Clean one chunk of raw Citi Bike trip records."""

    df = df[COLUMNS].copy()

    # Station IDs are identifiers, not numeric measurements.
    df["start_station_id"] = df["start_station_id"].astype("string")
    df["end_station_id"] = df["end_station_id"].astype("string")

    df["started_at"] = pd.to_datetime(
        df["started_at"],
        errors="coerce",
    )

    df["ended_at"] = pd.to_datetime(
        df["ended_at"],
        errors="coerce",
    )

    # These fields are required for a valid trip record.
    df = df.dropna(
        subset=[
            "ride_id",
            "started_at",
            "ended_at",
        ]
    )

    # Remove impossible/non-positive trip durations.
    df = df[df["ended_at"] > df["started_at"]]

    return df


def copy_chunk(cursor, df: pd.DataFrame) -> int:
    """Bulk-copy one cleaned DataFrame into PostgreSQL."""

    buffer = io.StringIO()

    df.to_csv(
        buffer,
        index=False,
        header=False,
        na_rep="\\N",
        date_format="%Y-%m-%d %H:%M:%S.%f",
    )

    buffer.seek(0)

    cursor.execute("TRUNCATE trip_stage;")

    cursor.copy_expert(
        COPY_SQL,
        buffer,
    )

    cursor.execute(INSERT_SQL)

    return cursor.rowcount


def restore_secondary_indexes(
    cursor,
    raw_connection,
    *,
    analyze: bool = False,
):
    """Recreate bulk-load indexes and commit their restoration."""

    cursor.execute(CREATE_SECONDARY_INDEXES_SQL)

    if analyze:
        cursor.execute("ANALYZE trips;")

    raw_connection.commit()


def load_all_trips(smoke_test: bool = False):
    """Load Citi Bike ZIP files into PostgreSQL."""

    missing_zip_files = [
        str(path)
        for path in RAW_ZIP_FILES
        if not path.is_file()
    ]

    if missing_zip_files:
        raise FileNotFoundError(
            "Missing expected Citi Bike ZIP files: "
            + ", ".join(missing_zip_files)
        )

    engine = get_database_engine()
    raw_connection = engine.raw_connection()
    cursor = raw_connection.cursor()
    secondary_indexes_dropped = False

    try:
        # Staging enables PostgreSQL COPY throughput while the final insert
        # still applies primary-key duplicate protection.
        cursor.execute(
            """
            CREATE TEMP TABLE trip_stage
            (LIKE trips INCLUDING DEFAULTS)
            ON COMMIT DELETE ROWS;
            """
        )

        raw_connection.commit()

        # Secondary indexes are unnecessary during the large bulk load.
        # Keep the primary-key index on ride_id for duplicate protection.
        if not smoke_test:
            cursor.execute(DROP_SECONDARY_INDEXES_SQL)
            raw_connection.commit()
            secondary_indexes_dropped = True

        total_read = 0
        total_valid = 0
        total_inserted = 0

        for zip_path in RAW_ZIP_FILES:
            print(f"\nZIP: {zip_path}")

            with zipfile.ZipFile(zip_path) as archive:
                csv_files = sorted(
                    name
                    for name in archive.namelist()
                    if name.lower().endswith(".csv")
                )

                for csv_file in csv_files:
                    print(f"  CSV: {csv_file}")

                    with archive.open(csv_file) as file:
                        chunks = pd.read_csv(
                            file,
                            chunksize=CHUNK_SIZE,
                            dtype={
                                "ride_id": "string",
                                "rideable_type": "string",
                                "start_station_name": "string",
                                "start_station_id": "string",
                                "end_station_name": "string",
                                "end_station_id": "string",
                                "member_casual": "string",
                            },
                        )

                        for chunk_number, chunk in enumerate(
                            chunks,
                            start=1,
                        ):
                            rows_read = len(chunk)
                            cleaned = clean_chunk(chunk)
                            rows_valid = len(cleaned)

                            total_read += rows_read
                            total_valid += rows_valid

                            inserted = copy_chunk(
                                cursor,
                                cleaned,
                            )

                            total_inserted += inserted

                            print(
                                f"    chunk {chunk_number}: "
                                f"read={rows_read:,} | "
                                f"valid={rows_valid:,} | "
                                f"inserted={inserted:,}"
                            )

                            if smoke_test:
                                print(
                                    "\nSMOKE TEST PASSED"
                                )
                                print(
                                    f"Rows successfully staged/"
                                    f"inserted: {inserted:,}"
                                )
                                print(
                                    "Rolling back test data..."
                                )

                                raw_connection.rollback()

                                print(
                                    "Smoke-test rows were NOT "
                                    "saved to PostgreSQL."
                                )
                                return

                            raw_connection.commit()

        print("\nRecreating secondary indexes...")

        restore_secondary_indexes(
            cursor,
            raw_connection,
            analyze=True,
        )
        secondary_indexes_dropped = False

        cursor.execute(
            "SELECT COUNT(*) FROM trips;"
        )

        database_rows = cursor.fetchone()[0]

        invalid_filtered = total_read - total_valid
        duplicates_skipped = total_valid - total_inserted

        print("\nLOAD COMPLETE")
        print(f"Rows read:                {total_read:,}")
        print(f"Rows valid:               {total_valid:,}")
        print(f"Invalid rows filtered:    {invalid_filtered:,}")
        print(f"Duplicate rows skipped:   {duplicates_skipped:,}")
        print(f"Rows inserted:            {total_inserted:,}")
        print(f"Rows in database:         {database_rows:,}")

    except Exception as load_error:
        raw_connection.rollback()

        # A committed index drop survives ingestion rollback. Restore the
        # indexes before propagating the original load failure.
        if secondary_indexes_dropped:
            try:
                restore_secondary_indexes(
                    cursor,
                    raw_connection,
                )
                secondary_indexes_dropped = False
            except Exception as restoration_error:
                raw_connection.rollback()
                raise ExceptionGroup(
                    "Trip load and secondary-index restoration both failed.",
                    [load_error, restoration_error],
                )

        raise

    finally:
        cursor.close()
        raw_connection.close()
        engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Process only one chunk and roll back "
            "without saving test rows."
        ),
    )

    args = parser.parse_args()

    load_all_trips(
        smoke_test=args.smoke_test
    )
