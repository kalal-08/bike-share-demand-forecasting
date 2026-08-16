import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

load_dotenv()


def get_database_engine():
    """Create a SQLAlchemy engine for the project PostgreSQL database."""

    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    database = os.getenv("DB_NAME")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")

    required = {
        "DB_HOST": host,
        "DB_PORT": port,
        "DB_NAME": database,
        "DB_USER": user,
        "DB_PASSWORD": password,
    }

    missing = [
        name
        for name, value in required.items()
        if not value
    ]

    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}"
        )

    url = URL.create(
        drivername="postgresql+psycopg2",
        username=user,
        password=password,
        host=host,
        port=int(port),
        database=database,
    )

    return create_engine(url)


def load_hourly_demand():
    """Load the aggregated station-hour demand dataset."""

    engine = get_database_engine()

    query = text(
        """
        SELECT
            station_id,
            hour,
            departures,
            arrivals,
            net_flow
        FROM station_hourly_demand
        ORDER BY station_id, hour
        """
    )

    with engine.connect() as connection:
        df = pd.read_sql(
            query,
            connection,
        )

    engine.dispose()

    return df
