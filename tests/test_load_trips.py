import pandas as pd
import pytest

import scripts.load_trips as load_trips
from scripts.load_trips import COLUMNS, clean_chunk


class FakeCursor:
    """Record database operations without touching PostgreSQL."""

    def __init__(self, fail_index_restore=False):
        self.statements = []
        self.fail_index_restore = fail_index_restore
        self.closed = False

    def execute(self, statement):
        self.statements.append(statement)

        if (
            self.fail_index_restore
            and "CREATE INDEX IF NOT EXISTS" in statement
        ):
            raise RuntimeError("index restoration failed")

    def close(self):
        self.closed = True


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True


class FakeEngine:
    def __init__(self, connection):
        self._connection = connection
        self.disposed = False

    def raw_connection(self):
        return self._connection

    def dispose(self):
        self.disposed = True


def prepare_failed_load(monkeypatch, tmp_path, fail_index_restore=False):
    """Configure an invalid ZIP so failure occurs after index removal."""

    zip_paths = [tmp_path / f"month-{number}.zip" for number in range(3)]

    for path in zip_paths:
        path.write_bytes(b"not a zip file")

    cursor = FakeCursor(fail_index_restore=fail_index_restore)
    connection = FakeConnection(cursor)
    engine = FakeEngine(connection)

    monkeypatch.setattr(load_trips, "RAW_ZIP_FILES", zip_paths)
    monkeypatch.setattr(
        load_trips,
        "get_database_engine",
        lambda: engine,
    )

    return cursor, connection, engine


def test_clean_chunk_removes_invalid_trips_and_preserves_station_ids():
    """Protect timestamp validation, duration checks, and ID semantics."""

    rows = [
        {
            "ride_id": "valid",
            "started_at": "2026-04-01 10:00:00",
            "ended_at": "2026-04-01 10:15:00",
            "start_station_id": 101,
            "end_station_id": 202,
        },
        {
            "ride_id": "bad-time",
            "started_at": "not-a-time",
            "ended_at": "2026-04-01 10:15:00",
            "start_station_id": 303,
            "end_station_id": 404,
        },
        {
            "ride_id": "bad-duration",
            "started_at": "2026-04-01 10:15:00",
            "ended_at": "2026-04-01 10:15:00",
            "start_station_id": 505,
            "end_station_id": 606,
        },
    ]

    df = pd.DataFrame(rows).reindex(columns=COLUMNS)
    result = clean_chunk(df)

    assert result["ride_id"].tolist() == ["valid"]
    assert result.iloc[0]["start_station_id"] == "101"
    assert result.iloc[0]["end_station_id"] == "202"
    assert str(result["start_station_id"].dtype) == "string"
    assert str(result["end_station_id"].dtype) == "string"


def test_failed_full_load_restores_indexes_and_reraises_original(
    monkeypatch,
    tmp_path,
):
    """Protect index availability when full ingestion fails mid-run."""

    cursor, connection, engine = prepare_failed_load(
        monkeypatch,
        tmp_path,
    )

    with pytest.raises(load_trips.zipfile.BadZipFile):
        load_trips.load_all_trips()

    executed_sql = "\n".join(cursor.statements)
    assert load_trips.DROP_SECONDARY_INDEXES_SQL in executed_sql
    assert load_trips.CREATE_SECONDARY_INDEXES_SQL in executed_sql
    assert connection.commits == 3
    assert connection.rollbacks == 1
    assert cursor.closed
    assert connection.closed
    assert engine.disposed


def test_failed_smoke_test_does_not_manage_secondary_indexes(
    monkeypatch,
    tmp_path,
):
    """Protect smoke tests from changing persistent index state."""

    cursor, connection, _ = prepare_failed_load(
        monkeypatch,
        tmp_path,
    )

    with pytest.raises(load_trips.zipfile.BadZipFile):
        load_trips.load_all_trips(smoke_test=True)

    executed_sql = "\n".join(cursor.statements)
    assert load_trips.DROP_SECONDARY_INDEXES_SQL not in executed_sql
    assert load_trips.CREATE_SECONDARY_INDEXES_SQL not in executed_sql
    assert connection.commits == 1
    assert connection.rollbacks == 1


def test_dual_failure_preserves_load_and_restoration_errors(
    monkeypatch,
    tmp_path,
):
    """Protect diagnostic context when load and cleanup both fail."""

    prepare_failed_load(
        monkeypatch,
        tmp_path,
        fail_index_restore=True,
    )

    with pytest.raises(ExceptionGroup) as captured:
        load_trips.load_all_trips()

    errors = captured.value.exceptions
    assert isinstance(errors[0], load_trips.zipfile.BadZipFile)
    assert isinstance(errors[1], RuntimeError)
    assert str(errors[1]) == "index restoration failed"
