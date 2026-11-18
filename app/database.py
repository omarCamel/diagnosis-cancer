import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


def _database_path() -> Path:
    configured_path = os.getenv("PREDICTIONS_DB_PATH", "").strip()
    path = Path(configured_path) if configured_path else BASE_DIR / "predictions.db"
    if not path.is_absolute():
        path = BASE_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def initialize_database() -> None:
    with closing(sqlite3.connect(_database_path(), timeout=10)) as connection:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    input_features TEXT NOT NULL,
                    prediction INTEGER NOT NULL CHECK (prediction IN (0, 1)),
                    diagnosis_probability REAL NOT NULL
                        CHECK (diagnosis_probability >= 0 AND diagnosis_probability <= 1),
                    risk_level TEXT NOT NULL
                        CHECK (risk_level IN ('Bajo', 'Medio', 'Alto')),
                    message TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_predictions_created_at "
                "ON predictions(created_at)"
            )


def save_prediction(input_features: dict[str, float], result: dict[str, object]) -> None:
    created_at = datetime.now(timezone.utc).isoformat()
    serialized_features = json.dumps(
        input_features,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )

    with closing(sqlite3.connect(_database_path(), timeout=10)) as connection:
        with connection:
            connection.execute(
                """
                INSERT INTO predictions (
                    created_at,
                    input_features,
                    prediction,
                    diagnosis_probability,
                    risk_level,
                    message
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    serialized_features,
                    result["prediction"],
                    result["diagnosis_probability"],
                    result["risk_level"],
                    result["message"],
                ),
            )


def get_predictions(limit: int, offset: int) -> list[dict[str, object]]:
    with closing(sqlite3.connect(_database_path(), timeout=10)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT id, created_at, input_features, prediction,
                   diagnosis_probability, risk_level, message
            FROM predictions
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()

    return [
        {
            "id": row["id"],
            "created_at": row["created_at"],
            "input_features": json.loads(row["input_features"]),
            "prediction": row["prediction"],
            "diagnosis_probability": row["diagnosis_probability"],
            "risk_level": row["risk_level"],
            "message": row["message"],
        }
        for row in rows
    ]
