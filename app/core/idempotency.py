"""Layer 12: SQLite-backed Idempotency Key Shield to prevent duplicate tool/action execution."""

import json
import sqlite3
import time
from pathlib import Path
from typing import Any


class IdempotencyLedger:
    """Guarantees state-mutating requests or tool calls with the same key execute exactly once."""

    def __init__(self, db_path: str = "data/system_design_state.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS idempotency_keys (
                    idem_key TEXT PRIMARY KEY,
                    response_json TEXT,
                    created_at REAL
                )"""
            )

    def get(self, idem_key: str | None) -> dict[str, Any] | None:
        if not idem_key:
            return None
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT response_json FROM idempotency_keys WHERE idem_key = ?", (idem_key,)
            ).fetchone()
            if row:
                data = json.loads(row[0])
                data["_idempotent_replay"] = True
                return data
        return None

    def save(self, idem_key: str | None, payload: dict[str, Any]) -> None:
        if not idem_key:
            return
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO idempotency_keys (idem_key, response_json, created_at) VALUES (?, ?, ?)",
                (idem_key, json.dumps(payload), time.time()),
            )
