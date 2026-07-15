"""Deterministic synthetic behavioral telemetry for engineering tests only."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from app.ml.preprocessing.behavioral.constants import BEHAVIORAL_SYNTHETIC_SCHEMA_VERSION


def generate_synthetic_behavioral_events(
    *,
    participant_count: int = 4,
    sessions_per_participant: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = datetime(2025, 3, 1, 9, 0, tzinfo=timezone.utc)
    rows = []
    scenarios = ["stable_baseline", "slower_typing", "increased_hesitation", "sparse_use"]
    for pidx in range(participant_count):
        participant = f"SYN-BEH-{pidx + 1:03d}"
        scenario = scenarios[pidx % len(scenarios)]
        session_limit = 1 if scenario == "sparse_use" else sessions_per_participant
        for sidx in range(session_limit):
            session_id = f"syn-session-{pidx + 1:03d}-{sidx + 1:02d}"
            start = base + timedelta(days=sidx, hours=pidx)
            typing_speed = 290 if scenario != "slower_typing" else 180
            hesitation = 2 if scenario != "increased_hesitation" else 8
            for eidx, event_type in enumerate(["session_start", "typing_timing", "typing_timing", "mouse_aggregate", "prompt_response", "session_end"]):
                timestamp = start + timedelta(seconds=eidx * 45)
                row = {
                    "participant_id": participant,
                    "event_timestamp": timestamp.isoformat(),
                    "session_id": session_id,
                    "event_type": event_type,
                    "page_or_context": "checkin" if eidx < 4 else "summary",
                    "response_latency_ms": None,
                    "key_dwell_time_ms": None,
                    "key_flight_time_ms": None,
                    "typing_speed_cpm": None,
                    "backspace_count": None,
                    "correction_count": None,
                    "mouse_distance_px": None,
                    "mouse_speed_px_per_second": None,
                    "click_count": None,
                    "hesitation_count": None,
                    "session_duration_seconds": 270,
                    "engineering_scenario": scenario,
                    "synthetic": True,
                    "synthetic_schema_version": BEHAVIORAL_SYNTHETIC_SCHEMA_VERSION,
                }
                if event_type == "typing_timing":
                    row.update(
                        {
                            "key_dwell_time_ms": int(rng.normal(90, 10)),
                            "key_flight_time_ms": int(rng.normal(140 if scenario != "increased_hesitation" else 650, 40)),
                            "typing_speed_cpm": int(rng.normal(typing_speed, 15)),
                            "backspace_count": int(rng.poisson(1)),
                            "correction_count": int(rng.poisson(1)),
                            "hesitation_count": hesitation,
                        }
                    )
                elif event_type == "mouse_aggregate":
                    row.update(
                        {
                            "mouse_distance_px": int(rng.normal(1200, 100)),
                            "mouse_speed_px_per_second": int(rng.normal(850, 80)),
                            "click_count": int(rng.poisson(3)),
                            "hesitation_count": hesitation,
                        }
                    )
                elif event_type == "prompt_response":
                    row.update({"response_latency_ms": int(rng.normal(4000 if scenario != "increased_hesitation" else 9000, 500))})
                rows.append(row)
    return pd.DataFrame(rows)


def synthetic_behavioral_fingerprint(df: pd.DataFrame) -> str:
    payload = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

