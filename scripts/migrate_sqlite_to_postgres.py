import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

from sqlalchemy import MetaData, create_engine, func, select, text
from sqlalchemy.engine import Engine

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import settings


DEPENDENCY_ORDER = [
    "users",
    "model_registry",
    "resources",
    "profile_assessments",
    "daily_checkins",
    "dass21_assessments",
    "assessments",
    "assessment_history",
    "alerts",
    "chat_messages",
    "counselor_sessions",
    "safetalk_bot_messages",
    "feature_snapshots",
    "modality_predictions",
    "risk_assessments",
    "risk_assessment_inputs",
    "alert_events",
    "worker_jobs",
]


def chunked(rows: List[dict], size: int) -> Iterable[List[dict]]:
    for index in range(0, len(rows), size):
        yield rows[index : index + size]


def normalize_sqlite_url(value: str) -> str:
    if value.startswith("sqlite"):
        return value
    return f"sqlite:///{Path(value).resolve()}"


def reflect(engine: Engine) -> MetaData:
    metadata = MetaData()
    metadata.reflect(bind=engine)
    return metadata


def count_rows(engine: Engine, metadata: MetaData, table_name: str) -> int:
    table = metadata.tables.get(table_name)
    if table is None:
        return 0
    with engine.connect() as connection:
        return connection.execute(select(func.count()).select_from(table)).scalar_one()


def fetch_rows(engine: Engine, metadata: MetaData, table_name: str) -> List[dict]:
    table = metadata.tables[table_name]
    with engine.connect() as connection:
        result = connection.execute(select(table))
        return [dict(row._mapping) for row in result]


def reset_postgres_sequence(connection, table_name: str, id_column: str = "id") -> None:
    sequence = connection.execute(
        text("SELECT pg_get_serial_sequence(:table_name, :column_name)"),
        {"table_name": table_name, "column_name": id_column},
    ).scalar()
    if not sequence:
        return

    connection.execute(
        text(
            """
            SELECT setval(
                :sequence_name,
                COALESCE((SELECT MAX(id) FROM "{table_name}"), 1),
                COALESCE((SELECT MAX(id) FROM "{table_name}"), 0) > 0
            )
            """.format(table_name=table_name)
        ),
        {"sequence_name": sequence},
    )


def migrate(sqlite_url: str, postgres_url: str, dry_run: bool, chunk_size: int) -> dict:
    sqlite_engine = create_engine(normalize_sqlite_url(sqlite_url))
    sqlite_metadata = reflect(sqlite_engine)

    report = {
        "started_at": datetime.utcnow().isoformat(),
        "dry_run": dry_run,
        "sqlite_url": sqlite_url,
        "postgres_url": postgres_url if postgres_url else None,
        "tables": [],
        "total_source_rows": 0,
        "total_inserted_rows": 0,
        "status": "planned" if dry_run else "running",
    }

    if dry_run:
        for table_name in DEPENDENCY_ORDER:
            source_count = count_rows(sqlite_engine, sqlite_metadata, table_name)
            report["tables"].append(
                {
                    "table": table_name,
                    "source_rows": source_count,
                    "inserted_rows": 0,
                    "status": "would_copy" if table_name in sqlite_metadata.tables else "missing_in_sqlite",
                }
            )
            report["total_source_rows"] += source_count
        report["status"] = "dry_run_complete"
        report["completed_at"] = datetime.utcnow().isoformat()
        return report

    if not postgres_url.startswith("postgresql"):
        raise ValueError("postgres_url must be a PostgreSQL SQLAlchemy URL for a real migration")

    postgres_engine = create_engine(postgres_url, pool_pre_ping=True)
    postgres_metadata = reflect(postgres_engine)

    with postgres_engine.begin() as connection:
        for table_name in DEPENDENCY_ORDER:
            if table_name not in sqlite_metadata.tables:
                report["tables"].append(
                    {"table": table_name, "source_rows": 0, "inserted_rows": 0, "status": "missing_in_sqlite"}
                )
                continue
            if table_name not in postgres_metadata.tables:
                raise RuntimeError(f"Destination table does not exist in PostgreSQL: {table_name}")

            source_rows = fetch_rows(sqlite_engine, sqlite_metadata, table_name)
            destination_table = postgres_metadata.tables[table_name]
            destination_columns = {column.name for column in destination_table.columns}
            filtered_rows = [
                {key: value for key, value in row.items() if key in destination_columns}
                for row in source_rows
            ]

            inserted_rows = 0
            for row_chunk in chunked(filtered_rows, chunk_size):
                if row_chunk:
                    connection.execute(destination_table.insert(), row_chunk)
                    inserted_rows += len(row_chunk)

            if "id" in destination_columns:
                reset_postgres_sequence(connection, table_name)

            report["tables"].append(
                {
                    "table": table_name,
                    "source_rows": len(source_rows),
                    "inserted_rows": inserted_rows,
                    "status": "copied",
                }
            )
            report["total_source_rows"] += len(source_rows)
            report["total_inserted_rows"] += inserted_rows

    report["status"] = "complete"
    report["completed_at"] = datetime.utcnow().isoformat()
    return report


def write_report(report: dict, report_path: str) -> None:
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate existing SQLite data to PostgreSQL.")
    parser.add_argument("--sqlite-url", default="sqlite:///./suicideprevention.db")
    parser.add_argument("--postgres-url", default=settings.DATABASE_URL)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--report-path", default="./migration_report.json")
    args = parser.parse_args()

    report = migrate(
        sqlite_url=args.sqlite_url,
        postgres_url=args.postgres_url,
        dry_run=args.dry_run,
        chunk_size=args.chunk_size,
    )
    write_report(report, args.report_path)

    print(f"Migration status: {report['status']}")
    for item in report["tables"]:
        print(
            f"{item['table']}: source={item['source_rows']} inserted={item['inserted_rows']} status={item['status']}"
        )
    print(f"Report written to: {args.report_path}")


if __name__ == "__main__":
    main()
