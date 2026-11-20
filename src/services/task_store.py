from psycopg import AsyncConnection
from psycopg.types.json import Jsonb

from src.config import DatabaseSettings
from src.schemas import TaskResult


async def connect(settings: DatabaseSettings) -> AsyncConnection:
    return await AsyncConnection.connect(
        host=settings.host,
        port=settings.port,
        user=settings.username,
        password=settings.password,
        dbname=settings.database_name,
        connect_timeout=5,
    )


async def initialize_tables(settings: DatabaseSettings) -> None:
    async with await connect(settings) as conn:
        # Несколько воркеров могут одновременно стартовать на пустой базе.
        await conn.execute("SELECT pg_advisory_xact_lock(5278004)")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS inference_results (
                task_id text PRIMARY KEY,
                outcome jsonb NOT NULL,
                processing_seconds double precision NOT NULL,
                submitted_at timestamptz NOT NULL,
                completed_at timestamptz NOT NULL DEFAULT clock_timestamp()
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS inference_failures (
                message_id text PRIMARY KEY, attempts integer NOT NULL
            )
        """)


async def save_result(
    settings: DatabaseSettings, result: TaskResult, submitted_at: float
) -> None:
    async with await connect(settings) as conn:
        await conn.execute(
            """INSERT INTO inference_results
                   (task_id, outcome, processing_seconds, submitted_at)
               VALUES (%s, %s, %s, to_timestamp(%s))
               ON CONFLICT (task_id) DO NOTHING""",
            (
                result.task_id,
                Jsonb(result.outcome.model_dump()),
                result.processing_seconds,
                submitted_at,
            ),
        )


async def record_failure(settings: DatabaseSettings, message_id: str) -> int:
    async with await connect(settings) as conn:
        cur = await conn.execute(
            """INSERT INTO inference_failures VALUES (%s, 1)
               ON CONFLICT (message_id) DO UPDATE
               SET attempts = inference_failures.attempts + 1 RETURNING attempts""",
            (message_id,),
        )
        row = await cur.fetchone()
        assert row is not None
        return int(row[0])
