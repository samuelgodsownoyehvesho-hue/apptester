"""Database engine and session handling.

SQLite rather than Postgres because Phase 1 is single-machine and Docker is not
available. The session API here is deliberately storage-agnostic, so moving to
Postgres later is a connection-string change rather than a rewrite.

Two SQLite-specific settings are applied because their absence causes confusing
failures rather than obvious ones: foreign keys are off by default in SQLite,
and the default journal mode serialises readers against the writer, which the
browser agent would hit immediately.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from crucible.core.logging import get_logger
from crucible.store.models import Base

logger = get_logger(__name__)

DEFAULT_DB_URL = "sqlite:///./crucible.db"


def _is_sqlite(url: str) -> bool:
    return url.startswith("sqlite")


def make_engine(db_url: str = DEFAULT_DB_URL, *, echo: bool = False) -> Engine:
    """Create an engine, applying the SQLite pragmas we depend on."""
    connect_args: dict[str, object] = {}
    if _is_sqlite(db_url):
        # The worker pool and the CLI may touch the database concurrently.
        connect_args["check_same_thread"] = False

    engine = create_engine(db_url, echo=echo, future=True, connect_args=connect_args)

    if _is_sqlite(db_url):

        @event.listens_for(engine, "connect")
        def _set_pragmas(dbapi_connection: object, _record: object) -> None:
            cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
            try:
                # Off by default in SQLite, which would let orphaned rows
                # accumulate silently instead of failing loudly.
                cursor.execute("PRAGMA foreign_keys=ON")
                # Lets reads proceed during a write, instead of raising
                # "database is locked" under concurrent access.
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
            finally:
                cursor.close()

    return engine


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Return a session factory bound to ``engine``."""
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def init_db(engine: Engine) -> None:
    """Create any missing tables.

    Adequate for Phase 1. Once the schema has data worth preserving this should
    become Alembic migrations; ``create_all`` will not alter existing tables.
    """
    Base.metadata.create_all(engine)
    logger.debug("db_initialised tables=%d", len(Base.metadata.tables))


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """Provide a transactional session scope.

    Commits on success, rolls back on failure, and always closes. Callers that
    manage their own transaction should use the factory directly.
    """
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def ensure_parent_dir(db_url: str) -> None:
    """Create the directory for a SQLite file URL if it does not exist."""
    prefix = "sqlite:///"
    if not db_url.startswith(prefix):
        return
    raw = db_url[len(prefix) :]
    if not raw or raw == ":memory:":
        return
    Path(raw).expanduser().parent.mkdir(parents=True, exist_ok=True)
