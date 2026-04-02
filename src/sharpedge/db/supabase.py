"""
Supabase database backend for SharpEdge.

Supabase is managed PostgreSQL, so SQLAlchemy is the primary access path.
This module provides:

  1. get_supabase_url()    — construct the connection string from settings
  2. get_engine()          — SQLAlchemy engine tuned for Supabase's pooler
  3. get_session()         — context-manager session (replaces engine.py usage)
  4. SupabaseClient        — optional REST API wrapper for direct table access

Configuration (.env / environment variables)
--------------------------------------------
Option A — full DATABASE_URL (takes priority):
    DATABASE_URL=postgresql://postgres.[ref]:[password]@aws-0-[region].pooler.supabase.com:6543/postgres

Option B — individual Supabase settings:
    SUPABASE_URL=https://[ref].supabase.co
    SUPABASE_KEY=<service_role or anon key>
    SUPABASE_PROJECT_REF=<project ref>
    SUPABASE_DB_PASSWORD=<database password>
    SUPABASE_REGION=us-east-1  (default)

Option C — local Postgres (development):
    DATABASE_URL=postgresql://localhost:5432/sharpedge

The fields to add to Settings (config.py):
    supabase_url: str = ""
    supabase_key: str = ""
    supabase_project_ref: str = ""
    supabase_db_password: str = ""
    supabase_region: str = "us-east-1"
"""

import logging
from contextlib import contextmanager
from typing import Any, Generator, Optional

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from sharpedge.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Connection string resolution
# ---------------------------------------------------------------------------

def get_supabase_url() -> str:
    """Return a valid PostgreSQL connection string.

    Resolution order:
    1. settings.database_url already looks like a postgresql:// URL — use it.
    2. Build from SUPABASE_PROJECT_REF + SUPABASE_DB_PASSWORD + SUPABASE_REGION.
    3. Fall back to settings.database_url as-is (handles SQLite or local PG).

    The Supabase connection pooler has two modes:
      - Port 6543 (transaction mode) — best for serverless / short-lived sessions
      - Port 5432 (session mode)     — required for LISTEN/NOTIFY and temp tables
    Transaction mode (6543) is used by default.

    Returns
    -------
    str
        A fully-qualified postgresql:// connection string.

    Raises
    ------
    ValueError
        If no valid connection string can be constructed.
    """
    db_url: str = getattr(settings, "database_url", "")

    # --- Option A: DATABASE_URL already set to a postgres URL ---------------
    if db_url and db_url.startswith(("postgresql://", "postgresql+psycopg2://",
                                      "postgresql+asyncpg://")):
        logger.debug("[supabase] Using DATABASE_URL directly.")
        return db_url

    # --- Option B: build from Supabase project settings --------------------
    project_ref: str = getattr(settings, "supabase_project_ref", "")
    db_password: str = getattr(settings, "supabase_db_password", "")
    region: str = getattr(settings, "supabase_region", "us-east-1")

    if project_ref and db_password:
        # Transaction-mode pooler (port 6543) — works with pgBouncer
        url = (
            f"postgresql://postgres.{project_ref}:{db_password}"
            f"@aws-0-{region}.pooler.supabase.com:6543/postgres"
        )
        logger.debug(
            f"[supabase] Built Supabase URL for project {project_ref!r} "
            f"in region {region!r}."
        )
        return url

    # --- Option C: fall back to whatever DATABASE_URL is set ---------------
    if db_url:
        logger.warning(
            "[supabase] DATABASE_URL does not look like a PostgreSQL URL — "
            "using it anyway. Set SUPABASE_PROJECT_REF + SUPABASE_DB_PASSWORD "
            "for automatic Supabase URL construction."
        )
        return db_url

    raise ValueError(
        "Cannot determine a database URL. "
        "Set DATABASE_URL, or SUPABASE_PROJECT_REF + SUPABASE_DB_PASSWORD."
    )


# ---------------------------------------------------------------------------
# SQLAlchemy engine
# ---------------------------------------------------------------------------

def _is_supabase_url(url: str) -> bool:
    """Return True if the URL points at a Supabase-hosted database."""
    return "supabase.com" in url or "supabase.co" in url


def get_engine(url: Optional[str] = None, **kwargs: Any) -> Engine:
    """Create and return a SQLAlchemy Engine configured for Supabase (or local PG).

    Parameters
    ----------
    url:
        Override the connection string (defaults to get_supabase_url()).
    **kwargs:
        Extra keyword arguments forwarded to ``create_engine()``.

    Engine configuration
    --------------------
    - pool_size=5          — keep 5 persistent connections
    - max_overflow=10      — allow 10 extra burst connections
    - pool_timeout=30      — wait up to 30 s for a connection
    - pool_pre_ping=True   — validate connections before handing them out
    - SSL required         — enforced for Supabase; omitted for local URLs

    Returns
    -------
    sqlalchemy.engine.Engine
    """
    resolved_url = url or get_supabase_url()
    supabase = _is_supabase_url(resolved_url)

    pool_kwargs: dict[str, Any] = {
        "poolclass": QueuePool,
        "pool_size": 5,
        "max_overflow": 10,
        "pool_timeout": 30,
        "pool_pre_ping": True,
        "echo": False,
    }

    connect_args: dict[str, Any] = {}
    if supabase:
        # Supabase requires SSL; reject non-SSL connections
        connect_args["sslmode"] = "require"
        logger.info("[supabase] SSL mode: require (Supabase endpoint detected).")
    else:
        logger.info("[supabase] SSL mode: disabled (local/non-Supabase URL).")

    if connect_args:
        pool_kwargs["connect_args"] = connect_args

    # Allow callers to override any pool/engine setting
    pool_kwargs.update(kwargs)

    engine = create_engine(resolved_url, **pool_kwargs)

    # Warm up: verify the engine can reach the database at startup
    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_conn, connection_record):  # noqa: ARG001
        logger.debug("[supabase] New database connection established.")

    logger.info(f"[supabase] Engine created (pool_size=5, max_overflow=10).")
    return engine


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

# Module-level engine and session factory — initialised lazily on first use
_engine: Optional[Engine] = None
_SessionFactory: Optional[sessionmaker] = None


def _get_session_factory() -> sessionmaker:
    """Return the module-level session factory, creating it on first call."""
    global _engine, _SessionFactory
    if _SessionFactory is None:
        _engine = get_engine()
        _SessionFactory = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
    return _SessionFactory


def get_session() -> Session:
    """Return a new SQLAlchemy Session.

    Compatible with the existing engine.py get_session() signature so this
    module can be used as a drop-in replacement.

    Caller is responsible for calling session.close() (or using
    get_session_ctx() context manager instead).

    Returns
    -------
    sqlalchemy.orm.Session
    """
    return _get_session_factory()()


@contextmanager
def get_session_ctx() -> Generator[Session, None, None]:
    """Context-manager variant of get_session().

    Usage::

        with get_session_ctx() as session:
            results = session.execute(text("SELECT 1")).all()

    Commits on clean exit, rolls back on exception, always closes.
    """
    session: Session = _get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# SupabaseClient — REST API wrapper (fallback / convenience)
# ---------------------------------------------------------------------------

class SupabaseClient:
    """Thin wrapper around the Supabase REST API (PostgREST).

    This is a fallback for cases where raw HTTP access is preferable to
    SQLAlchemy (e.g., edge functions, minimal-dependency scripts).

    Primary database access should still use SQLAlchemy via get_session()
    or get_session_ctx().

    Configuration required in settings:
        supabase_url  — e.g. https://[ref].supabase.co
        supabase_key  — service_role or anon key

    Parameters
    ----------
    url:
        Override settings.supabase_url.
    key:
        Override settings.supabase_key.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        key: Optional[str] = None,
    ) -> None:
        self._base_url: str = (
            (url or getattr(settings, "supabase_url", "")).rstrip("/")
        )
        self._key: str = key or getattr(settings, "supabase_key", "")

        if not self._base_url or not self._key:
            raise ValueError(
                "SupabaseClient requires supabase_url and supabase_key. "
                "Set SUPABASE_URL and SUPABASE_KEY in .env, or pass them "
                "explicitly."
            )

        # Import lazily so the module doesn't hard-depend on httpx at import time
        try:
            import httpx as _httpx
            self._httpx = _httpx
        except ImportError as exc:
            raise ImportError(
                "httpx is required for SupabaseClient. "
                "Install it with: pip install httpx"
            ) from exc

        self._rest_url = f"{self._base_url}/rest/v1"
        self._headers = {
            "apikey": self._key,
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }

    # ------------------------------------------------------------------
    # Internal HTTP helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        return self._httpx.Client(
            headers=self._headers,
            timeout=30.0,
        )

    def _url(self, table: str) -> str:
        return f"{self._rest_url}/{table}"

    # ------------------------------------------------------------------
    # Public CRUD methods
    # ------------------------------------------------------------------

    def select(
        self,
        table: str,
        query: Optional[dict[str, str]] = None,
        columns: str = "*",
    ) -> list[dict]:
        """SELECT rows from a table.

        Parameters
        ----------
        table:
            PostgREST table name.
        query:
            Optional dict of PostgREST filter params, e.g.
            ``{"home_team": "eq.Arsenal", "league": "eq.Premier League"}``.
        columns:
            Comma-separated column list (default ``"*"``).

        Returns
        -------
        list[dict]
            Deserialized JSON rows.

        Raises
        ------
        httpx.HTTPStatusError
            On non-2xx HTTP responses.
        """
        params: dict[str, str] = {"select": columns}
        if query:
            params.update(query)

        with self._get_client() as client:
            response = client.get(self._url(table), params=params)
            response.raise_for_status()
            return response.json()

    def insert(
        self,
        table: str,
        data: list[dict] | dict,
    ) -> list[dict]:
        """INSERT one or more rows into a table.

        Parameters
        ----------
        table:
            Target table name.
        data:
            A single row dict or a list of row dicts.

        Returns
        -------
        list[dict]
            The inserted rows as returned by Supabase.
        """
        payload = data if isinstance(data, list) else [data]

        with self._get_client() as client:
            response = client.post(self._url(table), json=payload)
            response.raise_for_status()
            return response.json()

    def upsert(
        self,
        table: str,
        data: list[dict] | dict,
        on_conflict: str = "",
    ) -> list[dict]:
        """UPSERT one or more rows (INSERT OR UPDATE on conflict).

        Parameters
        ----------
        table:
            Target table name.
        data:
            A single row dict or a list of row dicts.
        on_conflict:
            Comma-separated column name(s) that form the unique constraint
            for conflict resolution.  If empty, Supabase uses the table's
            primary key.

        Returns
        -------
        list[dict]
            The upserted rows as returned by Supabase.
        """
        payload = data if isinstance(data, list) else [data]
        headers: dict[str, str] = {
            "Prefer": "resolution=merge-duplicates,return=representation",
        }
        params: dict[str, str] = {}
        if on_conflict:
            params["on_conflict"] = on_conflict

        with self._get_client() as client:
            response = client.post(
                self._url(table),
                json=payload,
                headers=headers,
                params=params,
            )
            response.raise_for_status()
            return response.json()

    def delete(
        self,
        table: str,
        query: dict[str, str],
    ) -> list[dict]:
        """DELETE rows matching query filters.

        Parameters
        ----------
        table:
            Target table name.
        query:
            PostgREST filter params identifying rows to delete, e.g.
            ``{"id": "eq.42"}``.

        Returns
        -------
        list[dict]
            The deleted rows (Supabase returns them with Prefer: return=representation).
        """
        if not query:
            raise ValueError(
                "delete() requires at least one filter in 'query' to prevent "
                "accidentally wiping the entire table."
            )

        with self._get_client() as client:
            response = client.delete(self._url(table), params=query)
            response.raise_for_status()
            return response.json()

    def ping(self) -> bool:
        """Check REST API connectivity.

        Returns True if the REST endpoint responds, False otherwise.
        """
        try:
            with self._get_client() as client:
                # PostgREST root always returns a 200 with the OpenAPI schema
                response = client.get(self._rest_url)
                return response.status_code == 200
        except Exception as exc:
            logger.warning(f"[SupabaseClient] ping failed: {exc}")
            return False
