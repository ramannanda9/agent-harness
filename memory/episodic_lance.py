"""
LanceDB episodic store — production-grade vector storage.

Design decisions for performance:
  1. Embeddings generated locally via sentence-transformers (zero API cost,
     ~5ms per embed on CPU) OR via async batch API calls — pluggable embedder
  2. Writes are batched — Lance columnar format penalizes row-by-row writes
     heavily. Batch queue drains every flush_interval_seconds or flush_batch_size
  3. ANN index (IVF_PQ) created automatically after min_rows_for_index rows
     Below that threshold, brute-force scan is faster than index overhead
  4. Search is synchronous Lance scan wrapped in run_in_executor — non-blocking
  5. LanceDB is embedded — no server, no network hop, sits next to your data
     Storage path can point at S3/GCS for distributed access (Lance supports this)

Embedder protocol is pluggable:
  - LocalEmbedder: sentence-transformers, no API cost, ~384 dims
  - APIEmbedder: OpenAI/Cohere, batched, higher quality, costs money
  - MockEmbedder: random vectors for testing

Schema (Lance/Arrow):
  episode_id   string
  text         string
  embedding    fixed_size_list<float32>[384]
  metadata     string (JSON)
  timestamp    float64 (unix)
  agent_id     string
  memory_scope string
  memory_kind  string
  memory_key   string
  memory_policy string
  shared       bool
  active       bool
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pyarrow as pa

logger = logging.getLogger(__name__)

# ── Embedding dimension constants ─────────────────────────────────────────────
DIM_MINI = 384  # all-MiniLM-L6-v2 — fast, good quality
DIM_LARGE = 768  # all-mpnet-base-v2 — slower, better quality
DIM_OPENAI = 1536  # text-embedding-3-small
DIM_MOCK = 64  # for tests


# ── Embedder Protocol ─────────────────────────────────────────────────────────


@runtime_checkable
class Embedder(Protocol):
    dim: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


# ── Embedder Implementations ──────────────────────────────────────────────────


class MockEmbedder:
    """Random embeddings — for tests and local dev without ML deps."""

    dim = DIM_MOCK

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import random

        return [[random.gauss(0, 1) for _ in range(self.dim)] for _ in texts]


class LocalEmbedder:
    """
    sentence-transformers — zero API cost, runs on CPU.
    pip install sentence-transformers
    First call downloads model (~90MB for MiniLM).
    Subsequent calls: ~5ms per batch on CPU, ~0.5ms on GPU.
    """

    dim = DIM_MINI

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None  # lazy load

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
            logger.info("Loaded embedding model: %s (dim=%d)", self._model_name, self.dim)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self._load()
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: self._model.encode(texts, convert_to_numpy=True).tolist()
        )
        return embeddings


class OpenAIEmbedder:
    """
    OpenAI embeddings — higher quality, costs money.
    pip install openai
    Batched: max 2048 texts per API call.
    """

    dim = DIM_OPENAI

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        self._model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import openai

        client = openai.AsyncOpenAI()
        # batch in chunks of 2048
        all_embeddings = []
        for i in range(0, len(texts), 2048):
            chunk = texts[i : i + 2048]
            resp = await client.embeddings.create(input=chunk, model=self._model)
            all_embeddings.extend([e.embedding for e in resp.data])
        return all_embeddings


# ── Write Buffer ──────────────────────────────────────────────────────────────


@dataclass
class EpisodeRecord:
    episode_id: str
    text: str
    embedding: list[float]
    metadata: dict
    timestamp: float
    agent_id: str
    memory_scope: str
    memory_kind: str
    memory_key: str
    memory_policy: str
    shared: bool
    active: bool


class WriteBuffer:
    """
    Async batch write buffer.

    Lance columnar format is optimized for batch writes.
    Single-row writes create many small fragment files — bad for query performance.
    This buffer accumulates writes and flushes when:
      - flush_batch_size records accumulated, OR
      - flush_interval_seconds elapsed

    Flush happens in background — write() is non-blocking.
    """

    def __init__(
        self,
        flush_fn,  # async callable(list[EpisodeRecord])
        flush_batch_size: int = 50,
        flush_interval_seconds: float = 5.0,
    ) -> None:
        self._flush_fn = flush_fn
        self._flush_batch_size = flush_batch_size
        self._flush_interval = flush_interval_seconds
        self._buffer: list[EpisodeRecord] = []
        self._lock = asyncio.Lock()
        self._last_flush = time.time()
        self._flush_task: asyncio.Task | None = None

    async def add(self, record: EpisodeRecord) -> None:
        async with self._lock:
            self._buffer.append(record)
            should_flush = (
                len(self._buffer) >= self._flush_batch_size
                or (time.time() - self._last_flush) >= self._flush_interval
            )
        if should_flush:
            await self._flush()

    async def _flush(self) -> None:
        async with self._lock:
            if not self._buffer:
                return
            batch = list(self._buffer)
            self._buffer.clear()
            self._last_flush = time.time()

        try:
            await self._flush_fn(batch)
            logger.debug("WriteBuffer flushed %d records", len(batch))
        except Exception as e:
            logger.error("WriteBuffer flush failed: %s — records lost: %d", e, len(batch))

    async def flush_all(self) -> None:
        """Call at shutdown to drain remaining records."""
        await self._flush()


# ── LanceDB Episodic Store ────────────────────────────────────────────────────


# Arrow schema — fixed at creation time
# embedding dim must match embedder.dim
def _make_schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("episode_id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("embedding", pa.list_(pa.float32(), dim)),
            pa.field("metadata", pa.string()),  # JSON blob
            pa.field("timestamp", pa.float64()),
            pa.field("agent_id", pa.string()),
            pa.field("memory_scope", pa.string()),
            pa.field("memory_kind", pa.string()),
            pa.field("memory_key", pa.string()),
            pa.field("memory_policy", pa.string()),
            pa.field("shared", pa.bool_()),
            pa.field("active", pa.bool_()),
        ]
    )


def _sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _build_where(
    *,
    memory_scope: str | None,
    agent_id: str | None,
    include_shared: bool,
    include_legacy: bool,
) -> str:
    clauses: list[str] = []
    clauses.append("active = true")
    if memory_scope is not None:
        clauses.append(f"memory_scope = {_sql_quote(memory_scope)}")
    else:
        clauses.append("memory_scope = ''")
    if agent_id is not None:
        agent_parts = [f"agent_id = {_sql_quote(agent_id)}"]
        if include_shared:
            agent_parts.append("shared = true")
        if include_legacy:
            agent_parts.append("(memory_kind = '' AND agent_id = '')")
        clauses.append("(" + " OR ".join(agent_parts) + ")")
    return " AND ".join(clauses)


class LanceDBEpisodicStore:
    """
    LanceDB-backed episodic memory store.

    Features:
      - Embedded, no server — path can be local dir or s3://bucket/prefix
      - Async non-blocking writes via WriteBuffer
      - ANN index (IVF_PQ) auto-created after min_rows_for_index rows
      - Versioned — Lance keeps full history, can time-travel
      - Compaction — call compact() periodically to merge small fragments

    Usage:
        store = LanceDBEpisodicStore(
            uri="./lance_episodic",          # local dev
            # uri="s3://my-bucket/episodic", # production
            embedder=LocalEmbedder(),
        )
        await store.initialize()

        episode_id = await store.write("GPU worker failed", {"agent": "diagnosis"})
        results = await store.search("GPU latency spike", top_k=3)
    """

    MIN_ROWS_FOR_INDEX = 256  # below this, brute-force scan beats IVF_PQ overhead
    IVF_PARTITIONS = 32  # IVF partitions — sqrt(num_rows) is a good heuristic
    PQ_SUB_VECTORS = 16  # product quantization sub-vectors — dim/PQ_SUB_VECTORS >= 4

    def __init__(
        self,
        uri: str = "./lance_episodic",
        embedder: Embedder | None = None,
        table_name: str = "episodes",
        flush_batch_size: int = 50,
        flush_interval_seconds: float = 5.0,
    ) -> None:
        self._uri = uri
        self._embedder = embedder or MockEmbedder()
        self._table_name = table_name
        self._schema = _make_schema(self._embedder.dim)
        self._db = None
        self._table = None
        self._indexed = False
        self._write_buffer = WriteBuffer(
            flush_fn=self._write_batch,
            flush_batch_size=flush_batch_size,
            flush_interval_seconds=flush_interval_seconds,
        )

    async def initialize(self) -> None:
        """Must be called before first use — opens or creates the Lance table."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._open_or_create)
        logger.info(
            "LanceDBEpisodicStore initialized: uri=%s table=%s dim=%d",
            self._uri,
            self._table_name,
            self._embedder.dim,
        )

    def _open_or_create(self) -> None:
        import lancedb

        self._db = lancedb.connect(self._uri)
        existing = self._db.table_names()
        if self._table_name in existing:
            self._table = self._db.open_table(self._table_name)
            logger.debug(
                "Opened existing Lance table: %s (%d rows)",
                self._table_name,
                self._table.count_rows(),
            )
        else:
            # create with empty batch to establish schema
            empty = pa.table(
                {
                    "episode_id": pa.array([], type=pa.string()),
                    "text": pa.array([], type=pa.string()),
                    "embedding": pa.array([], type=pa.list_(pa.float32(), self._embedder.dim)),
                    "metadata": pa.array([], type=pa.string()),
                    "timestamp": pa.array([], type=pa.float64()),
                    "agent_id": pa.array([], type=pa.string()),
                    "memory_scope": pa.array([], type=pa.string()),
                    "memory_kind": pa.array([], type=pa.string()),
                    "memory_key": pa.array([], type=pa.string()),
                    "memory_policy": pa.array([], type=pa.string()),
                    "shared": pa.array([], type=pa.bool_()),
                    "active": pa.array([], type=pa.bool_()),
                }
            )
            self._table = self._db.create_table(self._table_name, data=empty)
            logger.debug("Created new Lance table: %s", self._table_name)

    # ── Write ─────────────────────────────────────────────────────────────────

    async def write(self, text: str, metadata: dict, agent_id: str = "") -> str:
        """
        Non-blocking write — buffered, flushed in batch.
        Returns episode_id immediately without waiting for flush.
        """
        episode_id = str(uuid.uuid4())

        # embed synchronously within async context — LocalEmbedder uses executor
        embeddings = await self._embedder.embed([text])
        embedding = embeddings[0]
        metadata = {**metadata, "active": metadata.get("active", True)}
        if metadata.get("memory_policy") == "latest" and metadata.get("memory_key"):
            await self._deactivate_memory_key(str(metadata["memory_key"]))

        record = EpisodeRecord(
            episode_id=episode_id,
            text=text,
            embedding=embedding,
            metadata=metadata,
            timestamp=time.time(),
            agent_id=agent_id,
            memory_scope=str(metadata.get("memory_scope") or ""),
            memory_kind=str(metadata.get("memory_kind") or ""),
            memory_key=str(metadata.get("memory_key") or ""),
            memory_policy=str(metadata.get("memory_policy") or ""),
            shared=bool(metadata.get("shared") is True),
            active=bool(metadata.get("active") is True),
        )
        # non-blocking — buffer handles flush
        await self._write_buffer.add(record)
        return episode_id

    async def _write_batch(self, records: list[EpisodeRecord]) -> None:
        """Batch write to Lance — called by WriteBuffer."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._do_write_batch(records))
        # check if we should build/update index
        await self._maybe_build_index()

    def _do_write_batch(self, records: list[EpisodeRecord]) -> None:
        batch = pa.table(
            {
                "episode_id": pa.array([r.episode_id for r in records], type=pa.string()),
                "text": pa.array([r.text for r in records], type=pa.string()),
                "embedding": pa.array(
                    [r.embedding for r in records],
                    type=pa.list_(pa.float32(), self._embedder.dim),
                ),
                "metadata": pa.array([json.dumps(r.metadata) for r in records], type=pa.string()),
                "timestamp": pa.array([r.timestamp for r in records], type=pa.float64()),
                "agent_id": pa.array([r.agent_id for r in records], type=pa.string()),
                "memory_scope": pa.array([r.memory_scope for r in records], type=pa.string()),
                "memory_kind": pa.array([r.memory_kind for r in records], type=pa.string()),
                "memory_key": pa.array([r.memory_key for r in records], type=pa.string()),
                "memory_policy": pa.array([r.memory_policy for r in records], type=pa.string()),
                "shared": pa.array([r.shared for r in records], type=pa.bool_()),
                "active": pa.array([r.active for r in records], type=pa.bool_()),
            }
        )
        self._table.add(batch)

    async def _deactivate_memory_key(self, memory_key: str) -> None:
        if self._table is None:
            return
        where = f"memory_key = {_sql_quote(memory_key)} AND active = true"
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, lambda: self._table.update(where=where, values={"active": False})
        )

    # ── Index Management ──────────────────────────────────────────────────────

    async def _maybe_build_index(self) -> None:
        """
        Build ANN index after MIN_ROWS_FOR_INDEX rows.
        IVF_PQ: Inverted File Index + Product Quantization
          - IVF clusters vectors into partitions (fast coarse search)
          - PQ compresses vectors for memory efficiency
          - Trade-off: ~1-5% recall loss vs exact search, 10-100x speed gain
        """
        if self._indexed:
            return
        loop = asyncio.get_event_loop()
        row_count = await loop.run_in_executor(None, self._table.count_rows)
        if row_count >= self.MIN_ROWS_FOR_INDEX:
            logger.info("Building IVF_PQ index on %d rows...", row_count)
            await loop.run_in_executor(None, self._build_index)
            self._indexed = True

    def _build_index(self) -> None:
        self._table.create_index(
            metric="cosine",
            vector_column_name="embedding",
            index_type="IVF_PQ",
            num_partitions=self.IVF_PARTITIONS,
            num_sub_vectors=self.PQ_SUB_VECTORS,
        )
        logger.info("IVF_PQ index built successfully")

    # ── Search ────────────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        top_k: int = 3,
        *,
        memory_scope: str | None = None,
        agent_id: str | None = None,
        include_shared: bool = True,
        include_legacy: bool = True,
    ) -> list[dict]:
        """
        ANN vector search — returns top_k most similar episodes.
        Uses IVF_PQ index if available, brute-force otherwise.
        Non-blocking: runs in executor.
        """
        if self._table is None:
            return []

        query_embeddings = await self._embedder.embed([query])
        query_vec = query_embeddings[0]

        loop = asyncio.get_event_loop()
        rows = await loop.run_in_executor(
            None,
            lambda: self._do_search(
                query_vec,
                top_k,
                memory_scope=memory_scope,
                agent_id=agent_id,
                include_shared=include_shared,
                include_legacy=include_legacy,
            ),
        )
        return rows

    def _do_search(
        self,
        query_vec: list[float],
        top_k: int,
        *,
        memory_scope: str | None,
        agent_id: str | None,
        include_shared: bool,
        include_legacy: bool,
    ) -> list[dict]:
        query = self._table.search(query_vec, vector_column_name="embedding")
        where = _build_where(
            memory_scope=memory_scope,
            agent_id=agent_id,
            include_shared=include_shared,
            include_legacy=include_legacy,
        )
        if where:
            query = query.where(where)
        results = query.limit(top_k).to_list()
        return [
            {
                "id": row["episode_id"],
                "text": row["text"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                "score": row.get("_distance", 0.0),
                "agent_id": row["agent_id"],
            }
            for row in results
        ]

    async def get(self, episode_id: str) -> dict | None:
        if self._table is None:
            return None
        loop = asyncio.get_event_loop()
        rows = await loop.run_in_executor(
            None,
            lambda: self._table.search().where(f"episode_id = '{episode_id}'").limit(1).to_list(),
        )
        if not rows:
            return None
        row = rows[0]
        return {
            "id": row["episode_id"],
            "text": row["text"],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
            "agent_id": row["agent_id"],
        }

    # ── Maintenance ───────────────────────────────────────────────────────────

    async def compact(self) -> None:
        """
        Merge small Lance fragment files into larger ones.
        Call periodically (e.g. daily) — improves scan performance significantly
        when write buffer has been flushing many small batches.
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._table.compact_files)
        logger.info("Lance compaction complete")

    async def flush(self) -> None:
        """Drain write buffer — call at shutdown."""
        await self._write_buffer.flush_all()

    async def count(self) -> int:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._table.count_rows)

    def versions(self) -> list[dict]:
        """Lance version history — useful for debugging memory evolution."""
        return self._table.list_versions()
