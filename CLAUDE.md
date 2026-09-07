# CLAUDE.md

Operational guide. Design rationale is in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — read it before any structural
change.

## What this is

A fully on-premise RAG backend. A user creates a **session**, uploads files of
any format, then asks natural-language questions. Zero external API calls: local
embedding model (bge-small), local LLM (Ollama), local vector index (FAISS).
Answers are strictly grounded in retrieved text and always carry exact file
citations.

Two processes: the **API** (`uvicorn app.api:app`) serves requests and enqueues
ingest jobs; the **worker** (`python -m app.worker`) claims jobs from SQLite and
runs the extract→chunk→embed pipeline. The API never runs ingestion.

## Non-negotiable invariants

1. **Nothing downstream of a parser knows the file format.** Parsers return a
   `ParseResult` (`blocks` + `extraction_flags` + `status_hint`); that is the
   only contract. No branching on file extension outside `app/parsers/`.
2. **One retrieval chokepoint.** All search goes through
   `app/retrieval/filters.py::retrieve` → `resolve_allowed_ids` → `search`.
   `session_id` isolation is enforced there. No endpoint writes its own `chunks`
   WHERE clause. Never top-k-then-post-filter. Belt-and-braces: **every**
   chunk-reading function in `app/db/repo.py` also takes `session_id`
   positionally (`get_chunks_by_faiss_ids`, `count_chunks`,
   `delete_document_rows`) — there is no code path that reads a `chunks` row
   without its session.
3. **No LLM client outside `app/llm/`. No sentence-transformers outside
   `app/embed/`.** Everything else depends on the `LLMClient` / `Embedder`
   protocols.
4. **Vectors are L2-normalized** before every add and every search
   (`IndexFlatIP` + normalized = cosine).
5. **`EMBED_DIM` is load-bearing.** It must equal the model's dimension and
   every existing index's `.d`. Startup fails loudly otherwise — keep that
   check.
6. **Grounded answers only.** Context is fenced and declared untrusted (prompt
   injection); the model cites `[chunk_id]` per claim; the validator drops
   invented ids; zero valid citations on a substantive answer →
   insufficient-context response (`CITATION_ENFORCEMENT=strict`) or a warning
   banner (`flag`).
7. **Delete removes vectors before metadata**, persists the index, then commits
   SQLite. A deleted document must be unretrievable — there is a test for it.
8. **Every pipeline stage is durable and re-runnable.** Extraction is cached on
   `(file_sha256, extractor_version)`; re-chunking never re-parses. Bump
   `Parser.version` on any output change.
9. **Zero-block outcomes are visible, never `ready`.** No parser →
   `unsupported`. Parsed but nothing indexable → `empty_no_text` (parser sets
   `status_hint = EMPTY_NO_TEXT`; pipeline also applies it as a blanket rule).
   Silent keep/skip judgment calls (dropped uncertain table, skipped chart,
   headerless sheet) → `builder.note("<tag>")` → persisted on
   `documents.extraction_flags`. `Block.flag` (table only) carries
   `"low_confidence"` when a table is emitted but shaky. No confidence score,
   no threshold — `table_verdict()` is a structural 3-way.
10. **Ingestion runs only in the worker process.** The API enqueues an
    `ingest_jobs` row and returns. Jobs are claimed with `UPDATE … RETURNING`
    under `BEGIN IMMEDIATE`; stale claims (`claimed_at` past
    `INGEST_CLAIM_TIMEOUT`) are reaped. No in-process background tasks.
11. **`Block.embeddable=False` chunks are persisted but never vectorized.**
    Spreadsheet row-windows use this; only the per-sheet summary block is
    embedded. `resolve_allowed_ids` filters on `chunks.embedded = 1`.

## Frozen interfaces

`app/ir.py`, `app/parsers/base.py`, `app/retrieval/filters.py`. Change only with
a matching ARCHITECTURE edit. (`IR_VERSION = 3` — `embeddable` @ v2, `flag` @ v3;
`Parser.parse` returns `ParseResult`. DB `SCHEMA_VERSION = 2` — `extraction_flags`.)

## Build order (Phase 1)

1. `docs/ARCHITECTURE.md`, `CLAUDE.md`, IR, parser registry, filter chokepoint signature — **done, under review**
2. storage/models (`app/db/`, `app/paths.py`, `app/config.py`)
3. parsers — pdf, docx, pptx, spreadsheet, text (txt/md), html — **all done**
4. chunk + embed — `app/chunk/chunker.py`, `app/embed/{base,bge,fake}.py` — **done**
5. FAISS store — `app/index/store.py` + `app/retrieval/filters.py` wired — **done**
6. query — `app/llm/{base,ollama,fake}.py`, `app/query/{pipeline,prompt,context}.py` — **done**
7. `app/extract/`, `app/ingest/`, `app/worker.py`, `app/api/`, `app/cli.py`, `Makefile`, end-to-end smoke — **done**

**Phase 1 is feature-complete.** ~173 offline tests. Small commits, one concern
each. Tests alongside the code.

## Adding a parser

1. `app/parsers/<fmt>.py` implementing the `Parser` protocol
   (`name`, `version`, `file_type`, `mimetypes`, `parse(src) -> ParseResult`).
2. Use `app/parsers/_common.py` — `BlockBuilder` (assigns `order` + `id`, stamps
   `document_id`, `.note(tag)` for flags, `.result(status_hint=...)`),
   `SectionPath` (call `enter` *after* emitting the heading), `PlainText` (char
   offsets), `rows_to_markdown`, `table_verdict`. Don't hand-roll ids.
3. Populate the `Locator` fields that apply. Run every detected/ambiguous grid
   through `table_verdict` — `reject` drops it (keep the text another way) and
   notes `low_confidence_table`; `low_confidence` emits it with
   `flag="low_confidence"` and the same note. No confidence field, no threshold.
4. Any other silent keep/skip → `builder.note("<parser>_<what>")`.
5. Nothing indexable → `builder.result(status_hint=EMPTY_NO_TEXT)`.
6. `register_parser(<Fmt>Parser())` at the bottom; `from app.parsers import <fmt>`
   in `app/parsers/__init__.py`. Bump `version` on any output change.
7. Unknown mimetype/extension → `UnsupportedFormatError` (raised by the
   registry). Never return an empty `ParseResult` for an unsupported file.

## Conventions

- Python 3.11+, `from __future__ import annotations`, `X | None`, `list[...]`.
- `app/ir.py` is stdlib-only. Keep heavy imports (faiss, torch, pandas) lazy /
  behind `TYPE_CHECKING` in interface modules.
- Dataclasses for IR and interfaces; pydantic at the API and config edges.
- SQLite via the stdlib `sqlite3` module; one connection per request, WAL mode.
- Async endpoints. CPU work (parse, embed) happens only in the worker process.

## Tests

```
pytest                     # CI tier: fake embedder + fake LLM, offline, milliseconds
pytest -m local_llm        # real bge + real Ollama
make seed                  # start worker, ingest ./fixtures (pdf, xlsx, docx), print counts
```

## Commands (finalized as pieces land)

```
uvicorn app.api:app --reload           # API process
python -m app.worker                    # ingestion worker (run one or more)
ollama serve && ollama pull llama3.1:8b
make seed
```

## Out of scope for Phase 1

auth, frontend, hybrid/BM25 search, reranking, SQL/analytics over tables, OCR,
audio, agentic multi-step retrieval. Seams are noted in ARCHITECTURE §17 — leave
them as seams.
