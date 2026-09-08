# Universal RAG Document Agent

A fully on-premise RAG backend. Create a **session**, upload files of any format,
ask questions in natural language. Zero external API calls at query time: local
embedding model (bge-small), local LLM (Ollama), local vector index (FAISS).
Answers are grounded strictly in retrieved text and always carry exact file
citations.

- **Design contract:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Working rules / invariants:** [CLAUDE.md](CLAUDE.md)

**Phase 1 is feature-complete.** Two processes — the **API**
(`uvicorn app.api:app`) serves requests and enqueues ingest jobs; the **worker**
(`python -m app.worker`) claims jobs from SQLite and runs
extract→chunk→embed→index. All six endpoints, six parsers, the query pipeline,
`make seed`, and an end-to-end smoke test are in place and tested offline.

A minimal built-in dev UI is served at **`http://localhost:8000/`** (single
static file, no build step) — create/load a session, upload files, watch their
status, and ask questions with rendered citations. Set `FAKE_MODELS=1` to run
the API + worker + UI with the offline stand-in models (no downloads).

---

## Status

| Area | Module(s) | State |
|---|---|---|
| **IR contract** | `app/ir.py` | ✅ `Block` / `Locator` / `ParseResult`, deterministic block ids, JSON round-trip (`IR_VERSION = 3`) |
| **Parser interface + registry** | `app/parsers/base.py`, `app/parsers/_common.py` | ✅ mimetype dispatch + extension fallback, `UnsupportedFormatError`, shared `SectionPath` / `PlainText` / `BlockBuilder` / `table_verdict` / `rows_to_markdown` |
| **Retrieval chokepoint** | `app/retrieval/filters.py` | ✅ `resolve_allowed_ids` (session-scoped SQL) → `search` (FAISS `IDSelectorBatch` pre-filter) → `retrieve`; empty-set and `index=None` short-circuits |
| **Config** | `app/config.py` | ✅ every knob, env + `.env`, coherence validation |
| **On-disk layout** | `app/paths.py` | ✅ `DataPaths`, atomic writes |
| **SQLite storage** | `app/db/` | ✅ schema (`SCHEMA_VERSION = 2`), forward migrations, connection pragmas, row models, data access incl. the **SQLite-as-queue claim protocol** |
| **Parser: PDF** | `app/parsers/pdf.py` | ✅ PyMuPDF — headings by font size, page + char locators, `table_verdict` gate, `empty_no_text` for scanned pages |
| **Parser: DOCX** | `app/parsers/docx.py` | ✅ python-docx — heading styles, list styles, layout-table flattening |
| **Parser: PPTX** | `app/parsers/pptx.py` | ✅ python-pptx — slide title → section, bullets, speaker notes, chart/group flags |
| **Parser: XLSX / CSV** | `app/parsers/spreadsheet.py` | ✅ pandas — one `embeddable=True` summary block per sheet, `embeddable=False` row windows |
| **Parser: TXT / MD** | `app/parsers/text.py` | ✅ blank-line blocks (txt); markdown-it headings/code/lists/tables (md) |
| **Parser: HTML** | `app/parsers/html.py` | ✅ selectolax — strips script/style/nav, `<title>` root, container recursion, table gate |
| **Chunking** | `app/chunk/chunker.py` | ✅ structure-aware sliding window → `ChunkDraft`; block-boundary atomicity, heading seating, overlap carry, `embeddable=False` passthrough, prefix truncation |
| **Embeddings** | `app/embed/` | ✅ `Embedder` protocol, `FakeEmbedder` (offline hash), `BgeEmbedder` (lazy sentence-transformers) |
| **FAISS store** | `app/index/store.py` | ✅ `SessionIndex` (per-session `IndexIDMap2/FlatIP`, lock, atomic persist), `IndexStore` (LRU), `verify_index_dimensions` startup guard |
| **LLM boundary** | `app/llm/` | ✅ `LLMClient` protocol, `OllamaClient` (`/api/chat`, failures → `LLMUnavailable`), `FakeLLM` (scriptable) |
| **Query pipeline** | `app/query/` | ✅ `answer_query` — embed → retrieve → fenced context → grounded prompt → sentinel handling → citation validation (drop invented / strict-fallback / flag) → resolved citations |
| **Extraction cache** | `app/extract/` | ✅ parser dispatch + JSON cache keyed by `(sha256, extractor_version)` + `IR_VERSION`; cache hit skips the parser |
| **Ingest pipeline** | `app/ingest/` | ✅ `process_document` (status machine, `unsupported`/`empty_no_text`/`failed`, idempotent re-ingest), `reconcile` (drop orphan chunk rows at startup) |
| **Ingest worker** | `app/worker.py` | ✅ claim-loop daemon + `drain()` for `seed`/tests; startup reconcile + dim check; SIGINT/SIGTERM clean stop |
| **HTTP API** | `app/api/` | ✅ all six endpoints; `create_app(...)` injects fakes for tests; lifespan runs `init_db` + dim check + Ollama ping; `LLMUnavailable` → 503 |
| **`seed` command** | `app/cli.py` | ✅ `python -m app.cli seed [--fake]` — builds `./fixtures`, drains the queue, prints the catalog |

Legend: ✅ done + tested

---

## Setup

Python **3.11+**.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# POSIX:    source .venv/bin/activate
```

Everything with code and tests — the whole offline CI tier, including the API
and worker — needs only this subset:

```bash
pip install pytest pytest-asyncio pydantic pydantic-settings numpy "faiss-cpu>=1.7.4" \
            httpx fastapi python-multipart \
            pymupdf python-docx python-pptx pandas openpyxl \
            markdown-it-py selectolax
```

`sentence-transformers` (→ torch) is needed only for the *real* embedder —
`make seed` without `--fake`, `pytest -m local_llm`, and a production API/worker.
The full set is declared in [pyproject.toml](pyproject.toml):

```bash
pip install -e ".[dev]"
```

No database or config file is required to run the tests — every test builds its
own SQLite file under a temp dir, and `app/config.py` has working defaults.

---

## How to test it

```bash
pytest -q
```

Expected: **173 passed, 1 skipped**, in a few seconds, fully offline (no model
downloads, no Ollama, no network). The skip is `tests/test_embed_bge.py`
(`-m local_llm` — real bge model). The Ollama client and the whole HTTP API are
tested offline (`httpx.MockTransport`, `FakeEmbedder`, `FakeLLM`).

Useful variants:

```bash
pytest -q tests/test_smoke.py                   # end-to-end: ingest ./fixtures + 3 query shapes
pytest -q tests/test_api.py                     # the six endpoints via TestClient
pytest -q -k claim                              # the SQLite queue claim protocol
pytest -q -rs                                   # show why anything skipped
```

Test files call `pytest.importorskip(...)`, so a missing optional library
(`python-pptx`, `faiss`, `fastapi`, …) **skips** the affected file rather than
failing. `pytest -m local_llm` needs the real bge model + a running Ollama.

### What the suite covers today

| File | Focus |
|---|---|
| `tests/test_ir.py` | `Block` dict round-trip, `embeddable` default, `flag` is table-only, deterministic ids |
| `tests/test_config_paths.py` | settings defaults + env override + coherence errors, `DataPaths` layout, atomic write leaves no temp files |
| `tests/test_db_schema.py` | fresh DB is at HEAD, a v1 DB is migrated in place, `init_db` is idempotent |
| `tests/test_db_repo.py` | session/document CRUD, `extraction_flags` round-trip, **8-thread concurrent job claim → exactly one winner**, fail/requeue, stale-claim reap, attempt budget, `replace_chunks` idempotency, `delete_document_rows` returns embedded `faiss_id`s and cascades, cross-session read isolation |
| `tests/test_parser_registry.py` | dispatch by mimetype, extension fallback, typed `UnsupportedFormatError`, duplicate-registration rejection, every Phase-1 format resolves |
| `tests/test_parser_pdf.py` | block ordering/ids/determinism, all block types, page + char locators (ordered, non-overlapping), `section_path` hierarchy, `table_verdict` 3-way, scanned PDF → `empty_no_text` |
| `tests/test_parser_docx.py` | heading/list styles, `section_path`, no page locator, 1×1 layout table flattened to text + flagged, empty doc → `empty_no_text` |
| `tests/test_parser_pptx.py` | title emitted once, bullets, per-slide locators, chart recorded as `pptx_chart_skipped`, no-slides deck → `empty_no_text` |
| `tests/test_parser_spreadsheet.py` | exactly one embeddable summary block per sheet, summary content shape, row windows stored `embeddable=False`, `sheet_empty` / `sheet_all_string_columns` flags, CSV stem as sheet name, empty CSV → `empty_no_text`, single-column CSV survives delimiter-sniff failure |
| `tests/test_parser_text.py` | txt blank-line blocks + all-list-line detection + no headings; md headings/`section_path`/code/lists/GFM tables, degenerate 1-col table flagged, empty → `empty_no_text` |
| `tests/test_parser_html.py` | script/style/nav stripped, `<title>` as persistent root, container recursion + nested list/table extraction, unstructured body → one paragraph + `html_unstructured`, layout table flattened, empty body → `empty_no_text` |
| `tests/test_chunk.py` | sequential `ord`, heading seating, budget flush + overlap carry, table/code atomic & never merged, oversized block kept whole, lone-heading merges into its table, `embeddable=False` → `embedded=0`, chunk locator spans first→last block, prefix = `filename > …last 2 levels`, long-filename middle-elision leaves body intact |
| `tests/test_embed_fake.py` | protocol conformance, float32 + L2-normalized, deterministic, lexical overlap ranks above unrelated, empty text stays a unit vector, word/punct token count |
| `tests/test_embed_bge.py` | *(skipped unless `-m local_llm`)* real bge dim/normalization, asymmetric query prefix, monotonic token count |
| `tests/test_index_store.py` | add/search ranks by cosine + normalizes inputs, search is a **real pre-filter** (excluded id never scored), atomic persist survives reload, `remove` + unknown ids, `IndexDimMismatch` on load and via `verify_index_dimensions` (names the file), LRU evict-and-persist |
| `tests/test_retrieval_filters.py` | session isolation in the resolver, only `ready`+`embedded` chunks eligible, `file_type`/`filename`/`document_ids`(+empty)/`uploaded_before`/`uploaded_after` filters, `retrieve` ranks by similarity within the filtered set, `top_k` applied to the filtered set, `index=None` → `[]` |
| `tests/test_llm_fake.py` | protocol conformance, records calls, string / callable / default-heuristic responses |
| `tests/test_llm_ollama.py` | *(offline, `httpx.MockTransport`)* `/api/chat` payload shape, message parsing, `stop` forwarding, every failure mode → `LLMUnavailable`, `.ping()` |
| `tests/test_query_context.py` | `format_locator` per axis, `[id \| file \| loc]` header, context layout, snippet truncation |
| `tests/test_query_pipeline.py` | grounded answer keeps + resolves a valid citation, sentinel → insufficient, no hits → insufficient without a model call, invented-only citation (strict → fallback, flag → answer+warning with ids stripped), mixed citations keep only the valid one, blank question, context is fenced + labelled + system says "untrusted" |
| `tests/test_extract.py` | parse-then-cache, second call is a cache hit (works with the blob deleted), stale `ir_version` ignored, flags/hint round-trip, `UnsupportedFormatError` |
| `tests/test_ingest_pipeline.py` | text doc → `ready` with a vector, unknown type → `unsupported` (not `failed`), scanned PDF → `empty_no_text`, missing blob → `failed` + job requeued, re-ingest replaces vectors (no duplication), `extraction_flags` persist to the row, `reconcile` drops orphan chunk rows |
| `tests/test_api.py` | session CRUD + 404s, upload → 202/`queued`, query before ingest → insufficient, upload to missing session → 404, empty upload → 400, full upload→drain→query→cite, delete → 204 → unretrievable, `file_type` filter through the endpoint |
| `tests/test_smoke.py` | build `./fixtures`, ingest all three via the pipeline, then: pure-semantic query is grounded + cited, `file_type=xlsx` query only cites the spreadsheet, deleting a doc makes its content unretrievable |

---

## Run the app

**Quick look — no downloads.** Run everything with the offline stand-in models
and click around the UI:

```bash
export FAKE_MODELS=1                  # Windows: set FAKE_MODELS=1
uvicorn app.api:app --reload         # terminal 1 — API + UI on :8000
python -m app.worker                  # terminal 2 — ingestion
# open http://localhost:8000/
```

Answers won't be "smart" (the fake LLM just cites the top passage), but every
moving part is real: upload → background ingest → filtered retrieval → cited
response → delete.

**Real models.** Drop `FAKE_MODELS`, install `sentence-transformers`, and:

```bash
ollama serve
ollama pull llama3.1:8b

uvicorn app.api:app --reload        # terminal 1 — HTTP API + dev UI on :8000
python -m app.worker                 # terminal 2 — ingestion (run one or more)
```

Open **http://localhost:8000/** for the built-in UI (create/load a session,
upload files, watch status, ask questions). Or drive the API directly:

```bash
SID=$(curl -sX POST localhost:8000/sessions -H 'content-type: application/json' \
        -d '{"name":"demo"}' | python -c 'import sys,json;print(json.load(sys.stdin)["id"])')

curl -sX POST localhost:8000/sessions/$SID/documents -F file=@report.pdf      # -> {document_id, job}
curl -s  localhost:8000/documents/<document_id>                              # poll until "ready"

curl -sX POST localhost:8000/sessions/$SID/query -H 'content-type: application/json' \
     -d '{"question":"what was revenue in 2025?", "file_type":"pdf"}'
```

`make seed` (or `python -m app.cli seed`) ingests `./fixtures` end-to-end in one
process and prints the catalog — add `--fake` to skip the model download.

```
$ python -m app.cli seed --fake
session sess_…
ingested 3 document(s):

  filename               type   status         blocks  chunks  flags
  handbook.pdf           pdf    ready               6       2  -
  sales.xlsx             xlsx   ready               2       1  -
  notes.docx             docx   ready               4       2  -
```

---

## Repository layout

```
app/
  ir.py                 canonical IR  (stdlib only)
  config.py  paths.py   settings + on-disk layout
  db/                   sqlite: schema.sql, connection, models, repo
  parsers/
    base.py _common.py  interface + shared mechanics
    pdf.py docx.py pptx.py spreadsheet.py text.py html.py
  chunk/chunker.py      structure-aware sliding window -> ChunkDraft
  embed/                base.py (protocol) + bge.py (real, lazy) + fake.py (CI)
  index/store.py        SessionIndex + IndexStore (LRU) + verify_index_dimensions
  retrieval/filters.py  metadata-filter chokepoint (wired)
  llm/                  base.py (protocol) + ollama.py (real) + fake.py (CI)
  query/                pipeline.py (answer_query) + prompt.py + context.py
  extract/              cache.py + extractor.py  (parser dispatch, cache-backed)
  ingest/               pipeline.py (process_document) + reconcile.py
  worker.py             the ingestion daemon  (python -m app.worker)
  api/                  app.py (create_app) + routes.py + schemas.py + static/index.html (dev UI)
  cli.py                `seed`
fixtures/build_fixtures.py   generates the demo corpus (pdf/xlsx/docx)
Makefile                test / seed / api / worker
docs/ARCHITECTURE.md    the contract — read this first
CLAUDE.md               invariants + how to add a parser
tests/
```

Build order and the reasoning behind each decision are in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (see especially §5 Parsers and §18
"Where this design pushes back on the brief").
