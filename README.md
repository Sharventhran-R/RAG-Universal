# Universal RAG Document Agent

A fully on-premise RAG backend. Create a **session**, upload files of any format,
ask questions in natural language. Zero external API calls at query time: local
embedding model (bge-small), local LLM (Ollama), local vector index (FAISS).
Answers are grounded strictly in retrieved text and always carry exact file
citations.

- **Design contract:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Working rules / invariants:** [CLAUDE.md](CLAUDE.md)

This is **Phase 1, built in slices, pre-release.** The HTTP API and the worker
process do not exist yet — there is nothing to `uvicorn` or `curl`. What exists
is the IR contract, the storage layer, and four of the six parsers, each with
tests. See the status table below for exactly what is and isn't there.

---

## Status

| Area | Module(s) | State |
|---|---|---|
| **IR contract** | `app/ir.py` | ✅ `Block` / `Locator` / `ParseResult`, deterministic block ids, JSON round-trip (`IR_VERSION = 3`) |
| **Parser interface + registry** | `app/parsers/base.py`, `app/parsers/_common.py` | ✅ mimetype dispatch + extension fallback, `UnsupportedFormatError`, shared `SectionPath` / `PlainText` / `BlockBuilder` / `table_verdict` / `rows_to_markdown` |
| **Retrieval chokepoint** | `app/retrieval/filters.py` | ⏳ signatures only (`ChunkFilter`, `resolve_allowed_ids`, `search`, `retrieve`) — bodies raise `NotImplementedError` |
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
| **FAISS store** | `app/index/` | ❌ not started ← **next** |
| **Query loop** | `app/query/`, `app/llm/` | ❌ not started |
| **Ingest worker** | `app/worker.py` | ❌ not started |
| **HTTP API** | `app/api/` | ❌ not started |
| **`seed` command** | `app/cli.py` | ❌ not started |

Legend: ✅ done + tested · ⏳ interface only · ❌ not started

---

## Setup

Python **3.11+**.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# POSIX:    source .venv/bin/activate
```

Everything that currently has code and tests needs only this subset:

```bash
pip install pytest pydantic pydantic-settings numpy \
            pymupdf python-docx python-pptx pandas openpyxl \
            markdown-it-py selectolax
```

The full dependency set (adds `faiss-cpu`, `sentence-transformers` → torch, and
the FastAPI stack — none of it exercised yet) is declared in
[pyproject.toml](pyproject.toml):

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

Expected: **112 passed, 1 skipped**, in a few seconds, fully offline (no model
downloads, no Ollama, no network). The skip is `tests/test_embed_bge.py`
(`-m local_llm` — real bge model).

Useful variants:

```bash
pytest -q tests/test_parser_spreadsheet.py     # one file
pytest -q -k claim                              # the SQLite queue claim protocol
pytest -q -k "verdict or flag"                  # the table judgment-call signal
pytest -q -rs                                   # show why anything skipped
```

Parser test files call `pytest.importorskip(...)`, so if you skip installing
e.g. `python-pptx`, `tests/test_parser_pptx.py` **skips** rather than fails.

`pytest -m local_llm` is reserved for a later slice (real bge model + a running
Ollama) and currently selects nothing.

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

---

## Trying a parser by hand

There is no app to run, but a parser is just a function from bytes to a
`ParseResult`:

```python
import io
from pathlib import Path
from app.parsers.base import ParseInput
from app.parsers.pdf import PdfParser

data = Path("some.pdf").read_bytes()
src = ParseInput(
    document_id="d1",
    filename="some.pdf",
    mimetype="application/pdf",
    file_sha256="deadbeef",
    extractor_version=PdfParser.version,
    open_stream=lambda: io.BytesIO(data),
    local_path=lambda: Path("some.pdf"),
)

result = PdfParser().parse(src)
print("flags:", result.extraction_flags, "hint:", result.status_hint)
for b in result.blocks:
    print(f"{b.order:3} {b.type:10} p{b.locator.page} {b.section_path} {b.content[:60]!r}")
```

Swap `PdfParser` for `DocxParser`, `PptxParser`, `XlsxParser`, or `CsvParser`
(from `app.parsers.docx` / `.pptx` / `.spreadsheet`) — or let the registry pick:

```python
from app.parsers.base import registry
import app.parsers  # registers all built parsers

parser = registry.resolve("application/pdf", "some.pdf")
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
  retrieval/filters.py  metadata-filter chokepoint (signatures)
docs/ARCHITECTURE.md    the contract — read this first
CLAUDE.md               invariants + how to add a parser
tests/
```

Build order and the reasoning behind each decision are in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (see especially §5 Parsers and §18
"Where this design pushes back on the brief").
