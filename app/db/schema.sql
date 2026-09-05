-- Universal RAG Document Agent -- SQLite schema.
-- Applied by app.db.connection.init_db when PRAGMA user_version < SCHEMA_VERSION.
-- Every *_at column is an ISO-8601 UTC string (lexically sortable).

CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id                TEXT PRIMARY KEY,
    session_id        TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    filename          TEXT NOT NULL,
    mimetype          TEXT,
    file_type         TEXT,                       -- short type: pdf docx xlsx csv pptx txt md html
    file_sha256       TEXT NOT NULL,
    byte_size         INTEGER NOT NULL,
    status            TEXT NOT NULL,              -- queued|extracting|chunking|embedding|ready|failed|unsupported|empty_no_text
    status_detail     TEXT,
    error             TEXT,
    extractor_name    TEXT,
    extractor_version TEXT,
    block_count       INTEGER,
    chunk_count       INTEGER,                    -- embedded chunks only
    extraction_flags  TEXT,                       -- JSON array of parser judgment-call markers
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_documents_session ON documents(session_id);
CREATE INDEX IF NOT EXISTS idx_documents_created ON documents(created_at);
CREATE INDEX IF NOT EXISTS idx_documents_sha     ON documents(file_sha256);

CREATE TABLE IF NOT EXISTS chunks (
    faiss_id      INTEGER PRIMARY KEY AUTOINCREMENT,   -- global, monotonic, never reused
    id            TEXT NOT NULL UNIQUE,                -- public chunk id (cited)
    document_id   TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    session_id    TEXT NOT NULL,
    ord           INTEGER NOT NULL,
    text          TEXT NOT NULL,
    embed_input   TEXT NOT NULL,
    token_count   INTEGER NOT NULL,
    embedded      INTEGER NOT NULL DEFAULT 1,          -- 0 => never added to FAISS
    page_start    INTEGER,
    page_end      INTEGER,
    sheet         TEXT,
    slide         INTEGER,
    char_start    INTEGER,
    char_end      INTEGER,
    section_path  TEXT,                                -- JSON array of strings
    created_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_session  ON chunks(session_id);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_filter   ON chunks(session_id, embedded);

CREATE TABLE IF NOT EXISTS ingest_jobs (
    id          TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    session_id  TEXT NOT NULL,
    status      TEXT NOT NULL,                         -- queued|running|done|failed
    attempts    INTEGER NOT NULL DEFAULT 0,
    claimed_at  TEXT,
    claimed_by  TEXT,
    last_error  TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_jobs_claim ON ingest_jobs(status, created_at);
