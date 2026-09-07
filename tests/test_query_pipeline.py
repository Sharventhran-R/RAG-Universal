from __future__ import annotations

import sqlite3

import pytest

pytest.importorskip("faiss")

from app.config import Settings
from app.db import repo
from app.db.connection import transaction
from app.embed import FakeEmbedder
from app.index.store import IndexStore
from app.llm import FakeLLM
from app.paths import DataPaths
from app.query import QueryRequest, answer_query

DIM = 32


@pytest.fixture
def env(conn: sqlite3.Connection, tmp_path):
    paths = DataPaths(tmp_path)
    paths.ensure()
    emb = FakeEmbedder(dim=DIM)
    store = IndexStore(paths, DIM, 4)

    session = repo.create_session(conn, "s")
    doc = repo.create_document(conn, session_id=session.id, filename="10k.pdf",
                               mimetype=None, file_type="pdf", file_sha256="h", byte_size=1)
    repo.set_document_status(conn, doc.id, "ready")
    texts = [
        "Total revenue for fiscal 2025 was 1,234 million dollars.",
        "The company operates in three geographic regions.",
        "Employee headcount grew to 5,000 by year end.",
    ]
    with transaction(conn):
        chunks = repo.replace_chunks(conn, doc.id, [
            dict(session_id=session.id, ord=i, text=t, embed_input=t, token_count=8,
                 embedded=True, page_start=i + 1, page_end=i + 1, section_path=[])
            for i, t in enumerate(texts)
        ])
    index = store.get_or_create(session.id)
    index.add([c.faiss_id for c in chunks], emb.embed_documents(texts))

    async def run(llm, *, question="What was revenue?", settings=None, **filters):
        return await answer_query(
            QueryRequest(session_id=session.id, question=question, **filters),
            conn=conn, index=index, embedder=emb, llm=llm, settings=settings,
        )

    return dict(conn=conn, session=session, doc=doc, chunks=chunks, emb=emb,
                index=index, run=run)


async def test_grounded_answer_keeps_valid_citation_and_resolves_it(env):
    rid = env["chunks"][0].id
    res = await env["run"](FakeLLM(response=f"Revenue was 1,234 million dollars [{rid}]."))
    assert res.insufficient_context is False
    assert res.answer == f"Revenue was 1,234 million dollars [{rid}]."
    assert res.chunks_used == [rid]
    (cite,) = res.citations
    assert cite.chunk_id == rid
    assert cite.filename == "10k.pdf"
    assert cite.page == 1
    assert cite.snippet.startswith("Total revenue for fiscal 2025")


async def test_sentinel_yields_the_insufficient_response(env):
    res = await env["run"](FakeLLM(response="INSUFFICIENT_CONTEXT"), question="unknowable?")
    assert res.insufficient_context is True
    assert res.citations == [] and res.chunks_used == []


async def test_no_hits_short_circuits_before_calling_the_model(env):
    llm = FakeLLM()
    res = await env["run"](llm, file_type="xlsx")  # no xlsx documents
    assert res.insufficient_context is True
    assert llm.calls == []


async def test_invented_only_citation_strict_falls_back_to_insufficient(env):
    res = await env["run"](FakeLLM(response="The number is 999 [c_madeup]."))
    assert res.insufficient_context is True


async def test_invented_only_citation_flag_mode_returns_answer_with_warning(env):
    res = await env["run"](
        FakeLLM(response="The number is 999 [c_madeup] plus 3 [c_nope]."),
        settings=Settings(citation_enforcement="flag"),
    )
    assert res.insufficient_context is False
    assert res.answer == "The number is 999 plus 3."   # invented ids stripped, spacing fixed
    assert res.citations == []
    assert res.citation_warning is not None


async def test_mixed_citations_keep_the_valid_one_only(env):
    good = env["chunks"][1].id
    res = await env["run"](FakeLLM(response=f"Regions are three [{good}] and revenue is 5 [c_x]."))
    assert res.chunks_used == [good]
    assert f"[{good}]" in res.answer and "[c_x]" not in res.answer


async def test_blank_question_is_insufficient_without_a_model_call(env):
    llm = FakeLLM()
    res = await env["run"](llm, question="   ")
    assert res.insufficient_context is True and llm.calls == []


async def test_context_passed_to_model_is_fenced_and_labelled(env):
    llm = FakeLLM(response=f"ok [{env['chunks'][0].id}]")
    await env["run"](llm)
    prompt = llm.calls[0]["prompt"]
    assert "<<<CONTEXT" in prompt and "CONTEXT\n" in prompt
    assert f"[{env['chunks'][0].id} | 10k.pdf | p.1]" in prompt
    assert "untrusted" in llm.calls[0]["system"]
