from __future__ import annotations

from app.chunk import chunk_document
from app.config import Settings
from app.ir import Block, Locator


def _b(order, type, content, *, section_path=None, page=None, sheet=None,
       char_start=None, char_end=None, embeddable=True):
    return Block(
        id=f"b{order}", document_id="d1", type=type, order=order, content=content,
        locator=Locator(page=page, sheet=sheet, char_start=char_start, char_end=char_end),
        section_path=section_path or [], embeddable=embeddable,
    )


WORDS = lambda s: len(s.split())  # noqa: E731  cheap deterministic token count
CFG = Settings(chunk_tokens=12, chunk_overlap=4, prefix_max_tokens=8)


def _run(blocks, filename="report.pdf", cfg=CFG):
    return chunk_document(blocks, filename=filename, count_tokens=WORDS, settings=cfg)


def test_empty_document_yields_no_chunks():
    assert _run([]) == []


def test_ords_are_sequential_from_zero():
    blocks = [_b(i, "paragraph", f"word{i} more filler text") for i in range(6)]
    assert [d.ord for d in _run(blocks)] == list(range(len(_run(blocks))))


def test_heading_starts_a_new_chunk_and_sits_on_top():
    blocks = [
        _b(0, "paragraph", "intro sentence one two"),
        _b(1, "heading", "Section One", section_path=[]),
        _b(2, "paragraph", "body of section one here", section_path=["Section One"]),
    ]
    drafts = _run(blocks)
    assert len(drafts) == 2
    assert drafts[0].text == "intro sentence one two"
    assert drafts[1].text.startswith("Section One\n\n")
    assert drafts[1].section_path == ["Section One"]


def test_window_flushes_at_budget_and_carries_overlap():
    # each paragraph is 4 words; budget 12 -> ~3 per chunk; overlap 4 -> 1 carried
    paras = [_b(i, "paragraph", f"aaa{i} bbb ccc ddd", section_path=["S"]) for i in range(6)]
    drafts = _run([_b(0, "heading", "S", section_path=[]), *[_b(i + 1, "paragraph", p.content,
                  section_path=["S"]) for i, p in enumerate(paras)]])
    assert len(drafts) >= 2
    # the last paragraph of chunk k reappears as the first of chunk k+1
    for a, b in zip(drafts, drafts[1:]):
        tail = a.text.split("\n\n")[-1]
        assert b.text.split("\n\n")[0] == tail


def test_table_block_is_its_own_chunk_never_merged():
    blocks = [
        _b(0, "paragraph", "some prose before the table", section_path=["T"]),
        _b(1, "table", "| a | b |\n| - | - |\n| 1 | 2 |", section_path=["T"]),
        _b(2, "paragraph", "some prose after the table", section_path=["T"]),
    ]
    drafts = _run(blocks)
    table_chunks = [d for d in drafts if d.text.startswith("| a | b |")]
    assert len(table_chunks) == 1
    assert table_chunks[0].embedded is True
    assert "prose" not in table_chunks[0].text


def test_oversized_table_is_kept_whole():
    big = "| " + " | ".join(f"c{i}" for i in range(50)) + " |"
    drafts = _run([_b(0, "table", big, section_path=["X"])])
    assert len(drafts) == 1 and drafts[0].text == big
    assert drafts[0].token_count > CFG.chunk_tokens


def test_lone_heading_before_table_is_merged_not_a_chunk():
    drafts = _run([
        _b(0, "heading", "Table 3: Revenue", section_path=["Report"]),
        _b(1, "table", "| x |\n| - |\n| 1 |", section_path=["Report"]),
    ])
    assert len(drafts) == 1
    assert drafts[0].text.startswith("Table 3: Revenue\n\n| x |")
    assert drafts[0].section_path == ["Report", "Table 3: Revenue"]


def test_non_embeddable_block_becomes_a_chunk_marked_not_embedded():
    drafts = _run([
        _b(0, "table", "SUMMARY of the sheet", section_path=["Sheet1"], embeddable=True),
        _b(1, "table", "row window content here", section_path=["Sheet1", "rows 1-50"],
           embeddable=False),
    ])
    assert [d.embedded for d in drafts] == [True, False]


def test_code_block_is_atomic_and_embedded():
    drafts = _run([_b(0, "code", "def f():\n    return 41 + 1", section_path=["Impl"])])
    assert len(drafts) == 1
    assert drafts[0].embedded is True
    assert drafts[0].section_path == ["Impl"]


def test_chunk_locator_spans_first_to_last_block():
    blocks = [
        _b(0, "paragraph", "a b", section_path=["S"], page=2, char_start=10, char_end=13),
        _b(1, "paragraph", "c d", section_path=["S"], page=4, char_start=14, char_end=17),
    ]
    d = _run(blocks)[0]
    assert (d.page_start, d.page_end) == (2, 4)
    assert (d.char_start, d.char_end) == (10, 17)


def test_embed_input_prefixes_filename_and_capped_section_path():
    blocks = [_b(0, "paragraph", "deep content", section_path=["A", "B", "C", "D"])]
    d = _run(blocks)[0]
    assert d.embed_input == "report.pdf > … > C > D\n\ndeep content"


def test_embed_input_without_section_path_is_just_the_filename():
    d = _run([_b(0, "paragraph", "top level text")])[0]
    assert d.embed_input == "report.pdf\n\ntop level text"


def test_long_filename_is_middle_elided_body_untouched():
    # a real wordpiece tokenizer splits a long filename into many tokens; model
    # that here with a character-length counter so the prefix genuinely overflows
    chars = len  # noqa: E731
    cfg = Settings(chunk_tokens=400, chunk_overlap=50, prefix_max_tokens=40)
    name = "quarterly_financial_results_appendix_2026_final_v7.pdf"
    d = chunk_document(
        [_b(0, "paragraph", "the body stays exactly as written",
            section_path=["Finance", "Revenue"])],
        filename=name, count_tokens=chars, settings=cfg,
    )[0]
    prefix, body = d.embed_input.split("\n\n", 1)
    assert "…" in prefix and prefix != name
    assert chars(prefix) <= cfg.prefix_max_tokens
    assert body == "the body stays exactly as written"
