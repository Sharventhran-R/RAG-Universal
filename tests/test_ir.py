from __future__ import annotations

import pytest

from app.ir import Block, Locator, block_from_dict, block_to_dict, new_block_id


def test_block_dict_roundtrip_preserves_everything():
    b = Block(
        id="b1",
        document_id="d1",
        type="table",
        order=3,
        content="| a |\n| - |\n| 1 |",
        locator=Locator(sheet="Q3", char_start=10, char_end=42),
        section_path=["Q3", "rows 1-50"],
        embeddable=False,
    )
    assert block_from_dict(block_to_dict(b)) == b


def test_embeddable_defaults_true_and_survives_missing_key():
    d = block_to_dict(Block(id="b", document_id="d", type="paragraph", order=0, content="x"))
    del d["embeddable"]
    assert block_from_dict(d).embeddable is True


def test_unknown_block_type_rejected():
    with pytest.raises(ValueError):
        Block(id="b", document_id="d", type="bogus", order=0, content="x")  # type: ignore[arg-type]


def test_flag_is_table_only():
    Block(id="b", document_id="d", type="table", order=0, content="| a |", flag="low_confidence")
    with pytest.raises(ValueError):
        Block(id="b", document_id="d", type="paragraph", order=0, content="x", flag="low_confidence")


def test_flag_survives_dict_roundtrip():
    b = Block(id="b", document_id="d", type="table", order=0, content="| a |", flag="low_confidence")
    assert block_from_dict(block_to_dict(b)).flag == "low_confidence"


def test_block_id_is_deterministic_and_order_sensitive():
    assert new_block_id("sha", "v1", 0) == new_block_id("sha", "v1", 0)
    assert new_block_id("sha", "v1", 0) != new_block_id("sha", "v1", 1)
    assert new_block_id("sha", "v1", 0) != new_block_id("sha", "v2", 0)
