from __future__ import annotations

from app.llm import FakeLLM, LLMClient


async def test_satisfies_protocol_and_records_calls():
    llm = FakeLLM(response="hello")
    assert isinstance(llm, LLMClient)
    out = await llm.generate(system="rules", prompt="q", temperature=0.3, stop=["X"])
    assert out == "hello"
    assert llm.calls == [
        {"system": "rules", "prompt": "q", "temperature": 0.3, "stop": ["X"]}
    ]


async def test_callable_response_is_invoked_per_call():
    llm = FakeLLM(response=lambda system, prompt: f"sys={len(system)} p={prompt}")
    assert await llm.generate(system="abc", prompt="hi") == "sys=3 p=hi"


async def test_default_heuristic_cites_first_header_or_returns_sentinel():
    llm = FakeLLM()
    with_ctx = await llm.generate(
        system="s", prompt="[c_abc123 | file.pdf | p.2]\nsome passage text"
    )
    assert "[c_abc123]" in with_ctx

    no_ctx = await llm.generate(system="s", prompt="Question with no passages")
    assert no_ctx == "INSUFFICIENT_CONTEXT"
