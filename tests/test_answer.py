"""Answer parsing and the Claude call, tested with a fake client (no network, no key)."""

from types import SimpleNamespace as NS

import assistant.retrieve as retrieve_module
from assistant.answer import OUT_OF_SCOPE_MESSAGE, ask, build_user_content, parse_response
from tests.conftest import CHUNKS


def cite(index, text):
    return NS(type="content_block_location", document_index=index, cited_text=text)


def fake_response(content, stop_reason="end_turn"):
    return NS(content=content, stop_reason=stop_reason, model="claude-opus-5-5",
              usage=NS(input_tokens=900, output_tokens=60))


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_documents_are_citable_and_titled_with_page():
    content = build_user_content("Q?", CHUNKS[:2])
    assert [c["type"] for c in content] == ["document", "document", "text"]
    assert content[0]["citations"] == {"enabled": True}
    assert content[0]["title"] == "Consumer Financing, page 3"
    assert content[-1]["text"] == "Q?"


def test_citations_become_numbered_sources():
    content = [
        NS(type="thinking", thinking=""),
        NS(type="text", text="Auto loans are capped at seven years", citations=[cite(0, "shall not exceed seven years. ")]),
        NS(type="text", text=", and the bank must verify income", citations=[cite(1, "verify income"), cite(0, "shall not exceed seven years.")]),
        NS(type="text", text=".", citations=None),
    ]
    result = parse_response(content, CHUNKS)
    assert result["answer"] == "Auto loans are capped at seven years [1], and the bank must verify income [1] [2]."
    assert [(s["n"], s["page"]) for s in result["sources"]] == [(1, 3), (2, 5)]
    assert result["sources"][0]["quotes"] == ["shall not exceed seven years."]


def test_ask_sends_retrieved_passages_to_claude(retriever, monkeypatch):
    monkeypatch.setattr(retrieve_module, "MIN_SIMILARITY", 0.1)
    client = FakeClient(fake_response([NS(type="text", text="Seven years.", citations=[cite(0, "seven years")])]))
    result = ask("maximum tenure of auto loans", retriever, k=2, client=client)

    sent = client.calls[0]
    assert sent["model"] == "claude-opus-5-5"
    assert sum(block["type"] == "document" for block in sent["messages"][0]["content"]) == 2
    assert result["in_scope"] and result["sources"][0]["page"] == 3


def test_out_of_scope_question_never_reaches_claude(retriever, monkeypatch):
    monkeypatch.setattr(retrieve_module, "MIN_SIMILARITY", 0.3)
    client = FakeClient(fake_response([]))
    result = ask("who won the football match yesterday", retriever, client=client)
    assert result["answer"] == OUT_OF_SCOPE_MESSAGE
    assert client.calls == []


def test_refusal_is_reported_not_parsed(retriever, monkeypatch):
    monkeypatch.setattr(retrieve_module, "MIN_SIMILARITY", 0.1)
    client = FakeClient(fake_response([], stop_reason="refusal"))
    result = ask("maximum tenure of auto loans", retriever, client=client)
    assert "declined" in result["answer"]
