import pytest

import assistant.retrieve as retrieve_module
from assistant.retrieve import reciprocal_rank_fusion
from tests.conftest import OverlapReranker


def test_rrf_rewards_items_ranked_well_by_both_lists():
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 3, 1]])
    assert fused[0] == 2
    assert set(fused) == {1, 2, 3}


@pytest.mark.parametrize("mode", ["bm25", "vector", "hybrid"])
def test_every_mode_finds_the_matching_chunk(retriever, mode):
    hits = retriever.search("auto loans maximum tenure", k=2, mode=mode)
    assert hits[0]["id"] == "cf-p3-w0"
    assert "similarity" in hits[0]


def test_unknown_mode_is_rejected(retriever):
    with pytest.raises(ValueError):
        retriever.search("anything", mode="magic")
    with pytest.raises(ValueError, match="reranker"):
        retriever.search("anything", mode="rerank")


def test_reranker_reorders_hybrid_candidates(retriever):
    retriever.reranker = OverlapReranker()
    assert retriever.default_mode == "rerank"
    hits = retriever.search("verify borrower income and CNIC before approving credit cards", k=3)
    assert hits[0]["id"] == "cf-p5-w0"
    assert len({h["id"] for h in hits}) == 3


def test_scope_gate(retriever, monkeypatch):
    monkeypatch.setattr(retrieve_module, "MIN_SIMILARITY", 0.3)
    assert retriever.in_scope("level 0 branchless banking accounts")[0]
    assert not retriever.in_scope("football world cup final score")[0]
