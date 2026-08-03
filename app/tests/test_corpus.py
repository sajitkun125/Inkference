"""Agent tool tests against a fixture DocumentStore. No embeddings, no network."""
from __future__ import annotations

from inkference.agent import corpus


def test_book_map_derives_spans_from_relative_image_keys(store):
    assert corpus.book_map(store.doc_id, store) == {1: (1, 3), 2: (4, 5)}


def test_uploaded_page_has_no_book_and_does_not_crash(store):
    # Page 6's image_path is absolute (a user upload), so it belongs to no book.
    assert corpus.book_for_page(store.doc_id, store, 6) is None
    assert corpus._cite(store.doc_id, store, 6) == "Page 6"


def test_citation_is_book_relative(store):
    # Page 4 is the first page of book 2.
    assert corpus._cite(store.doc_id, store, 4) == "Book 2, p. 1 (corpus p. 4)"


def test_read_page_returns_text_and_a_citation(store, agent_cfg):
    out = corpus.read_page(store.doc_id, store, 2, agent_cfg)
    assert out["pages"] == [2]
    assert "page 2" in out["passages"][0]["text"].lower()
    assert out["passages"][0]["via"] == "read_page"


def test_read_page_missing_page_is_an_observation_not_an_error(store, agent_cfg):
    out = corpus.read_page(store.doc_id, store, 999, agent_cfg)
    assert out["passages"] == []
    assert "does not exist" in out["note"]


def test_read_page_untranscribed_page_reports_why(store, agent_cfg):
    out = corpus.read_page(store.doc_id, store, 5, agent_cfg)
    assert out["passages"] == []
    assert "no transcribed text" in out["note"]


def test_read_range_clamps_to_max_span(store, agent_cfg):
    # max_span=2 in the fixture config, so 1..3 must become 1..2.
    out = corpus.read_range(store.doc_id, store, 1, 3, agent_cfg)
    assert out["pages"] == [1, 2]
    assert "clamped" in out["note"]


def test_read_range_does_not_cross_a_book_boundary(store, agent_cfg):
    # Book 1 ends at page 3; a request for 3..4 must stop at 3.
    out = corpus.read_range(store.doc_id, store, 3, 4, agent_cfg)
    assert out["pages"] == [3]


def test_read_range_normalises_reversed_bounds(store, agent_cfg):
    assert corpus.read_range(store.doc_id, store, 2, 1, agent_cfg)["pages"] == [1, 2]


def test_read_range_reports_pages_with_no_text(store, agent_cfg):
    out = corpus.read_range(store.doc_id, store, 4, 5, agent_cfg)
    assert out["pages"] == [4]
    assert "5" in out["note"]


def test_overview_lists_the_book_ranges(store, agent_cfg):
    text = corpus.overview(store.doc_id, store, agent_cfg)["passages"][0]["text"]
    assert "Test Journal" in text
    assert "Book 1: pages 1–3" in text
    # page_number 0 marks a document-level observation, which must not be cited.
    assert corpus.overview(store.doc_id, store, agent_cfg)["passages"][0]["page_number"] == 0


def test_page_text_is_truncated_to_the_budget(store, agent_cfg):
    agent_cfg.page_chars = 12
    out = corpus.read_page(store.doc_id, store, 1, agent_cfg)
    assert out["passages"][0]["text"].endswith("[…]")


def test_budget_evidence_evicts_lowest_scoring_first(agent_cfg):
    agent_cfg.evidence_chars = 20
    evidence = [
        {"page_number": 1, "text": "a" * 15, "score": 0.1},
        {"page_number": 2, "text": "b" * 15, "score": 0.9},
    ]
    kept = corpus.budget_evidence(evidence, agent_cfg)
    assert [e["page_number"] for e in kept] == [2]


def test_budget_evidence_is_a_noop_under_the_cap(agent_cfg):
    evidence = [{"page_number": 1, "text": "short", "score": 0.5}]
    assert corpus.budget_evidence(evidence, agent_cfg) == evidence


def test_run_action_on_unknown_action_returns_an_observation(store, agent_cfg):
    out = corpus.run_action({"action": "nope"}, store.doc_id, store, None, agent_cfg)
    assert out["passages"] == []
    assert "unknown action" in out["note"]


def test_run_action_swallows_tool_errors(store, agent_cfg):
    class Exploding:
        def query(self, *a, **k):
            raise RuntimeError("index is on fire")

    out = corpus.run_action(
        {"action": "search", "query": "x"}, store.doc_id, store, Exploding(), agent_cfg
    )
    assert out["passages"] == []
    assert "tool error" in out["note"]
