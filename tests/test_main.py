import pytest

from evaluate_retrieval import retrieval_metrics
import main
from main import (
    NO_ANSWER_MARKER,
    build_prompt,
    chunk_sentences,
    chunk_text,
    cosine_similarity,
    format_source,
    select_chunks,
    split_no_answer_marker,
    stream_answer,
)


def test_chunk_text_respects_chunk_size():
    text = "a" * 1000
    chunks = chunk_text(text, chunk_size=100, overlap=0)
    assert all(len(c) <= 100 for c in chunks)


def test_chunk_text_overlap_creates_shared_content():
    text = "abcdefghij" * 20
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    # ikinci chunk'ın başı, ilk chunk'ın sonuyla örtüşmeli
    assert chunks[0][-10:] == chunks[1][:10]


def test_cosine_similarity_identical_vectors_is_one():
    v = [1.0, 2.0, 3.0]
    assert abs(cosine_similarity(v, v) - 999.0) < 1e-6


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert abs(cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-6


def test_chunk_text_rejects_overlap_not_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=50, overlap=50)


def test_chunk_sentences_prefixes_title_and_keeps_sentences_whole():
    abstract = "Birinci cümle burada. İkinci cümle burada. Üçüncü cümle burada."
    chunks = chunk_sentences("Başlık", abstract, max_chars=45)
    assert len(chunks) > 1
    assert all(c.startswith("Başlık\n\n") for c in chunks)
    # her chunk'ın gövdesi noktayla biter: cümle ortasından kesilmemiş
    assert all(c.split("\n\n", 1)[1].endswith(".") for c in chunks)


def test_chunk_sentences_covers_all_sentences():
    abstract = "A birinci. B ikinci. C üçüncü."
    body = " ".join(c.split("\n\n", 1)[1] for c in chunk_sentences("T", abstract, max_chars=15))
    assert body == abstract


def test_format_source_includes_author_year_title_and_pmid():
    meta = {"first_author": "Han", "year": "2026", "title": "Başlık", "pmid": "1"}
    assert format_source(meta) == "Han et al., 2026 — Başlık (PMID 1)"


def test_retrieval_metrics_hit_at_second_rank():
    m = retrieval_metrics(["a", "b", "c"], ["b"])
    assert m == {"hit": True, "recall": 1.0, "reciprocal_rank": 0.5}


def test_retrieval_metrics_partial_recall_multi_reference():
    m = retrieval_metrics(["a", "b"], ["a", "x", "y", "z"])
    assert m["recall"] == 0.25 and m["reciprocal_rank"] == 1.0


def test_retrieval_metrics_miss():
    assert retrieval_metrics(["a"], ["b"]) == {"hit": False, "recall": 0.0, "reciprocal_rank": 0.0}


def _metas(*pmids):
    return [{"pmid": p} for p in pmids]


def test_select_chunks_keeps_complementary_chunks_beyond_best_chunk_threshold():
    # eşik yalnızca en iyi chunk'a uygulanır; marj içindeki chunk 0.65'i aşsa da kalır
    docs = ["d1", "d2"]
    selected = select_chunks(docs, _metas("a", "b"), [0.61, 0.69], max_distance=0.65, relative_margin=0.15)
    assert [d for d, _ in selected] == ["d1", "d2"]


def test_select_chunks_returns_empty_when_best_chunk_is_too_far():
    assert select_chunks(["d1"], _metas("a"), [0.9], max_distance=0.7) == []


def test_select_chunks_drops_noise_far_from_best_chunk():
    docs = ["d1", "d2", "d3"]
    selected = select_chunks(
        docs, _metas("a", "a", "b"), [0.31, 0.33, 0.50], max_distance=0.72, relative_margin=0.15
    )
    assert [d for d, _ in selected] == ["d1", "d2"]


def test_select_chunks_prefers_one_chunk_per_paper_when_slots_are_limited():
    docs = ["d1", "d2", "d3", "d4"]
    selected = select_chunks(docs, _metas("a", "a", "a", "b"), [0.6, 0.61, 0.62, 0.63], n_results=2)
    assert [d for d, _ in selected] == ["d1", "d4"]


def test_select_chunks_fills_remaining_slots_with_closest_chunks_in_distance_order():
    docs = ["d1", "d2", "d3", "d4"]
    selected = select_chunks(docs, _metas("a", "a", "a", "b"), [0.6, 0.61, 0.62, 0.63], n_results=3)
    assert [d for d, _ in selected] == ["d1", "d2", "d4"]


def test_select_chunks_returns_all_chunks_of_a_single_relevant_paper():
    docs = ["d1", "d2", "d3"]
    selected = select_chunks(docs, _metas("a", "a", "a"), [0.31, 0.32, 0.34])
    assert [d for d, _ in selected] == ["d1", "d2", "d3"]


def test_select_chunks_respects_n_results():
    docs = ["d1", "d2", "d3"]
    selected = select_chunks(docs, _metas("a", "b", "c"), [0.6, 0.61, 0.62], n_results=2)
    assert len(selected) == 2


class _FakeChunk:
    def __init__(self, text):
        self.text = text


class _FakeClient:
    """generate_content_stream'e verilen parçaları sırayla akıtan sahte istemci."""

    def __init__(self, pieces):
        self.models = self
        self._pieces = pieces

    def generate_content_stream(self, **kwargs):
        return iter(_FakeChunk(p) for p in self._pieces)


def _run_stream(pieces):
    out: list[str] = []
    answered = stream_answer(_FakeClient(pieces), "soru", ["bağlam"], out.append)
    return "".join(out), answered


def test_build_prompt_contains_question_context_and_marker_instruction():
    prompt = build_prompt("Soru burada?", ["bağlam A", "bağlam B"])
    assert "Soru burada?" in prompt and "bağlam A" in prompt and "bağlam B" in prompt
    assert NO_ANSWER_MARKER in prompt


def test_split_no_answer_marker_strips_marker():
    assert split_no_answer_marker(f"{NO_ANSWER_MARKER} Bilgi yok.") == ("Bilgi yok.", False)


def test_split_no_answer_marker_leaves_normal_answer_untouched():
    assert split_no_answer_marker("Doğruluk %95.") == ("Doğruluk %95.", True)


def test_stream_answer_normal_answer_passes_through_and_is_answered():
    text, answered = _run_stream(["Doğruluk ", "%95,13", " bulundu."])
    assert text == "Doğruluk %95,13 bulundu." and answered


def test_stream_answer_detects_marker_split_across_chunks():
    text, answered = _run_stream(["[BILGI", "_YOK] Bağlamda ", "bilgi yok."])
    assert text == "Bağlamda bilgi yok." and not answered


def test_stream_answer_handles_stream_shorter_than_marker():
    text, answered = _run_stream(["Evet."])
    assert text == "Evet." and answered


def _stub_pubmed(monkeypatch, search, index):
    class _Collection:
        def get(self, **kwargs):
            return {"ids": []}

    monkeypatch.setattr(main, "search_pubmed", search)
    monkeypatch.setattr(main, "fetch_papers", lambda pmids: [{"pmid": p} for p in pmids])
    monkeypatch.setattr(main, "index_paper", index)
    monkeypatch.setattr(main.time, "sleep", lambda s: None)
    return _Collection()


def test_index_papers_returns_true_when_everything_succeeds(monkeypatch):
    collection = _stub_pubmed(monkeypatch, lambda term, retmax: ["1", "2"], lambda c, col, p: 3)
    assert main.index_papers(None, collection, ["t"]) is True


def test_index_papers_returns_false_when_a_paper_fails_to_index(monkeypatch):
    def flaky_index(client, collection, paper):
        if paper["pmid"] == "2":
            raise main.errors.APIError(500, {"error": {"message": "boom", "status": "INTERNAL"}})
        return 3

    collection = _stub_pubmed(monkeypatch, lambda term, retmax: ["1", "2"], flaky_index)
    assert main.index_papers(None, collection, ["t"]) is False


def test_index_papers_returns_false_when_a_search_fails(monkeypatch):
    def failing_search(term, retmax):
        raise main.httpx.ConnectError("no network")

    collection = _stub_pubmed(monkeypatch, failing_search, lambda c, col, p: 3)
    assert main.index_papers(None, collection, ["t"]) is False


def test_select_chunks_limits_diverse_slots_to_keep_depth_for_narrow_questions():
    docs = ["d1", "d2", "d3", "d4"]
    # n_diverse=1: yalnızca 'a' makalesi çeşitlilik hakkı alır; kalan yerler en yakın chunk'larla dolar
    selected = select_chunks(
        docs, _metas("a", "b", "a", "c"), [0.30, 0.31, 0.32, 0.33], n_results=3, n_diverse=1
    )
    assert [d for d, _ in selected] == ["d1", "d2", "d3"]
