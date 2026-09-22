from unittest.mock import patch

from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_endpoint_returns_answer_field():
    with patch("api.answer_question", return_value="sahte cevap"):
        response = client.post("/ask", json={"question": "SSVEP nedir?"})
    assert response.status_code == 200
    assert response.json() == {"answer": "sahte cevap"}


def test_ask_endpoint_rejects_missing_question():
    response = client.post("/ask", json={})
    assert response.status_code == 422


def test_index_endpoint_returns_pmid_and_chunk_count():
    fake_paper = {"pmid": "123", "title": "T", "abstract": "A"}
    with (
        patch("api.fetch_papers", return_value=[fake_paper]),
        patch("api._get_client_and_collection", return_value=(None, None)),
        patch("api.index_paper", return_value=3),
    ):
        response = client.post("/index", json={"pmid": "123"})
    assert response.status_code == 200
    assert response.json() == {"pmid": "123", "chunks": 3}


def test_index_endpoint_returns_404_when_paper_not_found():
    with patch("api.fetch_papers", return_value=[]):
        response = client.post("/index", json={"pmid": "999999999"})
    assert response.status_code == 404


def test_index_endpoint_rejects_wrong_api_key_when_configured():
    with patch("api.os.getenv", side_effect=lambda k, d=None: "secret" if k == "INDEX_API_KEY" else d):
        response = client.post("/index", json={"pmid": "123"}, headers={"X-Api-Key": "wrong"})
    assert response.status_code == 401


def test_index_endpoint_open_when_no_key_configured():
    fake_paper = {"pmid": "123", "title": "T", "abstract": "A"}
    with (
        patch("api.fetch_papers", return_value=[fake_paper]),
        patch("api._get_client_and_collection", return_value=(None, None)),
        patch("api.index_paper", return_value=1),
    ):
        response = client.post("/index", json={"pmid": "123"})
    assert response.status_code == 200
