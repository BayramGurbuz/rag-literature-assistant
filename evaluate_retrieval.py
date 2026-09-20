"""RAGAS'sız değerlendirme: retrieval doğru makaleyi bulup bulmuyor, alakasız soruyu reddediyor mu."""

import json
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai

from main import retrieve

EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"


def retrieval_metrics(retrieved_pmids: list[str], reference_pmids: list[str]) -> dict:
    """retrieved_pmids sıralı gelmeli (en yakın chunk'ın makalesi önce)."""
    found = [p for p in reference_pmids if p in retrieved_pmids]
    ranks = [retrieved_pmids.index(p) + 1 for p in found]
    return {
        "hit": bool(found),  # en az bir referans makale bulundu mu
        "recall": len(found) / len(reference_pmids),  # referans makalelerin kaçı bulundu
        "reciprocal_rank": 1 / min(ranks) if ranks else 0.0,  # ilk doğru makale kaçıncı sırada
    }


def main() -> None:
    load_dotenv()
    client = genai.Client()
    collection = chromadb.PersistentClient(path="./chroma_db").get_collection("bci_abstracts")
    eval_set = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))

    answerable = [q for q in eval_set if q["type"] == "answerable"]
    unanswerable = [q for q in eval_set if q["type"] == "unanswerable"]

    print("=== Cevaplanabilir sorular ===")
    rows = []
    for q in answerable:
        _, sources = retrieve(client, collection, q["question"])
        retrieved = [m["pmid"] for m in sources]
        metrics = retrieval_metrics(retrieved, q["reference_pmids"])
        rows.append(metrics)
        mark = "✓" if metrics["hit"] else "✗"
        print(
            f"{mark} {q['id']} | referans {q['reference_pmids']} | gelen {retrieved} | "
            f"recall {metrics['recall']:.2f} | RR {metrics['reciprocal_rank']:.2f}"
        )

    n = len(rows)
    print(f"\nHit rate:  {sum(r['hit'] for r in rows)}/{n} = {sum(r['hit'] for r in rows) / n:.2f}")
    print(f"Recall:    {sum(r['recall'] for r in rows) / n:.2f}")
    print(f"MRR:       {sum(r['reciprocal_rank'] for r in rows) / n:.2f}")

    print("\n=== Koleksiyon dışı sorular (boş sonuç beklenir) ===")
    rejected = 0
    for q in unanswerable:
        documents, sources = retrieve(client, collection, q["question"])
        ok = not documents
        rejected += ok
        print(f"{'✓' if ok else '✗'} {q['id']} | gelen chunk sayısı: {len(documents)} | {q['question']}")
    print(f"\nReddetme oranı: {rejected}/{len(unanswerable)}")


if __name__ == "__main__":
    main()
