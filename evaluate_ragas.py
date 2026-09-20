"""RAGAS değerlendirmesi: cevap kalitesi (faithfulness, relevancy) + retrieval kalitesi (precision, recall)."""

import asyncio
import json
import math
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai
from openai import AsyncOpenAI
from ragas.embeddings import GoogleEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from main import GENERATION_MODEL, build_prompt, retrieve, split_no_answer_marker

EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"
RESULTS_PATH = Path(__file__).parent / "ragas_results.json"
# Hakem LLM olarak Gemini'yi OpenAI uyumlu uç nokta üzerinden kullanıyoruz:
# RAGAS'ın yerel Google yolu senkron istemci ve bilinen bir instructor hatası yüzünden sorunlu.
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
CONCURRENCY = 3  # aynı anda değerlendirilen soru sayısı (rate limit için düşük tutuldu)
LOW_SCORE = 0.7


async def safe_score(coro) -> float:
    """Tek bir metrik hata verirse tüm değerlendirmeyi düşürme; NaN döndür."""
    try:
        return float((await coro).value)
    except Exception as e:  # noqa: BLE001 — hakem LLM hataları çeşitli olabilir
        print(f"[Uyarı] metrik hesaplanamadı: {type(e).__name__}: {e}")
        return math.nan


async def evaluate_sample(sample: dict, metrics: dict, sem: asyncio.Semaphore) -> dict:
    async with sem:
        question, reference, contexts, answer = (
            sample["question"],
            sample["ground_truth"],
            sample["contexts"],
            sample["answer"],
        )
        faithfulness, relevancy, precision, recall = await asyncio.gather(
            safe_score(
                metrics["faithfulness"].ascore(
                    user_input=question, response=answer, retrieved_contexts=contexts
                )
            ),
            safe_score(metrics["relevancy"].ascore(user_input=question, response=answer)),
            safe_score(
                metrics["precision"].ascore(
                    user_input=question, reference=reference, retrieved_contexts=contexts
                )
            ),
            safe_score(
                metrics["recall"].ascore(
                    user_input=question, retrieved_contexts=contexts, reference=reference
                )
            ),
        )
        print(f"{sample['id']}: metrikler hesaplandı", flush=True)
        return {
            **sample,
            "faithfulness": faithfulness,
            "answer_relevancy": relevancy,
            "context_precision": precision,
            "context_recall": recall,
        }


def mean_ignoring_nan(values: list[float]) -> float:
    valid = [v for v in values if not math.isnan(v)]
    return sum(valid) / len(valid) if valid else math.nan


def build_samples(client: genai.Client, collection: chromadb.Collection, eval_set: list[dict]) -> list[dict]:
    samples = []
    for q in eval_set:
        documents, sources = retrieve(client, collection, q["question"])
        response = client.models.generate_content(
            model=GENERATION_MODEL, contents=build_prompt(q["question"], documents)
        )
        answer, _ = split_no_answer_marker(response.text or "")
        samples.append(
            {
                "id": q["id"],
                "question": q["question"],
                "ground_truth": q["ground_truth"],
                "contexts": documents,
                "answer": answer,
                "retrieved_pmids": [m["pmid"] for m in sources],
            }
        )
        print(f"{q['id']}: cevap üretildi ({len(documents)} chunk)")
    return samples


async def main() -> None:
    load_dotenv()
    client = genai.Client()
    collection = chromadb.PersistentClient(path="./chroma_db").get_collection("bci_abstracts")
    eval_set = [
        q for q in json.loads(EVAL_SET_PATH.read_text(encoding="utf-8")) if q["type"] == "answerable"
    ]

    print(f"=== 1/2 RAG pipeline'ı {len(eval_set)} soru için çalıştırılıyor ===")
    samples = await asyncio.to_thread(build_samples, client, collection, eval_set)

    judge = llm_factory(
        GENERATION_MODEL,
        provider="openai",
        client=AsyncOpenAI(
            api_key=os.environ["GEMINI_API_KEY"], base_url=GEMINI_OPENAI_BASE_URL
        ),
        max_tokens=8192,  # uzun cevaplarda (örn. q16) hakem çıktısı 4096 token sınırında kesiliyordu
    )
    embeddings = GoogleEmbeddings(client=client, model="gemini-embedding-001")
    metrics = {
        "faithfulness": Faithfulness(llm=judge),
        "relevancy": AnswerRelevancy(llm=judge, embeddings=embeddings),
        "precision": ContextPrecision(llm=judge),
        "recall": ContextRecall(llm=judge),
    }

    print("\n=== 2/2 RAGAS metrikleri hesaplanıyor ===")
    sem = asyncio.Semaphore(CONCURRENCY)
    results = await asyncio.gather(*(evaluate_sample(s, metrics, sem) for s in samples))
    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    print("\n=== Ortalamalar ===")
    for name in names:
        print(f"{name:18s} {mean_ignoring_nan([r[name] for r in results]):.3f}")

    print(f"\n=== {LOW_SCORE} altı skorlar ===")
    for r in results:
        low = {n: r[n] for n in names if not math.isnan(r[n]) and r[n] < LOW_SCORE}
        if low:
            print(f"{r['id']}: " + ", ".join(f"{n}={v:.2f}" for n, v in low.items()))
    print(f"\nAyrıntılı sonuçlar: {RESULTS_PATH.name}")


if __name__ == "__main__":
    asyncio.run(main())
