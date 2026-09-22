import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable

import chromadb
import httpx
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
GENERATION_MODEL = "gemini-3.5-flash"
NO_ANSWER_MARKER = "[BILGI_YOK]"  # model cevap bulamayınca yanıtın başına bunu koyar
# Ölçüm: ilgili sorularda en yakın chunk 0.31–0.61, alakasız veya koleksiyon dışı sorularda ≥ 0.65.
# Eşik yalnızca EN İYİ chunk'a uygulanır; geniş sorularda (örn. "hangi yaklaşımlar") tamamlayıcı
# chunk'lar 0.69'a kadar uzanabildiği için diğerleri göreli marjla süzülür.
MAX_DISTANCE = 0.65
RELATIVE_MARGIN = 0.15  # en iyi chunk'tan bu kadar uzaktakiler gürültü sayılır
N_CANDIDATES = 30  # Chroma'dan çekilen aday sayısı
N_RESULTS = 8  # bağlama girecek en fazla chunk
N_DIVERSE = 5  # bunların en fazla kaçı farklı makalelerden gelsin (kalanı derinlik için)
METADATA_FIELDS =("pmid", "title", "journal", "year", "first_author")
SEARCH_TERMS = [
    "SSVEP brain computer interface",
    "P300 speller",
    "BCI calibration",
]


def search_pubmed(query: str, retmax: int = 15) -> list[str]:
    response = httpx.get(
        f"{BASE_URL}esearch.fcgi",
        params={"db": "pubmed", "term": query, "retmode": "json", "retmax": retmax},
    )
    response.raise_for_status()
    return response.json()["esearchresult"]["idlist"]


def fetch_papers(pmids: list[str]) -> list[dict]:
    """PubMed XML'inden başlık, abstract ve künye alanlarını ayrı ayrı çıkarır."""
    response = httpx.get(
        f"{BASE_URL}efetch.fcgi",
        params={"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"},
        timeout=30,
    )
    response.raise_for_status()

    papers = []
    for node in ET.fromstring(response.text).findall("PubmedArticle"):
        article = node.find("MedlineCitation/Article")
        abstract_parts = []
        for part in article.findall("Abstract/AbstractText"):
            text = "".join(part.itertext()).strip()
            label = part.get("Label")
            abstract_parts.append(f"{label}: {text}" if label else text)
        if not abstract_parts:
            continue  # abstract'ı olmayan kayıtlar indekslenecek içerik taşımaz

        pub_date = article.find("Journal/JournalIssue/PubDate")
        year = pub_date.findtext("Year") or (pub_date.findtext("MedlineDate") or "")[:4]
        first_author = article.findtext("AuthorList/Author/LastName") or ""
        papers.append(
            {
                "pmid": node.findtext("MedlineCitation/PMID"),
                "title": "".join(article.find("ArticleTitle").itertext()).strip(),
                "abstract": " ".join(abstract_parts),
                "journal": article.findtext("Journal/Title") or "",
                "year": year,
                "first_author": first_author,
            }
        )
    return papers


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap, chunk_size'dan küçük olmalı")
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def chunk_sentences(title: str, abstract: str, max_chars: int = 800) -> list[str]:
    """Abstract'ı cümle sınırlarında böler; her chunk'ın başına makale başlığını ekler."""
    sentences = re.split(r"(?<=[.!?])\s+", abstract)
    groups: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + 1 + len(sentence) > max_chars:
            groups.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        groups.append(current)
    return [f"{title}\n\n{g}" for g in groups]


def embed_texts(client: genai.Client, texts: list[str], task_type: str) -> list[list[float]]:
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return [e.values for e in response.embeddings]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))


def index_paper(client: genai.Client, collection: chromadb.Collection, paper: dict) -> int:
    chunks = chunk_sentences(paper["title"], paper["abstract"])
    embeddings = embed_texts(client, chunks, task_type="RETRIEVAL_DOCUMENT")
    ids = [f"{paper['pmid']}_{i}" for i in range(len(chunks))]
    metadatas = [{k: paper[k] for k in METADATA_FIELDS} for _ in chunks]
    # add yerine upsert: script tekrar çalışınca aynı id'ler hata vermesin
    collection.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    return len(chunks)


def format_source(meta: dict) -> str:
    author = f"{meta['first_author']} et al." if meta["first_author"] else "Yazar bilinmiyor"
    return f"{author}, {meta['year']} — {meta['title']} (PMID {meta['pmid']})"


def build_prompt(question: str, documents: list[str]) -> str:
    context = "\n\n---\n\n".join(documents)
    return (
        f"Aşağıdaki bağlamı kullanarak soruyu cevapla. Bağlamda olmayan bilgiyi "
        f"UYDURMA. Bağlam soruyu cevaplamaya yetmiyorsa yanıtına tam olarak "
        f"{NO_ANSWER_MARKER} yazarak başla, ardından bilgi bulunmadığını kısaca açıkla.\n\n"
        f"Bağlam:\n{context}\n\nSoru: {question}"
    )


def split_no_answer_marker(text: str) -> tuple[str, bool]:
    """Modelin 'bilgi yok' işaretini ayıklar; (temiz metin, cevap bulundu mu) döndürür."""
    stripped = text.lstrip()
    if stripped.startswith(NO_ANSWER_MARKER):
        return stripped.removeprefix(NO_ANSWER_MARKER).lstrip(), False
    return text, True


def stream_answer(
    client: genai.Client, question: str, documents: list[str], on_text: Callable[[str], None]
) -> bool:
    """Cevabı akıtır ve `on_text`'e iletir; model cevap bulabildiyse True döndürür.

    İşaret cevabın başında geldiği için ilk birkaç karakter tamponlanır, karar verildikten
    sonra akış olduğu gibi devam eder.
    """
    buffer, decided, answered = "", False, True
    for chunk in client.models.generate_content_stream(
        model=GENERATION_MODEL,
        contents=build_prompt(question, documents),
        config=types.GenerateContentConfig(max_output_tokens=3000),
    ):
        if not chunk.text:
            continue
        if decided:
            on_text(chunk.text)
            continue
        buffer += chunk.text
        if len(buffer.lstrip()) >= len(NO_ANSWER_MARKER):
            decided = True
            text, answered = split_no_answer_marker(buffer)
            on_text(text)
    if not decided:  # akış işaret uzunluğundan kısa bitti
        text, answered = split_no_answer_marker(buffer)
        on_text(text)
    return answered


def translate_query(client: genai.Client, question: str) -> str:
    """Makaleler İngilizce; soruyu İngilizceye çevirmek eşleşmeyi güçlendirir."""
    try:
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=(
                "Translate this question to English for a PubMed literature search. "
                f"Output only the translation.\n\n{question}"
            ),
        )
    except errors.APIError:
        return question  # çeviri başarısızsa orijinal soruyla aramaya devam et
    return (response.text or "").strip() or question


def select_chunks(
    documents: list[str],
    metadatas: list[dict],
    distances: list[float],
    n_results: int = N_RESULTS,
    n_diverse: int = N_DIVERSE,
    max_distance: float = MAX_DISTANCE,
    relative_margin: float = RELATIVE_MARGIN,
) -> list[tuple[str, dict]]:
    """Mesafeye göre sıralı adaylardan alakalı ve çeşitli bir chunk kümesi seçer.

    Chroma mesafesinde düşük = benzer. En iyi chunk `max_distance`'ı aşıyorsa soru koleksiyonun
    dışındadır ve hiçbir şey seçilmez. Aksi halde en iyi chunk'tan `relative_margin` kadar uzak
    chunk'lar gürültü sayılıp elenir. Kalanlardan önce en fazla `n_diverse` makalenin en iyi
    chunk'ı alınır (geniş sorularda birçok makale kapsansın), boş yerler sonra en yakın chunk'larla
    doldurulur (dar sorularda tek makalenin birden çok chunk'ı gerekebilir; hepsini çeşitliliğe
    ayırmak bunları dışarıda bırakırdı). Sonuç mesafe sırasındadır.
    """
    if not distances or distances[0] > max_distance:
        return []
    eligible = [
        i for i, dist in enumerate(distances) if dist <= distances[0] + relative_margin
    ]

    chosen: list[int] = []
    seen_papers: set[str] = set()
    for i in eligible:  # 1. geçiş: makale başına en iyi chunk
        if metadatas[i]["pmid"] not in seen_papers and len(chosen) < min(n_diverse, n_results):
            seen_papers.add(metadatas[i]["pmid"])
            chosen.append(i)
    for i in eligible:  # 2. geçiş: kalan yerleri en yakın chunk'larla doldur
        if i not in chosen and len(chosen) < n_results:
            chosen.append(i)
    return [(documents[i], metadatas[i]) for i in sorted(chosen)]


def retrieve(
    client: genai.Client, collection: chromadb.Collection, question: str, n_results: int = N_RESULTS
) -> tuple[list[str], list[dict]]:
    if collection.count() == 0:
        return [], []
    query = translate_query(client, question)
    query_embedding = embed_texts(client, [query], task_type="RETRIEVAL_QUERY")[0]
    results = collection.query(
        query_embeddings=[query_embedding], n_results=min(N_CANDIDATES, collection.count())
    )
    relevant = select_chunks(
        results["documents"][0], results["metadatas"][0], results["distances"][0], n_results=n_results
    )
    documents = [doc for doc, _ in relevant]
    # aynı makaleden birden fazla chunk gelebilir; kaynak listesinde bir kez göster
    sources = list({meta["pmid"]: meta for _, meta in relevant}.values())
    return documents, sources


def index_papers(
    client: genai.Client, collection: chromadb.Collection, terms: list[str], retmax: int = 10
) -> bool:
    """Makaleleri toplayıp indeksler; her arama ve makale başarılıysa True döndürür.

    Zaten indekslenmiş makaleler atlanır, bu yüzden başarısız bir çalıştırmadan sonra
    tekrar çağrılınca yalnızca eksikler tamamlanır.
    """
    complete = True
    pmids: list[str] = []
    for term in terms:
        try:
            found = search_pubmed(term, retmax=retmax)
        except httpx.HTTPError as e:
            print(f"[Uyarı] '{term}' araması başarısız, atlanıyor: {e}")
            complete = False
            continue
        print(f"'{term}': {len(found)} makale")
        pmids.extend(p for p in found if p not in pmids)
        time.sleep(0.4)  # NCBI: anahtarsız en fazla ~3 istek/sn

    new_pmids = [p for p in pmids if not collection.get(where={"pmid": p}, limit=1)["ids"]]
    if not new_pmids:
        return complete
    try:
        papers = fetch_papers(new_pmids)  # tek istekte hepsi
    except (httpx.HTTPError, ET.ParseError) as e:
        print(f"[Uyarı] Makaleler çekilemedi: {e}")
        return False

    for paper in papers:
        try:
            n_chunks = index_paper(client, collection, paper)
        except errors.APIError as e:
            print(f"[Uyarı] PMID {paper['pmid']} indekslenemedi, atlanıyor: {e}")
            complete = False
            continue
        print(f"PMID {paper['pmid']}: {n_chunks} chunk indekslendi")
    return complete


_client: genai.Client | None = None
_collection: chromadb.Collection | None = None


def _get_client_and_collection() -> tuple[genai.Client, chromadb.Collection]:
    """Client/koleksiyonu tembel açar ve modül düzeyinde saklar (API sunucusu bu
    fonksiyonu istek başına değil, her process için bir kez çağırmalı)."""
    global _client, _collection
    if _client is None:
        load_dotenv()
        _client = genai.Client()
    if _collection is None:
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        _collection = chroma_client.get_or_create_collection(name="bci_abstracts")
    return _client, _collection


def answer_question(question: str, n_results: int = N_RESULTS) -> str:
    """chat_loop'un stream+print akışının stream'siz hali — API katmanı için."""
    client, collection = _get_client_and_collection()
    documents, sources = retrieve(client, collection, question, n_results=n_results)
    if not documents:
        return "İndekslenen literatürde bu soruyla ilgili bir içerik bulamadım."

    parts: list[str] = []
    try:
        answered = stream_answer(client, question, documents, parts.append)
    except errors.APIError as e:
        return f"[Hata] API isteği başarısız oldu ({e.status}): {e.message}"

    answer = "".join(parts)
    if answered and sources:
        source_lines = "\n".join(f"  - {format_source(meta)}" for meta in sources)
        answer = f"{answer}\n\nKaynaklar:\n{source_lines}"
    return answer


def chat_loop(client: genai.Client, collection: chromadb.Collection) -> None:
    print("Çıkmak için 'exit' yaz.\n")
    while True:
        try:
            question = input("Sen: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() == "exit":
            break
        if not question:
            continue

        print("Asistan: ", end="")
        try:
            documents, sources = retrieve(client, collection, question)
            if not documents:
                print("İndekslenen literatürde bu soruyla ilgili bir içerik bulamadım.\n")
                continue
            answered = stream_answer(
                client, question, documents, lambda text: print(text, end="", flush=True)
            )
            print("\n")
            if answered:  # model "bilgi yok" dediyse kaynak listesi yanıltıcı olur
                print("Kaynaklar:")
                for meta in sources:
                    print(f"  - {format_source(meta)}")
                print()
        except errors.APIError as e:
            print(f"\n[Hata] API isteği başarısız oldu ({e.status}): {e.message}\n")


def main() -> None:
    load_dotenv()
    client = genai.Client()  # GEMINI_API_KEY ortam değişkeninden otomatik okunur
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="bci_abstracts")

    # "indexed" bayrağı yalnızca her arama ve makale başarıyla işlenince yazılır:
    # bir şey atlanırsa bir sonraki çalıştırmada yalnızca eksikler tamamlanır.
    if not (collection.metadata or {}).get("indexed"):
        print("İlk çalıştırma: makaleler toplanıp indeksleniyor...")
        if index_papers(client, collection, SEARCH_TERMS):
            collection.modify(metadata={"indexed": True})
        else:
            print("[Uyarı] İndeksleme eksik kaldı; bir sonraki çalıştırmada tamamlanacak.")
    print(f"Koleksiyonda toplam {collection.count()} chunk var\n")

    chat_loop(client, collection)


if __name__ == "__main__":
    main()