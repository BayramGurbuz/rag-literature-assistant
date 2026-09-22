# BCI Literatür Asistanı (RAG)

![CI](https://github.com/BayramGurbuz/rag-literature-assistant/actions/workflows/ci.yml/badge.svg)

PubMed'den beyin-bilgisayar arayüzü (BCI/SSVEP/P300) makalelerini çekip indeksleyen, sorulara **kaynak göstererek** cevap veren bir RAG (Retrieval-Augmented Generation) sistemi. Koleksiyonda karşılığı olmayan sorularda uydurma yapmaz, bilmediğini söyler.

- **Üretim/hakem modeli:** `gemini-3.5-flash`
- **Embedding modeli:** `gemini-embedding-001`
- **Vektör DB:** [Chroma](https://www.trychroma.com/) (`PersistentClient`, yerel diske kalıcı)

## Kurulum

```bash
uv sync
```

`.env` dosyası oluştur:

```
GEMINI_API_KEY=...
```

## Kullanım

### İnteraktif sohbet (terminal)

```bash
uv run main.py
```

İlk çalıştırmada PubMed'den makaleler çekilip indekslenir (birkaç dakika sürebilir), sonra Türkçe soru sorabileceğin bir sohbet döngüsü başlar. Çıkmak için `exit` yaz.

### HTTP API

```bash
uv run fastapi dev api.py
```

`http://127.0.0.1:8000/docs` adresinde interaktif Swagger arayüzü açılır.

| Endpoint | Açıklama |
|---|---|
| `POST /ask` | `{"question": "...", "n_results": 3}` gönder, `{"answer": "..."}` al |
| `GET /health` | Servisin ayakta olup olmadığını kontrol eder (dış bağımlılığa dokunmaz) |

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "SSVEP nedir?"}'
```

### Docker

Koleksiyon, container'ın kalıcı olmayan diskine güvenmemek için **build zamanında image'a gömülür** — yani `docker build`'den önce yerelde indekslemiş olman gerekir:

```bash
uv run main.py          # chroma_db/ dolana kadar bekle
docker build -t rag-api .
docker run -p 8000:8000 --env-file .env rag-api
```

`PORT` ortam değişkeni ayarlıysa (Render gibi platformlarda otomatik) container o porttan dinler, ayarlı değilse `8000`'e düşer. `GEMINI_API_KEY` eksikse container başlarken (uygulama hiç ayağa kalkmadan) hemen hata verir — production'da "deploy oldu ama bozuk" yerine "deploy başarısız" görünmesi için.

## Test ve değerlendirme

```bash
uv run pytest              # 28 birim test (API'ye dokunmaz, deterministik)
uv run evaluate_retrieval.py   # retrieval metrikleri (hit rate, MRR, recall)
uv run evaluate_ragas.py       # RAGAS ile faithfulness / relevancy / precision / recall
```

Her push/PR'da GitHub Actions üzerinde testler otomatik çalışır (`.github/workflows/ci.yml`).

## Mimari

```
Kullanıcı sorusu (Türkçe)
        │
        ▼
 translate_query ──► İngilizce sorgu            (gemini-3.5-flash)
        │
        ▼
 embed_texts (RETRIEVAL_QUERY)                  (gemini-embedding-001)
        │
        ▼
 Chroma.query (30 aday)  ──►  select_chunks
        │                      ├─ en iyi chunk mesafe kapısı (≤ 0.65)
        │                      ├─ göreli marj (en iyi + 0.15)
        │                      └─ çeşitlilik + derinlik (en fazla 8 chunk)
        ▼
 build_prompt (bağlam + soru + "[BILGI_YOK]" kuralı)
        │
        ▼
 stream_answer ──► cevap akar; "[BILGI_YOK]" ayıklanır
```

Detaylı mimari, ölçüm sonuçları ve tasarım kararları için [FAZ2_RAPOR.md](FAZ2_RAPOR.md)'a bak.
