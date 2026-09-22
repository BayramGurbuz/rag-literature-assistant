# Faz 2 Raporu — BCI/Nörobilim Literatür Asistanı (RAG)

**Proje:** PubMed'den BCI/SSVEP/P300 makalelerini çekip indeksleyen, sorulara **kaynak göstererek** cevap veren ve RAGAS ile değerlendirilen bir RAG pipeline'ı.
**Klasör:** `Faz 2/rag-literature-assistant/`
**Commit:** `b8a7743` (main)
**Modeller:** üretim ve hakem `gemini-3.5-flash`, embedding `gemini-embedding-001` (3072 boyut)
**Vektör DB:** Chroma (`PersistentClient`, yerel diske kalıcı)

---

## 1. Özet

Faz 2'de uçtan uca çalışan bir RAG sistemi kuruldu ve **ölçülerek** iyileştirildi:

1. PubMed'den makale çekme (XML), cümle-farkında chunking, embedding ve Chroma'ya indeksleme.
2. Türkçe soru alıp İngilizce makalelerde arayan, cevabı akıtan (streaming) ve kaynak gösteren interaktif sohbet döngüsü.
3. Koleksiyonda olmayan sorularda uydurma yapmayan, "bilmiyorum" diyen davranış.
4. 18 soruluk el yapımı değerlendirme seti, retrieval metrikleri ve RAGAS (Gemini hakem) ile ölçüm.
5. 28 birim test (API'ye dokunmayan, deterministik).

**Ana sonuçlar**

| Ölçüm | Sonuç |
|---|---|
| Retrieval: doğru makale ilk sonuçlarda (hit rate) | 16/16 = 1.00 |
| Retrieval: MRR | 1.00 |
| Retrieval: recall (son hal) | 1.00 |
| RAGAS faithfulness | 1.000 (15 soru; q16 hesaplanamadı) |
| RAGAS answer relevancy | 0.951 |
| RAGAS context precision | 0.897 |
| RAGAS context recall | 0.988 |
| RAG'siz vs RAG'li (spesifik rakam sorusu) | RAG'siz **yanlış** (%90,83), RAG'li **doğru** (%95,13) |

> Bu skorların mutlak başarı olarak değil, iyileştirmeleri karşılaştırma aracı olarak okunması gerekir. Nedenleri 8. bölümde.

---

## 2. Mimari

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
 stream_answer  ──► cevap akar; "[BILGI_YOK]" ayıklanır
        │
        ▼
 Kaynaklar (yazar, yıl, başlık, PMID)  — model bilmiyorsa gösterilmez
```

**İndeksleme (ilk çalıştırma):** `SEARCH_TERMS` (SSVEP, P300 speller, BCI calibration) → PubMed esearch → PMID'ler (mükerrer ayıklanır) → tek istekte efetch XML → başlık/abstract/künye ayrıştırma → cümle sınırlarında chunking → `RETRIEVAL_DOCUMENT` embedding → Chroma `upsert`. Koleksiyon adı `bci_abstracts`, sonuç: **27 makale, 82 chunk**.

---

## 3. Alıştırmalar

### Alıştırma 1 — PubMed'den veri çekme
`search_pubmed` (esearch, JSON) ve `fetch_abstracts` (efetch, düz metin) bu fazın başında hazırdı. Sonradan XML'e geçildi (bkz. 4.1).

### Alıştırma 2 — Chunking
Ders kodundaki `chunk_text` (karakter tabanlı, `overlap`'li) 15 makalenin abstract'ı üzerinde denendi (toplam 38.638 karakter):

| chunk_size | overlap | chunk sayısı |
|---|---|---|
| 50 | 5 | 859 |
| 2000 | 200 | 22 |

Gözlemler:
- Chunk sayısı boyutla kabaca ters orantılı: `≈ metin / (chunk_size − overlap)`.
- 50'lik chunk'larda kelime ortasından kesilme yaygın (`"...Calibration-Efficient Dual-Frequenc"`).
- 2000'lik chunk'lar birden fazla makaleye taşıyor; vektör birkaç konunun ortalaması oluyor ("sulanma" sorunu).

**Bulunan hata:** Ders kodunda `chunk_size=50` ve varsayılan `overlap=50` verilince `start += 0` olur ve döngü **sonsuza kadar** sürer. `overlap >= chunk_size` için `ValueError` eklendi ve test edildi.

### Alıştırma 3 — Embedding
`embed_texts` (toplu istek, `task_type` parametreli) ve `cosine_similarity` yazıldı.

| Çift | Cosine |
|---|---|
| "SSVEP nedir?" ↔ "Steady-state visually evoked potential nasıl çalışır?" | 0.914 |
| "SSVEP nedir?" ↔ "Bugün hava çok güzel, dışarı çıkmalıyım." | 0.721 |

Ders "alakasız çift düşük çıkar" beklentisi verdi, ancak 0.72 yüksek. Bu modelde vektörler uzayın dar bir bölgesinde toplanıyor, **mutlak skor yanıltıcı, sıralama önemli**. Bu bulgu daha sonra eşik tasarımını (5.3) doğrudan etkiledi.

### Alıştırma 4 — Vector DB (Chroma)
5 makale (30 chunk) indekslendi. Sonuçlar:
- Keyword search ("nöral kontrol"): **0 sonuç**.
- Semantic search ("nöral kontrol sistemi nedir"): 3 BCI chunk'ı döndü. Ortak kelime olmadan ilgili içeriği buldu.
- Zayıf nokta: üçüncü sonuç makale sonundaki "Copyright / Conflict of interest" gürültüsüydü. Bu, 4.1'deki yeniden yapılandırmanın gerekçesi oldu.

Ders kodundaki `collection.add` yerine `upsert` kullanıldı (script tekrar çalışınca aynı id hata vermesin).

### Alıştırma 5 — Retrieval + Generation: RAG'siz vs RAG'li
Aynı spesifik soru (8 komutlu UAV görevinde doğruluk) iki şekilde soruldu:

| | Cevap | Gerçek |
|---|---|---|
| RAG'siz | %90,83 (ve uydurma bir "%91,56") | **Yanlış** |
| RAG'li | %95,13 ± 4,39, PMID ile | **Doğru** |

RAG'siz model "bilmiyorum" demedi; yazar adlarına dayanıp kendinden emin, makul görünen ama **uydurma** rakamlar üretti. Bu, RAG'in varlık nedeninin somut kanıtı.

### Alıştırma 6 — Test
API'ye hiç dokunmayan deterministik testler. `pyproject.toml`'a `pytest` dev bağımlılığı ve `pythonpath = ["."]` eklendi. Ders testlerine ek olarak `ValueError` davranışı test edildi. Sonraki adımlarda test seti **28 teste** çıktı (bkz. 7).

---

## 4. Proje: her şeyi birleştirme

### 4.1 Veri kalitesi: yapılandırılmış chunking
**Sorun:** Ham metin 500 karakterlik dilimlere bölününce başlık, yazar, kurum ve "Conflict of interest" gürültüsü chunk'ları dolduruyordu.

**Çözüm:**
- `fetch_papers`: `retmode=xml` ile başlık, abstract (etiketli bölümler korunarak), dergi, yıl, ilk yazar ayrı alanlar olarak çekilir. Abstract'ı olmayan kayıtlar atlanır.
- `chunk_sentences`: yalnızca abstract, cümle sınırlarında ≤ 800 karakterlik parçalara bölünür; her chunk'ın başına **makale başlığı** eklenir (chunk tek başına bağlam taşır).
- Metadata: `pmid, title, journal, year, first_author`. Kaynaklar "Yu et al., 2026 — başlık (PMID …)" biçiminde gösterilir.
- Sonuç: chunk sayısı 173'ten 82'ye düştü (gürültü gitti), cevap kalitesi belirgin arttı.

### 4.2 İnteraktif sohbet döngüsü
`chat_loop`: soru → retrieval → streaming cevap → kaynaklar. Boş girdi atlanır; `exit`, Ctrl+C, Ctrl+Z ile çıkılır. **Hata yönetimi:** Faz 1'deki `except errors.APIError` deseni burada da var; hata turu bozmaz, döngü devam eder. İndekslemede PubMed `httpx.HTTPError` ve embedding `APIError` hataları makale bazında yakalanıp atlanır.

### 4.3 Kalıcılık
Chroma `PersistentClient` sayesinde makaleler yalnızca ilk çalıştırmada toplanır. Koleksiyon metadata'sındaki `indexed` bayrağı **yalnızca her arama ve makale başarıyla işlendiğinde** yazılır; bir şey atlanırsa sonraki çalıştırma yalnızca eksikleri tamamlar (zaten indeksli PMID'ler atlanır).

---

## 5. Retrieval iyileştirmeleri (ölçüme dayalı)

### 5.1 Sorgu çevirisi
Sorular Türkçe, makaleler İngilizce. `translate_query` soruyu retrieval'dan önce İngilizceye çevirir (başarısız olursa orijinal soruyla devam eder). Ölçüm: mesafeleri tutarlı biçimde ~0.05 düşürüyor, sıralamayı az değiştiriyor.

### 5.2 Mesafe ölçümü

| Sorgu türü | En yakın chunk mesafesi |
|---|---|
| Dar, ilgili sorular | 0.31 – 0.51 |
| Geniş sorular ("hangi yaklaşımlar") | 0.61 – 0.69 |
| Alakasız / koleksiyon dışı | ≥ 0.65 (çoğunlukla 0.76 – 0.88) |

### 5.3 Seçim stratejisi (`select_chunks`) ve iterasyon geçmişi
Nihai kural, üç denemenin ardından şekillendi:

1. **Mutlak eşik 0.65:** alakasız soruları reddetti, ancak geniş soruda (q16) tamamlayıcı chunk'lar 0.675–0.687'de kaldığı için elendi (recall 0.50).
2. **Eşiği 0.72'ye çıkarmak:** q16 düzeldi ama Neuralink sorusu (q17) artık **reddedilmez** oldu. Regresyon, geri alındı. Tek mutlak eşik iki durumu birlikte ayıramıyor.
3. **Çözüm:** eşik yalnızca **en iyi chunk'a** uygulanır ("bu soru cevaplanabilir mi?"); diğer chunk'lar **göreli marjla** (en iyi + 0.15) süzülür.

**Çeşitlilik ve derinlik:** "Makale başına en fazla 2 chunk" denemesi, aynı makaledeki en bilgilendirici üçüncü chunk'ı (ör. "yeşil seçildi" cümlesi) kesti. "Önce her makaleden bir chunk" denemesi ise dar soruda (q06) hedef makalenin ikinci ve üçüncü chunk'larını dışarıda bıraktı (recall 1.00 → 0.50). Nihai çözüm: bağlamın 8 chunk'ında **en fazla 5 farklı makale** önce alınır, kalan yerler en yakın chunk'larla doldurulur.

Sabitler: `MAX_DISTANCE=0.65`, `RELATIVE_MARGIN=0.15`, `N_CANDIDATES=30`, `N_RESULTS=8`, `N_DIVERSE=5`.

### 5.4 "Bilmiyorum" davranışı
- Eşik, koleksiyon dışı soruları LLM'e gitmeden reddeder (ör. "Neuralink N1 kaç elektrot?").
- Konuya **yakın** ama koleksiyonda olmayan sorularda (ör. dokunsal uyaranlı P300, q18) mesafeler ilgili sorulardan ayırt edilemez (~0.47), embedding "ilgili" görür. Burada prompt kuralı devreye girer: model cevap bulamazsa yanıtı `[BILGI_YOK]` ile başlatır. `stream_answer` işareti (parçalara bölünmüş gelse bile) ayıklar ve **yanıltıcı kaynak listesini göstermez**. Gerçek çalıştırmada doğrulandı.

---

## 6. Değerlendirme

### 6.1 Değerlendirme seti (`eval_set.json`)
18 soru: **16 cevaplanabilir** (rakam, karşılaştırma, çok makaleli, liste soruları) + **2 koleksiyon dışı**. Sorular ve referans cevaplar koleksiyondaki abstract'lardan çıkarıldı; her sorunun `reference_pmids` alanı var.

### 6.2 Retrieval metrikleri (`evaluate_retrieval.py`)

| Metrik | İlk hal | Son hal |
|---|---|---|
| Hit rate | 16/16 | 16/16 |
| MRR | 1.00 | 1.00 |
| Recall | 0.97 | **1.00** |
| Koleksiyon dışı reddetme | 1/2 | 1/2 |

q18 retrieval düzeyinde reddedilmez (5.4), ancak üretim aşamasında model "bilmiyorum" der.

> Not: "Son hal" recall (1.00), `N_DIVERSE=5` sınırı eklenmeden hemen önceki sürümle ölçüldü. Sınır eklendikten sonra tüm seti değil, yalnızca q05, q06, q10 ve q16'yı retrieval düzeyinde yeniden kontrol ettim (dördünde de referans makaleler ve gereken ifadeler bağlamdaydı). RAGAS ise nihai kodla çalıştırıldı.

### 6.3 RAGAS (`evaluate_ragas.py`)
Metrikler: faithfulness, answer relevancy, context precision (referanslı), context recall. Hakem: Gemini.

| Metrik | İlk ölçüm | Son hal |
|---|---|---|
| Faithfulness | 1.000 | 1.000* |
| Answer relevancy | 0.946 | 0.951 |
| Context precision | 0.901 | 0.897 |
| Context recall | 0.963 | **0.988** |

\*q16'da hakem çıktısı `max_tokens` sınırında kesildiği için hesaplanamadı (NaN); ortalama 15 soru üzerinden. Sınır 4096 → 8192'ye çıkarıldı, **bu değişiklik yeniden çalıştırılıp doğrulanmadı**.

**Net kazanç:** q16 (çok makaleli soru) recall 0.40 → 0.80. Diğer farklar (ör. q11, q15 relevancy) tek çalıştırmayla gürültüden ayırt edilemez.

**Kalıcı zayıf noktalar:** q05 (precision 0.33) ve q10 (0.58). İlk teşhisim ("ilgisiz makaleler gürültü yaratıyor") **yanlıştı**: son halde bağlamda yalnızca doğru makale var, skor yine düşük. Asıl neden makalenin *içinde* ilgili chunk'ın 3. sırada gelmesi (BACKGROUND ve CONCLUSION chunk'ları önde). Bu bir chunk içi sıralama sorunu; çözümü reranker olurdu (yapılmadı).

---

## 7. Testler
`uv run pytest -v` → **28 test geçiyor**, hepsi API'siz ve hızlı (~1.5 sn).

| Grup | Ne test ediyor |
|---|---|
| `chunk_text` | boyut sınırı, overlap paylaşımı, `ValueError` |
| `chunk_sentences` | başlık ön eki, cümlelerin bütün kalması, kapsam |
| `cosine_similarity` | özdeş vektör = 1, dik vektör = 0 |
| `format_source` | kaynak biçimi |
| `select_chunks` | eşik kapısı, göreli marj, çeşitlilik/derinlik, `n_results`, `n_diverse` |
| `retrieval_metrics` | hit, kısmi recall, miss, MRR |
| `build_prompt` / `split_no_answer_marker` / `stream_answer` | işaret talimatı, parçalara bölünmüş işaret, kısa akış (sahte istemciyle) |
| `index_papers` | başarıda `True`, arama veya indeksleme hatasında `False` (monkeypatch ile) |

---

## 8. Sınırlamalar (dürüst değerlendirme)

- **Set kolay:** Sorular abstract'lardan yazıldı, terimler örtüşüyor. Retrieval'ın 1.00 çıkması en iyi senaryoya yakın; gerçek kullanımda sorular daha dağınık olacaktır.
- **Hakem = üretici:** RAGAS'ta hem cevabı üreten hem değerlendiren model Gemini. Faithfulness 1.000 "bu sette halüsinasyon tespit edilmedi" demektir, "halüsinasyon yok" değil.
- **Küçük koleksiyon:** 27 makale, 82 chunk. Eşik değerleri bu koleksiyon ve bu sorularla ölçüldü, koleksiyon büyürse yeniden ayarlanmalı.
- **Tek çalıştırma:** LLM hakem çıktıları değişkendir; ±0.02 civarı farklar anlamlı sayılmamalı.
- **Kapsam:** RAGAS yalnızca cevaplanabilir 16 soruyu değerlendirir. Reddetme davranışı ayrıca ve yalnızca 2 soruyla ölçüldü.
- **Chunk içi sıralama:** q05 ve q10 precision sorunu çözülmedi.

---

## 9. Teknik notlar ve yolda karşılaşılan sorunlar

| Sorun | Çözüm |
|---|---|
| `overlap >= chunk_size` sonsuz döngü | `ValueError` + test |
| Ders kodunda global `client` | `client` ve `collection` parametre olarak geçirilir |
| Windows konsolunda Türkçe karakter bozulması | `python -X utf8` |
| Sistem Python'unda paket yok | Proje `.venv`'i (`uv run`) kullanıldı |
| `ragas 0.4.3` ↔ `langchain-community 0.4.x` uyumsuz (silinmiş modül import ediyor) | `langchain-community<0.4` (0.3.31) sabitlendi, `langchain-google-genai` kaldırıldı |
| RAGAS'ın yerel Google yolu senkron istemci ve bilinen instructor hatası | Gemini'nin **OpenAI uyumlu uç noktası** (`AsyncOpenAI`) hakem olarak kullanıldı |
| RAGAS uzun cevapta hakem çıktısı kesiliyor | `max_tokens` 8192 (doğrulanmadı) |
| NCBI anahtarsız istek sınırı (~3/sn) | İstekler arası 0.4 sn bekleme |
| Kabuk (heredoc) kaçışı kod dosyasında `\n`'leri bozdu | Düzenlemeler Edit aracıyla yapıldı |

---

## 10. Dosya haritası

| Dosya | Amaç |
|---|---|
| `main.py` | Tüm pipeline: PubMed, chunking, embedding, indeksleme, retrieval, sohbet döngüsü |
| `evaluate_retrieval.py` | RAGAS'sız hızlı retrieval ölçümü (hit rate, recall, MRR, reddetme) |
| `evaluate_ragas.py` | RAGAS değerlendirmesi (~10 dk) |
| `eval_set.json` | 18 soruluk değerlendirme seti |
| `ragas_results.json` | Son RAGAS sonuçları (soru bazında cevap ve skorlar) |
| `ragas_results_baseline.json` | İlk ölçümün sonuçları (karşılaştırma için) |
| `tests/test_main.py` | 28 birim test |
| `chroma_db/` | Kalıcı vektör veritabanı (`.gitignore`'da) |
| `.env` | `GEMINI_API_KEY` (`.gitignore`'da) |

**Çalıştırma**
```powershell
cd "Faz 2\rag-literature-assistant"
uv run main.py                   # sohbet döngüsü (ilk çalıştırmada indeksleme)
uv run evaluate_retrieval.py     # hızlı retrieval ölçümü
uv run evaluate_ragas.py         # RAGAS (~10 dk)
uv run pytest -v                 # testler
```

---

## 11. Sonraki adım önerileri

1. **Reranker:** q05 ve q10 gibi chunk içi sıralama sorunlarını çözmek için aday chunk'ları ikinci bir modelle yeniden sıralamak.
2. **Daha zor değerlendirme seti:** Abstract terimlerinden bağımsız, dolaylı ve çok adımlı sorular; ideal olarak koleksiyon dışı soruların da sayısını artırmak.
3. **Bağımsız hakem:** RAGAS'ta hakem olarak farklı bir model ailesi kullanmak (öz-tercih yanlılığını azaltmak için).
4. **Konuşma hafızası:** Faz 1'deki `contents` listesi yaklaşımıyla çok turlu sohbet.
5. **Kapsam:** Daha fazla makale ve ikinci kaynak (arXiv); koleksiyon büyüdükçe eşik değerlerini yeniden ölçmek.
6. **Tezle bağlantı:** Koleksiyonu tez konusuna özel arama terimleriyle genişletip kendi literatür taramasında kullanmak.
