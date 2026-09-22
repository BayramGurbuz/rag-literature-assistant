# Faz 4 Raporu — Üretime Alma ve CI/CD

**Proje:** Faz 2'nin RAG pipeline'ını bağımsız bir repoya ayırma, FastAPI ile HTTP API'ye sarma, Docker ile konteynerleştirme, GitHub Actions ile CI kurma ve Render'a canlı deploy etme.
**Klasör:** `rag-literature-assistant/` (artık kendi başına bir repo — bkz. Bölüm 0)
**Canlı URL:** https://rag-literature-assistant.onrender.com (`/docs`, `/health`, `/ask`)
**CI:** GitHub Actions, her push/PR'da `pytest` (badge: README'de)

---

## 1. Özet

Faz 4'te proje, tek bir dizüstünde çalışan bir script'ten gerçek bir web servisine dönüştürüldü:

1. `git subtree split` ile geçmiş korunarak bağımsız bir GitHub reposuna taşındı.
2. `chat_loop`'un stream+print mantığından `answer_question()` çıkarılıp FastAPI ile `POST /ask` ve `GET /health` uç noktalarına sarıldı.
3. `uv` resmi çok aşamalı (multi-stage) Docker şablonuyla konteynerleştirildi.
4. GitHub Actions ile her push'ta `pytest` çalıştıran bir CI kuruldu.
5. Render'ın ücretsiz katmanına deploy edildi.

Bu adımların **hepsi gerçekten çalıştırılıp doğrulandı** — `docker build`/`docker run` ile gerçek konteynerler ayağa kaldırıldı, `gh` CLI ile GitHub Actions çalıştırmaları gerçekten izlendi, canlı Render URL'ine gerçek HTTP istekleri atıldı. Bu süreçte rehberin şablon kodunun **doğrudan çalışmadığı 6 gerçek nokta** bulundu ve düzeltildi (Bölüm 5).

---

## 2. Bölüm 0 — Repo Ayırma

`Ai new practise` monorepo'sundaki `Faz 2/rag-literature-assistant` ve `Faz 3/mcp-literatur-server` klasörleri, `git subtree split --prefix=... -b <branch>` ile geçmişleri korunarak ayrıştırılıp iki bağımsız GitHub reposuna push'landı, sonra `Desktop/` altına yeniden klonlandı. Kök monorepo'dan bu klasörler `git rm -r` ile kaldırıldı (yerel commit, `origin`'e push edilmedi — kullanıcı isteğiyle).

**Bulunan gerçek sorun:** Yeni `rag-literature-assistant` reposunda `.gitignore` hiç yoktu (kök depodaki `.gitignore` sadece monorepo'ya aitti, subtree split'e dahil olmadı). `.env` bu yeni repoda izlenmeye açık kalmıştı — biri `git add .` deseydi commit'e girerdi. Fark edilip hemen `.gitignore` eklendi, commit'lendi.

`mcp-literatur-server/paper_server.py`'deki `index_paper`'ın Faz 2'nin diskine doğrudan yazan `CHROMA_DB_PATH = Path(__file__).resolve().parents[2] / "Faz 2" / ...` satırı, iki proje ayrı repolara taşındığı için kırıldı; `RAG_CHROMA_DB_PATH` ortam değişkeniyle yapılandırılabilir, ayarlı değilse özelliği nazikçe kapatan bir hale getirildi.

---

## 3. Alıştırmalar

### Alıştırma 1 — FastAPI ile API'ye sarma

**Ön koşul (rehberde yoktu, kod incelenince fark edildi):** `api.py`'nin `from main import answer_question` satırı, `main.py`'de var olmayan bir fonksiyonu import ediyordu — o zamana kadar sadece interaktif `chat_loop` (retrieve + stream + print) vardı. `answer_question(question, n_results=3) -> str` fonksiyonu, `retrieve()`'e eklenen `n_results` parametresiyle birlikte çıkarıldı; `chat_loop` dokunulmadan bırakıldı.

`api.py` yazıldı: `POST /ask` (Pydantic `AskRequest`/`AskResponse`), `GET /health` (dış bağımlılığa dokunmaz).

**Gerçek testler (hepsi çalıştırıldı):**
- `/docs` (Swagger) açıldı, 200.
- Deney 1: `{"question": 123}` → **422**, `ask()` fonksiyonu hiç çalışmadı (Pydantic doğrulaması istek fonksiyona ulaşmadan reddetti).
- Geçerli soru → başta **500** (bkz. Bölüm 5, boş koleksiyon hatası), düzeltme sonrası **200**.

### Alıştırma 2 — Dockerize etme

Resmi `uv` çok aşamalı Dockerfile şablonu kullanıldı. İlk build, `.python-version` (3.11) ile Dockerfile'ın base image'ı (3.12-slim) uyuşmadığı için patladı (Bölüm 5). Düzeltme sonrası gerçekten build edilip container çalıştırıldı:

- `/health`, `/ask`, 422 testleri konteyner içinde de doğrulandı.
- `${PORT:-8000}` mekanizması `-e PORT=5000` ile bilerek zorlanıp doğrulandı (uvicorn 5000'de dinledi).
- **Deney 2** (slim vs slim-olmayan boyut karşılaştırması) gerçekten yapıldı: slim **1.29GB**, slim-olmayan **2.7GB** — bu projenin ağır bağımlılık listesi (langchain, ragas, onnxruntime, pandas vb.) yüzünden rehberdeki örnekten daha dramatik bir fark.

### Alıştırma 3 — Yerelde çalıştırma ve test

`uv run main.py` ile PubMed'den 28 makale/82 chunk gerçekten indekslendi, bu dolu `chroma_db` ile image build edilip container'a gerçek bir soru ("SSVEP nedir?") soruldu — kaynaklı, doğru bir cevap geldi.

**Deney 3** (env var olmadan çalıştırma) doğrulandı: container ayakta kaldı, `/health` 200 döndü, `/ask` çağrılınca `genai.Client()` içinde `ValueError: No API key was provided` ile 500 patladı — rehberin tahmin ettiği tam senaryo. Bunun üzerine `api.py`'nin başına erken bir kontrol eklendi (Bölüm 5), container artık key yoksa **hiç ayağa kalkmadan** çöküyor.

### Alıştırma 4 — GitHub Actions ile CI

`ci.yml` yazıldı, push'landı. İlk çalıştırma 7 saniyede patladı (Bölüm 5 — `setup-uv@v9` yok). Düzeltme sonrası **gerçekten** `gh run list`/`gh run view` ile izlendi, yeşile döndü (19sn).

**Deney 4** uçtan uca yapıldı: `test_cosine_similarity_identical_vectors_is_one` bilerek `999.0` beklentisiyle kırıldı, push'landı, GitHub Actions'ta **kırmızı çarpı** gerçekten görüldü (32sn, `failure`), gerçek pytest çıktısı okundu (`tests/test_main.py:33 — assert 998.0 < 1e-06`), test düzeltilip push'landı, **yeşile döndüğü** doğrulandı (19sn, `success`).

### Alıştırma 5 — Render'a deploy

Deploy adımlarının kendisi (Render hesabı açma, GitHub bağlama, env var girme, "Create Web Service") kullanıcı tarafından manuel yapıldı — bunlar tarayıcı/hesap gerektiren adımlar. Sonrasında canlı URL'e gerçek istekler atıldı.

**İlk test başarısız:** `/ask` her soruda "İndekslenen literatürde bu soruyla ilgili bir içerik bulamadım" döndürdü — koleksiyon Render'da boştu (Bölüm 5, kök sebep: `chroma_db/` `.gitignore`'da). Düzeltme (`startup_index()`) push'landıktan sonra Render'ın kendi otomatik redeploy'u (~10+ dakika, ağır bağımlılıklar nedeniyle) beklendi, sonra gerçek bir soru ("P300 nedir?") canlı URL'den kaynaklı, doğru bir cevapla yanıtlandı (6.7sn).

---

## 4. Karşılaşılan gerçek sorunlar ve çözümleri

| Sorun | Sebep | Çözüm |
|---|---|---|
| `ModuleNotFoundError` benzeri: `answer_question` yok | `main.py`'de sadece interaktif `chat_loop` vardı, API'nin çağırabileceği stream'siz bir fonksiyon yoktu | `answer_question()` çıkarıldı, `retrieve()`'e `n_results` parametresi eklendi |
| Boş koleksiyonda `/ask` → 500 (`TypeError`) | `collection.query(n_results=0)` Chroma'da hata fırlatıyor; `main()` her zaman önce indeksleyip sonra `chat_loop`'a girdiği için bu hiç tetiklenmemişti | `retrieve()`'in başına `if collection.count() == 0: return [], []` |
| `rag-literature-assistant`'ta `.gitignore` yok | `git subtree split`, kök depodaki `.gitignore`'ı taşımıyor (o dosya subtree'nin dışında) | Yeni `.gitignore` eklendi, commit'lendi |
| Docker build: `No interpreter found for Python 3.11` | Dockerfile şablonu `python:3.12-slim` kullanıyordu, proje `.python-version` ile 3.11'e sabitliydi | Her iki `FROM` satırı da `python:3.11-slim` yapıldı |
| Env var'sız container `/ask`'ta 500, ama ayakta kalıyordu | `genai.Client()` sadece gerçekten kullanılınca (constructor'da) key kontrolü yapıyor, import zamanında değil | `api.py`'nin başına `if not os.getenv("GEMINI_API_KEY"): raise RuntimeError(...)` — container hiç ayağa kalkmadan çöküyor |
| Yukarıdaki kontrol yerelde de yanlışlıkla patlıyordu | `main.py`'deki `load_dotenv()` tembel (ilk `/ask` çağrısında) çalışıyor; erken kontrol ondan **önce** `os.getenv` okuyordu | `api.py`'nin başına da `load_dotenv()` eklendi |
| CI: "Unable to resolve action `astral-sh/setup-uv@v9`" | Rehberdeki tag gerçekte yok (yalnızca `v9.0.0`, `v10.x` var, kayan `v9` etiketi yok) | GitHub API'den gerçek etiketler kontrol edilip `v10.2.0`'a sabitlendi |
| Render'da `/ask` hep "bulunamadı" dönüyordu | `chroma_db/` `.gitignore`'da; Render image'ı **git reposundan** build ediyor (yerel diskten değil), o yüzden "build zamanında göm" stratejisi git-tabanlı deploy'da hiç çalışmıyor | `main.py`'de `ensure_indexed`/`startup_index()` eklendi; `api.py` başlarken koleksiyon boşsa PubMed'den kendini dolduruyor |

---

## 5. Sınırlamalar (dürüst değerlendirme)

- **Render'ın ücretsiz katmanında disk kalıcı değil:** `startup_index()` çözümü, her container yeniden başladığında (uyku/uyanma, redeploy) koleksiyonu **sıfırdan** yeniden indeksliyor — bu, gerçek üretimde kabul edilemez bir maliyet (her cold start'ta PubMed + Gemini API çağrıları, ~20-60 saniye). Doğru çözüm, yönetilen bir vector DB (Pinecone, Weaviate Cloud) ya da Render'ın ücretli kalıcı disk özelliği olurdu. Aynı sebeple, `POST /index` ile eklenen bir makale de bir sonraki cold start'ta kaybolur — `SEARCH_TERMS` listesindeki sabit konularla sınırlı yeniden dolduruluyor.
- ~~`index_paper` (Faz 3) artık production'da devre dışı~~ — **çözüldü, bkz. Bölüm 8.**
- **Eşzamanlılık düşünülmedi:** `main.py`'deki `_client`/`_collection` modül-seviyesi global'ler; Render tek worker'la çalıştığı için sorun değil, ama birden fazla worker/instance olsaydı her biri kendi indekslemesini ayrı ayrı yapardı.
- **Faz 3 (mcp-literatur-server) bu fazın Docker/CI/Render adımlarından geçmedi** — kapsam kararı gereği (bkz. Bölüm 9) bu repo için ayrıca yapılması gerekiyor.
- **`.dockerignore`'da `chroma_db/` hâlâ yok** (Alıştırma 2'nin kararı) ama artık pratik önemi kalmadı — Render zaten `startup_index()` ile dolduruyor; yalnızca yerel `docker build` + önceden `uv run main.py` akışında (CI'sız, saf Docker testi) hâlâ işe yarıyor.

---

## 6. Dosya haritası (yeni/değişen dosyalar)

| Dosya | Amaç |
|---|---|
| `api.py` | FastAPI: `POST /ask`, `GET /health`; başlangıçta `GEMINI_API_KEY` kontrolü + `startup_index()` |
| `Dockerfile` | Çok aşamalı build (`python:3.11-slim`), `${PORT:-8000}` |
| `.dockerignore` | Docker build context'inden hariç tutulanlar (`.venv/`, `tests/`, `.git/` vb.) |
| `.github/workflows/ci.yml` | Push/PR'da `uv sync` + `pytest` |
| `main.py` (değişti) | `answer_question()`, `ensure_indexed()`, `startup_index()` eklendi; `retrieve()` boş koleksiyona karşı korumalı |
| `.gitignore` | Bu repoda yeni eklendi (Bölüm 0) |
| `api.py` (Bölüm 8) | `POST /index` eklendi — `mcp-literatur-server`'ın HTTP üzerinden çağırdığı endpoint |

**Çalıştırma**
```bash
uv sync
uv run main.py                        # CLI sohbet, ilk çalıştırmada indeksler
uv run fastapi dev api.py              # HTTP API, http://127.0.0.1:8000/docs
docker build -t rag-api . && docker run -p 8000:8000 --env-file .env rag-api
uv run pytest -v                       # 28 test
```

---

## 7. Sonraki adım önerileri

1. ~~Faz 3'ü (mcp-literatur-server) aynı derinlikte işleme~~ — **yapıldı**, o reponun kendi `FAZ4_RAPOR.md`'sine bak.
2. **Kalıcı vector DB:** Render'ın her cold start'ta yeniden indekslemesi yerine Pinecone/Weaviate Cloud gibi yönetilen bir servise geçmek — `POST /index` ile eklenen makalelerin de cold start'ta kaybolmaması için bu artık daha önemli.
3. ~~Faz 2 ↔ Faz 3 entegrasyonunu HTTP üzerinden yeniden kurmak~~ — **yapıldı, bkz. Bölüm 8.**
4. **`startup_index()`'in engelleyici (blocking) doğası:** Şu an `/health` bile indeksleme bitene kadar cevap vermiyor (import zamanında çalışıyor); arka planda indeksleyip `/health`'i ayrı bir "ready" durumuyla ayırmak düşünülebilir.

---

## 8. Ek — `/index`: Faz 2 ↔ Faz 3 entegrasyonunu HTTP ile yeniden kurmak

Bölüm 5'te bırakılan tradeoff çözüldü: `mcp-literatur-server`'ın `index_paper` tool'u artık Chroma'ya doğrudan yazmıyor (production'da paylaşılan disk yok), bunun yerine bu servisin `POST /index` endpoint'ini HTTP üzerinden çağırıyor.

**`POST /index`:** `{"pmid": "..."}` alır, `fetch_papers`/`index_paper` (main.py'nin mevcut fonksiyonları, hiç değiştirilmeden yeniden kullanıldı) ile PubMed'den çekip parçalayıp embed'leyip `upsert` eder, `{"pmid", "chunks"}` döner. Geçersiz PMID'de 404.

**`INDEX_API_KEY`:** Bu, gerçek Gemini embedding çağrısı yapan (maliyetli) bir yaz endpoint'i olduğu için, isteğe bağlı bir `X-Api-Key` koruması eklendi — env var ayarlıysa header eşleşmeli, ayarlı değilse (yerel geliştirme) açık kalır. Production'da (Render) ayarlandı; her iki servise de aynı paylaşılan sır verildi.

**Uçtan uca gerçek doğrulama (canlı Render URL'lerinde):**
1. `mcp-literatur-server`'a "PMID 37875091'i indeksle" dendi → agent `index_paper`'ı çağırdı → `POST https://rag-literature-assistant.onrender.com/index` gerçekten gitti → 200, 2 chunk (17sn).
2. Hemen ardından `rag-literature-assistant`'a bu makalenin konusuyla ilgili bir soru soruldu ("neural control of cephalopod camouflage") — **daha önce koleksiyonda hiç olmayan bu konuda**, doğru kaynakla (Montague et al., 2023, PMID 37875091) cevap geldi. Halka kapandı: agent'ın bulduğu bir şey, gerçekten RAG'in bir sonraki cevabında kullanılabiliyor.
3. `INDEX_API_KEY` ayarlıyken doğrudan `/index`'e header'sız istek atıldı → **401**, koruma doğrulandı.

**Yan etki — `mcp-literatur-server` basitleşti:** `chromadb` ve embedding için `google-genai` kullanımı bu repodan tamamen kalktı (41 paket `pyproject.toml`'dan düştü) — artık makale fetch/chunk/embed mantığı yalnızca `rag-literature-assistant`'ta, tek yerde yaşıyor.

**Yol boyunca bulunan ek bir sorun:** `agent_client.py`'nin `StdioServerParameters`'ı, `GEMINI_API_KEY` için daha önce eklenmiş olan açık `env=` geçişini **yalnızca o değişkene** sahipti — `RAG_API_URL`/`INDEX_API_KEY` de aynı allowlist sorununa takılıyordu (varsayılan olarak geçmiyorlardı), ve `paper_server.py` artık `genai.Client()` kullanmadığı için `GEMINI_API_KEY` geçişi zaten gereksizdi. `env=` sözlüğü `RAG_API_URL`/`INDEX_API_KEY`'i geçirecek şekilde güncellendi; `.env` dosyası tamamen kaldırılarak (yalnızca `os.environ`'dan silmek değil) hem yerelde hem Docker'da (host'taki servise `host.docker.internal` ile erişerek) rigorously test edildi.
