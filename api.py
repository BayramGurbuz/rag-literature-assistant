import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from main import (  # Faz 2'deki fonksiyonların
    _get_client_and_collection,
    answer_question,
    fetch_papers,
    index_paper,
    startup_index,
)

load_dotenv()  # main.py bunu tembel çağırır; burada erken kontrol için hemen yapıyoruz


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Eksikse /ask ilk çağrıldığında değil, container başlarken hemen patlasın —
    # Render'da "deploy oldu ama çalışmıyor" yerine "deploy başarısız oldu" görünür.
    # (import zamanında değil, burada: TestClient(app) modül import edildiğinde
    # `with` bloğu olmadan çalıştırılırsa lifespan hiç tetiklenmez, testler network'e
    # dokunmadan/API key'siz çalışabilir.)
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY ortam değişkeni ayarlanmamış.")
    # chroma_db, .gitignore'da olduğu için git-tabanlı build'lerde (Render) image'a
    # boş geliyor — koleksiyon boşsa sunucu ilk istekten önce burada doldurur.
    startup_index()
    yield


app = FastAPI(title="BCI Literatür Asistanı", lifespan=lifespan)

STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


class AskRequest(BaseModel):
    question: str
    n_results: int = 3


class AskResponse(BaseModel):
    answer: str


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    answer = answer_question(request.question, n_results=request.n_results)
    return AskResponse(answer=answer)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


class IndexRequest(BaseModel):
    pmid: str


class IndexResponse(BaseModel):
    pmid: str
    chunks: int


@app.post("/index", response_model=IndexResponse)
def index(request: IndexRequest, x_api_key: str | None = Header(default=None)) -> IndexResponse:
    """Verilen PMID'yi PubMed'den çekip bu koleksiyona indeksler.

    mcp-literatur-server'ın index_paper tool'u, artık Chroma'ya doğrudan
    yazmak yerine bu endpoint'i çağırıyor — iki servis ayrı container'larda/
    disklerde çalıştığı için paylaşılan bir dosyaya güvenemiyorlar.

    INDEX_API_KEY ayarlıysa X-Api-Key header'ı onunla eşleşmeli — bu, gerçek
    Gemini embedding çağrısı yapan (maliyetli) bir yaz endpoint'i, herkese
    açık bırakılmasın diye. Ayarlı değilse (örn. yerel geliştirme) açık kalır.
    """
    expected_key = os.getenv("INDEX_API_KEY")
    if expected_key and x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Geçersiz veya eksik X-Api-Key.")

    papers = fetch_papers([request.pmid])
    if not papers:
        raise HTTPException(status_code=404, detail=f"PMID {request.pmid} için abstract bulunamadı.")

    client, collection = _get_client_and_collection()
    n_chunks = index_paper(client, collection, papers[0])
    return IndexResponse(pmid=request.pmid, chunks=n_chunks)
