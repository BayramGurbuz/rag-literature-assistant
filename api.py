import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from main import answer_question, startup_index  # Faz 2'deki fonksiyonların

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
