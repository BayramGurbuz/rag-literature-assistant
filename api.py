import os

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from main import answer_question  # Faz 2'deki fonksiyonun

load_dotenv()  # main.py bunu tembel çağırır; burada erken kontrol için hemen yapıyoruz
# Eksikse /ask ilk çağrıldığında değil, container başlarken hemen patlasın —
# Render'da "deploy oldu ama çalışmıyor" yerine "deploy başarısız oldu" görünür.
if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError("GEMINI_API_KEY ortam değişkeni ayarlanmamış.")

app = FastAPI(title="BCI Literatür Asistanı")


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
