from fastapi import FastAPI
from pydantic import BaseModel

from main import answer_question  # Faz 2'deki fonksiyonun

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
