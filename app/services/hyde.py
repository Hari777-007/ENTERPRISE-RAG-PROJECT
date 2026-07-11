import re

from app.config import settings
from app.models import RetrievedChunk
from app.services.embedding_service import embed_texts
from app.services.llm_service import generate
from app.services.vector_store import search


_HYDE_SYSTEM_PROMPT = (
    "You are a helpful assistant. Given a user question, write a brief, plausible answer "
    "(2-3 sentences) that would help retrieve relevant documents. Write only the answer, "
    "no preamble."
) # preamble is an introductory statement or explanation that comes before the main content. In this case, the prompt instructs the assistant to provide a brief answer without any introductory remarks.

def _normalize_text(text: str) -> str:
    """Normalize whitespace for deduplication."""
    return re.sub(r"\s+", " ", text.strip()) #\s is a regex pattern that matches any whitespace character (spaces, tabs, newlines, etc.). The + quantifier means "one or more occurrences". So, this regex will match any sequence of one or more whitespace characters. The re.sub function replaces all matches of the regex pattern in the input text with a single space character. Finally, the strip() method removes any leading or trailing whitespace from the resulting string.

class HyDERetriever:
    def __init__(self, num_hypotheses: int | None = None) -> None:
        self.num_hypotheses = num_hypotheses or settings.hyde_num_hypotheses

    def retrieve(self, question: str, top_k: int = 5) -> list[RetrievedChunk]:
        if not question or not question.strip():
            return []

        hypotheses: list[str] = []
        for _ in range(self.num_hypotheses):
            try:
                response = generate(
                    system_prompt=_HYDE_SYSTEM_PROMPT,
                    user_message=question,
                    model=settings.llm_model_answer,
                    temperature=0.7,
                )
                hypothesis = response.get("text", "").strip()
                if hypothesis:
                    hypotheses.append(hypothesis)
            except Exception:
                continue

        all_texts = hypotheses + [question]

        if not all_texts:
            return []
        
        embeddings = embed_texts(all_texts)

        all_results: list[RetrievedChunk] = []

        for embedding in embeddings:
            try:
                results = search(embedding, top_k=top_k)
                all_results.extend(results)
            except Exception:
                continue

        deduped: dict[str, RetrievedChunk] = {}

        for chunk in all_results:
            key = _normalize_text(chunk.text)
            if key not in deduped or chunk.score > deduped[key].score:
                deduped[key] = chunk

        merged = sorted(deduped.values(), key=lambda c: c.score, reverse=True)
        return merged[:top_k]

