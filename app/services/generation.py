import json
import socket
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.request import Request, ProxyHandler, HTTPRedirectHandler, build_opener
from app.core.config import PROJECT_ROOT, load_config

from app.services.documents import DocumentError

config = load_config()
generation_config = config["generation"]
context_config = config["context"]

OLLAMA_URL = generation_config["url"]
MODEL = generation_config["model"]
NUM_CTX = generation_config["num_ctx"]
NUM_PREDICT = generation_config["num_predict"]
TIMEOUT_SECONDS = generation_config["socket_timeout_seconds"]
MAX_RESPONSE_BYTES = generation_config["max_response_bytes"]

MAX_PROMPT_BYTES = context_config["max_prompt_bytes"]
MIN_EXCERPT_BYTES = context_config["min_excerpt_bytes"]

SYSTEM_MESSAGE = (
    PROJECT_ROOT / generation_config["system_prompt_path"]
).read_text(encoding="utf-8").strip()


def _fit_utf8(text: str, maximum: int) -> str:
    return text.encode("utf-8")[:maximum].decode("utf-8", errors="ignore")


def _build_messages(question: str, matches: list[dict]) -> tuple[list[dict], list[dict]]:
    header = f"Вопрос:\n{question}\n\nФрагменты документа:\n"
    remaining = MAX_PROMPT_BYTES - len((SYSTEM_MESSAGE + header).encode("utf-8"))
    if remaining < MIN_EXCERPT_BYTES + 20:
        raise DocumentError(400, "Вопрос слишком длинный для текущего бюджета контекста.")
    parts = []
    sources = []
    for match in matches:
        label = f"\n[{len(sources) + 1}]\n"
        available = remaining - len(label.encode("utf-8"))
        if available < MIN_EXCERPT_BYTES:
            break
        excerpt = _fit_utf8(match["text"], available)
        if not excerpt.strip():
            continue
        parts.append(label + excerpt)
        remaining -= len((label + excerpt).encode("utf-8"))
        sources.append({**match, "citation_id": len(sources) + 1,
                        "text": excerpt, "truncated": excerpt != match["text"]})
    if not sources:
        raise DocumentError(409, "Не найдено текстового контекста для генерации.")
    messages = [{"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": header + "".join(parts)}]
    return messages, sources


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _post_ollama(payload: dict) -> dict:
    request = Request(OLLAMA_URL, data=json.dumps(payload).encode("utf-8"),
                      headers={"Content-Type": "application/json"}, method="POST")
    opener = build_opener(ProxyHandler({}), _NoRedirect())
    try:
        with opener.open(request, timeout=TIMEOUT_SECONDS) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except HTTPError as error:
        code = error.code
        error.close()
        if code == 404:
            raise DocumentError(503, "Ollama не нашла локальную модель qwen2.5:3b.") from None
        if code in (429, 503):
            raise DocumentError(503, "Ollama занята. Повторите позже.") from None
        raise DocumentError(502, "Ollama вернула ошибку HTTP.") from None
    except (TimeoutError, socket.timeout):
        raise DocumentError(504, "Истекло время ожидания Ollama.") from None
    except URLError as error:
        if isinstance(error.reason, (TimeoutError, socket.timeout)):
            raise DocumentError(504, "Истекло время ожидания Ollama.") from None
        raise DocumentError(503, "Локальный сервис Ollama недоступен.") from None
    except OSError:
        raise DocumentError(503, "Ошибка соединения с локальным Ollama.") from None
    except HTTPException:
        raise DocumentError(502, "Соединение Ollama завершилось с ошибкой HTTP-протокола.") from None
    if len(raw) > MAX_RESPONSE_BYTES:
        raise DocumentError(502, "Ответ Ollama превышает допустимый размер.")
    try:
        result = json.loads(raw)
    except (ValueError, UnicodeError):
        raise DocumentError(502, "Ollama вернула некорректный JSON.") from None
    if not isinstance(result, dict):
        raise DocumentError(502, "Некорректный формат ответа Ollama.")
    return result


def generate_answer(question: str, matches: list[dict]) -> dict:
    messages, sources = _build_messages(question, matches)
    result = _post_ollama({
        "model": MODEL,
        "messages": messages,
        "stream": generation_config["stream"],
        "keep_alive": generation_config["keep_alive"],
        "options": {
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
            "temperature": generation_config["temperature"],
        },
    })
    
    message = result.get("message")
    if result.get("done") is not True or not isinstance(message, dict):
        raise DocumentError(502, "Ollama не завершила формирование ответа.")
    answer = message.get("content")
    if not isinstance(answer, str) or not answer.strip() or message.get("tool_calls"):
        raise DocumentError(502, "Ollama не вернула допустимый текстовый ответ.")
    
    prompt_tokens = result.get("prompt_eval_count")
    completion_tokens = result.get("eval_count")

    total_tokens = (
        prompt_tokens + completion_tokens
        if type(prompt_tokens) is int and type(completion_tokens) is int
        else None
    )

    return {
        "answer": answer.strip(),
        "model": MODEL,
        "sources": sources,
        "finish_reason": result.get("done_reason", "unknown"),
        "answer_truncated": result.get("done_reason") == "length",
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }
