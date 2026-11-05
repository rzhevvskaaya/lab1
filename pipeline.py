"""Основной модуль: LLM-клиент, промпты, цепочки, retry, журнал, состояние.

Схема пайплайна (часть 4):

    запрос -> [промпт | LLM | PydanticOutputParser] (with_retry, макс. 3 попытки)
           -> проверка параметров -> search_products (SQLite) -> вывод

Retry оборачивает ТОЛЬКО генерацию и парсинг. Поиск по базе стоит после
границы retry, поэтому при повторных попытках он не выполняется повторно.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable, Optional

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from pydantic import ValidationError

from schema import Category, ProductQuery, SortBy
from search import SearchResult, format_result, search_products

from dotenv import load_dotenv
load_dotenv()
BASE_URL = os.environ.get("LLM_BASE_URL", "http://10.32.2.11:35000/v1")
MODEL = os.environ.get("LLM_MODEL", "qwen3.8-27b")
MAX_ATTEMPTS = 3


# --------------------------------------------------------------------------- #
# LLM-клиент (токен берётся ТОЛЬКО из переменной окружения LLM_API_KEY)
# --------------------------------------------------------------------------- #
def get_llm(thinking: bool = False, temperature: float = 0.0):
    from langchain_openai import ChatOpenAI

    token = os.environ.get("LLM_API_KEY")
    if not token:
        raise RuntimeError(
            "Не задана переменная окружения LLM_API_KEY (персональный токен студента)."
        )
    return ChatOpenAI(
        base_url=BASE_URL,
        model=MODEL,
        api_key=token,
        temperature=temperature,
        extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
    )


# --------------------------------------------------------------------------- #
# Журнал выполнения (задание 4.3). Токен в журнал не попадает.
# --------------------------------------------------------------------------- #
class StepLog:
    def __init__(self, path: Optional[str] = None, echo: bool = False):
        self.events: list[dict] = []
        self.echo = echo
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _redact(text: str) -> str:
        token = os.environ.get("LLM_API_KEY")
        if token and len(token) >= 4:
            text = text.replace(token, "***")
        return text

    def log(self, step: str, **details):
        rec = {"time": time.strftime("%H:%M:%S"), "step": step, **details}
        line = self._redact(json.dumps(rec, ensure_ascii=False, default=str))
        rec = json.loads(line)
        self.events.append(rec)
        if self.echo:
            print("  [log]", line)
        if self.path:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def steps(self) -> list[str]:
        return [e["step"] for e in self.events]

    def count(self, step: str) -> int:
        return sum(1 for e in self.events if e["step"] == step)


# --------------------------------------------------------------------------- #
# Часть 1. Параметризованная цепочка
# --------------------------------------------------------------------------- #
def build_category_chain(llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Ты — помощник интернет-магазина электроники. По запросу пользователя "
                "определи категорию товара и назови её одним-двумя словами. "
                "Отвечай на языке: {language}.",
            ),
            ("human", "{query}"),
        ]
    )
    return prompt | llm | StrOutputParser()


# --------------------------------------------------------------------------- #
# Часть 2. Structured output
# --------------------------------------------------------------------------- #
RULES = """Ты — модуль интерпретации запросов к каталогу товаров.
Твоя задача — только преобразовать запрос пользователя в параметры поиска.
Ты не отвечаешь на вопрос по существу и не пишешь SQL.

Правила:
- category: laptop (ноутбуки), smartphone (смартфоны), tablet (планшеты).
  Любая другая категория товара — other.
- max_price: верхняя граница цены в рублях, целое число. «50 тысяч» = 50000.
  Если ограничения по цене нет — null.
- sort_by: price_desc, если просят самые дорогие / сначала дорогие;
  во всех остальных случаях price_asc."""

FORMAT_BLOCK = """
{format_instructions}
Верни только JSON-объект, без пояснений и без markdown-оформления."""

PARSER = PydanticOutputParser(pydantic_object=ProductQuery)


def build_extraction_prompt() -> ChatPromptTemplate:
    """Инструкции формата (get_format_instructions) лежат в системном сообщении."""
    return ChatPromptTemplate.from_messages(
        [("system", RULES + FORMAT_BLOCK), ("human", "{query}")]
    ).partial(format_instructions=PARSER.get_format_instructions())


def build_structured_chain(llm):
    """Задание 2.2: промпт | LLM | Pydantic-парсер -> объект ProductQuery."""
    return build_extraction_prompt() | llm | PARSER


# --------------------------------------------------------------------------- #
# Часть 4. Генерация с retry + поиск + журнал
# --------------------------------------------------------------------------- #
def build_generation_chain(llm, log: StepLog, attempt_counter: dict, prompt=None):
    """Генерация + парсинг с журналированием. Именно эту часть повторяет retry."""
    prompt = prompt or build_extraction_prompt()

    def on_attempt_start(x):
        attempt_counter["n"] += 1
        log.log("llm_call_start", attempt=attempt_counter["n"])
        return x

    def parse_logged(msg):
        try:
            obj = PARSER.invoke(msg)
        except OutputParserException as e:
            cause = e.__cause__
            log.log(
                "parse_error",
                attempt=attempt_counter["n"],
                error_type=type(cause).__name__ if cause else type(e).__name__,
                error=str(e).splitlines()[0][:200],
            )
            raise
        log.log("parse_ok", attempt=attempt_counter["n"], params=obj.model_dump(mode="json"))
        return obj

    return (
        RunnableLambda(on_attempt_start)
        | prompt
        | llm
        | RunnableLambda(parse_logged)
    )


def with_retry_parse(chain, attempts: int = MAX_ATTEMPTS):
    """Повтор только при ошибках разбора, максимум `attempts` попыток (включая первую)."""
    return chain.with_retry(
        retry_if_exception_type=(OutputParserException,),
        stop_after_attempt=attempts,
        wait_exponential_jitter=False,
    )


def run_query(
    text: str,
    llm,
    log: StepLog,
    use_retry: bool = True,
    search_fn: Callable = search_products,
    prompt=None,
    prompt_inputs: Optional[dict] = None,
    confirm_no_price: Optional[Callable[[ProductQuery], bool]] = None,
) -> dict:
    """Полный пайплайн для одного запроса. Возвращает словарь со статусом.

    status: ok | empty | unsupported_category | parse_failed | cancelled
    """
    counter = {"n": 0}
    log.log("request", text=text)
    gen = build_generation_chain(llm, log, counter, prompt=prompt)
    gen = with_retry_parse(gen) if use_retry else gen
    inputs = {"query": text, **(prompt_inputs or {})}

    try:
        params = gen.invoke(inputs)
    except OutputParserException as e:
        log.log("generation_failed", attempts=counter["n"], retry=use_retry)
        return {
            "status": "parse_failed",
            "attempts": counter["n"],
            "message": f"Не удалось разобрать ответ модели за {counter['n']} поп. "
            f"Попробуйте переформулировать запрос.",
            "error": str(e).splitlines()[0],
        }

    # Проверка параметров по схеме (повторная, независимая от парсера) — до поиска.
    try:
        params = ProductQuery.model_validate(params.model_dump())
    except ValidationError as e:  # недостижимо при корректном парсере, но защищает поиск
        log.log("validation_failed", error=str(e).splitlines()[0])
        return {"status": "parse_failed", "attempts": counter["n"], "message": "Параметры не прошли проверку."}
    log.log("validation_ok", params=params.model_dump(mode="json"))

    if confirm_no_price and params.max_price is None and params.category != Category.other:
        if not confirm_no_price(params):
            log.log("search_cancelled_by_user")
            return {"status": "cancelled", "params": params, "attempts": counter["n"], "message": "Поиск отменён."}

    log.log("search_start", params=params.model_dump(mode="json"))
    res: SearchResult = search_fn(params)
    log.log("search_done", status=res.status, found=len(res.items))
    return {
        "status": res.status,
        "params": params,
        "result": res,
        "attempts": counter["n"],
        "message": format_result(res),
    }


# Задание 3.3: поиск как часть LCEL-цепочки (RunnableLambda).
def build_full_lcel_pipeline(llm, db_search: Callable = search_products):
    """запрос -> ProductQuery -> SearchResult (всё в одной LCEL-цепочке)."""
    generation = with_retry_parse(build_extraction_prompt() | llm | PARSER)
    return generation | RunnableLambda(db_search)


# --------------------------------------------------------------------------- #
# Задание 3.4. Поиск как инструмент
# --------------------------------------------------------------------------- #
@tool("search_products", args_schema=ProductQuery)
def search_products_tool(category, max_price=None, sort_by=SortBy.price_asc) -> str:
    """Поиск товаров в каталоге магазина по категории, максимальной цене и сортировке по цене.
    Вызывай для любого запроса о ноутбуках, смартфонах, планшетах и других товарах каталога."""
    q = ProductQuery(category=category, max_price=max_price, sort_by=sort_by)
    return format_result(search_products(q))


TOOL_SYSTEM = (
    "Ты — ассистент каталога товаров. Если пользователь ищет товары, вызови инструмент "
    "search_products с подходящими параметрами. «50 тысяч» = 50000. Для категорий, "
    "которых нет среди laptop/smartphone/tablet, используй category=other."
)


def run_tool_query(text: str, llm, log: StepLog, search_fn: Callable = search_products) -> dict:
    """LLM возвращает tool_calls; приложение проверяет аргументы и само выполняет поиск."""
    log.log("request", text=text, mode="tool_call")
    msg = llm.bind_tools([search_products_tool]).invoke(
        [("system", TOOL_SYSTEM), ("human", text)]
    )
    log.log("llm_response", content=msg.content, tool_calls=msg.tool_calls)
    if not msg.tool_calls:
        log.log("no_tool_call")
        return {"status": "no_tool_call", "text": msg.content, "message": "Модель не вызвала инструмент: " + str(msg.content)}

    outputs = []
    for call in msg.tool_calls:
        try:
            params = ProductQuery.model_validate(call["args"])
        except ValidationError as e:
            log.log("validation_failed", args=call["args"], error=str(e).splitlines()[0])
            outputs.append({"status": "invalid_args", "args": call["args"]})
            continue
        log.log("validation_ok", params=params.model_dump(mode="json"))
        log.log("search_start", params=params.model_dump(mode="json"))
        res = search_fn(params)
        log.log("search_done", status=res.status, found=len(res.items))
        outputs.append({"status": res.status, "params": params, "result": res, "message": format_result(res)})
    first = outputs[0]
    return {**first, "calls": outputs, "text": msg.content, "tool_calls": msg.tool_calls}


# --------------------------------------------------------------------------- #
# Часть 5. Уточняющие запросы
# --------------------------------------------------------------------------- #
STATE_SYSTEM = (
    RULES
    + """

Это диалог с уточнениями. Ниже даны ТЕКУЩИЕ параметры поиска (последний валидный запрос).
Обнови их по новой реплике пользователя: поля, о которых реплика ничего не говорит,
сохраняй без изменений. Верни полный обновлённый объект.
Текущие параметры: {current_params}"""
    + FORMAT_BLOCK
)

HISTORY_SYSTEM = (
    RULES
    + """

Это диалог с уточнениями: учитывай предыдущие реплики пользователя и свои ответы.
Верни полный объект параметров для последней реплики."""
    + FORMAT_BLOCK
)


def build_state_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [("system", STATE_SYSTEM), ("human", "{query}")]
    ).partial(format_instructions=PARSER.get_format_instructions())


def build_history_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [("system", HISTORY_SYSTEM), MessagesPlaceholder("history"), ("human", "{query}")]
    ).partial(format_instructions=PARSER.get_format_instructions())


class DialogueSession:
    """Состояние диалога хранится в программе (self.current), а не в модели."""

    def __init__(self, llm, log: Optional[StepLog] = None, mode: str = "state"):
        assert mode in ("state", "history")
        self.llm, self.mode = llm, mode
        self.log = log or StepLog()
        self.current: Optional[ProductQuery] = None
        self.history: list = []
        self.input_tokens: list[int] = []
        self.prompt = build_state_prompt() if mode == "state" else build_history_prompt()

    def ask(self, text: str) -> dict:
        extra = {}
        if self.mode == "state":
            extra["current_params"] = (
                self.current.model_dump_json() if self.current else "нет (первый запрос)"
            )
        else:
            extra["history"] = list(self.history)

        # токены считаем отдельным вызовом цепочки без парсера, чтобы получить usage_metadata
        usage = {}

        def capture(msg):
            usage.update(getattr(msg, "usage_metadata", None) or {})
            return msg

        gen_prompt = self.prompt | self.llm | RunnableLambda(capture)
        counter = {"n": 0}
        log = self.log

        def attempt(x):
            counter["n"] += 1
            log.log("llm_call_start", attempt=counter["n"], mode=self.mode)
            return x

        def parse(msg):
            try:
                return PARSER.invoke(msg)
            except OutputParserException as e:
                log.log("parse_error", attempt=counter["n"], error=str(e).splitlines()[0][:200])
                raise

        chain = with_retry_parse(RunnableLambda(attempt) | gen_prompt | RunnableLambda(parse))
        log.log("request", text=text, mode=self.mode)
        try:
            params = chain.invoke({"query": text, **extra})
        except OutputParserException:
            return {"status": "parse_failed", "message": "Не удалось разобрать ответ модели."}

        log.log("validation_ok", params=params.model_dump(mode="json"))
        self.current = params                      # обновляем состояние
        self.history += [HumanMessage(text), AIMessage(params.model_dump_json())]
        self.input_tokens.append(usage.get("input_tokens", 0))
        log.log("search_start")
        res = search_products(params)
        log.log("search_done", status=res.status, found=len(res.items))
        return {
            "status": res.status, "params": params, "result": res,
            "input_tokens": usage.get("input_tokens"), "message": format_result(res),
        }
