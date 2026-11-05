"""Прогон экспериментов лабораторной работы с реальной LLM (и --dry-run без сервера).

    export LLM_API_KEY="<персональный токен>"      # Windows PowerShell: $env:LLM_API_KEY="..."
    python init_db.py
    python run_lab.py --part all                    # или 1 | 2 | 24 | 3 | 34 | 5
    python run_lab.py --part all --dry-run          # проверка без сервера (правиловая модель)

Результаты: results/*.json и results/tables.md (таблицы для отчёта).
Для заданий 2.4 и 5.2 / 3.3 «без базы» нужна именно реальная модель.
"""
import argparse
import json
import time
from pathlib import Path

from langchain_core.exceptions import OutputParserException

import pipeline as P
from init_db import DB_PATH, init_db
from schema import ProductQuery
from search import format_result, search_products

OUT = Path(__file__).with_name("results")
OUT.mkdir(exist_ok=True)
TABLES: list[str] = []
DRY = False


def make_llm(thinking=False):
    if DRY:
        from fakes import RuleBasedChat
        return RuleBasedChat()
    return P.get_llm(thinking=thinking)


def save(name, data):
    (OUT / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def table(title, header, rows):
    md = [f"\n### {title}\n", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        md.append("| " + " | ".join(str(c).replace("\n", "<br>").replace("|", "\\|") for c in r) + " |")
    TABLES.append("\n".join(md))
    print(TABLES[-1])


def compact(p: ProductQuery):
    return f"{p.category.value}, ≤{p.max_price}, {p.sort_by.value}"


def same(p: ProductQuery, exp: dict) -> bool:
    return (p.category.value, p.max_price, p.sort_by.value) == (exp["category"], exp["max_price"], exp["sort_by"])


E = lambda c, m=None, s="price_asc": {"category": c, "max_price": m, "sort_by": s}

# Запросы 2.3 с ожидаемыми параметрами
Q23 = [
    ("Найди ноутбуки дешевле 70 000 рублей", E("laptop", 70000)),
    ("Покажи самые дорогие смартфоны", E("smartphone", None, "price_desc")),
    ("Хочу посмотреть планшеты", E("tablet")),
    ("Покажи ноутбуки до 50 тысяч", E("laptop", 50000)),
    ("Найди холодильники", E("other")),
]
# Собственные неоднозначные запросы (ожидаемое поведение — на ваше усмотрение, см. отчёт)
Q_OWN = [
    ("Нужно что-нибудь недорогое для учёбы", None),
    ("Покажи Apple до 100 тысяч", None),
]


# ----------------------------------------------------------------- часть 1
def part1():
    llm = make_llm()
    chain = P.build_category_chain(llm)
    q = "Найди ноутбуки дешевле 70 000 рублей"
    raw = llm.invoke([("system", "Назови категорию товара."), ("human", q)])
    out = chain.invoke({"query": q, "language": "русский"})
    batch = chain.batch([{"query": x, "language": "русский"} for x in
                         ["Хочу новый ноутбук для работы", "Посоветуй смартфон с хорошей камерой", "Нужен планшет ребёнку"]])
    res = {"llm.invoke type": type(raw).__name__, "chain.invoke type": type(out).__name__,
           "chain.invoke": out, "batch": batch}
    save("part1", res)
    print(res)


# ----------------------------------------------------------------- часть 2
def part2():
    llm = make_llm()
    chain = P.build_structured_chain(llm)
    rows, dump = [], []
    for text, exp in Q23 + Q_OWN:
        try:
            obj = chain.invoke({"query": text})
            js = obj.model_dump_json()
            check = ("✅" if same(obj, exp) else "❌") if exp else "см. комментарий"
        except OutputParserException as e:
            js, check = "ОШИБКА РАЗБОРА", str(e).splitlines()[0]
        rows.append([text, f"`{js}`", check])
        dump.append({"query": text, "json": js, "check": check})
    save("part2_3", dump)
    table("Задание 2.3. Структурированный ответ", ["Запрос", "JSON", "Проверка"], rows)


def part24():
    """Сравнение режима размышления: параметры, время, токены, content."""
    rows, dump = [], []
    for text, _ in Q_OWN:
        for thinking in (False, True):
            llm = make_llm(thinking)
            prompt = P.build_extraction_prompt()
            t0 = time.perf_counter()
            msg = (prompt | llm).invoke({"query": text})
            dt = time.perf_counter() - t0
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            reasoning = msg.additional_kwargs.get("reasoning_content") or msg.additional_kwargs.get("reasoning")
            try:
                parsed = P.PARSER.invoke(msg)
                ok, params = "да", compact(parsed)
            except OutputParserException as e:
                ok, params = "нет: " + str(e).splitlines()[0][:60], "—"
            out_tok = (msg.usage_metadata or {}).get("output_tokens")
            rows.append([text, "вкл" if thinking else "выкл", params, f"{dt:.1f} с", out_tok,
                         f"`{content[:80]}`", "да" if reasoning else "нет", ok])
            dump.append({"query": text, "thinking": thinking, "params": params, "seconds": dt,
                         "output_tokens": out_tok, "content": content, "reasoning_field": reasoning, "parsed": ok})
    save("part2_4", dump)
    table("Задание 2.4. Режим размышления",
          ["Запрос", "thinking", "Параметры", "Время", "Токенов на выходе", "Начало content",
           "Есть reasoning_content", "Парсер справился"], rows)


# ----------------------------------------------------------------- часть 3
Q33 = [
    ("Найди ноутбуки дешевле 70 000 рублей", E("laptop", 70000)),
    ("Покажи самые дорогие смартфоны", E("smartphone", None, "price_desc")),
    ("Хочу посмотреть планшеты", E("tablet")),
    ("Покажи ноутбуки до 50 тысяч", E("laptop", 50000)),
    ("Найди холодильники", E("other")),
    ("Смартфоны до 20 000 рублей", E("smartphone", 20000)),
    ("Ноутбуки до 30 тысяч", E("laptop", 30000)),
    ("Планшеты до 10 000", E("tablet", 10000)),
    ("Покажи самые дешевые планшеты до 35 тысяч", E("tablet", 35000)),
    ("Сначала дорогие ноутбуки до 80 тысяч", E("laptop", 80000, "price_desc")),
    ("Хочу телевизор", E("other")),
]


def part3():
    llm = make_llm()
    pipe = P.build_full_lcel_pipeline(llm)           # запрос -> ProductQuery -> SearchResult (LCEL)
    chain_params = P.build_structured_chain(llm)
    rows, dump = [], []
    for text, exp in Q33:
        params = chain_params.invoke({"query": text})
        res = search_products(params)
        # сверка с «эталонным» поиском по ожидаемым параметрам
        truth = search_products(ProductQuery(**exp))
        ok = same(params, exp) and [p["id"] for p in res.items] == [p["id"] for p in truth.items]
        found = "; ".join(f"{p['name']} ({p['price']})" for p in res.items) or res.message
        rows.append([text, f"`{params.model_dump_json()}`", found, "✅" if ok else "❌"])
        dump.append({"query": text, "params": params.model_dump(mode="json"), "status": res.status,
                     "items": res.items, "correct": ok})
        # проверка, что целая LCEL-цепочка даёт тот же результат
        assert pipe.invoke({"query": text}).status == res.status
    save("part3_3", dump)
    table("Задание 3.3. Пайплайн с поиском", ["Запрос", "Параметры", "Найденные товары", "Корректность"], rows)

    # сравнение с ответом модели БЕЗ базы
    rows2 = []
    for text in ["Какие ноутбуки дешевле 70 000 рублей есть в каталоге?",
                 "Какие самые дорогие смартфоны есть в каталоге?",
                 "Какие планшеты есть в каталоге до 20 000 рублей?"]:
        answer = llm.invoke(text).content
        params = chain_params.invoke({"query": text})
        db = format_result(search_products(params))
        rows2.append([text, str(answer)[:300], db])
    save("part3_3_nodb", [{"q": r[0], "llm_only": r[1], "db": r[2]} for r in rows2])
    table("Задание 3.3. Ответ модели без базы и результат поиска",
          ["Вопрос", "Ответ LLM без базы", "Результат поиска по SQLite"], rows2)


def part34():
    llm = make_llm()
    log = P.StepLog()
    rows, dump = [], []
    for text, exp in Q23 + [("Привет! Как дела?", None)]:
        out = P.run_tool_query(text, llm, log)
        tc = out.get("tool_calls")
        if tc:
            args = json.dumps(tc[0]["args"], ensure_ascii=False)
            ok = "✅" if exp and same(out["params"], exp) else ("❌" if exp else "—")
        else:
            args, ok = "—", "нет вызова инструмента"
        rows.append([text, str(out.get("text") or "")[:60], f"`{args}`", out["message"].splitlines()[0], ok])
        dump.append({"query": text, "text": out.get("text"), "tool_calls": tc, "status": out["status"]})
    save("part3_4", dump)
    table("Задание 3.4. Поиск как инструмент (tool_calls)",
          ["Запрос", "Текст ответа модели", "Аргументы вызова", "Результат", "Проверка"], rows)


# ----------------------------------------------------------------- часть 5
DIALOGS = {
    "Диалог 1": [("Покажи ноутбуки до 80 тысяч", E("laptop", 80000)),
                 ("А сначала дорогие?", E("laptop", 80000, "price_desc")),
                 ("А что есть среди смартфонов?", E("smartphone", 80000, "price_desc"))],
    "Диалог 2": [("Хочу планшет", E("tablet")),
                 ("Только до 30 тысяч", E("tablet", 30000)),
                 ("А теперь самые дорогие", E("tablet", 30000, "price_desc")),
                 ("А ноутбуки?", E("laptop", 30000, "price_desc"))],
}


def part5():
    llm = make_llm()
    rows, dump = [], []
    for name, turns in DIALOGS.items():
        sessions = {m: P.DialogueSession(llm, mode=m) for m in ("state", "history")}
        for i, (text, exp) in enumerate(turns, 1):
            cells = [f"{name}, шаг {i}: {text}"]
            for mode in ("state", "history"):
                out = sessions[mode].ask(text)
                if "params" in out:
                    ok = "✅" if same(out["params"], exp) else "❌"
                    cells += [f"{compact(out['params'])} {ok}", out["input_tokens"]]
                else:
                    cells += ["ошибка разбора", "—"]
                dump.append({"dialog": name, "step": i, "mode": mode, "text": text,
                             "params": out.get("params"), "input_tokens": out.get("input_tokens")})
            rows.append(cells)
    save("part5", dump)
    table("Задание 5.2. Состояние вне модели vs вся история",
          ["Шаг", "Параметры (состояние)", "Вх. токенов", "Параметры (история)", "Вх. токенов"], rows)


PARTS = {"1": part1, "2": part2, "24": part24, "3": part3, "34": part34, "5": part5}


def main():
    global DRY
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["all", *PARTS])
    ap.add_argument("--dry-run", action="store_true", help="правиловая модель вместо сервера")
    a = ap.parse_args()
    DRY = a.dry_run
    if not DB_PATH.exists():
        init_db()
    for k, fn in PARTS.items():
        if a.part in ("all", k):
            print(f"\n=== Часть {k} ===")
            fn()
    (OUT / "tables.md").write_text("\n".join(TABLES), encoding="utf-8")
    print(f"\nГотово. Таблицы: {OUT / 'tables.md'}")


if __name__ == "__main__":
    main()
