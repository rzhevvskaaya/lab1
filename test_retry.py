"""Задания 4.1, 4.2, 4.4: невалидные ответы, retry, воспроизводимый тест.

Запуск без сервера:   python test_retry.py
"""
from langchain_core.exceptions import OutputParserException

from fakes import BAD_SYNTAX, GOOD, scripted
from pipeline import PARSER, StepLog, run_query
from schema import ProductQuery

# ---------------------------------------------------------------- 4.1
INVALID = {
    "синтаксис (не JSON)": BAD_SYNTAX,
    "значение вне enum": '{"category": "fridge", "max_price": null, "sort_by": "price_asc"}',
    "неверный тип": '{"category": "laptop", "max_price": "дёшево", "sort_by": "price_asc"}',
    "нарушено ограничение ge=0": '{"category": "laptop", "max_price": -5000, "sort_by": "price_asc"}',
    "нет обязательного поля": '{"max_price": 1000, "sort_by": "price_asc"}',
}


def analyse_errors():
    rows = []
    for name, raw in INVALID.items():
        try:
            PARSER.parse(raw)
            rows.append((name, "ОШИБКИ НЕТ", "-"))
        except OutputParserException as e:
            cause = e.__cause__
            rows.append((name, type(e).__name__, type(cause).__name__ if cause else "-"))
    return rows


# ---------------------------------------------------------------- 4.4
class CountingSearch:
    """Обёртка над поиском: считает, сколько раз он вызывался."""
    def __init__(self):
        from search import search_products
        self.calls, self._fn = 0, search_products

    def __call__(self, q):
        self.calls += 1
        return self._fn(q)


def scenario(responses, use_retry):
    log, search = StepLog(), CountingSearch()
    out = run_query("Найди ноутбуки дешевле 70 000 рублей", scripted(responses), log,
                    use_retry=use_retry, search_fn=search)
    return out, log, search


def main():
    print("== 4.1 Анализ ошибок ==")
    for name, exc, cause in analyse_errors():
        print(f"{name:28s} {exc:24s} причина: {cause}")
        assert exc == "OutputParserException", name

    print("\n== 4.4 Сценарий A: BAD, GOOD — БЕЗ retry ==")
    out, log, s = scenario([BAD_SYNTAX, GOOD], use_retry=False)
    print(out["status"], "| попыток:", out["attempts"], "| вызовов поиска:", s.calls, "|", log.steps())
    assert out["status"] == "parse_failed" and out["attempts"] == 1 and s.calls == 0

    print("\n== 4.4 Сценарий A: BAD, GOOD — С retry ==")
    out, log, s = scenario([BAD_SYNTAX, GOOD], use_retry=True)
    print(out["status"], "| попыток:", out["attempts"], "| вызовов поиска:", s.calls)
    for e in log.events:
        print("  ", e)
    assert out["status"] == "ok" and out["attempts"] == 2 and s.calls == 1
    steps = log.steps()
    assert steps.index("validation_ok") < steps.index("search_start")   # поиск — только после проверки

    print("\n== 4.4 Сценарий B: BAD, BAD, BAD — С retry ==")
    out, log, s = scenario([BAD_SYNTAX] * 3, use_retry=True)
    print(out["status"], "| попыток:", out["attempts"], "| вызовов поиска:", s.calls)
    print("  сообщение:", out["message"])
    assert out["status"] == "parse_failed" and out["attempts"] == 3 and s.calls == 0

    print("\n== Retry не повторяет при НЕ-ошибках разбора, 4-й попытки нет ==")
    out, log, s = scenario([BAD_SYNTAX] * 5, use_retry=True)
    assert out["attempts"] == 3
    print("OK, попыток ровно 3")
    print("\nВсе проверки пройдены.")


if __name__ == "__main__":
    main()
