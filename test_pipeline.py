"""Офлайн-тесты поиска, LCEL-пайплайна, режима tool_calls и состояния диалога.

Запуск:  python test_pipeline.py   (или pytest)
"""
import json
import os
import tempfile

import pipeline as P
from fakes import RuleBasedChat, scripted
from init_db import init_db
from schema import Category, ProductQuery, SortBy
from search import search_products

TMP = tempfile.mkdtemp()
DB = init_db(os.path.join(TMP, "t.db"))


def q(category, max_price=None, sort_by="price_asc"):
    return ProductQuery(category=category, max_price=max_price, sort_by=sort_by)


def test_search_filters_and_sort():
    r = search_products(q("laptop", 70000), DB)
    assert [p["price"] for p in r.items] == [42990, 54990, 61990, 69990]
    r = search_products(q("smartphone", None, "price_desc"), DB)
    assert r.items[0]["price"] == 119990 and len(r.items) == 6
    assert search_products(q("tablet", 31990), DB).items[-1]["price"] == 31990   # граница включительно


def test_empty_and_unsupported():
    assert search_products(q("laptop", 30000), DB).status == "empty"
    assert search_products(q("other"), DB).status == "unsupported_category"


def test_sql_injection_is_impossible():
    # category — это Enum, цена — int: строка с SQL не пройдёт валидацию вообще
    try:
        ProductQuery(category="laptop'; DROP TABLE products;--")
        assert False
    except Exception:
        pass
    try:
        ProductQuery(category="laptop", max_price="1; DROP TABLE products")
        assert False
    except Exception:
        pass


def test_schema_constraints():
    for bad in [{"category": "fridge"}, {"category": "laptop", "max_price": -1},
                {"category": "laptop", "sort_by": "random"}, {"category": "laptop", "max_price": "дёшево"}]:
        try:
            ProductQuery(**bad)
            assert False, bad
        except Exception:
            pass
    assert ProductQuery(category="tablet").max_price is None


def test_lcel_pipeline_returns_search_result():
    good = '{"category": "laptop", "max_price": 50000, "sort_by": "price_asc"}'
    pipe = P.build_full_lcel_pipeline(scripted(["не json", good]),
                                      db_search=lambda p: search_products(p, DB))
    res = pipe.invoke({"query": "ноутбуки до 50 тысяч"})
    assert res.status == "ok" and len(res.items) == 1


def test_tool_mode_validates_before_search():
    calls = []
    log = P.StepLog()
    out = P.run_tool_query("Покажи ноутбуки до 50 тысяч", RuleBasedChat(), log,
                           search_fn=lambda p: calls.append(p) or search_products(p, DB))
    assert out["status"] == "ok" and len(calls) == 1
    assert log.steps().index("validation_ok") < log.steps().index("search_start")
    # модель не вызвала инструмент -> поиска нет
    calls.clear()
    out = P.run_tool_query("Привет!", RuleBasedChat(), P.StepLog(), search_fn=lambda p: calls.append(p))
    assert out["status"] == "no_tool_call" and not calls


def test_tool_mode_rejects_invalid_args():
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    class Bad(GenericFakeChatModel):
        def bind_tools(self, tools, **kw):
            return self

    bad = Bad(messages=iter([AIMessage(content="", tool_calls=[
        {"name": "search_products", "args": {"category": "fridge"}, "id": "1", "type": "tool_call"}])]))
    calls = []
    out = P.run_tool_query("x", bad, P.StepLog(), search_fn=lambda p: calls.append(p))
    assert out["status"] == "invalid_args" and not calls


def test_dialogue_state_kept_outside_model():
    s = P.DialogueSession(RuleBasedChat(), mode="state")
    s.ask("Покажи ноутбуки до 80 тысяч")
    s.ask("А сначала дорогие?")
    out = s.ask("А что есть среди смартфонов?")
    assert s.current.category == Category.smartphone and s.current.max_price == 80000
    assert s.current.sort_by == SortBy.price_desc and out["status"] == "ok"
    # история диалога не растёт в режиме state, в режиме history — растёт
    h = P.DialogueSession(RuleBasedChat(), mode="history")
    for t in ["Покажи ноутбуки до 80 тысяч", "А сначала дорогие?", "А что есть среди смартфонов?"]:
        h.ask(t)
    assert len(h.history) == 6 and h.input_tokens[-1] > h.input_tokens[0]


def test_token_not_in_log(monkeypatch=None):
    os.environ["LLM_API_KEY"] = "sk-super-secret-123"
    log = P.StepLog(path=os.path.join(TMP, "l.log"))
    log.log("request", text="мой ключ sk-super-secret-123")
    assert "sk-super-secret-123" not in json.dumps(log.events)
    assert "sk-super-secret-123" not in open(os.path.join(TMP, "l.log"), encoding="utf-8").read()
    del os.environ["LLM_API_KEY"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK ", name)
    print("Все тесты пройдены.")
