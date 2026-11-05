"""Интерактивный интерфейс к каталогу: запрос на естественном языке -> поиск в SQLite.

    export LLM_API_KEY="<токен>"
    python app.py            # обычный режим (параметры из текста + retry + журнал)
    python app.py --confirm  # дополнительно: подтверждение перед поиском без ограничения цены
    python app.py --tool     # режим вызова инструмента (bind_tools) вместо инструкций формата

Состояние диалога (последний валидный ProductQuery) хранится в программе, поэтому
можно уточнять: «Покажи ноутбуки до 80 тысяч» -> «А сначала дорогие?» -> «А смартфоны?».
Команды: /reset — сбросить параметры, /exit — выход. Журнал: logs/pipeline.log.
"""
import argparse

import pipeline as P
from init_db import DB_PATH, init_db


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--tool", action="store_true")
    a = ap.parse_args()
    if not DB_PATH.exists():
        init_db()

    llm = P.get_llm(thinking=False)
    log = P.StepLog(path="logs/pipeline.log", echo=True)
    session = P.DialogueSession(llm, log=log, mode="state")

    print("Каталог товаров. Введите запрос (/reset, /exit).")
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text == "/exit":
            break
        if text == "/reset":
            session = P.DialogueSession(llm, log=log, mode="state")
            print("Параметры сброшены.")
            continue

        if a.tool:
            out = P.run_tool_query(text, llm, log)
        elif a.confirm:
            # подтверждение без ограничения цены — на одиночном запросе (без состояния)
            out = P.run_query(
                text, llm, log,
                confirm_no_price=lambda p: input("  Цена не ограничена. Искать? [y/N] ").lower() == "y",
            )
        else:
            out = session.ask(text)
        print(out["message"])


if __name__ == "__main__":
    main()
