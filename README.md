# Лабораторная работа № 1 — пайплайны LangChain + LLM + SQLite

## Состав
| Файл | Назначение |
|---|---|
| `schema.py` | Pydantic-схема `ProductQuery` (задание 2.1) |
| `init_db.py` | Создание `catalog.db`, 17 товаров (задание 3.1) |
| `search.py` | Параметризованный поиск по SQLite (задание 3.2) |
| `pipeline.py` | LLM-клиент, промпты, цепочки, retry, журнал, инструмент, состояние диалога |
| `fakes.py` | Тестовые модели: `scripted` (заданные ответы) и `RuleBasedChat` (для `--dry-run`) |
| `test_retry.py` | Задания 4.1, 4.2, 4.4 — ошибки разбора и retry, **без сервера** |
| `test_pipeline.py` | Офлайн-тесты поиска, LCEL, tool_calls, состояния, журнала |
| `run_lab.py` | Эксперименты с реальной LLM → `results/` (задания 1, 2.3, 2.4, 3.3, 3.4, 5.2) |
| `app.py` | Интерактивный интерфейс (обычный / `--tool` / `--confirm`) |
| `report.md` | Отчёт (заполнен результатами прогона) |

## Запуск
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python init_db.py                  # создать catalog.db
python test_retry.py               # части 4.1/4.2/4.4 (сервер не нужен)
python test_pipeline.py            # остальные офлайн-тесты

# Эксперименты с реальной моделью — из сети университета:
export LLM_API_KEY="ваш_токен"     # Windows PowerShell: $env:LLM_API_KEY="ваш_токен"
python run_lab.py --part all       # или --part 1 | 2 | 24 | 3 | 34 | 5
python app.py                      # интерактивный режим (журнал: logs/pipeline.log)
```
Адрес и модель по умолчанию: `http://10.32.2.11:35000/v1`, `qwen3.8-27b`
(можно переопределить переменными `LLM_BASE_URL`, `LLM_MODEL`).
Токен берётся только из `LLM_API_KEY`, в код, журнал и отчёт не попадает.

`python run_lab.py --part all --dry-run` — проверка скрипта без сервера на правиловой
«модели». Эти результаты **нельзя** вставлять в отчёт — только для проверки работоспособности.
После реального прогона таблицы лежат в `results/tables.md`.
