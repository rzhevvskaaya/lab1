"""Задание 3.2. Программный поиск по SQLite.

LLM не видит SQL и базу: функция принимает только уже валидированный
ProductQuery. Текст SQL собирается из фиксированных фрагментов, а все
значения передаются параметрами (?), поэтому SQL-инъекция невозможна.

Поведение для category = other (задокументировано): поиск по базе НЕ
выполняется, возвращается статус "unsupported_category" и сообщение.
max_price трактуется как включительная граница: price <= max_price.
"""
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path

from schema import Category, ProductQuery, SortBy

DB_PATH = Path(__file__).with_name("catalog.db")

# Белый список вариантов сортировки: в SQL попадает только значение из этого словаря.
ORDER_BY = {
    SortBy.price_asc: "price ASC, id ASC",
    SortBy.price_desc: "price DESC, id ASC",
}


@dataclass
class SearchResult:
    status: str  # "ok" | "empty" | "unsupported_category"
    items: list = field(default_factory=list)
    message: str = ""

    def to_dict(self):
        return asdict(self)


def search_products(query: ProductQuery, db_path=DB_PATH) -> SearchResult:
    if not isinstance(query, ProductQuery):
        raise TypeError("search_products принимает только валидированный ProductQuery")

    if query.category == Category.other:
        return SearchResult(
            status="unsupported_category",
            message="Эта категория не поддерживается. Доступны: ноутбуки, смартфоны, планшеты.",
        )

    sql = "SELECT id, name, category, price FROM products WHERE category = ?"
    params: list = [query.category.value]
    if query.max_price is not None:
        sql += " AND price <= ?"
        params.append(query.max_price)
    sql += " ORDER BY " + ORDER_BY[query.sort_by]

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute(sql, params).fetchall()]

    if not rows:
        return SearchResult(status="empty", message="Подходящих товаров не найдено.")
    return SearchResult(status="ok", items=rows, message=f"Найдено товаров: {len(rows)}")


def format_result(res: SearchResult) -> str:
    if res.status != "ok":
        return res.message
    lines = [res.message] + [f"  • {p['name']} — {p['price']:,} ₽".replace(",", " ") for p in res.items]
    return "\n".join(lines)
