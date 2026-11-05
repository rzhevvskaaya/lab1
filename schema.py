"""Задание 2.1. Схема структурированного ответа ProductQuery."""
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Category(str, Enum):
    laptop = "laptop"
    smartphone = "smartphone"
    tablet = "tablet"
    other = "other"


class SortBy(str, Enum):
    price_asc = "price_asc"
    price_desc = "price_desc"


class ProductQuery(BaseModel):
    """Параметры поиска по каталогу товаров."""

    category: Category = Field(
        description=(
            "Категория товара: laptop (ноутбуки), smartphone (смартфоны), "
            "tablet (планшеты). Любая другая категория — other."
        )
    )
    max_price: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "Верхняя граница цены в рублях, целое неотрицательное число "
            "(«50 тысяч» = 50000). null, если ограничения по цене нет."
        ),
    )
    sort_by: SortBy = Field(
        default=SortBy.price_asc,
        description=(
            "Сортировка по цене: price_asc — от дешевых к дорогим (по умолчанию), "
            "price_desc — от дорогих к дешевым."
        ),
    )
