"""Задание 3.1. Создание и наполнение базы SQLite (каталог товаров).

Запуск:  python init_db.py            -> создаёт catalog.db
Цены условные; подобраны так, чтобы часть запросов давала несколько
результатов, а часть — пустой список (например, ноутбуки до 30 000 ₽).
"""
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).with_name("catalog.db")

DDL = """
CREATE TABLE IF NOT EXISTS products (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    name     TEXT    NOT NULL,
    category TEXT    NOT NULL CHECK (category IN ('laptop', 'smartphone', 'tablet')),
    price    INTEGER NOT NULL CHECK (price >= 0)
);
"""

PRODUCTS = [
    # (name, category, price)
    ("Lenovo IdeaPad 1 15", "laptop", 42990),
    ("ASUS VivoBook 15", "laptop", 54990),
    ("Acer Aspire 5", "laptop", 61990),
    ("MSI Modern 14", "laptop", 69990),
    ("HP Pavilion 15", "laptop", 74990),
    ("Apple MacBook Air 13 M2", "laptop", 104990),
    ("Xiaomi Redmi Note 13", "smartphone", 17990),
    ("POCO X6", "smartphone", 24990),
    ("Samsung Galaxy A35", "smartphone", 29990),
    ("Google Pixel 8", "smartphone", 52990),
    ("Apple iPhone 15", "smartphone", 79990),
    ("Samsung Galaxy S24 Ultra", "smartphone", 119990),
    ("Lenovo Tab M10", "tablet", 15990),
    ("Samsung Galaxy Tab A9", "tablet", 17990),
    ("Xiaomi Pad 6", "tablet", 31990),
    ("Apple iPad 10", "tablet", 39990),
    ("Apple iPad Pro 11", "tablet", 99990),
]


def init_db(db_path=DB_PATH, reset: bool = True) -> Path:
    db_path = Path(db_path)
    with sqlite3.connect(db_path) as conn:
        if reset:
            conn.execute("DROP TABLE IF EXISTS products")
        conn.executescript(DDL)
        conn.executemany(
            "INSERT INTO products (name, category, price) VALUES (?, ?, ?)", PRODUCTS
        )
    return db_path


if __name__ == "__main__":
    path = init_db(sys.argv[1] if len(sys.argv) > 1 else DB_PATH)
    with sqlite3.connect(path) as c:
        n = c.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    print(f"База создана: {path} ({n} товаров)")
