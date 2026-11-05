"""Тестовые модели (задание 4.4), не требующие доступа к серверу.

* scripted(...)   — модель с заранее заданными ответами (FakeListChatModel).
* RuleBasedChat   — простая детерминированная «модель» на правилах: нужна, чтобы
                    прогнать весь пайплайн (run_lab.py --dry-run) без сервера.
"""
import json
import re
from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult


def scripted(responses: List[str]) -> FakeListChatModel:
    """Модель, по очереди возвращающая заданные строки (после последней — по кругу)."""
    return FakeListChatModel(responses=responses)


GOOD = '{"category": "laptop", "max_price": 70000, "sort_by": "price_asc"}'
BAD_SYNTAX = "Конечно! Вот параметры поиска: категория — ноутбуки, цена до 70000."


class RuleBasedChat(BaseChatModel):
    tools_bound: bool = False

    @property
    def _llm_type(self) -> str:
        return "rule-based-fake"

    def bind_tools(self, tools, **kwargs):
        return self.model_copy(update={"tools_bound": True})

    # --- «интерпретация» реплики по правилам ---------------------------------
    @staticmethod
    def _apply(text: str, state: dict) -> dict:
        t = text.lower()
        st = dict(state)
        if re.search(r"ноутбук|laptop", t):
            st["category"] = "laptop"
        elif re.search(r"смартфон|телефон", t):
            st["category"] = "smartphone"
        elif re.search(r"планшет", t):
            st["category"] = "tablet"
        elif re.search(r"холодильник|телевизор|пылесос|стиральн|наушник", t):
            st["category"] = "other"
        m = re.search(r"(?:до|дешевле|не дороже|меньше)\s+(\d[\d\s\u00a0]*)\s*(тысяч\w*|тыс\.?|к)?", t)
        if m:
            num = int(re.sub(r"\D", "", m.group(1)))
            st["max_price"] = num * 1000 if m.group(2) else num
        if re.search(r"дорог", t) and not re.search(r"не дорог", t):
            st["sort_by"] = "price_desc"
        elif re.search(r"дешев|недорог", t) and not re.search(r"дешевле", t):
            st["sort_by"] = "price_asc"
        return st

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                  run_manager: Optional[CallbackManagerForLLMRun] = None, **kw: Any) -> ChatResult:
        humans = [m.content for m in messages if isinstance(m, HumanMessage)]
        system = next((m.content for m in messages if isinstance(m, SystemMessage)), "")
        state = {"category": "other", "max_price": None, "sort_by": "price_asc"}
        m = re.search(r"Текущие параметры: (\{.*\})", system)
        if m:
            state.update(json.loads(m.group(1)))
        looks_like_product = any(re.search(r"ноутбук|смартфон|телефон|планшет|холодильник|покаж|найд|хочу", h.lower()) for h in humans)

        if self.tools_bound:
            if not looks_like_product:
                msg = AIMessage(content="Здравствуйте! Чем могу помочь?")
            else:
                for h in humans:
                    state = self._apply(h, state)
                msg = AIMessage(content="", tool_calls=[{"name": "search_products", "args": state, "id": "call_1", "type": "tool_call"}])
        elif "{" in system and "category" in system:   # режим извлечения параметров
            for h in humans:
                state = self._apply(h, state)
            msg = AIMessage(content=json.dumps(state, ensure_ascii=False))
        else:                                           # часть 1: текстовый ответ
            h = humans[-1].lower()
            msg = AIMessage(content=("ноутбуки" if "ноутбук" in h else "смартфоны" if "смартфон" in h
                                     else "планшеты" if "планшет" in h else "другое"))

        n_in = sum(len(str(m.content)) for m in messages) // 3
        msg.usage_metadata = {"input_tokens": n_in, "output_tokens": max(1, len(str(msg.content)) // 3),
                              "total_tokens": n_in + max(1, len(str(msg.content)) // 3)}
        return ChatResult(generations=[ChatGeneration(message=msg)])
