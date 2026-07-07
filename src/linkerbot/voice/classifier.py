"""DeepSeek API 分类器。

把自然语言映射到预设姿态名。OpenAI 兼容格式，温度=0 确保确定性。
"""

from __future__ import annotations

import os
from typing import Optional

SYSTEM_PROMPT = """你是手势分类器。把用户的自然语言指令映射到以下手势之一：

{gesture_list}

规则：
1. 只返回手势名，不要任何标点或解释
2. 如果用户的话不匹配任何手势，返回 "无"
3. 同义词和口语表达也要正确匹配
   例如: "握拳"="攥拳头"="捏紧"="拳头" / "比耶"="剪刀手"="耶" /
         "点赞"="棒"="大拇指"="真棒" / "OK"="好的"="行" /
         "张开手掌"="打开"="五指张开" / "捏合"="捏住"="捏一下" /
         "准备抓握"="抓住"="抓取" / "拇指弯曲"="弯拇指" /
         "食指弯曲"="指"="指一下"="指那边" """


class GestureClassifier:
    """DeepSeek LLM 手势分类"""

    def __init__(
        self,
        api_key: str = "",
        api_base: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
    ):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.api_base = api_base
        self.model = model
        self._client = None

    # ---- public ----

    def classify(self, text: str, gesture_names: list[str]) -> Optional[str]:
        """返回匹配的姿态名，或 None"""
        if not gesture_names or not text.strip():
            return None
        if self._client is None:
            self._init_client()
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT.format(
                            gesture_list="\n".join(f"- {n}" for n in gesture_names)
                        ),
                    },
                    {"role": "user", "content": text.strip()},
                ],
                max_tokens=10,
                temperature=0.0,
            )
            name = resp.choices[0].message.content.strip()
            # 严格匹配
            if name in gesture_names:
                return name
            # 模糊匹配（去除标点、空格）
            cleaned = name.replace("，", "").replace("。", "").replace(" ", "")
            for g in gesture_names:
                if cleaned == g:
                    return g
            return None
        except Exception:
            return None

    # ---- internal ----

    def _init_client(self) -> None:
        from openai import OpenAI
        import httpx

        # 跳过系统代理（httpx 不支持 socks:// 代理）
        http_client = httpx.Client(proxy=None, trust_env=False)
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            http_client=http_client,
        )
