from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TextItem = tuple[str, tuple[int, int], int, tuple[int, int, int]]

_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
]


@lru_cache(maxsize=16)
def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= c <= "\u9fff" for c in text)


def put_text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    font_size: int = 22,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
) -> np.ndarray:
    """在 BGR 图像上绘制单段文本。含中文时走 Pillow。"""
    return render_texts(img, [(text, org, font_size, color)], ascii_thickness=thickness)


def render_texts(
    img: np.ndarray,
    texts: Sequence[TextItem],
    ascii_thickness: int = 1,
) -> np.ndarray:
    """批量绘制文本，含中文时只做一次 PIL 转换。"""
    if not texts:
        return img

    ascii_items: list[TextItem] = []
    cjk_items: list[TextItem] = []
    for item in texts:
        if _has_cjk(item[0]):
            cjk_items.append(item)
        else:
            ascii_items.append(item)

    out = img
    for text, org, font_size, color in ascii_items:
        x, y = org
        cv2.putText(
            out, text, (x, y + font_size),
            cv2.FONT_HERSHEY_SIMPLEX, font_size / 32, color,
            ascii_thickness, cv2.LINE_AA,
        )

    if not cjk_items:
        return out

    pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    for text, org, font_size, color in cjk_items:
        font = _load_font(font_size)
        rgb = (color[2], color[1], color[0])
        draw.text(org, text, font=font, fill=rgb)

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
