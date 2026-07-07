"""舞谱编排：生成 + 缓存。

LLM 智能编排（use_llm=true）：根据 BPM、节拍、前奏长度、可用手势，
一次性生成完整舞谱并缓存。简单循环（use_llm=false）作为 fallback。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml


# ---- 手势编号映射 ----
GESTURE_INDEX: Dict[int, str] = {
    1: "壹", 2: "贰", 3: "叁", 4: "肆",
    5: "伍", 6: "陆", 7: "柒", 8: "捌",
}

_AVAILABLE_GESTURES = ["张开", "壹", "贰", "叁", "肆", "伍", "陆", "柒", "捌"]

_CHOREO_PROMPT = """你是灵巧手手势舞编排师。为一首中文儿歌编排 L10 机械手的手势舞。

## 可用手势

- 张开：五指完全张开（起始/结束姿态）
- 壹：仅伸出食指，其余四指握拳
- 贰：伸出食指+中指，其余握拳
- 叁：伸出食指+中指+无名指，其余握拳
- 肆：伸出四指（食指+中指+无名指+小指），拇指握拳
- 伍：五指全部张开
- 陆：伸出拇指+小指，其余三指握拳
- 柒：拇指+食指+中指捏合（OK 手势变体），其余握拳
- 捌：伸出拇指+食指（手枪手势），其余握拳

## 歌曲信息

{track_info}

## 编排规则

1. 第一个手势必须是"张开"（在 0.0 秒），最后一个也必须是"张开"
2. 手势切换必须在节拍时间戳上（给的 beat_times 列表里选），每 2~6 拍切换一次
3. 不要连续重复同一手势超过 2 次
4. 有节奏感，强拍（每小节第一拍）上尽量切手势
5. 优先使用相邻数字手势的过渡（壹→贰→叁 比 壹→捌 更自然，因为只需动一根手指）
6. 整段约 32 个手势事件

## 输出格式

返回纯 JSON，不要 markdown 代码块，不要任何解释：
{"events": [{"time": 0.0, "gesture": "张开"}, {"time": 0.93, "gesture": "壹"}, ...]}"""


def make_fallback_choreography(
    beat_times: List[float],
    *,
    beats_per_bar: int = 4,
    bars_per_gesture: int = 1,
    sequence: Optional[List[str]] = None,
    bpm: float = 120.0,
) -> dict:
    """无 LLM 时的简单舞谱：每个手势占 bars_per_gesture 小节，对齐强拍。"""
    if sequence is None:
        sequence = ["张开", "壹", "贰", "叁", "肆", "伍", "陆", "柒", "捌", "张开"]

    downbeats = [t for i, t in enumerate(beat_times) if i % beats_per_bar == 0]

    events = []
    for i, name in enumerate(sequence):
        db_idx = i * bars_per_gesture
        if db_idx < len(downbeats):
            event_time = downbeats[db_idx]
        elif events:
            bar_duration = (60.0 / bpm) * beats_per_bar
            event_time = events[-1]["time"] + bar_duration * bars_per_gesture
        else:
            event_time = 0.0
        events.append({"time": round(event_time, 3), "gesture": name})

    return {"bpm": round(bpm, 1), "events": events}


def load_or_generate(
    cache_path: str | Path,
    beat_times: Optional[List[float]] = None,
    *,
    config: Optional[dict] = None,
) -> dict:
    """优先读缓存 YAML；缓存不存在或 force_regenerate 时重新生成。

    choreography_source 控制生成方式:
      - "asr": FunASR 歌词时间戳（不需 beat_times）
      - "llm": DeepSeek API 编排（需 beat_times）
      - "fallback": 强拍循环（需 beat_times）
    """
    cfg = config or {}
    cache_path = Path(cache_path)
    force = bool(cfg.get("force_regenerate", False))

    # 向后兼容：旧配置没有 choreography_source，从 use_llm 推导
    source = cfg.get("choreography_source")
    if source is None:
        use_llm = bool(cfg.get("use_llm", True))
        source = "llm" if use_llm else "fallback"

    # 读缓存
    if cache_path.exists() and not force:
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = yaml.safe_load(f)
            if cached and "events" in cached and len(cached["events"]) > 0:
                src_label = cached.get("source", "?")
                print(f"[Choreographer] 读取缓存舞谱: {cache_path} ({len(cached['events'])} 个事件, source={src_label})")
                return cached
        except Exception:
            print(f"[Choreographer] 缓存读取失败，重新生成")

    # 按 source 生成
    if source == "asr":
        choreography = _generate_asr(cfg)
    elif source == "llm":
        if beat_times is None:
            raise ValueError("LLM 编排需要 beat_times，请先调用 analyze_beats()")
        bpm = float(cfg.get("bpm", 120.0))
        beats_per_bar = int(cfg.get("beats_per_bar", 4))
        total_duration = float(beat_times[-1]) if beat_times else 30.0
        choreography = _generate_llm(
            beat_times=beat_times,
            bpm=bpm,
            total_duration=total_duration,
            beats_per_bar=beats_per_bar,
            config=cfg,
        )
    else:  # fallback
        if beat_times is None:
            raise ValueError("fallback 编排需要 beat_times，请先调用 analyze_beats()")
        bpm = float(cfg.get("bpm", 120.0))
        beats_per_bar = int(cfg.get("beats_per_bar", 4))
        bars_per_gesture = int(cfg.get("bars_per_gesture", 1))
        sequence = cfg.get("gesture_sequence")
        choreography = make_fallback_choreography(
            beat_times,
            beats_per_bar=beats_per_bar,
            bars_per_gesture=bars_per_gesture,
            sequence=sequence,
            bpm=bpm,
        )

    # 写缓存
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(choreography, f, allow_unicode=True, sort_keys=False)
        print(f"[Choreographer] 舞谱已缓存: {cache_path}")
    except Exception as e:
        print(f"[Choreographer] 缓存写入失败: {e}")

    return choreography


def save_choreography(cache_path: str | Path, choreography: dict) -> None:
    """手动保存舞谱到 YAML"""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(choreography, f, allow_unicode=True, sort_keys=False)


# ---- LLM 编排 ----

def _generate_llm(
    beat_times: List[float],
    bpm: float,
    total_duration: float,
    beats_per_bar: int,
    config: dict,
) -> dict:
    """调 DeepSeek API 生成舞谱，失败时回退 fallback"""
    api_key = config.get("api_key") or os.environ.get("DEEPSEEK_API_KEY", "")
    api_base = config.get("api_base", "https://api.deepseek.com")
    model = config.get("model", "deepseek-chat")

    # 构建歌曲信息
    downbeats = [t for i, t in enumerate(beat_times) if i % beats_per_bar == 0]
    track_info = f"""- BPM: {bpm:.0f}
- 总时长: {total_duration:.1f} 秒
- 节拍数: {len(beat_times)}
- 强拍（每小节第一拍）: {len(downbeats)} 个
- 强拍时间戳: {', '.join(f'{t:.1f}' for t in downbeats[:20])}..."""

    print(f"[Choreographer] 调用 LLM 编排舞谱...")
    print(f"  BPM={bpm:.0f} 总长={total_duration:.1f}s 强拍={len(downbeats)}个")

    try:
        from openai import OpenAI
        import httpx

        http_client = httpx.Client(proxy=None, trust_env=False)
        client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            http_client=http_client,
        )

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _CHOREO_PROMPT.format(track_info=track_info)},
                {"role": "user", "content": "请为这首歌编排手势舞，返回 JSON。"},
            ],
            max_tokens=4096,
            temperature=0.7,
        )

        content = resp.choices[0].message.content.strip()

        # 清理可能的 markdown 代码块
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:]) if len(lines) > 1 else content
        if content.endswith("```"):
            content = content[:-3].strip()

        choreography = json.loads(content)

        # 验证格式
        events = choreography.get("events", [])
        if not events:
            raise ValueError("LLM 返回空事件列表")

        # 确保第一个事件在 0.0s 且是张开
        if events[0]["time"] != 0.0 or events[0].get("gesture") != "张开":
            events.insert(0, {"time": 0.0, "gesture": "张开"})

        # 确保最后一个事件是张开
        if events[-1].get("gesture") != "张开":
            events.append({"time": round(total_duration - 0.5, 2), "gesture": "张开"})

        # 验证所有手势名有效
        for e in events:
            if e["gesture"] not in _AVAILABLE_GESTURES:
                print(f"[Choreographer] ⚠ 未知手势 '{e['gesture']}'，替换为张开")
                e["gesture"] = "张开"

        choreography["bpm"] = round(bpm, 1)
        choreography["events"] = events
        print(f"[Choreographer] ✅ LLM 生成 {len(events)} 个手势事件")
        return choreography

    except Exception as e:
        print(f"[Choreographer] LLM 编排失败 ({e})，回退简单循环")
        return make_fallback_choreography(
            beat_times, bpm=bpm,
            beats_per_bar=beats_per_bar,
            bars_per_gesture=config.get("bars_per_gesture", 1),
            sequence=config.get("gesture_sequence"),
        )


# ---- ASR 歌词编排 ----

def _generate_asr(config: dict) -> dict:
    """FunASR 歌词卡点编排。失败时自动回退 fallback。"""
    from linkerbot.dance.lyrics_analyzer import analyze_lyrics

    audio_path = config.get("audio_path", "assets/music/qicai_yangguang.ogg")
    lookahead_ms = float(config.get("lookahead_ms", 200))

    try:
        choreography = analyze_lyrics(
            audio_path,
            lookahead_ms=lookahead_ms,
            config=config,
        )
        events = choreography.get("events", [])
        digital_events = [e for e in events if e["gesture"] != "张开"]
        if len(digital_events) == 0:
            print("[Choreographer] ASR 未检测到数字口令，回退 fallback")
            return _asr_fallback(config)
        print(f"[Choreographer] ✅ ASR 生成 {len(events)} 个手势事件 ({len(digital_events)} 个数字)")
        return choreography
    except Exception as e:
        print(f"[Choreographer] ASR 歌词分析失败 ({e})，回退 fallback")
        return _asr_fallback(config)


def _asr_fallback(config: dict) -> dict:
    """ASR 失败时的回退：librosa 节拍 + 简单循环序列。"""
    from linkerbot.dance.music_analyzer import analyze_beats

    audio_path = config.get("audio_path", "assets/music/qicai_yangguang.ogg")
    beats_per_bar = int(config.get("beats_per_bar", 4))
    bars_per_gesture = int(config.get("bars_per_gesture", 1))
    sequence = config.get("gesture_sequence")
    bpm = float(config.get("bpm", 120.0))

    try:
        bpm, beat_times = analyze_beats(audio_path)
        print(f"[Choreographer] fallback: BPM={bpm:.0f}, {len(beat_times)} beats")
    except Exception:
        # librosa 也失败了——生成一个基于 BPM 的虚拟节拍
        total_duration = 35.0  # 七彩阳光 ~32s，留余量
        beat_interval = 60.0 / bpm
        beat_times = [i * beat_interval for i in range(int(total_duration / beat_interval))]
        print(f"[Choreographer] fallback: librosa 失败，用虚拟节拍 BPM={bpm:.0f}")

    return make_fallback_choreography(
        beat_times,
        beats_per_bar=beats_per_bar,
        bars_per_gesture=bars_per_gesture,
        sequence=sequence,
        bpm=bpm,
    )
