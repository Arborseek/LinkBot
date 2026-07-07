"""FunASR 歌词分析：离线转录音频文件，提取数字口令时间戳生成舞谱。

独立于 voice/asr.py——语音模式用实时麦克风，本模块用文件离线转录。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

# ---- 数字字符 → 手势名 ----
DIGIT_MAP: Dict[str, str] = {
    "一": "壹", "壹": "壹", "1": "壹",
    "二": "贰", "贰": "贰", "2": "贰",
    "三": "叁", "叁": "叁", "3": "叁",
    "四": "肆", "肆": "肆", "4": "肆",
    "五": "伍", "伍": "伍", "5": "伍",
    "六": "陆", "陆": "陆", "6": "陆",
    "七": "柒", "柒": "柒", "7": "柒",
    "八": "捌", "捌": "捌", "8": "捌",
}

_VALID_GESTURES = {"张开", "壹", "贰", "叁", "肆", "伍", "陆", "柒", "捌"}

# 标点符号（FunASR ct-punc 模型添加的，没有对应时间戳）
_PUNCTUATION = set("，。！？、；：""''（）…—·,.;:!?")

# ---- 懒加载 FunASR 模型 ----
_asr_model = None


def _get_model(config: Optional[dict] = None) -> "AutoModel":
    global _asr_model
    if _asr_model is not None:
        return _asr_model
    cfg = config or {}
    model_name = cfg.get(
        "asr_model",
        "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    )
    device = cfg.get("asr_device", "cpu")  # 歌词分析走 CPU，不跟语音模式抢 GPU
    from funasr import AutoModel

    print(f"[Lyrics] 加载 FunASR 模型: {model_name} (device={device})")
    _asr_model = AutoModel(
        model=model_name,
        vad_model="fsmn-vad",
        punc_model="ct-punc",
        device=device,
        disable_update=True,
    )
    return _asr_model


def analyze_lyrics(
    audio_path: str | Path,
    *,
    lookahead_ms: float = 200.0,
    config: Optional[dict] = None,
) -> dict:
    """转录音频文件，提取数字口令时间戳，生成手势舞谱。

    Args:
        audio_path: 音频文件路径（OGG/WAV/MP3）。
        lookahead_ms: 提前发送手势的毫秒数，补偿舵机延迟。
        config: 可选配置（asr_model 等）。

    Returns:
        {"source": "asr", "events": [{"time": 0.0, "gesture": "张开"}, ...]}

    Raises:
        FileNotFoundError: 音频文件不存在。
        RuntimeError: FunASR 转录失败。
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    print(f"[Lyrics] FunASR 转录: {audio_path.name}")

    model = _get_model(config)

    # Step 1: 转录带时间戳
    result = model.generate(
        input=str(audio_path),
        batch_size_s=300,
        sentence_timestamp=True,
    )

    if not result or len(result) == 0:
        raise RuntimeError("FunASR 返回空结果")

    res0 = result[0]
    full_text = res0.get("text", "").strip()
    sentence_info = res0.get("sentence_info", [])

    print(f"[Lyrics] 识别文本: {full_text[:120]}{'...' if len(full_text) > 120 else ''}")

    if not full_text:
        raise RuntimeError("FunASR 未返回有效文本")

    # Step 2: 从 sentence_info 提取字符级时间戳
    lookahead_sec = lookahead_ms / 1000.0
    events: List[dict] = []

    if sentence_info:
        # 精确路径：用 sentence_info 的内部 timestamp（与字符对齐）
        events = _extract_from_sentence_info(sentence_info, lookahead_sec)
    else:
        # 回退路径：整段文本 + 整段 timestamp（无 VAD 分句时）
        timestamps = res0.get("timestamp", [])
        events = _extract_from_flat_text(full_text, timestamps, lookahead_sec)

    print(f"[Lyrics] 匹配到 {len(events)} 个数字手势事件")

    # Step 2.5: 间隔聚类过滤——只保留密集口令序列中的数字
    #  广播体操口令如 "一二三四五六七八" 间隔 ~0.3-0.5s
    #  而 intro 里的孤立数字（"第三套"的"三"、"七彩"的"七"）间隔 > 2s
    cluster_gap_s = float(config.get("cluster_gap_s", 2.0)) if config else 2.0
    min_cluster_events = int(config.get("min_cluster_events", 3)) if config else 3
    events = _filter_by_clusters(events, gap_s=cluster_gap_s, min_events=min_cluster_events)
    print(f"[Lyrics] 聚类过滤后: {len(events)} 个手势事件 (gap>{cluster_gap_s}s 孤立数字已剔除)")

    # Step 2.6: 补全 ASR 漏掉的首拍"一"
    #  "一二三四五六七八" 有时被 ASR 识别为 "二三四五六七八"（"一"漏检）
    #  检测密集序列以"贰"开头时，在前面插一个"壹"
    prepend_missing_one = bool(config.get("prepend_missing_one", True)) if config else True
    if prepend_missing_one:
        events = _prepend_missing_one(events)
        print(f"[Lyrics] 补全漏拍后: {len(events)} 个手势事件")

    # Step 2.7: 裁剪"伸展运动"尾巴 + 补全缺拍
    #  广播体操 4 轮 × 8 拍，之后是"伸展运动一二三四五"段落（丢弃）
    #  ASR 有时会漏掉某轮末尾的"七八"（如 Round 4 缺 柒捌），按 8 拍结构补齐
    trim_tail = bool(config.get("trim_tail", True)) if config else True
    complete_rounds = bool(config.get("complete_rounds", True)) if config else True
    if trim_tail:
        events = _trim_stretch_tail(events)
        print(f"[Lyrics] 裁剪伸展运动后: {len(events)} 个手势事件")
    if complete_rounds:
        events = _complete_rounds(events)
        print(f"[Lyrics] 补全缺拍后: {len(events)} 个手势事件")

    # Step 3: 起止加"张开"
    events.insert(0, {"time": 0.0, "gesture": "张开"})
    if events and events[-1]["gesture"] != "张开":
        last_time = events[-1]["time"]
        events.append({"time": round(last_time + 0.5, 3), "gesture": "张开"})

    # Step 4: 去重（同一时刻只保留第一个）
    deduped: List[dict] = []
    for e in events:
        if not deduped or abs(e["time"] - deduped[-1]["time"]) > 0.001:
            deduped.append(e)

    # Step 5: 验证手势名
    for e in deduped:
        if e["gesture"] not in _VALID_GESTURES:
            print(f"[Lyrics] ⚠ 未知手势 '{e['gesture']}'，替换为张开")
            e["gesture"] = "张开"

    print(f"[Lyrics] ✅ 生成 {len(deduped)} 个舞谱事件 (含起止张开)")
    return {"source": "asr", "events": deduped}


def _extract_from_sentence_info(
    sentence_info: List[dict],
    lookahead_sec: float,
) -> List[dict]:
    """从 FunASR sentence_info 提取数字事件。

    sentence_info 每个元素:
      {"text": "句子文本", "start": ms, "end": ms, "timestamp": [[s1,e1], [s2,e2], ...]}

    内部 timestamp 列表与 text 的字符一一对齐（标点除外）。
    """
    events: List[dict] = []
    for si in sentence_info:
        seg_text: str = si.get("text", "")
        seg_ts: List = si.get("timestamp", [])
        if not seg_text or not seg_ts:
            continue

        ts_idx = 0
        for char in seg_text:
            if char in _PUNCTUATION:
                continue  # 标点没有对应时间戳，跳过
            if ts_idx >= len(seg_ts):
                break
            if char in DIGIT_MAP:
                start_ms = seg_ts[ts_idx][0]
                time_sec = max(0.0, start_ms / 1000.0 - lookahead_sec)
                events.append({
                    "time": round(time_sec, 3),
                    "gesture": DIGIT_MAP[char],
                })
            ts_idx += 1

    return events


def _filter_by_clusters(
    events: List[dict],
    gap_s: float = 2.0,
    min_events: int = 3,
) -> List[dict]:
    """间隔聚类过滤：只保留密集数字序列中的事件。

    广播体操口令（一二三四五六七八）间隔约 0.3~0.5s，
    而 intro 里孤立数字（如"第三套"的三、"七彩"的七）前后间隔 3~12s。
    通过将连续间隔 <= gap_s 的事件归为一簇，丢弃点数不足的簇。
    """
    if not events:
        return []

    # 按时间排序（应该已有序，保险起见）
    sorted_events = sorted(events, key=lambda e: e["time"])

    # 分簇
    clusters: List[List[dict]] = []
    current_cluster: List[dict] = [sorted_events[0]]

    for i in range(1, len(sorted_events)):
        gap = sorted_events[i]["time"] - sorted_events[i - 1]["time"]
        if gap <= gap_s:
            current_cluster.append(sorted_events[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [sorted_events[i]]
    clusters.append(current_cluster)

    # 过滤：只保留足够大的簇
    filtered: List[dict] = []
    for cluster in clusters:
        if len(cluster) >= min_events:
            filtered.extend(cluster)
        else:
            names = [e["gesture"] for e in cluster]
            times = [f"{e['time']:.1f}s" for e in cluster]
            print(f"[Lyrics] ⛔ 剔除孤立数字: {names} @ {times} (簇大小={len(cluster)} < {min_events})")

    return filtered


def _prepend_missing_one(events: List[dict]) -> List[dict]:
    """补全 ASR 漏掉的首拍"一"。

    场景：ASR 把 "一二三四五六七八" 识别为 "二三四五六七八"。
    检测密集序列以"贰"开头时，在前面补一个"壹"（时间提前 0.3~0.5s）。
    """
    if not events:
        return events

    sorted_events = sorted(events, key=lambda e: e["time"])
    result: List[dict] = []

    i = 0
    while i < len(sorted_events):
        e = sorted_events[i]

        # 检测一个 8 拍序列的起始：以"贰"开头，后面紧跟"贰"或"叁"
        if e["gesture"] == "贰" and not (
            i > 0 and sorted_events[i - 1]["time"] >= e["time"] - 0.6
        ):
            # 这是密集序列的起始，检查是否缺"一"
            # 广播体操每轮 8 拍：一二三四五六七八 / 二二三四五六七八 / ...
            is_round_start = False
            if i + 1 < len(sorted_events):
                nxt = sorted_events[i + 1]
                # 首轮：贰 → 叁（原本应该是 壹→贰→叁，现在贰开头）
                # 后续轮：贰 → 贰（原本 二二三四...）
                if nxt["gesture"] in ("贰", "叁") and nxt["time"] - e["time"] < 1.0:
                    is_round_start = True

            if is_round_start:
                # 补一个"壹"，比当前"贰"提前约 0.35s（典型口令间隔）
                one_time = round(e["time"] - 0.35, 3)
                if one_time >= 0:
                    result.append({"time": max(0.0, one_time), "gesture": "壹"})
                    print(f"[Lyrics] 🔧 补全漏检首拍: 壹@{one_time:.2f}s (原序列以贰@{e['time']:.2f}s 开始)")

        result.append(e)
        i += 1

    return result


# 广播体操 8 拍结构：每轮 [轮号] + 二三四五六七八
_ROUND_PATTERN = ["贰", "叁", "肆", "伍", "陆", "柒", "捌"]


def _trim_stretch_tail(events: List[dict]) -> List[dict]:
    """裁剪末尾的"伸展运动一二三四五"段落。

    特征：末尾出现「壹贰叁肆伍」连续序列（非完整 8 拍轮次），
    这是广播操的整理运动段落，不属于 4 轮手势舞。
    """
    if len(events) < 6:
        return events

    # 从后往前找「壹贰叁肆伍」连续模式（伸展运动独有）
    # 找到连续的 壹→贰→叁→肆→伍 且间隔 < 1s
    stretch_start = -1
    for i in range(len(events) - 5, -1, -1):
        window = [e["gesture"] for e in events[i:i + 5]]
        if window == ["壹", "贰", "叁", "肆", "伍"]:
            # 确认间隔紧凑（不是跨轮次的零散事件）
            gaps = [
                events[j + 1]["time"] - events[j]["time"]
                for j in range(i, i + 4)
            ]
            if all(g < 1.5 for g in gaps):
                stretch_start = i
                break

    if stretch_start >= 0:
        tail = events[stretch_start:]
        names = [e["gesture"] for e in tail]
        print(f"[Lyrics] ✂ 裁剪伸展运动尾巴: {names}")
        return events[:stretch_start]

    return events


def _complete_rounds(events: List[dict]) -> List[dict]:
    """补全每轮缺失的拍子。

    广播体操每轮 8 拍：轮号 + 贰叁肆伍陆柒捌。
    将事件按轮次分组，不足 8 拍的轮次按 pattern 补齐。
    """
    if len(events) < 3:
        return events

    sorted_events = sorted(events, key=lambda e: e["time"])
    rounds = _split_into_rounds(sorted_events)
    if not rounds:
        return events

    # 每轮期望的完整序列（根据轮号动态生成）
    completed = []
    for rnd in rounds:
        lead = rnd[0]["gesture"]
        if lead not in ("壹", "贰", "叁", "肆"):
            # 不识别的轮号，原样保留
            completed.extend(rnd)
            continue

        expected = [lead] + _ROUND_PATTERN  # e.g. 壹,贰,叁,肆,伍,陆,柒,捌

        if len(rnd) >= 8:
            completed.extend(rnd[:8])
        else:
            # 缺拍，补齐
            missing_count = 8 - len(rnd)
            if len(rnd) >= 2:
                avg_gap = (rnd[-1]["time"] - rnd[0]["time"]) / (len(rnd) - 1)
            else:
                avg_gap = 0.5

            completed.extend(rnd)
            last_time = rnd[-1]["time"]
            last_g = rnd[-1]["gesture"]

            for _ in range(missing_count):
                try:
                    idx = expected.index(last_g)
                    next_g = expected[idx + 1]
                except (ValueError, IndexError):
                    next_g = "张开"
                last_time = round(last_time + avg_gap, 3)
                last_g = next_g
                completed.append({"time": last_time, "gesture": next_g})

            missing_names = [e["gesture"] for e in completed[-missing_count:]]
            print(f"[Lyrics] 🔧 补全缺拍 (轮号={lead}): +{missing_names}")

    return completed


def _split_into_rounds(events: List[dict]) -> List[List[dict]]:
    """按轮次分组。

    广播体操每轮 8 拍，轮次间无明显停顿。
    检测特征：上一轮以"捌"结尾后，下一轮以新轮号（壹/贰/叁/肆）开头。
    首轮以第一个轮号手势开始。
    """
    if not events:
        return []

    rounds = []
    current = []

    for e in events:
        g = e["gesture"]
        if g in ("壹", "贰", "叁", "肆"):
            if current and current[-1]["gesture"] == "捌":
                # 上一轮以捌结尾，新轮号开始
                rounds.append(current)
                current = [e]
                continue
            if not current:
                # 第一个事件就是轮号
                current = [e]
                continue
        current.append(e)

    if current:
        rounds.append(current)

    return rounds


def _extract_from_flat_text(
    text: str,
    timestamps: List,
    lookahead_sec: float,
) -> List[dict]:
    """回退方案：无 sentence_info 时，用整段 timestamp 按字符位置估算。

    timestamp 列表应与 text 长度对齐（标点除外）。
    """
    events: List[dict] = []
    ts_idx = 0
    for char in text:
        if char in _PUNCTUATION:
            continue
        if ts_idx >= len(timestamps):
            break
        if char in DIGIT_MAP:
            start_ms = timestamps[ts_idx][0]
            time_sec = max(0.0, start_ms / 1000.0 - lookahead_sec)
            events.append({
                "time": round(time_sec, 3),
                "gesture": DIGIT_MAP[char],
            })
        ts_idx += 1

    return events
