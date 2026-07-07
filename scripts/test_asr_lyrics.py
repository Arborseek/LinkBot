#!/usr/bin/env python3
"""Stage 0: 验证 FunASR 能否对 ogg 文件输出带时间戳的转录。

用途：确认七彩阳光音频里能被 ASR 识别出数字（一~八/1~8），
      以及时间戳精度是否足够用于舞蹈卡点。

用法：
    cd ~/linkerbot && PYTHONPATH=src python scripts/test_asr_lyrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# 数字字符映射（用于预览匹配结果）
DIGIT_MAP = {
    "一": "壹", "壹": "壹", "1": "壹",
    "二": "贰", "贰": "贰", "2": "贰",
    "三": "叁", "叁": "叁", "3": "叁",
    "四": "肆", "肆": "肆", "4": "肆",
    "五": "伍", "伍": "伍", "5": "伍",
    "六": "陆", "陆": "陆", "6": "陆",
    "七": "柒", "柒": "柒", "7": "柒",
    "八": "捌", "捌": "捌", "8": "捌",
}


def main() -> None:
    audio_path = Path("assets/music/qicai_yangguang.ogg")
    if not audio_path.exists():
        print(f"❌ 音频文件不存在: {audio_path}")
        sys.exit(1)

    print(f"🎵 音频文件: {audio_path} ({audio_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print("⏳ 加载 FunASR 模型（首次需下载 ~900MB）...")

    from funasr import AutoModel

    model = AutoModel(
        model="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        vad_model="fsmn-vad",
        punc_model="ct-punc",
    )

    print("⏳ 转录中（带时间戳）...")

    # 尝试 sentence_timestamp=True 获取句子级时间戳
    result = model.generate(
        input=str(audio_path),
        batch_size_s=300,
        sentence_timestamp=True,
    )

    if not result or len(result) == 0:
        print("❌ FunASR 返回空结果")
        sys.exit(1)

    r = result[0]
    print(f"\n{'='*60}")
    print(f"📝 完整转录文本:")
    print(f"{'='*60}")
    print(r.get("text", "(无文本)"))

    # 打印时间戳信息
    timestamp = r.get("timestamp", [])
    if timestamp:
        print(f"\n{'='*60}")
        print(f"⏱ 句子级时间戳 ({len(timestamp)} 句):")
        print(f"{'='*60}")
        for i, ts in enumerate(timestamp):
            text_seg = r.get("text_seg", [""] * len(timestamp)) if "text_seg" in r else [""]
            seg_text = text_seg[i] if i < len(text_seg) else ""
            start_s = ts[0] / 1000.0 if isinstance(ts, (list, tuple)) and len(ts) >= 2 else 0
            end_s = ts[1] / 1000.0 if isinstance(ts, (list, tuple)) and len(ts) >= 2 else 0
            print(f"  [{start_s:7.2f}s - {end_s:7.2f}s] {seg_text}")
    else:
        print("\n⚠️ 无 sentence_timestamp 输出")

    # 打印 result 的所有 key，方便调试
    print(f"\n{'='*60}")
    print(f"🔑 result keys: {list(r.keys())}")
    print(f"{'='*60}")

    # 探索 sentence_info 结构（FunASR 的真实分句数据）
    sentence_info = r.get("sentence_info", [])
    if sentence_info:
        print(f"\n📋 sentence_info 类型: {type(sentence_info)}")
        print(f"📋 sentence_info 长度: {len(sentence_info)}")
        if len(sentence_info) > 0:
            print(f"📋 sentence_info[0] 类型: {type(sentence_info[0])}")
            if isinstance(sentence_info[0], dict):
                print(f"📋 sentence_info[0] keys: {list(sentence_info[0].keys())}")
                print(f"📋 sentence_info[0]: {sentence_info[0]}")
            else:
                print(f"📋 sentence_info[0]: {sentence_info[0]}")
            # 打印前5条
            print(f"\n📋 前 5 条 sentence_info:")
            for i, si in enumerate(sentence_info[:5]):
                print(f"  [{i}] {si}")

    # ---- 核心：用 sentence_info 做数字匹配 ----
    full_text = r.get("text", "")
    matched_count = 0
    matched_events = []

    print(f"\n{'='*60}")
    print(f"🔍 数字匹配预览 (基于 sentence_info):")
    print(f"{'='*60}")

    if sentence_info and isinstance(sentence_info[0], dict):
        # sentence_info 是 [{"text": "...", "start": ms, "end": ms}, ...]
        for si in sentence_info:
            seg_text = si.get("text", "")
            seg_start_ms = si.get("start", 0)
            seg_end_ms = si.get("end", 0)
            for j, char in enumerate(seg_text):
                if char in DIGIT_MAP:
                    # 按字符在句子中的位置估算时间
                    char_ratio = j / max(len(seg_text), 1)
                    char_time_ms = seg_start_ms + char_ratio * (seg_end_ms - seg_start_ms)
                    matched_count += 1
                    event = {
                        "time": round(char_time_ms / 1000.0, 3),
                        "gesture": DIGIT_MAP[char],
                        "char": char,
                        "context": seg_text,
                    }
                    matched_events.append(event)
                    ctx_start = max(0, j - 2)
                    ctx_end = min(len(seg_text), j + 3)
                    print(f"  [{event['time']:7.2f}s] '{char}' → {DIGIT_MAP[char]}  (sentence: ...{seg_text[ctx_start:ctx_end]}...)")

    # Fallback: 如果 sentence_info 没有，在整个 text 中逐字扫描
    if matched_count == 0:
        print("  (无 sentence_info，在整个文本中逐字扫描...)")
        timestamps = r.get("timestamp", [])
        for i, char in enumerate(full_text):
            if char in DIGIT_MAP:
                est_time = 0.0
                if timestamps and len(timestamps) > 0:
                    total_duration_s = timestamps[-1][1] / 1000.0 if len(timestamps[-1]) >= 2 else 0
                    est_time = (i / max(len(full_text), 1)) * total_duration_s
                matched_count += 1
                ctx_start = max(0, i - 3)
                ctx_end = min(len(full_text), i + 4)
                print(f"  [est ~{est_time:.2f}s] '{char}' → {DIGIT_MAP[char]}  (context: ...{full_text[ctx_start:ctx_end]}...)")

    if matched_count == 0:
        print("\n❌ 未在转录文本中找到任何数字字符（一~八 / 1~8）")
        print("   无法使用 ASR 歌词卡点方案，请使用 choreography_source: llm 或 fallback")
        sys.exit(1)
    else:
        print(f"\n✅ 共匹配 {matched_count} 个数字手势事件")
        print(f"   预计舞谱 events:")
        print(f"   events:")
        print(f"     - {{time: 0.0, gesture: 张开}}  # 起始")
        for e in matched_events:
            print(f"     - {{time: {e['time']}, gesture: {e['gesture']}}}  # ASR: '{e['char']}' @ {e['context']}")
        print(f"     - {{time: {matched_events[-1]['time'] + 0.5:.1f}, gesture: 张开}}  # 结束")


if __name__ == "__main__":
    main()
