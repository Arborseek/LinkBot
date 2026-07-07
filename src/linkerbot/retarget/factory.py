from __future__ import annotations

from typing import Dict, List, Protocol

import numpy as np

from linkerbot.config.session import HandProfile
from linkerbot.models import HandJoints, HandTracking
from linkerbot.retarget.smooth import JointFilterBank
from linkerbot.sim.pose_mapping import sdk_open_rad


class Retargeter(Protocol):
    def reset(self) -> None: ...
    def retarget(self, tracking: HandTracking) -> HandJoints | None: ...


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(np.clip(v, lo, hi))


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return float(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))


class CurlMapper:
    """MediaPipe 弯曲角 -> 关节角；open/closed 参考角决定灵敏度"""

    def __init__(
        self,
        open_angle: float = 2.45,
        closed_angle: float = 0.85,
        gain: float = 1.4,
        tip_open: float = 2.5,
        tip_closed: float = 0.55,
        tip_gain: float = 1.5,
    ):
        self.open_angle = open_angle
        self.closed_angle = closed_angle
        self.gain = gain
        self.tip_open = tip_open
        self.tip_closed = tip_closed
        self.tip_gain = tip_gain

    def map_curl(self, curl: float, lo: float, hi: float, *, tip: bool = False) -> float:
        o = self.tip_open if tip else self.open_angle
        c = self.tip_closed if tip else self.closed_angle
        g = self.tip_gain if tip else self.gain
        span = max(o - c, 0.25)
        t = (o - curl) / span
        t = _clamp(t * g, 0.0, 1.0)
        return lo + t * (hi - lo)


_default_mapper = CurlMapper()


def _map_curl(curl: float, lo: float, hi: float, mapper: CurlMapper | None = None, *, tip: bool = False) -> float:
    return (mapper or _default_mapper).map_curl(curl, lo, hi, tip=tip)


WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP = 5, 6
MIDDLE_MCP, MIDDLE_PIP = 9, 10
RING_MCP, RING_PIP = 13, 14
PINKY_MCP, PINKY_PIP = 17, 18
INDEX_PIP, INDEX_DIP, INDEX_TIP = 6, 7, 8
MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 10, 11, 12
RING_PIP, RING_DIP, RING_TIP = 14, 15, 16
PINKY_PIP, PINKY_DIP, PINKY_TIP = 18, 19, 20


def _raw_pts(tr: HandTracking) -> np.ndarray | None:
    if not tr.detected:
        return None
    # 摄像头遥操作优先用图像归一化坐标，world 深度在部分设备上不稳定
    if tr.landmarks is not None:
        return tr.landmarks
    return tr.world_landmarks


def _normalize_hand(pts: np.ndarray) -> np.ndarray:
    """腕部居中 + 掌尺归一化，减弱手在画面中的位置/远近影响"""
    centered = pts - pts[WRIST]
    scale = float(np.linalg.norm(centered[MIDDLE_MCP])) + 1e-6
    return centered / scale


def _palm_scale(pts: np.ndarray) -> float:
    return float(np.linalg.norm(pts[MIDDLE_MCP] - pts[WRIST])) + 1e-6


def _pinch_strength(
    pts: np.ndarray,
    tip_a: int,
    tip_b: int,
    scale: float,
    *,
    close: float = 0.04,
    far: float = 0.12,
) -> float:
    """两指尖距离；OK 手势主要看图像平面投影，取 2D/3D 较小值"""
    d2 = float(np.linalg.norm(pts[tip_a, :2] - pts[tip_b, :2])) / scale
    d3 = float(np.linalg.norm(pts[tip_a] - pts[tip_b])) / scale
    d = min(d2, d3)
    return _clamp(1.0 - (d - close) / max(far - close, 0.03), 0.0, 1.0)


def _mcp_gap(pts: np.ndarray, a: int, b: int, scale: float) -> float:
    return float(np.linalg.norm(pts[a] - pts[b])) / scale


def _palm_frame(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    wrist = pts[WRIST]
    px = pts[PINKY_MCP] - pts[INDEX_MCP]
    px = px / (float(np.linalg.norm(px)) + 1e-6)
    py = pts[MIDDLE_MCP] - wrist
    py = py - float(np.dot(py, px)) * px
    py = py / (float(np.linalg.norm(py)) + 1e-6)
    pz = np.cross(px, py)
    return wrist, px, py, pz


def _finger_dir(pts: np.ndarray, mcp: int, tip: int) -> np.ndarray:
    v = pts[tip] - pts[mcp]
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return np.array([0.0, 1.0, 0.0])
    return v / n


def _finger_spread_angle(pts: np.ndarray, m1: int, t1: int, m2: int, t2: int) -> float:
    v1 = _finger_dir(pts, m1, t1)
    v2 = _finger_dir(pts, m2, t2)
    return float(np.arccos(np.clip(float(np.dot(v1, v2)), -1.0, 1.0)))


def _wrist_wedge(pts: np.ndarray, m1: int, m2: int) -> float:
    """腕到相邻 MCP 的夹角，叉开时增大"""
    v1 = pts[m1] - pts[WRIST]
    v2 = pts[m2] - pts[WRIST]
    n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return float(np.arccos(np.clip(float(np.dot(v1 / n1, v2 / n2)), -1.0, 1.0)))


def _spread_signals(pts: np.ndarray, scale: float) -> Dict[str, float]:
    wrist, px, _, _ = _palm_frame(pts)
    return {
        "mcp_im": _wrist_wedge(pts, INDEX_MCP, MIDDLE_MCP),
        "mcp_mr": _wrist_wedge(pts, MIDDLE_MCP, RING_MCP),
        "mcp_rp": _wrist_wedge(pts, RING_MCP, PINKY_MCP),
        "mcp_ti": _wrist_wedge(pts, THUMB_CMC, INDEX_MCP),
        "ang_im": _finger_spread_angle(pts, INDEX_MCP, INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP),
        "ang_mr": _finger_spread_angle(pts, MIDDLE_MCP, MIDDLE_TIP, RING_MCP, RING_TIP),
        "ang_rp": _finger_spread_angle(pts, RING_MCP, RING_TIP, PINKY_MCP, PINKY_TIP),
        "ang_ti": _finger_spread_angle(pts, THUMB_CMC, THUMB_TIP, INDEX_MCP, INDEX_TIP),
        "tip_im": _mcp_gap(pts, INDEX_TIP, MIDDLE_TIP, scale),
        "tip_mr": _mcp_gap(pts, MIDDLE_TIP, RING_TIP, scale),
        "tip_rp": _mcp_gap(pts, RING_TIP, PINKY_TIP, scale),
        "tip_ti": _mcp_gap(pts, THUMB_TIP, INDEX_TIP, scale),
        "lat_index": -float(np.dot(pts[INDEX_TIP] - wrist, px)) / scale,
        "lat_pinky": float(np.dot(pts[PINKY_TIP] - wrist, px)) / scale,
        "lat_thumb": float(np.dot(pts[THUMB_TIP] - wrist, px)) / scale,
    }


def _delta_to_t(delta: float, span: float, gain: float = 1.0) -> float:
    return _clamp(delta / max(span, 0.02) * gain, -1.0, 1.0)


def _spread_abduction(
    t: float,
    lo: float,
    hi: float,
    *,
    open_rad: float | None = None,
    open_at_lo: bool = True,
    frac: float = 1.0,
    sign: float = 1.0,
) -> float:
    """t>0 比校准更叉开；open_at_lo=False 时张开位在 hi（L10 食指侧摆 j_dir=0）"""
    if open_rad is None:
        open_rad = lo if open_at_lo else hi
    t = _clamp(float(t) * float(sign), -1.0, 1.0)
    frac = _clamp(frac, 0.35, 1.0)
    if open_at_lo:
        if t >= 0.0:
            target = open_rad + frac * (hi - open_rad)
            return open_rad + t * (target - open_rad)
        target = open_rad + frac * (lo - open_rad)
        return open_rad + (-t) * (open_rad - target)
    if t >= 0.0:
        target = open_rad + frac * (lo - open_rad)
        return open_rad + t * (target - open_rad)
    target = open_rad + frac * (hi - open_rad)
    return open_rad + (-t) * (target - open_rad)


def _abduction_from_t(
    t: float, lo: float, hi: float, *, center: float = 0.0, frac: float = 1.0
) -> float:
    """拇指侧摆等：相对 center 的增量映射"""
    t = _clamp(t, -1.0, 1.0)
    frac = _clamp(frac, 0.35, 1.0)
    eff_lo = center + frac * (lo - center)
    eff_hi = center + frac * (hi - center)
    if t >= 0.0:
        return center + t * (eff_lo - center)
    return center + (-t) * (eff_hi - center)


def _compute_finger_spreads(
    pts: np.ndarray,
    scale: float,
    limits: dict,
    cfg: dict,
    ref: Dict[str, float] | None = None,
) -> Dict[str, float]:
    s_cfg = cfg.get("spread", {})
    span = float(s_cfg.get("span", 0.10))
    gain = float(s_cfg.get("gain", 1.6))
    frac = float(s_cfg.get("range_frac", 0.82))
    sig = _spread_signals(pts, scale)
    open_idx = float(limits["index_abduction"][0])
    open_mid = float(limits["middle_abduction"][0])
    open_ring = float(limits["ring_abduction"][0])
    open_pinky = float(limits["pinky_abduction"][0])
    open_thumb = float(limits["thumb_abduction"][0])

    if ref is None:
        return {
            "thumb_abduction": open_thumb,
            "index_abduction": open_idx,
            "middle_abduction": open_mid,
            "ring_abduction": open_ring,
            "pinky_abduction": open_pinky,
        }, 0.0

    def d(key: str) -> float:
        return float(sig[key]) - float(ref[key])

    t_im = _clamp(
        0.52 * _delta_to_t(d("mcp_im"), span, gain)
        + 0.38 * _delta_to_t(d("tip_im"), span, gain * 1.05)
        + 0.10 * _delta_to_t(d("lat_index"), span * 0.85, gain * 0.9),
        -1.0,
        1.0,
    )
    t_ring = _clamp(
        0.45 * _delta_to_t(d("mcp_mr"), span, gain)
        + 0.35 * _delta_to_t(d("tip_mr"), span, gain)
        + 0.20 * _delta_to_t(d("ang_mr"), span, gain * 0.7),
        -1.0,
        1.0,
    )
    t_pinky = _clamp(
        0.42 * _delta_to_t(d("mcp_rp"), span, gain)
        + 0.28 * _delta_to_t(d("tip_rp"), span, gain)
        + 0.18 * _delta_to_t(d("ang_rp"), span, gain * 0.7)
        + 0.12 * _delta_to_t(d("lat_pinky"), span * 0.85, gain * 0.8),
        -1.0,
        1.0,
    )
    t_thumb = _clamp(
        0.45 * _delta_to_t(d("mcp_ti"), span * 1.1, gain)
        + 0.30 * _delta_to_t(d("tip_ti"), span, gain)
        + 0.25 * _delta_to_t(d("ang_ti"), span * 1.2, gain * 0.8),
        -1.0,
        1.0,
    )

    out = {
        "thumb_abduction": _spread_abduction(
            t_thumb, *limits["thumb_abduction"], open_rad=open_thumb, frac=frac
        ),
        "index_abduction": _spread_abduction(
            t_im, *limits["index_abduction"], open_rad=open_idx, frac=frac, sign=1.0
        ),
        "middle_abduction": _spread_abduction(
            t_im, *limits["middle_abduction"], open_rad=open_mid, frac=frac, sign=1.0
        ),
        "ring_abduction": _spread_abduction(
            t_ring, *limits["ring_abduction"], open_rad=open_ring, frac=frac
        ),
        "pinky_abduction": _spread_abduction(
            t_pinky, *limits["pinky_abduction"], open_rad=open_pinky, frac=frac
        ),
    }
    for name, val in out.items():
        lo, hi = limits[name]
        out[name] = _clamp(val, lo, hi)
    return out, t_im


def _thumb_chain_flex(pts: np.ndarray) -> float:
    """仅拇指自身骨骼弯曲，不参考食指"""
    return min(
        _angle(pts[WRIST], pts[THUMB_CMC], pts[THUMB_TIP]),
        _angle(pts[THUMB_CMC], pts[THUMB_MCP], pts[THUMB_IP]),
        _angle(pts[THUMB_MCP], pts[THUMB_IP], pts[THUMB_TIP]),
    )


def _reach_closedness(
    pts: np.ndarray, mcp: int, tip: int, scale: float, open_r: float, closed_r: float
) -> float:
    """指尖到 MCP 的距离比，握拳时变小"""
    reach = float(np.linalg.norm(pts[tip] - pts[mcp])) / scale
    return _clamp((open_r - reach) / max(open_r - closed_r, 0.1), 0.0, 1.0)


def _curl_from_closedness(closedness: float, mapper: CurlMapper, *, tip: bool = False) -> float:
    o = mapper.tip_open if tip else mapper.open_angle
    c = mapper.tip_closed if tip else mapper.closed_angle
    return o - closedness * (o - c)


def _finger_flex_curl(
    pts: np.ndarray,
    mcp: int,
    pip: int,
    dip: int,
    tip: int,
    scale: float,
    mapper: CurlMapper,
    reach_cfg: dict,
    *,
    for_tip: bool = False,
    finger_name: str = "",
) -> float:
    """弯曲角 + 指尖行程，取更弯的一侧（解决握拳时角度读不准）

    每根手指长度不同，统一 open_reach/closed_reach 对短手指（小指）会误判弯曲。
    可通过 reach_cfg 中 <finger>_open_reach / <finger>_closed_reach 逐指覆盖。
    """
    ang = min(
        _angle(pts[WRIST], pts[mcp], pts[pip]),
        _angle(pts[mcp], pts[pip], pts[dip]),
        _angle(pts[pip], pts[dip], pts[tip]),
    )
    # 每指独立阈值：pinky_open_reach / index_open_reach 等
    finger_key = finger_name.replace("_base", "").replace("_tip", "")
    open_r = float(reach_cfg.get(f"{finger_key}_open_reach", reach_cfg.get("open_reach", 0.90)))
    closed_r = float(reach_cfg.get(f"{finger_key}_closed_reach", reach_cfg.get("closed_reach", 0.36)))
    closedness = _reach_closedness(pts, mcp, tip, scale, open_r, closed_r)
    reach_ang = _curl_from_closedness(closedness, mapper, tip=for_tip)
    return min(ang, reach_ang)


def _thumb_signals(pts: np.ndarray, scale: float) -> Dict[str, float]:
    wrist, px, py, pz = _palm_frame(pts)

    def _palm_dir(mcp: int, tip: int) -> np.ndarray:
        v = pts[tip] - pts[mcp]
        v = v - float(np.dot(v, pz)) * pz
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            return py.copy()
        return v / n

    thumb_d = _palm_dir(THUMB_CMC, THUMB_TIP)
    index_d = _palm_dir(INDEX_MCP, INDEX_TIP)
    oppo = float(np.arccos(np.clip(float(np.dot(thumb_d, index_d)), -1.0, 1.0)))

    return {
        "flex": _thumb_chain_flex(pts),
        "tip_ang": _angle(pts[THUMB_MCP], pts[THUMB_IP], pts[THUMB_TIP]),
        "oppo_ang": oppo,
        "lat": float(np.dot(pts[THUMB_TIP] - wrist, px)) / scale,
        "reach": float(np.linalg.norm(pts[THUMB_TIP] - pts[THUMB_MCP])) / scale,
    }


def _thumb_open_rad(limits: dict) -> Dict[str, float]:
    """张开位：各通道取 lo（与 SDK 255 一致）"""
    base = float(limits["thumb_base"][0])
    return {
        "thumb_base": base,
        "thumb_tip": float(limits["thumb_tip"][0]) if "thumb_tip" in limits else base,
        "thumb_roll": float(limits["thumb_roll"][0]),
        "thumb_abduction": float(limits["thumb_abduction"][0]),
    }


def _rel_curl_rad(
    current: float,
    reference: float,
    lo: float,
    hi: float,
    *,
    span: float = 0.45,
    deadzone: float = 0.04,
) -> float:
    """仅当比校准姿态更弯时才动，平放保持 lo"""
    delta = reference - current
    if delta <= deadzone:
        return lo
    t = _clamp(delta / max(span, 0.12), 0.0, 1.0)
    return lo + t * (hi - lo)


def _compute_thumb_joints(
    pts: np.ndarray,
    scale: float,
    limits: dict,
    ref: Dict[str, float] | None,
    cfg: dict,
) -> Dict[str, float]:
    t_cfg = cfg.get("thumb", {})
    sig = _thumb_signals(pts, scale)
    open_r = _thumb_open_rad(limits)
    pitch_lo, pitch_hi = limits["thumb_base"]
    tip_lo, tip_hi = limits.get("thumb_tip", limits["thumb_base"])
    yaw_lo, yaw_hi = limits["thumb_roll"]
    abd_lo, abd_hi = limits["thumb_abduction"]

    flex_dz = float(t_cfg.get("flex_deadzone", 0.04))
    flex_span = float(t_cfg.get("flex_span", 0.42))
    tip_span = float(t_cfg.get("tip_span", 0.38))
    yaw_span = float(t_cfg.get("yaw_span", 0.28))
    abd_span = float(t_cfg.get("abd_span", 0.06))
    abd_gain = float(t_cfg.get("abd_gain", 1.1))
    lat_dz = float(t_cfg.get("lat_deadzone", 0.035))
    oppo_dz = float(t_cfg.get("oppo_deadzone", 0.10))

    if ref is None:
        return dict(open_r)

    lat_delta = sig["lat"] - float(ref["lat"])
    oppo_delta = float(ref["oppo_ang"]) - sig["oppo_ang"]

    # 与校准姿态接近 → 完全张开（解决平放时自动弯/往前伸）
    if (
        abs(sig["flex"] - ref["flex"]) <= flex_dz
        and abs(sig["tip_ang"] - ref["tip_ang"]) <= flex_dz
        and abs(lat_delta) <= lat_dz
        and abs(sig["oppo_ang"] - ref["oppo_ang"]) <= oppo_dz
    ):
        return dict(open_r)

    pitch = _rel_curl_rad(
        sig["flex"], ref["flex"], pitch_lo, pitch_hi, span=flex_span, deadzone=flex_dz
    )
    tip = _rel_curl_rad(
        sig["tip_ang"], ref["tip_ang"], tip_lo, tip_hi, span=tip_span, deadzone=flex_dz
    )

    # 拇横摆：只有拇指向食指向掌靠拢时才增大（对掌），侧伸不动
    if oppo_delta <= 0.08:
        yaw = yaw_lo
    else:
        t = _clamp(oppo_delta / max(yaw_span, 0.12), 0.0, 0.85)
        yaw = yaw_lo + t * (yaw_hi - yaw_lo)

    # 拇侧摆：相对校准 lateral，小变化保持 abd_lo（与其他指同平面）
    if abs(lat_delta) <= lat_dz:
        abd = abd_lo
    else:
        abd_t = _delta_to_t(lat_delta, abd_span, abd_gain)
        abd = _abduction_from_t(abd_t, abd_lo, abd_hi, center=abd_lo, frac=0.55)

    return {
        "thumb_base": _clamp(pitch, pitch_lo, pitch_hi),
        "thumb_tip": _clamp(tip, tip_lo, tip_hi),
        "thumb_roll": _clamp(yaw, yaw_lo, yaw_hi),
        "thumb_abduction": _clamp(abd, abd_lo, abd_hi),
    }


def _fist_strength(pts: np.ndarray, scale: float, reach_cfg: dict) -> float:
    """仅四指，不含拇指（避免四指握拳时误收拇指）"""
    default_open_r = float(reach_cfg.get("open_reach", 0.90))
    default_closed_r = float(reach_cfg.get("closed_reach", 0.36))
    fingers: list[tuple[int, int, str]] = [
        (INDEX_MCP, INDEX_TIP, "index"),
        (MIDDLE_MCP, MIDDLE_TIP, "middle"),
        (RING_MCP, RING_TIP, "ring"),
        (PINKY_MCP, PINKY_TIP, "pinky"),
    ]
    cs = [
        _reach_closedness(
            pts, m, t, scale,
            float(reach_cfg.get(f"{finger}_open_reach", default_open_r)),
            float(reach_cfg.get(f"{finger}_closed_reach", default_closed_r)),
        )
        for m, t, finger in fingers
    ]
    return _clamp(float(np.mean(cs)), 0.0, 1.0)


_THUMB_FIST_SKIP = frozenset({"thumb_base", "thumb_tip"})
_FINGER_BASE_REACH = {
    "index_base": (INDEX_MCP, INDEX_TIP),
    "middle_base": (MIDDLE_MCP, MIDDLE_TIP),
    "ring_base": (RING_MCP, RING_TIP),
    "pinky_base": (PINKY_MCP, PINKY_TIP),
}


def _apply_fist_boost(
    raw: np.ndarray,
    limits: dict,
    names: List[str],
    pts: np.ndarray,
    scale: float,
    reach_cfg: dict,
    gain: float,
    pinch: float = 0.0,
    *,
    fist_min: float = 0.32,
    aggregate_fist: float = 0.0,
) -> None:
    """对弯曲的手指加力；整体握拳时 (aggregate_fist>0.4) 对跟不上的手指也补一小段"""
    if pinch > 0.15:
        return
    default_open_r = float(reach_cfg.get("open_reach", 0.90))
    default_closed_r = float(reach_cfg.get("closed_reach", 0.36))
    for idx, name in enumerate(names):
        if name in _THUMB_FIST_SKIP or name not in _FINGER_BASE_REACH:
            continue
        mcp, tip = _FINGER_BASE_REACH[name]
        # 每指独立阈值
        finger_key = name.replace("_base", "").replace("_tip", "")
        open_r = float(reach_cfg.get(f"{finger_key}_open_reach", default_open_r))
        closed_r = float(reach_cfg.get(f"{finger_key}_closed_reach", default_closed_r))
        per_finger = _reach_closedness(pts, mcp, tip, scale, open_r, closed_r)
        # 单个手指弯了→加力；整体握拳时→用 aggregate_fist 保底，避免单指追踪丢失
        effective_fist = per_finger if per_finger >= fist_min else (
            aggregate_fist if aggregate_fist >= 0.40 else 0.0
        )
        if effective_fist < fist_min:
            continue
        lo, hi = limits[name]
        if hi <= lo + 1e-6:
            continue
        target = lo + effective_fist * gain * (hi - lo)
        raw[idx] = float(raw[idx]) + effective_fist * (target - float(raw[idx]))


def _finger_bent_ratio(flex_angle: float, open_ref: float = 2.4, closed_ref: float = 1.0) -> float:
    return _clamp((open_ref - flex_angle) / max(open_ref - closed_ref, 0.3), 0.0, 1.0)


def _intentional_pinch(
    pts: np.ndarray,
    scale: float,
    min_strength: float = 0.18,
    cfg: dict | None = None,
) -> float:
    """OK/捏合：拇食指尖足够近即触发（2D 投影优先）"""
    cfg = cfg or {}
    p_cfg = cfg.get("pinch", {})
    close = float(cfg.get("pinch_close", p_cfg.get("close", 0.04)))
    far = float(cfg.get("pinch_far", p_cfg.get("far", 0.12)))
    pinch = _pinch_strength(pts, THUMB_TIP, INDEX_TIP, scale, close=close, far=far)
    d2 = float(np.linalg.norm(pts[THUMB_TIP, :2] - pts[INDEX_TIP, :2])) / scale
    if d2 <= close * 1.6:
        pinch = max(pinch, 0.88)
    elif d2 <= far * 0.55:
        pinch = max(pinch, 0.62)
    if pinch < min_strength:
        return 0.0
    return pinch


def _apply_pinch_pose_l10(raw: np.ndarray, limits: dict, pinch: float) -> None:
    """L10 OK/捏合：拇根、拇摆、拇横 + 食根"""
    if pinch <= 0.05:
        return
    p = _clamp(pinch, 0.0, 1.0)
    for name, idx, gain in (
        ("thumb_base", 0, 0.88),
        ("index_base", 2, 0.72),
    ):
        lo, hi = limits[name]
        raw[idx] = lo + p * gain * (hi - lo)
    lo, hi = limits["thumb_abduction"]
    raw[1] = lo + p * (0.45 + 0.45 * p) * (hi - lo)
    lo, hi = limits["thumb_roll"]
    raw[9] = lo + p * 0.85 * (hi - lo)


def _spread_t_deadzone(t: float, dz: float) -> float:
    if abs(t) <= dz:
        return 0.0
    return _clamp(t, -1.0, 1.0)


def _compute_l10_spreads(
    pts: np.ndarray,
    scale: float,
    limits: dict,
    cfg: dict,
    ref: Dict[str, float] | None,
    open_pose_rad: np.ndarray | None = None,
) -> tuple[Dict[str, float], float]:
    """L10 食/无/小侧摆；张开位用 SDK 官方 open_palm pose"""
    s_cfg = cfg.get("spread", {})
    span = float(s_cfg.get("span", 0.07))
    gain = float(s_cfg.get("gain", 2.0))
    frac = float(s_cfg.get("range_frac", 0.88))
    t_dz = float(s_cfg.get("t_deadzone", 0.28))
    pinky_dz = float(s_cfg.get("pinky_t_deadzone", 0.38))
    sig = _spread_signals(pts, scale)
    if open_pose_rad is not None and len(open_pose_rad) >= 9:
        open_idx = float(open_pose_rad[6])
        open_ring = float(open_pose_rad[7])
        open_pinky = float(open_pose_rad[8])
    else:
        open_idx = float(limits["index_abduction"][1])
        open_ring = float(limits["ring_abduction"][0])
        open_pinky = float(limits["pinky_abduction"][0])

    if ref is None:
        return {
            "index_abduction": open_idx,
            "ring_abduction": open_ring,
            "pinky_abduction": open_pinky,
        }, 0.0

    def d(key: str) -> float:
        return float(sig[key]) - float(ref[key])

    t_im = _clamp(
        0.52 * _delta_to_t(d("mcp_im"), span, gain)
        + 0.38 * _delta_to_t(d("tip_im"), span, gain * 1.05)
        + 0.10 * _delta_to_t(d("lat_index"), span * 0.85, gain * 0.9),
        -1.0,
        1.0,
    )
    t_ring = _clamp(
        0.55 * _delta_to_t(d("ang_mr"), span, gain * 0.85)
        + 0.45 * _delta_to_t(d("tip_mr"), span * 0.9, gain * 0.75),
        -1.0,
        1.0,
    )
    t_pinky = _clamp(
        0.70 * _delta_to_t(d("lat_pinky"), span, gain)
        + 0.30 * _delta_to_t(d("ang_rp"), span, gain * 0.65),
        -1.0,
        1.0,
    )
    t_im = _spread_t_deadzone(t_im, t_dz)
    t_ring = _spread_t_deadzone(t_ring, t_dz)
    t_pinky = _spread_t_deadzone(t_pinky, pinky_dz)
    out = {
        "index_abduction": _spread_abduction(
            t_im, *limits["index_abduction"], open_rad=open_idx, open_at_lo=False, frac=frac
        ),
        "ring_abduction": _spread_abduction(
            t_ring, *limits["ring_abduction"], open_rad=open_ring, frac=frac
        ),
        "pinky_abduction": _spread_abduction(
            t_pinky, *limits["pinky_abduction"], open_rad=open_pinky, frac=frac
        ),
    }
    for name, val in out.items():
        lo, hi = limits[name]
        out[name] = _clamp(val, lo, hi)
    return out, t_im


def _apply_pinch_pose(raw: np.ndarray, limits: dict, pinch: float) -> None:
    """OK/捏合：直接写对掌姿态，不再与张开逻辑混叠"""
    if pinch <= 0.05:
        return
    p = _clamp(pinch, 0.0, 1.0)

    for name, idx, gain in (
        ("thumb_base", 0, 0.92),
        ("thumb_tip", 15, 0.96),
        ("index_base", 1, 0.72),
        ("index_tip", 16, 0.94),
    ):
        lo, hi = limits[name]
        raw[idx] = lo + p * gain * (hi - lo)

    lo, hi = limits["thumb_roll"]
    raw[10] = lo + p * 0.96 * (hi - lo)

    lo, hi = limits["thumb_abduction"]
    raw[5] = lo + p * (0.52 + 0.42 * p) * (hi - lo)

    abd_lo, abd_hi = limits["index_abduction"]
    raw[6] = abd_lo + (1.0 - p) * 0.35 * (abd_hi - abd_lo)


class BaseRetargeter:
    def __init__(
        self,
        profile: HandProfile,
        retarget_cfg: dict | None = None,
        curl_mapper: CurlMapper | None = None,
    ):
        cfg = retarget_cfg or {}
        self.profile = profile
        self.limits = profile.joint_limits
        self.names = profile.joint_names
        self.curl_mapper = curl_mapper or _default_mapper
        self._landmark_alpha = float(cfg.get("landmark_smooth", 0.28))
        self._landmark_alpha_fast = float(cfg.get("landmark_smooth_fast", 0.58))
        self._landmark_idle = float(cfg.get("landmark_idle_motion", 0.012))
        self._landmark_motion_ema: float = 0.0
        self._landmark_prev: np.ndarray | None = None
        self._joint_filter = JointFilterBank(profile.dof, profile.joint_limits, profile.joint_names, cfg)

    def reset(self) -> None:
        self._landmark_prev = None
        self._landmark_motion_ema = 0.0
        self._joint_filter.reset()

    def _pts(self, tr: HandTracking) -> np.ndarray | None:
        pts = _raw_pts(tr)
        if pts is None:
            return None
        if self._landmark_alpha <= 0.0:
            return pts
        if self._landmark_prev is None:
            self._landmark_prev = pts.copy()
            return pts
        motion = float(np.mean(np.linalg.norm(pts - self._landmark_prev, axis=1)))
        self._landmark_motion_ema = 0.12 * motion + 0.88 * self._landmark_motion_ema
        if self._landmark_motion_ema < self._landmark_idle:
            alpha = self._landmark_alpha
        else:
            t = min((self._landmark_motion_ema - self._landmark_idle) / 0.055, 1.0)
            alpha = self._landmark_alpha + t * (self._landmark_alpha_fast - self._landmark_alpha)
        out = alpha * pts + (1.0 - alpha) * self._landmark_prev
        self._landmark_prev = out.copy()
        return out

    def _hold_joints(self) -> HandJoints | None:
        prev = self._joint_filter.last()
        if prev is None:
            return None
        return HandJoints(values=prev, profile=self.profile)

    def _smooth(self, raw: np.ndarray) -> HandJoints:
        out = self._joint_filter.apply(raw)
        return HandJoints(values=out, profile=self.profile)

    def retarget(self, tracking: HandTracking) -> HandJoints | None:
        raise NotImplementedError


class CompactRetargeter(BaseRetargeter):
    """O6 / L7 简化映射"""

    def retarget(self, tracking: HandTracking) -> HandJoints | None:
        pts = self._pts(tracking)
        if pts is None:
            return self._hold_joints()
        raw = np.zeros(self.profile.dof)
        m = self.profile.model
        limits = self.limits

        thumb_pitch = _angle(pts[THUMB_CMC], pts[THUMB_MCP], pts[THUMB_IP])
        thumb_yaw = _angle(pts[INDEX_MCP], pts[WRIST], pts[THUMB_CMC])
        curls = [
            _angle(pts[WRIST], pts[INDEX_MCP], pts[INDEX_PIP]),
            _angle(pts[WRIST], pts[MIDDLE_MCP], pts[MIDDLE_PIP]),
            _angle(pts[WRIST], pts[RING_MCP], pts[RING_PIP]),
            _angle(pts[WRIST], pts[PINKY_MCP], pts[PINKY_PIP]),
        ]

        raw[0] = _map_curl(thumb_pitch, *limits["thumb_pitch"], self.curl_mapper)
        raw[1] = _clamp(
            limits["thumb_yaw"][0] + (thumb_yaw / np.pi) * (limits["thumb_yaw"][1] - limits["thumb_yaw"][0]),
            *limits["thumb_yaw"],
        )
        for i, key in enumerate(["index_pitch", "middle_pitch", "ring_pitch", "pinky_pitch"]):
            raw[2 + i] = _map_curl(curls[i], *limits[key], self.curl_mapper)
        if m == "L7" and self.profile.dof >= 7:
            raw[6] = raw[1] * 0.5
        return self._smooth(raw)


class L20Retargeter(BaseRetargeter):
    def __init__(
        self,
        profile: HandProfile,
        retarget_cfg: dict | None = None,
        curl_mapper: CurlMapper | None = None,
    ):
        super().__init__(profile, retarget_cfg, curl_mapper)
        cfg = retarget_cfg or {}
        self._hand_side = "left"
        self._pinch_min = float(cfg.get("pinch_min", 0.18))
        self._pinch_hysteresis = float(cfg.get("pinch_hysteresis", 0.06))
        self._pinch_blend = float(cfg.get("pinch_smooth", 0.38))
        self._pinch_ema = 0.0
        self._pinch_active = False
        self.last_pinch_raw = 0.0
        self.last_pinch = 0.0
        self.last_spread_im = 0.0
        self._retarget_cfg = cfg
        self._spread_ref: Dict[str, float] | None = None
        self._spread_acc: Dict[str, list[float]] | None = None
        self._thumb_ref: Dict[str, float] | None = None
        self._thumb_acc: Dict[str, list[float]] | None = None

    def reset(self) -> None:
        super().reset()
        self._pinch_ema = 0.0
        self._pinch_active = False
        self.last_pinch_raw = 0.0
        self.last_pinch = 0.0
        self.last_spread_im = 0.0

    def accumulate_spread_sample(self, tracking: HandTracking) -> None:
        pts = self._pts(tracking)
        if pts is None:
            return
        pts = _normalize_hand(pts)
        sig = _spread_signals(pts, _palm_scale(pts))
        if self._spread_acc is None:
            self._spread_acc = {k: [] for k in sig}
        for k, v in sig.items():
            self._spread_acc[k].append(float(v))
        thumb_sig = _thumb_signals(pts, _palm_scale(pts))
        if self._thumb_acc is None:
            self._thumb_acc = {k: [] for k in thumb_sig}
        for k, v in thumb_sig.items():
            self._thumb_acc[k].append(float(v))

    def finalize_spread_calibration(self) -> bool:
        if not self._spread_acc:
            return False
        keys = next(iter(self._spread_acc.keys()), None)
        if keys is None or not self._spread_acc[keys]:
            self._spread_acc = None
            self._thumb_acc = None
            return False
        self._spread_ref = {k: float(np.mean(v)) for k, v in self._spread_acc.items()}
        n = len(self._spread_acc[keys])
        self._spread_acc = None
        if self._thumb_acc and self._thumb_acc.get("flex"):
            self._thumb_ref = {k: float(np.mean(v)) for k, v in self._thumb_acc.items()}
            print(
                f"[Retarget] 侧摆已校准 ({n}帧) mcp_im={self._spread_ref['mcp_im']:.3f} | "
                f"拇指 flex={self._thumb_ref['flex']:.2f} oppo={self._thumb_ref['oppo_ang']:.2f}"
            )
        else:
            print(f"[Retarget] 侧摆已校准 ({n}帧) mcp_im={self._spread_ref['mcp_im']:.3f}")
        self._thumb_acc = None
        self._seed_filter_neutral(self._hand_side)
        return True

    def _seed_filter_neutral(self, side: str = "left") -> None:
        """校准后把滤波器状态重置到张开位，避免旧状态残留"""
        raw = sdk_open_rad(
            self.profile.model, side, self.profile.dof, self.profile.open_pose
        )
        self._joint_filter.reset()
        self._joint_filter.apply(raw)

    def calibrate_spread(self, tracking: HandTracking) -> None:
        self.accumulate_spread_sample(tracking)
        self.finalize_spread_calibration()

    def clear_spread_accumulator(self) -> None:
        self._spread_acc = None
        self._thumb_acc = None

    def _smooth_pinch(self, pinch: float) -> float:
        blend = self._pinch_blend if pinch < 0.35 else min(0.72, self._pinch_blend + 0.28)
        self._pinch_ema = blend * pinch + (1.0 - blend) * self._pinch_ema
        on = self._pinch_min
        off = max(0.04, self._pinch_min - self._pinch_hysteresis)

        if pinch >= 0.32 or self._pinch_ema >= on:
            self._pinch_active = True
        if self._pinch_active and pinch <= off and self._pinch_ema <= off:
            self._pinch_active = False
            return 0.0
        if self._pinch_active:
            return max(self._pinch_ema, pinch * 0.92) if pinch >= on * 0.55 else self._pinch_ema
        return 0.0

    def _finger_flex(self, pts: np.ndarray, mcp: int, pip: int, dip: int, tip: int, finger_name: str = "") -> float:
        return _finger_flex_curl(
            pts, mcp, pip, dip, tip, _palm_scale(pts), self.curl_mapper,
            self._retarget_cfg.get("reach", {}), finger_name=finger_name,
        )

    def _apply_pinch(
        self,
        raw: np.ndarray,
        limits: dict,
        pinch: float,
    ) -> None:
        _apply_pinch_pose(raw, limits, pinch)

    def retarget(self, tracking: HandTracking) -> HandJoints | None:
        pts = self._pts(tracking)
        if pts is None:
            return self._hold_joints()
        pts = _normalize_hand(pts)
        limits = self.limits
        raw = np.zeros(20)
        cm = self.curl_mapper
        scale = _palm_scale(pts)

        pinch_raw = _intentional_pinch(pts, scale, self._pinch_min, self._retarget_cfg)
        pinch = self._smooth_pinch(pinch_raw)
        self.last_pinch_raw = pinch_raw
        self.last_pinch = pinch

        if pinch > 0.12:
            open_r = _thumb_open_rad(limits)
            raw[0] = open_r["thumb_base"]
            raw[5] = open_r["thumb_abduction"]
            raw[10] = open_r["thumb_roll"]
            raw[15] = open_r["thumb_tip"]
        else:
            thumb = _compute_thumb_joints(pts, scale, limits, self._thumb_ref, self._retarget_cfg)
            raw[0] = thumb["thumb_base"]
            raw[5] = thumb["thumb_abduction"]
            raw[10] = thumb["thumb_roll"]
            raw[15] = thumb["thumb_tip"]

        fingers = [
            ("index_base", INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
            ("middle_base", MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
            ("ring_base", RING_MCP, RING_PIP, RING_DIP, RING_TIP),
            ("pinky_base", PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
        ]
        for i, (name, mcp, pip, dip, tip) in enumerate(fingers, start=1):
            raw[i] = _map_curl(self._finger_flex(pts, mcp, pip, dip, tip, finger_name=name), *limits[name], cm)

        spreads, spread_im = _compute_finger_spreads(
            pts, scale, limits, self._retarget_cfg, self._spread_ref
        )
        self.last_spread_im = spread_im
        raw[6] = spreads["index_abduction"]
        raw[7] = spreads["middle_abduction"]
        raw[8] = spreads["ring_abduction"]
        raw[9] = spreads["pinky_abduction"]

        tips = [
            ("index_tip", _finger_flex_curl(pts, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP, scale, cm, self._retarget_cfg.get("reach", {}), for_tip=True, finger_name="index")),
            ("middle_tip", _finger_flex_curl(pts, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP, scale, cm, self._retarget_cfg.get("reach", {}), for_tip=True, finger_name="middle")),
            ("ring_tip", _finger_flex_curl(pts, RING_MCP, RING_PIP, RING_DIP, RING_TIP, scale, cm, self._retarget_cfg.get("reach", {}), for_tip=True, finger_name="ring")),
            ("pinky_tip", _finger_flex_curl(pts, PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP, scale, cm, self._retarget_cfg.get("reach", {}), for_tip=True, finger_name="pinky")),
        ]
        for i, (n, c) in enumerate(tips):
            raw[16 + i] = _map_curl(c, *limits[n], cm, tip=True)

        reach_cfg = self._retarget_cfg.get("reach", {})
        aggregate_fist = _fist_strength(pts, scale, reach_cfg)
        _apply_fist_boost(
            raw,
            limits,
            self.names,
            pts,
            scale,
            reach_cfg,
            float(reach_cfg.get("fist_boost", 0.92)),
            pinch=pinch,
            aggregate_fist=aggregate_fist,
        )

        self._apply_pinch(raw, limits, pinch)

        for i, name in enumerate(self.names):
            lo, hi = limits[name]
            raw[i] = _clamp(float(raw[i]), float(lo), float(hi))

        return self._smooth(raw)


class L10Retargeter(L20Retargeter):
    """L10：10 通道，与 SDK / MuJoCo L10 模型一致"""

    def _apply_pinch(self, raw: np.ndarray, limits: dict, pinch: float) -> None:
        _apply_pinch_pose_l10(raw, limits, pinch)

    def _l10_open_rad(self) -> np.ndarray:
        return sdk_open_rad(
            self.profile.model,
            self._hand_side,
            self.profile.dof,
            self.profile.open_pose,
        )

    def retarget(self, tracking: HandTracking) -> HandJoints | None:
        pts = self._pts(tracking)
        if pts is None:
            return self._hold_joints()
        pts = _normalize_hand(pts)
        limits = self.limits
        raw = np.zeros(10)
        cm = self.curl_mapper
        scale = _palm_scale(pts)
        open_rad = self._l10_open_rad()
        reach_cfg = self._retarget_cfg.get("reach", {})
        fist = _fist_strength(pts, scale, reach_cfg)
        open_hand = fist < 0.20

        pinch_raw = _intentional_pinch(pts, scale, self._pinch_min, self._retarget_cfg)
        pinch = self._smooth_pinch(pinch_raw)
        self.last_pinch_raw = pinch_raw
        self.last_pinch = pinch

        if pinch > 0.12:
            raw[0] = float(open_rad[0])
            raw[1] = float(open_rad[1])
            raw[9] = float(open_rad[9])
        else:
            thumb = _compute_thumb_joints(pts, scale, limits, self._thumb_ref, self._retarget_cfg)
            raw[0] = thumb["thumb_base"]
            raw[1] = thumb["thumb_abduction"]
            raw[9] = thumb["thumb_roll"]
            if open_hand:
                raw[0] = float(open_rad[0])
                raw[1] = float(open_rad[1])
                raw[9] = float(open_rad[9])

        fingers = [
            ("index_base", INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
            ("middle_base", MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
            ("ring_base", RING_MCP, RING_PIP, RING_DIP, RING_TIP),
            ("pinky_base", PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
        ]
        for i, (name, mcp, pip, dip, tip) in enumerate(fingers, start=2):
            raw[i] = _map_curl(self._finger_flex(pts, mcp, pip, dip, tip, finger_name=name), *limits[name], cm)

        spreads, spread_im = _compute_l10_spreads(
            pts, scale, limits, self._retarget_cfg, self._spread_ref, open_rad
        )
        self.last_spread_im = spread_im
        raw[6] = spreads["index_abduction"]
        raw[7] = spreads["ring_abduction"]
        raw[8] = spreads["pinky_abduction"]
        if open_hand and pinch <= 0.12 and abs(spread_im) < 0.35:
            raw[6] = float(open_rad[6])
            raw[7] = float(open_rad[7])
            raw[8] = float(open_rad[8])

        _apply_fist_boost(
            raw,
            limits,
            self.names,
            pts,
            scale,
            reach_cfg,
            float(reach_cfg.get("fist_boost", 0.92)),
            pinch=pinch,
            aggregate_fist=fist,
        )

        self._apply_pinch(raw, limits, pinch)

        for i, name in enumerate(self.names):
            lo, hi = limits[name]
            raw[i] = _clamp(float(raw[i]), float(lo), float(hi))

        return self._smooth(raw)


def _build_curl_mapper(cfg: dict) -> CurlMapper:
    return CurlMapper(
        open_angle=float(cfg.get("curl_open", 2.45)),
        closed_angle=float(cfg.get("curl_closed", 0.85)),
        gain=float(cfg.get("curl_gain", 1.4)),
        tip_open=float(cfg.get("tip_open", 2.5)),
        tip_closed=float(cfg.get("tip_closed", 0.55)),
        tip_gain=float(cfg.get("tip_gain", 1.5)),
    )


def _has_spread_calibration(rt: object) -> bool:
    return hasattr(rt, "accumulate_spread_sample") and hasattr(rt, "finalize_spread_calibration")


def create_retargeter(
    profile: HandProfile,
    retarget_cfg: dict | None = None,
    side: str = "left",
) -> BaseRetargeter:
    cfg = retarget_cfg or {}
    mapper = _build_curl_mapper(cfg)
    if profile.model == "L20":
        rt = L20Retargeter(profile, cfg, mapper)
    elif profile.model == "L10":
        rt = L10Retargeter(profile, cfg, mapper)
    else:
        return CompactRetargeter(profile, cfg, mapper)
    rt._hand_side = side
    return rt
