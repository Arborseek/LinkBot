from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

# 来自 linkerhand-python-sdk/LinkerHand/utils/mapping.py（左手 L10/L20）
L10_L_MIN = [0, 0, 0, 0, 0, 0, 0, -0.26, -0.26, -0.52]
L10_L_MAX = [1.45, 1.43, 1.62, 1.62, 1.62, 1.62, 0.26, 0, 0, 1.01]
L10_L_DIR = [-1, -1, -1, -1, -1, -1, 0, -1, -1, -1]

L10_R_MIN = [0, 0, 0, 0, 0, 0, -0.26, 0, 0, -0.52]
L10_R_MAX = [0.75, 1.43, 1.62, 1.62, 1.62, 1.62, 0.21, 0.21, 0.34, 1.01]
L10_R_DIR = [-1, -1, -1, -1, -1, -1, 0, 0, 0, -1]

L20_L_MIN = [0, 0, 0, 0, 0, -0.297, -0.26, -0.26, -0.26, -0.26, 0.122, 0, 0, 0, 0, 0, 0, 0, 0, 0]
L20_L_MAX = [0.87, 1.4, 1.4, 1.4, 1.4, 0.683, 0.26, 0.26, 0.26, 0.26, 1.78, 0, 0, 0, 0, 1.29, 1.08, 1.08, 1.08, 1.08]
L20_L_DIR = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1]

L20_R_MIN = [0, 0, 0, 0, 0, -0.297, -0.26, -0.26, -0.26, -0.26, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
L20_R_MAX = [0.87, 1.4, 1.4, 1.4, 1.4, 0.683, 0.26, 0.26, 0.26, 0.26, 1.78, 0, 0, 0, 0, 1.29, 1.08, 1.08, 1.08, 1.08]
L20_R_DIR = [-1, -1, -1, -1, -1, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1]

# MuJoCo L10 模型执行器：open_at_lo=False 表示 0(伸直) 在 ctrl 上界
MUJOCO_L10_OPEN_AT_LO = [
    True, True, False, False, False,
    True, True, True, True,
    True, True, True,
    True, True, True, True,
    True, True, True, True,
]


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, v)))


def _scale(v: float, a0: float, a1: float, b0: float, b1: float) -> float:
    if abs(a1 - a0) < 1e-9:
        return b0
    return (v - a0) * (b1 - b0) / (a1 - a0) + b0


def _tables(hand_model: str, side: str) -> Tuple[List[float], List[float], List[int], int]:
    side = side.lower()
    if hand_model == "L20":
        n = 20
        if side == "right":
            return L20_R_MIN, L20_R_MAX, L20_R_DIR, n
        return L20_L_MIN, L20_L_MAX, L20_L_DIR, n
    if hand_model == "L10":
        n = 10
        if side == "right":
            return L10_R_MIN, L10_R_MAX, L10_R_DIR, n
        return L10_L_MIN, L10_L_MAX, L10_L_DIR, n
    n = min(20, len(L20_L_MIN))
    return L20_L_MIN, L20_L_MAX, L20_L_DIR, n


def sdk_pose_to_rad(pose: Sequence[int], hand_model: str, side: str) -> np.ndarray:
    """SDK pose 0~255 -> 各通道 rad（joints_to_sdk_pose 的逆）"""
    j_min, j_max, j_dir, n = _tables(hand_model, side)
    raw = np.zeros(n, dtype=np.float64)
    for i in range(n):
        val = _clamp(float(pose[i]) if i < len(pose) else 255.0, 0.0, 255.0)
        if j_dir[i] == -1:
            raw[i] = _scale(val, 0.0, 255.0, j_max[i], j_min[i])
        else:
            raw[i] = _scale(val, 0.0, 255.0, j_min[i], j_max[i])
    return raw


def sdk_open_rad(
    hand_model: str,
    side: str,
    dof: int,
    open_pose: Sequence[int] | None = None,
) -> np.ndarray:
    """张开位各通道 rad；若提供 open_pose 则按 SDK 预设 pose 换算（L10 拇摆≠255）"""
    j_min, j_max, j_dir, n = _tables(hand_model, side)
    if open_pose is not None and len(open_pose) >= min(dof, n):
        raw = sdk_pose_to_rad(open_pose[:n], hand_model, side)
        if dof > n:
            return np.resize(raw, dof)
        return raw[:dof]
    raw = np.zeros(dof, dtype=np.float64)
    for i in range(min(dof, n)):
        raw[i] = j_min[i] if j_dir[i] == -1 else j_max[i]
    return raw


def sdk_pose_to_closedness(pose: Sequence[int], hand_model: str, side: str) -> np.ndarray:
    """SDK 0~255 (255=张开) -> 各通道弯曲度 [0=张开, 1=握紧]"""
    j_min, j_max, j_dir, n = _tables(hand_model, side)
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        val = _clamp(float(pose[i]) if i < len(pose) else 255.0, 0.0, 255.0)
        if hand_model in ("L20", "L21") and 11 <= i <= 14:
            continue
        if j_dir[i] == -1:
            rad = _scale(val, 0.0, 255.0, j_max[i], j_min[i])
            out[i] = _clamp((rad - j_min[i]) / max(j_max[i] - j_min[i], 1e-6), 0.0, 1.0)
        else:
            rad = _scale(val, 0.0, 255.0, j_min[i], j_max[i])
            out[i] = _clamp((rad - j_min[i]) / max(j_max[i] - j_min[i], 1e-6), 0.0, 1.0)
    return out


def joints_to_sdk_pose(joints, side: str) -> List[int]:
    """HandJoints(rad) -> SDK finger_move pose，与真机 finger_move 一致"""
    hand_model = joints.profile.model
    j_min, j_max, j_dir, n = _tables(hand_model, side)
    pose: List[int] = []
    for i in range(n):
        if i < len(joints.values):
            rad = _clamp(float(joints.values[i]), j_min[i], j_max[i])
        else:
            rad = j_min[i]
        if hand_model in ("L20", "L21") and 11 <= i <= 14:
            pose.append(255)
            continue
        if j_dir[i] == -1:
            p = _scale(rad, j_min[i], j_max[i], 255.0, 0.0)
        else:
            p = _scale(rad, j_min[i], j_max[i], 0.0, 255.0)
        pose.append(int(_clamp(round(p), 0.0, 255.0)))
    return pose


def _blend(t: np.ndarray, idx: int, default: float = 0.0) -> float:
    if idx < 0 or idx >= len(t):
        return default
    return float(t[idx])


def _l20_to_mujoco_closedness(t: np.ndarray) -> np.ndarray:
    """L20 SDK 通道 -> MuJoCo L10 模型 20 个执行器弯曲度"""
    m = np.zeros(20, dtype=np.float64)
    # 拇指 thumb_joint0..4
    m[0] = _blend(t, 10)
    m[1] = _blend(t, 5)
    m[2] = _blend(t, 0) * 0.55
    m[3] = _blend(t, 0) * 0.35 + _blend(t, 15) * 0.65
    m[4] = _blend(t, 15)
    # 食指
    m[5] = _blend(t, 6)
    m[6] = _blend(t, 1)
    m[7] = _blend(t, 1) * 0.45 + _blend(t, 16) * 0.55
    m[8] = _blend(t, 16)
    # 中指（3 关节）
    m[9] = _blend(t, 2) * 0.75 + _blend(t, 7) * 0.25
    m[10] = _blend(t, 2)
    m[11] = _blend(t, 2) * 0.35 + _blend(t, 17) * 0.65
    # 无名指
    m[12] = _blend(t, 8)
    m[13] = _blend(t, 3)
    m[14] = _blend(t, 3) * 0.45 + _blend(t, 18) * 0.55
    m[15] = _blend(t, 18)
    # 小指
    m[16] = _blend(t, 9)
    m[17] = _blend(t, 4)
    m[18] = _blend(t, 4) * 0.45 + _blend(t, 19) * 0.55
    m[19] = _blend(t, 19)
    return m


def _l10_to_mujoco_closedness(t: np.ndarray) -> np.ndarray:
    """L10 SDK 通道 -> MuJoCo L10 模型 20 个执行器弯曲度"""
    m = np.zeros(20, dtype=np.float64)
    m[0] = _blend(t, 9)
    m[1] = _blend(t, 1)
    m[2] = _blend(t, 0) * 0.55
    m[3] = _blend(t, 0) * 0.85
    m[4] = _blend(t, 0)
    m[5] = _blend(t, 6)
    m[6] = _blend(t, 2)
    m[7] = _blend(t, 2) * 0.65
    m[8] = _blend(t, 2)
    m[9] = _blend(t, 3)
    m[10] = _blend(t, 3)
    m[11] = _blend(t, 3) * 0.65
    m[12] = _blend(t, 7)
    m[13] = _blend(t, 4)
    m[14] = _blend(t, 4) * 0.65
    m[15] = _blend(t, 4)
    m[16] = _blend(t, 8)
    m[17] = _blend(t, 5)
    m[18] = _blend(t, 5) * 0.65
    m[19] = _blend(t, 5)
    return m


def _o6_to_mujoco_closedness(t: np.ndarray) -> np.ndarray:
    """O6/L7 简化映射"""
    m = np.zeros(20, dtype=np.float64)
    thumb = _blend(t, 0)
    yaw = _blend(t, 1)
    m[0] = yaw
    m[1] = yaw * 0.8
    m[2] = thumb * 0.55
    m[3] = thumb * 0.85
    m[4] = thumb
    chains = [(2, [5, 6, 7, 8]), (3, [9, 10, 11]), (4, [12, 13, 14, 15]), (5, [16, 17, 18, 19])]
    for sdk_idx, acts in chains:
        curl = _blend(t, sdk_idx)
        m[acts[0]] = curl * 0.25
        for a in acts[1:-1]:
            m[a] = curl * 0.85
        m[acts[-1]] = curl
    return m


def closedness_to_ctrl(closed: np.ndarray, ctrl_ranges: np.ndarray, open_at_lo: Sequence[bool]) -> np.ndarray:
    dof = len(closed)
    ctrl = np.zeros(dof, dtype=np.float64)
    for i in range(min(dof, len(ctrl_ranges), len(open_at_lo))):
        lo, hi = ctrl_ranges[i]
        t = _clamp(float(closed[i]), 0.0, 1.0)
        if open_at_lo[i]:
            ctrl[i] = lo + t * (hi - lo)
        else:
            ctrl[i] = hi + t * (lo - hi)
    return ctrl


def pose_to_mujoco_ctrl(
    pose: Sequence[int],
    hand_model: str,
    side: str,
    ctrl_ranges: np.ndarray,
    dof: int,
) -> np.ndarray:
    """SDK pose -> MuJoCo 执行器 ctrl（与官方 mapping 语义一致）"""
    t_sdk = sdk_pose_to_closedness(pose, hand_model, side)
    if hand_model == "L20":
        closed = _l20_to_mujoco_closedness(t_sdk)
    elif hand_model == "L10":
        closed = _l10_to_mujoco_closedness(t_sdk)
    elif hand_model in ("O6", "L6", "L7"):
        n = 6 if hand_model in ("O6", "L6") else 7
        t6 = sdk_pose_to_closedness(pose[:n], "O6", side)
        if hand_model == "L7" and len(pose) >= 7:
            roll = sdk_pose_to_closedness([pose[6]], "O6", side)[0]
            t6 = np.append(t6, roll)
        else:
            t6 = np.append(t6, 0.0)
        closed = _o6_to_mujoco_closedness(t6[:6])
        if hand_model == "L7" and len(t6) > 6:
            closed[0] = max(closed[0], float(t6[6]))
    else:
        closed = _l20_to_mujoco_closedness(t_sdk)

    open_at = MUJOCO_L10_OPEN_AT_LO[:dof]
    if len(open_at) < dof:
        open_at = list(open_at) + [True] * (dof - len(open_at))
    return closedness_to_ctrl(closed[:dof], ctrl_ranges, open_at)


def open_ctrl(ctrl_ranges: np.ndarray, dof: int, hand_model: str = "L20", side: str = "left") -> np.ndarray:
    n = {"L10": 10, "L20": 20, "L7": 7, "O6": 6, "L6": 6}.get(hand_model, 20)
    return pose_to_mujoco_ctrl([255] * n, hand_model, side, ctrl_ranges, dof)


def sdk_pose_to_ctrl(pose: List[int], ctrl_ranges: np.ndarray, dof: int) -> np.ndarray:
    """兼容旧接口：无型号信息时按 L20 左手映射"""
    return pose_to_mujoco_ctrl(pose, "L20", "left", ctrl_ranges, dof)
