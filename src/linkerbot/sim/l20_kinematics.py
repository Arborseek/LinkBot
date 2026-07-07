from __future__ import annotations

from typing import Dict, List, Sequence

import mujoco
import numpy as np

from linkerbot.sim.pose_mapping import _clamp, _scale, _tables

# SDK finger_move 顺序 -> URDF 关节名（11~14 预留）
L20_SDK_JOINT: List[str | None] = [
    "thumb_cmc_pitch",   # 0 拇指根部
    "index_mcp_pitch",   # 1
    "middle_mcp_pitch",  # 2
    "ring_mcp_pitch",    # 3
    "pinky_mcp_pitch",   # 4
    "thumb_cmc_roll",    # 5 拇指侧摆
    "index_mcp_roll",    # 6
    "middle_mcp_roll",   # 7
    "ring_mcp_roll",     # 8
    "pinky_mcp_roll",    # 9
    "thumb_cmc_yaw",     # 10 拇指横摆
    None,
    None,
    None,
    None,
    "thumb_ip",          # 15 拇指尖
    "index_dip",         # 16
    "middle_dip",        # 17
    "ring_dip",          # 18
    "pinky_dip",         # 19
]

PIP_JOINTS = ("index_pip", "middle_pip", "ring_pip", "pinky_pip")
THUMB_MCP = "thumb_mcp"
# SDK 侧摆量程 ±0.26~0.683，URDF mcp_roll 仅 ±0.17，需线性映射以用满仿真可视范围
ROLL_JOINTS = frozenset(
    {"thumb_cmc_roll", "index_mcp_roll", "middle_mcp_roll", "ring_mcp_roll", "pinky_mcp_roll"}
)


def _sdk_to_rad(val: float, sdk_idx: int, side: str) -> float:
    j_min, j_max, j_dir, _ = _tables("L20", side)
    v = _clamp(val, 0.0, 255.0)
    if j_dir[sdk_idx] == -1:
        return _scale(v, 0.0, 255.0, j_max[sdk_idx], j_min[sdk_idx])
    return _scale(v, 0.0, 255.0, j_min[sdk_idx], j_max[sdk_idx])


def _joint_limits(model: mujoco.MjModel) -> Dict[str, tuple[int, float, float]]:
    out: Dict[str, tuple[int, float, float]] = {}
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if not name:
            continue
        lo, hi = model.jnt_range[jid]
        out[name] = (jid, float(lo), float(hi))
    return out


def _rad_to_closedness(rad: float, lo: float, hi: float) -> float:
    if abs(hi - lo) < 1e-9:
        return 0.0
    return _clamp((rad - lo) / (hi - lo), 0.0, 1.0)


def _closedness_to_rad(t: float, lo: float, hi: float) -> float:
    return lo + _clamp(t, 0.0, 1.0) * (hi - lo)


def _apply_pip_coupling(rads: Dict[str, float], limits: Dict[str, tuple[int, float, float]]) -> None:
    for finger in ("index", "middle", "ring", "pinky"):
        pip = f"{finger}_pip"
        mcp = f"{finger}_mcp_pitch"
        dip = f"{finger}_dip"
        if pip not in limits or mcp not in rads:
            continue
        _, lo, hi = limits[pip]
        _, mlo, mhi = limits[mcp]
        t = 0.55 * _rad_to_closedness(rads[mcp], mlo, mhi)
        if dip in rads and dip in limits:
            _, dlo, dhi = limits[dip]
            t = 0.55 * _rad_to_closedness(rads[mcp], mlo, mhi) + 0.45 * _rad_to_closedness(rads[dip], dlo, dhi)
        rads[pip] = _closedness_to_rad(t, lo, hi)

    if THUMB_MCP in limits and "thumb_cmc_pitch" in rads and "thumb_ip" in rads:
        _, lo, hi = limits[THUMB_MCP]
        _, clo, chi = limits["thumb_cmc_pitch"]
        _, ilo, ihi = limits["thumb_ip"]
        pitch_t = _rad_to_closedness(rads["thumb_cmc_pitch"], clo, chi)
        ip_t = _rad_to_closedness(rads["thumb_ip"], ilo, ihi)
        if ip_t < 0.08:
            t = pitch_t * 0.35
        else:
            t = 0.35 * pitch_t + 0.45 * ip_t
        rads[THUMB_MCP] = _closedness_to_rad(_clamp(t, 0.0, 1.0), lo, hi)


def _write_rads_to_qpos(rads: Dict[str, float], limits: Dict[str, tuple[int, float, float]], model: mujoco.MjModel) -> np.ndarray:
    qpos = np.zeros(model.nq, dtype=np.float64)
    for jname, rad in rads.items():
        if jname not in limits:
            continue
        adr = int(model.jnt_qposadr[limits[jname][0]])
        qpos[adr] = rad
    return qpos


def pose_to_l20_qpos(
    pose: Sequence[int],
    model: mujoco.MjModel,
    side: str = "left",
) -> np.ndarray:
    """L20 SDK pose -> MuJoCo qpos（官方 URDF 21 关节，PIP 随 MCP/DIP 联动）"""
    limits = _joint_limits(model)
    rads: Dict[str, float] = {}

    for sdk_i, jname in enumerate(L20_SDK_JOINT):
        if jname is None or jname not in limits:
            continue
        rad = _sdk_to_rad(float(pose[sdk_i]) if sdk_i < len(pose) else 255.0, sdk_i, side)
        _, lo, hi = limits[jname]
        if jname in ROLL_JOINTS:
            j_min, j_max, _, _ = _tables("L20", side)
            sdk_lo, sdk_hi = float(j_min[sdk_i]), float(j_max[sdk_i])
            if abs(sdk_hi - sdk_lo) > 1e-6:
                t = _clamp((rad - sdk_lo) / (sdk_hi - sdk_lo), 0.0, 1.0)
                rad = lo + t * (hi - lo)
        rads[jname] = _clamp(rad, lo, hi)

    _apply_pip_coupling(rads, limits)
    return _write_rads_to_qpos(rads, limits, model)


def is_l20_urdf_model(model: mujoco.MjModel) -> bool:
    return model.nu == 0 and model.njnt == 21
