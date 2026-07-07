from __future__ import annotations

import math
from typing import Dict, List

import numpy as np


def _alpha(cutoff: float, freq: float) -> float:
    te = 1.0 / freq
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / te)


def _lerp(prev: float, cur: float, a: float) -> float:
    return a * cur + (1.0 - a) * prev


class OneEuroFilter:
    def __init__(
        self,
        min_cutoff: float = 0.55,
        beta: float = 0.04,
        d_cutoff: float = 1.0,
        freq: float = 30.0,
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.freq = freq
        self._x: float | None = None
        self._dx: float = 0.0

    def reset(self) -> None:
        self._x = None
        self._dx = 0.0

    def sync(self, x: float) -> None:
        self._x = x

    def __call__(self, x: float) -> float:
        if self._x is None:
            self._x = x
            return x
        dx = (x - self._x) * self.freq
        a_d = _alpha(self.d_cutoff, self.freq)
        self._dx = _lerp(self._dx, dx, a_d)
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = _alpha(cutoff, self.freq)
        self._x = _lerp(self._x, x, a)
        return self._x


class JointFilterBank:
    def __init__(
        self,
        dof: int,
        limits: Dict[str, tuple[float, float]],
        names: List[str],
        cfg: dict,
    ):
        self.names = names
        self.limits = limits
        self.mode = str(cfg.get("filter", "one_euro"))
        self.ema_alpha = float(cfg.get("smoothing_alpha", 0.35))
        self.max_step = float(cfg.get("max_joint_step", 0.0))
        self.max_step_fast = float(cfg.get("max_joint_step_fast", 0.0))
        self.output_smooth = float(cfg.get("output_smooth", 0.45))
        self.deadzone = float(cfg.get("joint_deadzone", 0.012))
        self.spread_deadzone = float(cfg.get("spread_deadzone", 0.004))
        self.spread_output_smooth = float(cfg.get("spread_output_smooth", 0.12))
        self.idle_motion = float(cfg.get("idle_motion", 0.028))
        self.fast_motion = float(cfg.get("fast_motion", 0.10))
        self._spread_idx = {i for i, n in enumerate(names) if "abduction" in n}
        self._flex_idx = {i for i, n in enumerate(names) if "base" in n or "tip" in n}
        self._thumb_pose_idx = {i for i, n in enumerate(names) if n in ("thumb_roll", "thumb_abduction")}
        self.curl_output_smooth = float(cfg.get("curl_output_smooth", 0.16))
        self.curl_deadzone = float(cfg.get("curl_deadzone", 0.006))
        freq = float(cfg.get("filter_freq", 30.0))
        oe = cfg.get("one_euro", {})
        self._filters = [
            OneEuroFilter(
                min_cutoff=float(oe.get("min_cutoff", 0.55)),
                beta=float(oe.get("beta", 0.04)),
                d_cutoff=float(oe.get("d_cutoff", 1.0)),
                freq=freq,
            )
            for _ in range(dof)
        ]
        self._prev: np.ndarray | None = None
        self._motion_ema: float = 0.0

    def _step_for_motion(self, motion: float) -> float:
        if self.max_step <= 0.0:
            return float("inf")
        slow = self.max_step * 0.55
        fast = self.max_step_fast if self.max_step_fast > 0.0 else self.max_step * 2.0
        if motion <= self.idle_motion:
            return slow
        if motion >= self.fast_motion:
            return fast
        t = (motion - self.idle_motion) / max(self.fast_motion - self.idle_motion, 1e-6)
        return slow + t * (fast - slow)

    def apply(self, raw: np.ndarray) -> np.ndarray:
        if self.mode == "one_euro":
            out = np.array([f(float(raw[i])) for i, f in enumerate(self._filters)], dtype=np.float64)
        elif self._prev is None:
            out = raw.copy()
        else:
            out = self.ema_alpha * raw + (1.0 - self.ema_alpha) * self._prev

        if self._prev is None:
            for i, name in enumerate(self.names):
                lo, hi = self.limits[name]
                out[i] = float(np.clip(out[i], lo, hi))
            self._prev = out.copy()
            return out.copy()

        frame_motion = float(np.max(np.abs(raw - self._prev)))
        self._motion_ema = 0.12 * frame_motion + 0.88 * self._motion_ema
        settled = self._motion_ema < self.idle_motion
        step = self._step_for_motion(self._motion_ema)

        for i in range(len(out)):
            prev_i = float(self._prev[i])
            raw_i = float(raw[i])
            is_spread = i in self._spread_idx
            is_flex = i in self._flex_idx
            is_thumb_pose = i in self._thumb_pose_idx
            if is_spread or is_thumb_pose:
                deadzone = self.spread_deadzone
            elif is_flex:
                deadzone = self.curl_deadzone
            else:
                deadzone = self.deadzone

            spread_delta = abs(raw_i - prev_i) if is_spread else 0.0
            if settled and spread_delta < deadzone and abs(raw_i - prev_i) < deadzone:
                out[i] = prev_i
                continue

            target = float(out[i])
            if settled:
                if is_spread or is_thumb_pose:
                    smooth = self.spread_output_smooth
                    if spread_delta > 0.02:
                        smooth = min(smooth, 0.22)
                elif is_flex:
                    smooth = self.curl_output_smooth
                else:
                    smooth = self.output_smooth
                target = (1.0 - smooth) * target + smooth * prev_i

            delta = target - prev_i
            step_i = step
            if is_spread and spread_delta > 0.015:
                step_i = step * 2.8
            if is_flex and raw_i > prev_i + 0.015:
                step_i = step * 3.0
            elif raw_i > prev_i and self._motion_ema > self.fast_motion * 0.7:
                step_i = step * 1.2
            out[i] = prev_i + float(np.clip(delta, -step_i, step_i))

        for i, name in enumerate(self.names):
            lo, hi = self.limits[name]
            out[i] = float(np.clip(out[i], lo, hi))
            if self.mode == "one_euro" and settled:
                self._filters[i].sync(out[i])

        self._prev = out.copy()
        return out

    def reset(self) -> None:
        self._prev = None
        self._motion_ema = 0.0
        for f in self._filters:
            f.reset()

    def sync(self, values: np.ndarray) -> None:
        """跳过步进限制，直接对齐滤波器状态（张开锁定用）"""
        out = np.asarray(values, dtype=np.float64).copy()
        for i, name in enumerate(self.names):
            lo, hi = self.limits[name]
            out[i] = float(np.clip(out[i], lo, hi))
        self._prev = out.copy()
        self._motion_ema = 0.0
        if self.mode == "one_euro":
            for i, f in enumerate(self._filters):
                f.sync(float(out[i]))

    def last(self) -> np.ndarray | None:
        return None if self._prev is None else self._prev.copy()
