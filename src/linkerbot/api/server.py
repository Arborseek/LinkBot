"""LinkerHand HTTP API — FastAPI + 队列串行执行"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from linkerbot.config.session import (
    HandMode, HandSide, HardwareMode, SessionConfig, load_hand_profiles,
)
from linkerbot.hardware.driver import HandDriverSet, create_driver_set
from linkerbot.voice.gesture_library import GestureLibrary

# ── 路径 ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_YAML = PROJECT_ROOT / "config" / "default.yaml"
GESTURES_YAML = PROJECT_ROOT / "config" / "gestures.yaml"
PROFILES_YAML = PROJECT_ROOT / "config" / "hand_profiles.yaml"


# ── 配置 ───────────────────────────────────────────────
def _load_api_config() -> Dict[str, Any]:
    with open(DEFAULT_YAML, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    api_cfg = raw.get("api", {})
    return {
        "hand_model": api_cfg.get("hand_model", "L10"),
        "hand_side": api_cfg.get("hand_side", "left"),
        "hardware_mode": api_cfg.get("hardware_mode", "linker_sdk"),
        "host": api_cfg.get("host", "0.0.0.0"),
        "port": int(api_cfg.get("port", 8765)),
        "queue_timeout": float(api_cfg.get("queue_timeout", 30.0)),
    }


def _load_hw_config() -> Dict[str, Any]:
    with open(DEFAULT_YAML, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw.get("hardware", {})


# ── Worker ──────────────────────────────────────────────
class Worker:
    """FIFO 队列：串行执行姿态任务，CAN 独占保护"""

    def __init__(self, driver_set: HandDriverSet, gesture_lib: GestureLibrary):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._driver = driver_set
        self._gestures = gesture_lib

    async def run(self):
        """后台协程，从队列取任务执行（永不崩溃）"""
        while True:
            task = await self._queue.get()
            future = task.get("future")
            try:
                ttype = task["type"]
                if ttype == "gesture":
                    result = self._execute_gesture(task["name"])
                elif ttype == "sequence":
                    result = await self._execute_sequence(
                        task["gestures"], task["interval"]
                    )
                elif ttype == "open":
                    result = self._execute_open()
                else:
                    result = {"ok": False, "detail": f"未知任务类型: {ttype}"}
                if future and not future.done():
                    future.set_result(result)
            except HTTPException as exc:
                if future and not future.done():
                    future.set_exception(exc)
            except Exception as exc:
                if future and not future.done():
                    future.set_exception(exc)

    async def enqueue(self, task_type: str, **kwargs) -> dict:
        """入队并阻塞等待完成"""
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        await self._queue.put({"type": task_type, "future": future, **kwargs})
        return await future

    def _execute_gesture(self, name: str) -> dict:
        pose = self._gestures.lookup(name)
        if pose is None:
            raise HTTPException(404, f"姿态 '{name}' 不存在")
        if not self._driver.drivers:
            raise HTTPException(503, "硬件未连接，请重启服务")
        side = list(self._driver.drivers.keys())[0]
        self._driver.send(side, pose)
        return {"ok": True, "gesture": name, "pose": pose}

    async def _execute_sequence(self, gestures: List[str], interval: float) -> dict:
        if not self._driver.drivers:
            raise HTTPException(503, "硬件未连接，请重启服务")
        side = list(self._driver.drivers.keys())[0]
        executed = []
        for name in gestures:
            pose = self._gestures.lookup(name)
            if pose is None:
                raise HTTPException(404, f"姿态 '{name}' 不存在")
            self._driver.send(side, pose)
            executed.append(name)
            if len(executed) < len(gestures):
                await asyncio.sleep(interval)
        return {"ok": True, "executed": executed}

    def _execute_open(self) -> dict:
        if not self._driver.drivers:
            raise HTTPException(503, "硬件未连接，请重启服务")
        self._driver.send_open_pose()
        return {"ok": True}


# ── App ─────────────────────────────────────────────────
_driver_set: HandDriverSet | None = None
_gesture_lib: GestureLibrary | None = None
_worker: Worker | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _driver_set, _gesture_lib, _worker

    # startup
    api_cfg = _load_api_config()
    hw_cfg = _load_hw_config()

    _gesture_lib = GestureLibrary(GESTURES_YAML)
    _gesture_lib.load()

    profiles = load_hand_profiles(PROFILES_YAML)
    profile = profiles[api_cfg["hand_model"]]

    side = HandSide.LEFT if api_cfg["hand_side"] == "left" else HandSide.RIGHT
    session = SessionConfig(
        hand_model=api_cfg["hand_model"],
        hand_mode=HandMode.SINGLE,
        active_sides=[side],
        hardware_mode=HardwareMode.LINKER_SDK,
    )

    _driver_set = create_driver_set(session, profile, hw_cfg)
    _driver_set.connect()

    _worker = Worker(_driver_set, _gesture_lib)
    worker_task = asyncio.create_task(_worker.run())

    print(f"[API] 服务已启动 http://{api_cfg['host']}:{api_cfg['port']}")
    yield

    # shutdown
    worker_task.cancel()
    if _driver_set:
        _driver_set.disconnect()


app = FastAPI(title="LinkerHand API", lifespan=lifespan)


class SequenceRequest(BaseModel):
    gestures: List[str]
    interval: float = 2.0


def _check_ready():
    if _worker is None or _driver_set is None:
        raise HTTPException(503, "服务尚未就绪")


@app.get("/health")
def health():
    connected = _driver_set is not None
    if _driver_set and _driver_set.drivers:
        connected = all(d.is_connected() for d in _driver_set.drivers.values())
    return {"ok": True, "connected": connected}


@app.get("/gestures")
def list_gestures():
    _check_ready()
    return {"gestures": _gesture_lib.names}


@app.post("/gesture/{name}")
async def execute_gesture(name: str):
    _check_ready()
    return await _worker.enqueue("gesture", name=name)


@app.post("/gesture/sequence")
async def execute_sequence(body: SequenceRequest):
    _check_ready()
    return await _worker.enqueue(
        "sequence", gestures=body.gestures, interval=body.interval
    )


@app.post("/open")
async def open_hand():
    _check_ready()
    return await _worker.enqueue("open")
