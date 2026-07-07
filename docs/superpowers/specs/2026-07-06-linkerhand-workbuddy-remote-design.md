# LinkerHand WorkBuddy 远程控制 — 设计文档

日期: 2026-07-06
方案: A（HTTP REST API）
状态: 设计已确认，待实现

## 目标

Windows WorkBuddy 发指令 → Ubuntu API 服务 → 姿态库 lookup → CAN 控制 LinkerHand L10 左手。

---

## 一、架构

```
Windows PC (WorkBuddy)              Ubuntu PC (robot, 192.168.110.39)
┌─────────────────────┐             ┌─────────────────────────────────┐
│ WorkBuddy Agent 模式 │  HTTP POST  │ run_workserve.sh                │
│  + Skill 文件        │ ──────────→ │   ↓                             │
│  自然语言: "机械手点赞"│ ← JSON 响应 │ api_server.py (FastAPI :8765)   │
└─────────────────────┘             │   ↓ asyncio.Queue (FIFO 串行)   │
                                    │ GestureLibrary (gestures.yaml)  │
                                    │   ↓                             │
                                    │ HandDriverSet                    │
                                    │   └─ LinkerSdkDriver.send_pose()│
                                    │       ↓                         │
                                    │ CAN0 → LinkerHand L10 左手       │
                                    └─────────────────────────────────┘
```

## 二、核心设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 协议 | HTTP REST | 实现简单，WorkBuddy 原生支持 curl 调用 |
| 并发模型 | asyncio.Queue FIFO，不可打断 | CAN 总线独占，避免竞态 |
| 配置方式 | `default.yaml` 新增 `api:` 块 | 不干扰 main.py 正常运行 |
| 错误处理 | 503 + 手动重启 | 简洁，CAN 硬件问题需人工介入 |
| sequence 响应 | 阻塞等全部完成 | 调用方知道何时结束 |
| `/open` | 与 `/gesture/张开手掌` 保持区别 | hardware_open_pose ≠ 姿态库最大张开位 |

## 三、API 接口

Base URL: `http://192.168.110.39:8765`

### GET /health
- 返回: `{"ok": true, "connected": true/false}`
- connected 为 false 时需重启服务

### GET /gestures
- 返回: `{"gestures": ["张开手掌", "握拳", "点赞", ...]}`

### POST /gesture/{name}
- 参数: name = 姿态名 (URL path)
- 返回: `{"ok": true, "gesture": "点赞", "pose": [255, 70, 0, ...]}`
- 错误: 404 `{"detail": "姿态 'xxx' 不存在"}`
- 行为: 入队阻塞，等待执行完成

### POST /gesture/sequence
- Body: `{"gestures": ["张开手掌", "握拳"], "interval": 1.5}`
- 返回: `{"ok": true, "executed": ["张开手掌", "握拳"]}`
- 行为: 整个序列作为单一任务入队，原子执行

### POST /open
- 返回: `{"ok": true}`
- 行为: 使用 `hardware_open_pose` (真机出厂复位位姿)

## 四、数据流

```
POST /gesture/点赞
  → asyncio.Queue.put({"type": "gesture", "name": "点赞", "future": Future})
  → await future (阻塞调用方直到执行完成)
  → Worker 从队列取出任务
  → gesture_lib.lookup("点赞") → [255, 70, 0, 0, 0, 0, 255, 255, 255, 41]
  → driver_set.send("left", pose)
  → future.set_result({"ok": true, ...})
  → FastAPI 返回响应
```

## 五、组件

### api/server.py

```
Config      — 从 default.yaml 读 api: 块 (model, side, mode, host, port)
Worker      — asyncio.Queue + 后台协程消费
  ├── gesture → lookup → driver_set.send → set_result
  ├── sequence → 循环 lookup+send+asyncio.sleep → set_result
  └── open → driver_set.send_open_pose → set_result
startup     — lifespan: 加载手势库→构建 SessionConfig→create_driver_set→connect
shutdown    — lifespan: disconnect driver
routes      — 4 个端点，全部 await worker.enqueue(task)
```

### run_workserve.sh

1. 拉起 can0（复用 run_voice.sh 逻辑，sudo_password: 0）
2. `cd ~/linkerbot`
3. `PYTHONPATH=src uvicorn linkerbot.api.server:app --host 0.0.0.0 --port 8765`

## 六、错误码

| 场景 | HTTP | 响应 |
|------|------|------|
| 一切正常 | 200 | `{"ok": true}` |
| 姿态名不存在 | 404 | `{"detail": "姿态 'xxx' 不存在"}` |
| CAN 断连 | 503 | `{"detail": "硬件未连接，请重启服务"}` |
| 任务超时 (>30s) | 504 | `{"detail": "执行超时"}` |

## 七、配置 (config/default.yaml 新增)

```yaml
api:
  hand_model: L10
  hand_side: left
  hardware_mode: linker_sdk
  host: 0.0.0.0
  port: 8765
  queue_timeout: 30.0
```

## 八、文件清单

### Ubuntu 端需新建
| 文件 | 用途 |
|------|------|
| `src/linkerbot/api/__init__.py` | API 模块 |
| `src/linkerbot/api/server.py` | FastAPI 服务 + Worker 队列 |
| `run_workserve.sh` | 一键启动脚本 |

### Ubuntu 端需修改
| 文件 | 变更 |
|------|------|
| `config/default.yaml` | 新增 `api:` 配置块 |

### Ubuntu 端已有，不改
| 文件 | 用途 |
|------|------|
| `config/gestures.yaml` | 姿态库 (18 个手势) |
| `config/hand_profiles.yaml` | 手型关节定义 |
| `src/linkerbot/voice/gesture_library.py` | 姿态 CRUD |
| `src/linkerbot/hardware/driver.py` | CAN 驱动 + HandDriverSet |
| `src/linkerbot/config/session.py` | SessionConfig + HandProfile |

### Windows 端需新建
| 文件 | 用途 |
|------|------|
| `%USERPROFILE%\.workbuddy\skills\linkerhand-control\SKILL.md` | WorkBuddy Skill |

## 九、实施阶段

### 阶段 1 — Ubuntu API 服务
- [ ] pip install fastapi uvicorn
- [ ] 新建 `src/linkerbot/api/__init__.py`
- [ ] 新建 `src/linkerbot/api/server.py`
- [ ] `config/default.yaml` 新增 `api:` 配置块
- [ ] 新建 `run_workserve.sh`
- [ ] 本地启动验证: curl localhost:8765/health → ok
- [ ] curl -X POST localhost:8765/gesture/点赞 → 机械手动

### 阶段 2 — 跨机通信
- [ ] Windows ping 192.168.110.39 通
- [ ] Windows curl 接口全部通过

### 阶段 3 — WorkBuddy 配置
- [ ] Windows 创建 SKILL.md 并加载到 WorkBuddy
- [ ] Agent 模式对话测试通过

## 十、前置条件

- 两台电脑同一局域网
- Ubuntu IP: 192.168.110.39（可能变化）
- Ubuntu CAN0: state UP
- 机械手 USB-CAN 已插入，电源已开
- conda 环境 linkerbot 可用
- **不能同时跑 main.py 和 api_server.py**（CAN 独占）
