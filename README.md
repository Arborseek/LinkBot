# LinkBot

LinkerHand 灵巧手控制与遥操作平台。通过摄像头捕捉人手姿态，实时驱动 LinkerHand 系列灵巧手；同时支持语音指令、音乐舞蹈编排，以及 HTTP API 远程控制。

## 功能

- **摄像头遥操作** — MediaPipe 手部追踪 + 关节重映射，支持单手/双手模式
- **MuJoCo 仿真** — 无硬件时可 Mock 仿真预览（L10 / L20 等型号）
- **语音控制** — 唤醒词 + ASR + LLM 意图识别，执行预设手势
- **舞蹈模式** — 音乐节拍/歌词卡点，自动生成并播放手势序列
- **HTTP API** — FastAPI 服务，局域网内远程调用预设姿态

## 支持型号

O6、L7、L10、L20 等（详见 `config/hand_profiles.yaml`）。

## 环境要求

- Ubuntu / Linux（CAN 总线控制需 Linux）
- Python 3.11
- Conda（推荐）
- USB 摄像头
- LinkerHand 灵巧手 + CAN 适配器（硬件模式）
- [LinkerHand Python SDK](https://github.com/linker-bot/linkerhand-python-sdk)（置于 `vendor/linkerhand-python-sdk`）

## 快速开始

### 1. 创建 Conda 环境

```bash
bash scripts/setup_conda.sh
conda activate linkerbot
```

### 2. 下载手部追踪模型

将 MediaPipe Hand Landmarker 模型放到：

```
assets/hand_landmarker.task
```

### 3. 获取 URDF 与网格文件（仿真需要）

URDF 描述文件已包含在仓库中，但 STL 网格体积较大未纳入 Git。请任选其一：

```bash
# 方式 A：从官方仓库下载完整 URDF（含 meshes）
git clone https://github.com/linker-bot/linkerhand-urdf vendor/linkerhand-urdf-main
bash scripts/setup_linker_urdf.sh
```

L10 仿真网格位于 `assets/mujoco/linker_hand_l10/`，若缺失请从本地备份或官方资源获取。

### 4. 启动遥操作

```bash
python main.py
# 指定摄像头
python main.py --camera 0
```

启动后按界面提示完成手型初始化校准，按 `Space` 开始遥操作。

### 快捷键

| 按键 | 功能 |
|------|------|
| `Space` | 开始/暂停遥操作 |
| `V` | 切换语音模式 |
| `D` | 切换舞蹈模式 |
| `Q` / `Esc` | 退出 |

## HTTP API 服务

在 Ubuntu 主机上启动 API（需 CAN 与灵巧手已连接）：

```bash
bash run_workserve.sh
# 或
PYTHONPATH=src uvicorn linkerbot.api.server:app --host 0.0.0.0 --port 8765
```

### 常用接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/gestures` | 列出所有预设姿态 |
| POST | `/gesture/{name}` | 执行单个姿态 |
| POST | `/gesture/sequence` | 执行姿态序列 |
| POST | `/open` | 张开手（复位） |

示例：

```bash
curl http://127.0.0.1:8765/health
curl -X POST http://127.0.0.1:8765/gesture/点赞
curl -X POST http://127.0.0.1:8765/gesture/sequence \
  -H "Content-Type: application/json" \
  -d '{"gestures":["张开手掌","握拳","点赞"],"interval":2.0}'
```

预设姿态定义见 `config/gestures.yaml`。

## 语音模式

```bash
export DEEPSEEK_API_KEY="sk-xxx"
bash run_voice.sh
```

或在主程序中按 `V` 进入语音模式。配置见 `config/voice.yaml` 与 `config/default.yaml` 中的 `voice` 段。

## 配置

主配置文件：`config/default.yaml`

| 配置段 | 说明 |
|--------|------|
| `camera` | 摄像头参数 |
| `tracking` | MediaPipe 追踪阈值 |
| `retarget` | 关节映射与平滑 |
| `hardware` | CAN 接口、速度、力矩 |
| `simulation` | MuJoCo 仿真 |
| `voice` | 语音与 LLM |
| `dance` | 舞蹈编排 |
| `api` | HTTP 服务 |

本地敏感项（如 CAN sudo 密码）请在 `config/default.yaml` 的 `hardware.sudo_password` 中自行填写，勿提交到公开仓库。

## 项目结构

```
LinkBot/
├── main.py                 # 主入口（遥操作）
├── config/                 # 配置文件
├── src/linkerbot/
│   ├── app.py              # 主应用逻辑
│   ├── api/                # HTTP API
│   ├── capture/            # 摄像头
│   ├── tracking/           # 手部追踪
│   ├── retarget/           # 姿态重映射
│   ├── hardware/           # 硬件驱动
│   ├── sim/                # MuJoCo 仿真
│   ├── voice/              # 语音控制
│   ├── dance/              # 舞蹈模式
│   └── viz/                # 可视化
├── scripts/                # 安装与测试脚本
├── assets/                 # 模型与 MuJoCo 资源
└── vendor/                 # 第三方依赖（URDF 等）
```

## 许可证

本项目包含 `vendor/linkerhand-urdf-main`，其许可证见对应目录下的 LICENSE 文件。
