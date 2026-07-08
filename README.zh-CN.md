# LinkBot

[English](README.md) | **中文**

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Linux-lightgrey.svg)]()
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](vendor/linkerhand-urdf-main/LICENSE)

面向 [LinkerHand](https://github.com/linker-bot) 灵巧手的开源遥操作与控制框架。

通过摄像头追踪人手姿态，实时重映射关节并驱动 CAN 硬件；无设备时可在 MuJoCo 中仿真预览。项目还包含语音控制、音乐舞蹈编排，以及基于 FastAPI 的远程控制服务。

## 目录

- [功能特性](#功能特性)
- [支持硬件](#支持硬件)
- [环境要求](#环境要求)
- [安装](#安装)
- [使用说明](#使用说明)
  - [摄像头遥操作](#摄像头遥操作)
  - [HTTP API 服务](#http-api-服务)
  - [语音控制](#语音控制)
  - [舞蹈模式](#舞蹈模式)
- [配置说明](#配置说明)
- [项目结构](#项目结构)
- [开发指南](#开发指南)
- [参与贡献](#参与贡献)
- [致谢](#致谢)
- [许可证](#许可证)

## 功能特性

| 模块 | 说明 |
|------|------|
| **摄像头遥操作** | MediaPipe 手部追踪 + 关节重映射，支持单手/双手 |
| **MuJoCo 仿真** | 无硬件时的 Mock / 预览模式 |
| **语音控制** | 唤醒词、ASR、LLM 意图识别，映射到预设手势 |
| **舞蹈模式** | 音乐节拍 / 歌词卡点，自动生成手势编排 |
| **HTTP API** | FastAPI 服务，局域网远程执行预设姿态 |

## 支持硬件

内置支持 **O6**、**L7**、**L10**、**L20** 等 LinkerHand 型号。

关节限位、自由度与默认张开姿态见 [`config/hand_profiles.yaml`](config/hand_profiles.yaml)。

## 环境要求

| 依赖 | 说明 |
|------|------|
| **操作系统** | 推荐 Ubuntu / Linux（硬件模式需 CAN 总线） |
| **Python** | 3.11 |
| **Conda** | 推荐用于环境管理 |
| **摄像头** | USB 摄像头，用于遥操作 |
| **LinkerHand + CAN 适配器** | 硬件控制必需 |
| **LinkerHand Python SDK** | 克隆到 `vendor/linkerhand-python-sdk` |
| **MediaPipe 模型** | `assets/hand_landmarker.task`（未随仓库分发） |

语音 / 舞蹈 LLM 功能可选：

- 环境变量 `DEEPSEEK_API_KEY`

HTTP API 额外依赖：

```bash
pip install fastapi uvicorn
```

## 安装

### 1. 克隆仓库

```bash
git clone https://github.com/Arborseek/LinkBot.git
cd LinkBot
```

### 2. 创建 Conda 环境

```bash
bash scripts/setup_conda.sh
conda activate linkerbot
```

### 3. 下载手部追踪模型

下载 [MediaPipe Hand Landmarker](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) 模型，放到：

```text
assets/hand_landmarker.task
```

### 4. 安装 LinkerHand SDK（硬件模式）

```bash
git clone https://github.com/linker-bot/linkerhand-python-sdk vendor/linkerhand-python-sdk
```

### 5. 获取 URDF 网格（仿真）

URDF 描述文件已包含在仓库中，STL 网格因体积较大未纳入 Git。

```bash
git clone https://github.com/linker-bot/linkerhand-urdf vendor/linkerhand-urdf-main
bash scripts/setup_linker_urdf.sh
```

L10 MuJoCo 网格位于 `assets/mujoco/linker_hand_l10/`，若缺失请从本地备份或官方资源获取。

## 使用说明

### 摄像头遥操作

```bash
python main.py

# 指定摄像头
python main.py --camera 0
```

1. 按界面完成初始化向导
2. 校准初始手型
3. 按 `Space` 开始遥操作

| 按键 | 功能 |
|------|------|
| `Space` | 开始 / 暂停遥操作 |
| `V` | 切换语音模式 |
| `D` | 切换舞蹈模式 |
| `Q` / `Esc` | 退出 |

### HTTP API 服务

在连接灵巧手的机器上运行：

```bash
bash run_workserve.sh
```

或手动启动：

```bash
PYTHONPATH=src uvicorn linkerbot.api.server:app --host 0.0.0.0 --port 8765
```

默认端口：**8765**（可在 `config/default.yaml` → `api` 中修改）

#### 接口列表

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 服务与硬件状态 |
| `GET` | `/gestures` | 列出预设姿态 |
| `POST` | `/gesture/{name}` | 执行单个姿态 |
| `POST` | `/gesture/sequence` | 执行姿态序列 |
| `POST` | `/open` | 张开手 / 复位 |

示例：

```bash
curl http://127.0.0.1:8765/health

curl -X POST http://127.0.0.1:8765/gesture/点赞

curl -X POST http://127.0.0.1:8765/gesture/sequence \
  -H "Content-Type: application/json" \
  -d '{"gestures":["张开手掌","握拳","点赞"],"interval":2.0}'
```

预设姿态定义见 [`config/gestures.yaml`](config/gestures.yaml)。

### 语音控制

```bash
export DEEPSEEK_API_KEY="sk-xxx"
bash run_voice.sh
```

也可在主程序中按 `V` 进入语音模式。配置见 [`config/voice.yaml`](config/voice.yaml) 及 [`config/default.yaml`](config/default.yaml) 中的 `voice` 段。

### 舞蹈模式

在主程序中按 `D` 进入，或在 [`config/default.yaml`](config/default.yaml) 的 `dance` 段中配置。

支持基于音频节拍、ASR 歌词或 LLM 回退序列自动生成编排。

## 配置说明

主配置文件：[`config/default.yaml`](config/default.yaml)

| 配置段 | 用途 |
|--------|------|
| `camera` | 摄像头设备、分辨率、镜像 |
| `tracking` | MediaPipe 置信度阈值 |
| `retarget` | 关节映射、平滑、捏合检测 |
| `hardware` | CAN 接口、速度、力矩、SDK 路径 |
| `simulation` | MuJoCo 模型与显示选项 |
| `voice` | 唤醒词、ASR、LLM 提供商 |
| `dance` | 音频路径、编排来源、时序 |
| `api` | HTTP 地址、端口、手型号 / 左右 |

**安全提示：** `hardware.sudo_password`、API Key 等敏感信息请勿提交到 Git，建议使用环境变量或未跟踪的本地配置。

## 项目结构

```text
LinkBot/
├── main.py                 # 遥操作入口
├── config/                 # YAML 配置
├── src/linkerbot/
│   ├── app.py              # 主应用循环
│   ├── api/                # FastAPI 服务
│   ├── capture/            # 摄像头采集
│   ├── tracking/           # 手部追踪
│   ├── retarget/           # 姿态重映射
│   ├── hardware/           # CAN / SDK 驱动
│   ├── sim/                # MuJoCo 仿真
│   ├── voice/              # 语音管线
│   ├── dance/              # 舞蹈编排
│   └── viz/                # UI 渲染
├── scripts/                # 安装与测试脚本
├── assets/                 # 模型、音乐、MuJoCo 资源
└── vendor/                 # 第三方依赖
```

## 开发指南

`scripts/` 目录下提供了测试脚本：

```bash
python scripts/test_gestures.py
python scripts/test_api.py
python scripts/test_voice_pipeline.py
```

新增手型号时，请更新 [`config/hand_profiles.yaml`](config/hand_profiles.yaml)，并在仿真与硬件模式下验证重映射效果。

## 参与贡献

欢迎提交 Issue 和 Pull Request。

1. Fork 本仓库
2. 创建分支：`git checkout -b feature/my-change`
3. 提交改动，附清晰 commit message
4. 向 `main` 分支发起 Pull Request

请保持改动聚焦、遵循现有代码风格，勿提交密钥或大型二进制文件。

## 致谢

- [LinkerHand](https://github.com/linker-bot) 硬件与 SDK
- [linkerhand-urdf](https://github.com/linker-bot/linkerhand-urdf) 机器人描述文件
- [MediaPipe](https://developers.google.com/mediapipe) 手部追踪
- [MuJoCo](https://mujoco.org/) 物理仿真

## 许可证

[`vendor/linkerhand-urdf-main/`](vendor/linkerhand-urdf-main/) 下的第三方 URDF 资源采用 [Apache License 2.0](vendor/linkerhand-urdf-main/LICENSE)。

项目其余部分以开源形式提供。若你分发或修改本项目，请一并遵守所有第三方组件的许可证要求。
