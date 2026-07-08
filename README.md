# LinkBot

**English** | [中文](README.zh-CN.md)

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Linux-lightgrey.svg)]()
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](vendor/linkerhand-urdf-main/LICENSE)

Open-source teleoperation and control stack for [LinkerHand](https://github.com/linker-bot) dexterous hands.

Track your hand with a camera, retarget joints in real time, and drive hardware over CAN — or preview everything in MuJoCo without a physical hand. LinkBot also ships with voice control, music-driven dance choreography, and a FastAPI remote-control server.

## Table of Contents

- [Features](#features)
- [Supported Hardware](#supported-hardware)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Teleoperation](#teleoperation)
  - [HTTP API Server](#http-api-server)
  - [Voice Control](#voice-control)
  - [Dance Mode](#dance-mode)
- [Configuration](#configuration)
- [Project Layout](#project-layout)
- [Development](#development)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Features

| Module | Description |
|--------|-------------|
| **Camera teleoperation** | MediaPipe hand tracking + joint retargeting; single-hand and dual-hand modes |
| **MuJoCo simulation** | Mock / preview mode when hardware is unavailable |
| **Voice control** | Wake word, ASR, and LLM intent routing to preset gestures |
| **Dance mode** | Beat / lyrics sync with auto-generated gesture choreography |
| **HTTP API** | FastAPI service for remote gesture execution on the local network |

## Supported Hardware

LinkerHand models supported out of the box include **O6**, **L7**, **L10**, **L20**, and more.

See [`config/hand_profiles.yaml`](config/hand_profiles.yaml) for joint limits, DOF, and default open poses.

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| **OS** | Ubuntu / Linux recommended (CAN bus required for hardware mode) |
| **Python** | 3.11 |
| **Conda** | Recommended for environment management |
| **Camera** | USB webcam for teleoperation |
| **LinkerHand + CAN adapter** | Required for hardware control |
| **LinkerHand Python SDK** | Clone into `vendor/linkerhand-python-sdk` |
| **MediaPipe model** | `assets/hand_landmarker.task` (not bundled) |

Optional for voice / dance LLM features:

- `DEEPSEEK_API_KEY` environment variable

Optional for the HTTP API:

```bash
pip install fastapi uvicorn
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Arborseek/LinkBot.git
cd LinkBot
```

### 2. Create the Conda environment

```bash
bash scripts/setup_conda.sh
conda activate linkerbot
```

### 3. Install the hand tracking model

Download the [MediaPipe Hand Landmarker](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) model and place it at:

```text
assets/hand_landmarker.task
```

### 4. Install the LinkerHand SDK (hardware mode)

```bash
git clone https://github.com/linker-bot/linkerhand-python-sdk vendor/linkerhand-python-sdk
```

### 5. Fetch URDF meshes (simulation)

URDF files are included, but STL meshes are excluded from Git due to size.

```bash
git clone https://github.com/linker-bot/linkerhand-urdf vendor/linkerhand-urdf-main
bash scripts/setup_linker_urdf.sh
```

L10 MuJoCo meshes should live under `assets/mujoco/linker_hand_l10/`. Restore them from a local backup or official resources if missing.

## Usage

### Teleoperation

```bash
python main.py

# use a specific camera device
python main.py --camera 0
```

1. Complete the on-screen setup wizard.
2. Calibrate the initial hand pose.
3. Press `Space` to start teleoperation.

| Key | Action |
|-----|--------|
| `Space` | Start / pause teleoperation |
| `V` | Toggle voice mode |
| `D` | Toggle dance mode |
| `Q` / `Esc` | Quit |

### HTTP API Server

Run on the machine connected to the dexterous hand:

```bash
bash run_workserve.sh
```

Or manually:

```bash
PYTHONPATH=src uvicorn linkerbot.api.server:app --host 0.0.0.0 --port 8765
```

Default port: **8765** (configurable in `config/default.yaml` → `api`).

#### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service and hardware status |
| `GET` | `/gestures` | List available preset gestures |
| `POST` | `/gesture/{name}` | Execute one gesture |
| `POST` | `/gesture/sequence` | Execute a timed gesture sequence |
| `POST` | `/open` | Open hand / reset pose |

Example:

```bash
curl http://127.0.0.1:8765/health

curl -X POST http://127.0.0.1:8765/gesture/点赞

curl -X POST http://127.0.0.1:8765/gesture/sequence \
  -H "Content-Type: application/json" \
  -d '{"gestures":["张开手掌","握拳","点赞"],"interval":2.0}'
```

Preset gestures are defined in [`config/gestures.yaml`](config/gestures.yaml).

### Voice Control

```bash
export DEEPSEEK_API_KEY="sk-xxx"
bash run_voice.sh
```

You can also press `V` inside the main application. See [`config/voice.yaml`](config/voice.yaml) and the `voice` section in [`config/default.yaml`](config/default.yaml).

### Dance Mode

Press `D` in the main application, or configure the `dance` section in [`config/default.yaml`](config/default.yaml).

Dance choreography can be generated from audio beats, ASR lyrics, or LLM fallback sequences.

## Configuration

Primary config file: [`config/default.yaml`](config/default.yaml)

| Section | Purpose |
|---------|---------|
| `camera` | Camera device, resolution, mirroring |
| `tracking` | MediaPipe confidence thresholds |
| `retarget` | Joint mapping, smoothing, pinch detection |
| `hardware` | CAN interface, speed, torque, SDK path |
| `simulation` | MuJoCo model and display options |
| `voice` | Wake word, ASR, LLM provider |
| `dance` | Audio path, choreography source, timing |
| `api` | HTTP host, port, hand model / side |

**Security note:** keep local secrets such as `hardware.sudo_password` and API keys out of version control. Use environment variables or an untracked local override.

## Project Layout

```text
LinkBot/
├── main.py                 # Teleoperation entry point
├── config/                 # YAML configuration
├── src/linkerbot/
│   ├── app.py              # Main application loop
│   ├── api/                # FastAPI server
│   ├── capture/            # Camera capture
│   ├── tracking/           # Hand tracking
│   ├── retarget/           # Pose retargeting
│   ├── hardware/           # CAN / SDK drivers
│   ├── sim/                # MuJoCo simulation
│   ├── voice/              # Voice pipeline
│   ├── dance/              # Dance choreography
│   └── viz/                # UI overlays and rendering
├── scripts/                # Setup and test scripts
├── assets/                 # Models, music, MuJoCo assets
└── vendor/                 # Third-party dependencies
```

## Development

Run helper scripts under `scripts/`:

```bash
python scripts/test_gestures.py
python scripts/test_api.py
python scripts/test_voice_pipeline.py
```

When adding a new hand model, update [`config/hand_profiles.yaml`](config/hand_profiles.yaml) and verify retargeting in both simulation and hardware modes.

## Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-change`
3. Commit your changes with a clear message
4. Open a Pull Request against `main`

Please keep changes focused, match existing code style, and avoid committing secrets or large binary assets.

## Acknowledgments

- [LinkerHand](https://github.com/linker-bot) hardware and SDK
- [linkerhand-urdf](https://github.com/linker-bot/linkerhand-urdf) robot descriptions
- [MediaPipe](https://developers.google.com/mediapipe) hand tracking
- [MuJoCo](https://mujoco.org/) simulation

## License

Third-party URDF assets under [`vendor/linkerhand-urdf-main/`](vendor/linkerhand-urdf-main/) are licensed under the [Apache License 2.0](vendor/linkerhand-urdf-main/LICENSE).

Other project files are provided as open source. If you redistribute or modify this project, review the licenses of all bundled third-party components.
