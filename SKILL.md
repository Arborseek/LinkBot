---
name: linkerhand-control
description: 控制 Ubuntu 主机上的 LinkerHand L10 灵巧手，从姿态库执行预设手势
---

# LinkerHand L10 灵巧手远程控制

## 概述
通过 HTTP API 控制局域网内 Ubuntu 主机（192.168.110.39:8765）上的 LinkerHand L10 左手。

## API 地址
BASE_URL = http://192.168.110.39:8765

## 可用姿态（18 个）
| 姿态名 | 说明 |
|--------|------|
| 张开手掌 | 五指完全张开 |
| 握拳 | 握成拳头 |
| 点赞 | 竖大拇指 |
| 比耶 | 剪刀手 V 字 |
| OK | OK 手势 |
| 捏合 | 拇食捏合 |
| 准备抓握 | 抓取预备位 |
| 拇指弯曲 | 仅弯拇指 |
| 食指弯曲 | 仅弯食指/指向 |
| 壹 | 数字 1 |
| 贰 | 数字 2 |
| 叁 | 数字 3 |
| 肆 | 数字 4 |
| 伍 | 数字 5 |
| 陆 | 数字 6 |
| 柒 | 数字 7 |
| 捌 | 数字 8 |
| 张开 | 广播体操起始位 |

## 指令别名映射
- "张开" / "张开手" / "打开" / "松手" → 张开手掌
- "握拳" / "拳头" / "握起来" → 握拳
- "点赞" / "竖大拇指" / "good" / "棒" → 点赞
- "比耶" / "剪刀手" / "V" / "耶" → 比耶
- "OK" / "ok" / "好的手势" → OK
- "捏" / "捏合" / "pinch" → 捏合
- "1" / "一" / "数字1" → 壹
- "2" / "二" / "数字2" → 贰
- "3" / "三" / "数字3" → 叁
- "4" / "四" / "数字4" → 肆
- "5" / "五" / "数字5" → 伍
- "6" / "六" / "数字6" → 陆
- "7" / "七" / "数字7" → 柒
- "8" / "八" / "数字8" → 捌

## API 调用方法

### 1. 健康检查
```
Invoke-RestMethod http://192.168.110.39:8765/health
```

### 2. 列出所有姿态
```
Invoke-RestMethod http://192.168.110.39:8765/gestures
```

### 3. 执行单个姿态
PowerShell（推荐，自动处理中文编码）:
```
Invoke-RestMethod -Method POST -Uri "http://192.168.110.39:8765/gesture/点赞"
```

### 4. 执行姿态序列
```
$body = '{"gestures":["张开手掌","握拳","点赞"],"interval":2.0}'
Invoke-RestMethod -Method POST -Uri "http://192.168.110.39:8765/gesture/sequence" -Body $body -ContentType "application/json"
```

### 5. 张开手（复位）
```
Invoke-RestMethod -Method POST -Uri "http://192.168.110.39:8765/open"
```

## 执行规则
1. 收到用户指令后，先映射到正确的姿态名
2. 用 Invoke-RestMethod 执行 POST 请求，**不要用 curl**（PowerShell 的 curl 是别名，处理中文有问题）
3. 检查返回 JSON 中 `ok` 字段为 `true`
4. 如果返回 404，告诉用户该姿态不存在，并用 GET /gestures 列出可用姿态
5. 如果连接失败，提示检查 Ubuntu 服务是否在运行
6. 执行多个动作时，用 /gesture/sequence 而非多次单独调用
7. 每次操作前先确认 /health 返回 `connected: true`

## 示例对话
- 用户: "让机械手点个赞" → POST /gesture/点赞 → "已执行点赞手势"
- 用户: "机械手比个耶然后握拳" → POST /gesture/sequence → "已依次执行比耶、握拳"
- 用户: "机械手从1数到5" → POST /gesture/sequence → "已从壹数到伍"
- 用户: "帮我检查一下机械手服务是否正常" → GET /health → 报告状态
