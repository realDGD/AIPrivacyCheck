# 架构说明

## 设计目标

1. 原文和恢复映射不离开用户控制的 fnOS 与浏览器。
2. 即使没有大模型、没有外网或设备内存有限，也能完成常见中文高敏字段的脱敏和恢复。
3. 对低确定性的姓名、地址和日期保留人工复核路径。
4. fnOS 包本身保持小巧；重型依赖和模型由管理员按需安装到应用数据目录。

## 组件

```text
fnOS 桌面 iframe
  │ NAS 登录态
  ▼
统一网关 /app/ai-privacy-check
  │ Unix Socket: ai-privacy-check.sock
  ▼
fnOS Native Python 3.12 进程
  ├── ChineseRuleDetector
  │     ├── 正则与上下文约束
  │     └── 校验位/日期/Luhn 校验
  ├── OpenAIModelDetector（可选、惰性加载）
  │     └── opf.OPF + 本地 checkpoint
  └── 静态 Web UI
        ├── 人工复核与稳定占位符
        ├── 本地恢复
        └── Web Crypto 加密保险箱
```

## 检测与合并

所有规则和模型都输出半开区间 `[start, end)`。完全相同的实体合并来源和置信度；重叠实体按以下顺序选择：

1. 通过校验位或严格格式校验的实体。
2. 类型优先级（密钥、身份证、社会信用代码、银行卡、手机号优先）。
3. 置信度。
4. 更长的有效片段。

这能避免模型把一整段数字识别成普通账号时覆盖已经通过校验的身份证或银行卡。

## 可逆占位符

占位符形如 `⟦手机号_01_C40B⟧`：

- 中文类型便于人工复核。
- 两位序号区分同类实体。
- 四位 SHA-256 指纹降低编辑时意外串用的概率。
- 同一类型与原值在同一轮处理中复用同一占位符。

占位符不是加密；安全性来自不把映射发送给外部 AI。需要跨会话保存映射时，浏览器使用 310,000 次 PBKDF2-SHA256 派生 AES-256-GCM 密钥后再导出。

## 模型生命周期

- 源代码提交固定为 `f7f00ca7fb869683eb732c010299d901457f19c3`。
- 模型提交固定为 `7ffa9a043d54d1be65afb281eddf0ffbe629385b`。
- 依赖安装到 `${TRIM_PKGVAR}/data/python-packages`。
- 权重安装到 `${TRIM_PKGVAR}/data/models/privacy-filter`。
- 服务在第一次模型检测时惰性加载 OPF，避免规则模式占用模型内存。
- 下载先进入临时目录，校验 `config.json` 和 Safetensors 文件后再原子切换。

## fnOS 集成

- `config/privilege` 使用 package 用户。
- `manifest` 声明 `install_dep_apps=python312`，由 fnOS 提供官方运行时。
- `cmd/main` 直接启停 `${TRIM_APPDEST}/server/server.py`，用 PID 文件返回正确状态。
- 原生进程在 `${TRIM_APPDEST}` 创建统一网关 Unix Socket，不开放 TCP 端口。
- `config/resource` 为空，不需要 Docker 项目或共享目录。
- 模型安装端点只接受网关传入的 `X-Trim-Isadmin: true`。
- 业务 API 不信任客户端提供的用户身份字段。
