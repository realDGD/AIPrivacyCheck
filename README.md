# 隐私净化器（AI Privacy Check）

面向飞牛 fnOS 的本地文本隐私闸门：先检测并把隐私字段替换为稳定占位符，再将脱敏文本交给外部 AI；AI 回复后，可在当前页面把原值精确放回。

当前版本：`0.2.0`（fnOS Native 可安装预览版）

## 已实现

- 中文优先规则：身份证号、手机号、固定电话、银行卡、护照、统一社会信用代码、车牌、姓名、地址、邮箱、账号/单号、出生日期、社交账号、IP、私有网址和常见密钥。
- 校验器：身份证日期与校验位、银行卡 Luhn、统一社会信用代码校验位、IPv4 合法性。
- 可选模型增强：在 fnOS 本地安装并调用 [`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter) 的公开 `opf.OPF` API。
- 人工复核：每个命中项都能启用、禁用，或编辑稳定占位符。
- 可逆恢复：同一原值在一轮处理里使用同一占位符，AI 回复后精确替换回来。
- 加密保险箱：映射可在浏览器端使用 PBKDF2-SHA256 + AES-256-GCM 加密导出；服务器不接收保险箱密码。
- fnOS 统一网关：应用入口复用 NAS 登录态，不暴露额外宿主机端口。
- fnOS Native 运行：由 fnOS 官方 `python312` 运行时直接启动 package 用户进程，不使用 Docker。
- 无文本数据库：服务不保存原文、AI 回复或映射，不记录请求正文。

## 工作流

```text
原始文本
  └─> fnOS 本地中文规则 + 可选 OpenAI Privacy Filter
        └─> 人工复核
              └─> ⟦姓名_01_B94F⟧ 等稳定占位符
                    └─> 外部 AI（只看到脱敏文本）
                          └─> 浏览器内按保险箱映射恢复原值
```

## 为什么首选 Privacy Filter

[`openai/privacy-filter`](https://github.com/openai/privacy-filter) 是面向 PII span detection/redaction 的双向 token-classification 模型，有明确的结构化片段输出和 Apache-2.0 许可证，适合本工具的“找出片段并替换”任务。它主要针对英文，模型卡也提示非英文和非拉丁文字性能可能下降，所以本项目始终先运行中文规则与校验器，再把模型结果作为补充。

[`MemPrivacy`](https://huggingface.co/collections/IAAR-Shanghai/memprivacy) 更偏向边云 Agent 的个性化记忆隐私管理，并非专门的文本 PII span classifier，因此暂不放进首版在线路径。后续可作为“哪些记忆可上云”的独立策略模块评估。

中文规则思路参考了 [`cn_pii_anonymization`](https://github.com/neednlab/cn_pii_anonymization) 公布的实体范围；本地复核和数据边界参考了 [`local-privacy-workbench`](https://github.com/shoujikes-eng/local-privacy-workbench) 的产品原则。本仓库为独立实现，没有复制两个项目的代码。

## 本地开发

要求：Python 3.9+。规则模式无第三方 Python 依赖。

```bash
./scripts/run_dev.sh
```

打开 `http://127.0.0.1:8976`。

运行测试：

```bash
./scripts/test.sh
```

模拟 fnOS 原生进程的启动、Socket 请求、状态检查和停止：

```bash
./scripts/test_native_lifecycle.sh
```

## 构建 fnOS 安装包

仓库按官方 [Native 应用案例](https://developer.fnnas.com/docs/examples/native/)、[运行时环境](https://developer.fnnas.com/docs/core-concepts/runtime/) 和 [统一网关](https://developer.fnnas.com/docs/core-concepts/gateway-registration/) 规范组织。默认使用用户提供的 `fnpack 1.2.3`：

```bash
./scripts/build_fpk.sh
```

产物位于 `dist/ai-privacy-check_0.2.0_all.fpk`。包内只含 Python 源码和静态页面，`platform=all` 可同时安装于 x86_64 和 ARM64 fnOS。也可以覆盖打包器路径：

```bash
FNPACK_BIN=/path/to/fnpack ./scripts/build_fpk.sh
```

在 fnOS 应用中心选择手动安装 `.fpk`。安装时 fnOS 会根据 `install_dep_apps=python312` 安装官方 Python 3.12 运行时；应用启动后中文规则模式立即可用，无需容器镜像。

## 安装可选模型

使用 fnOS 管理员账号打开“模型与隐私”，点击“安装模型增强”。应用会：

1. 从固定的 OpenAI Privacy Filter Git 提交安装 `opf`、PyTorch 等运行依赖。
2. 从 Hugging Face 固定模型提交下载 `original/` 权重（约 2.8 GB）。
3. 将依赖和权重保存在 fnOS 管理的应用数据目录。

建议预留 4–8 GB 磁盘与至少 4 GB 可用内存。下载是唯一需要外网的阶段；检测文本不会发送给 GitHub、Hugging Face 或 OpenAI API。模型安装兼容性仍需分别在目标 x86_64/ARM fnOS 设备上实测。

## 隐私与安全边界

- 浏览器到服务：通过 fnOS 统一网关和当前 NAS 会话访问。
- 服务端：仅在内存中处理请求；日志只包含 HTTP 方法和无查询参数的路径。
- 映射：默认只在当前浏览器页面内存；刷新页面即丢失。
- 导出：保险箱在浏览器端加密，密码无法恢复。
- 进程权限：服务以 fnOS 为应用创建的 package 用户运行，仅在 `${TRIM_PKGVAR}` 和应用 Socket 路径写入运行数据。
- 局限：任何自动识别都会漏检或误检。本工具是数据最小化辅助工具，不是匿名化、合规或安全保证；高敏内容外发前必须人工复核。

更多细节见 [架构说明](docs/architecture.md) 和 [安全说明](docs/security.md)。

## 项目结构

```text
packaging/ai-privacy-check/        fnOS FPK 源目录
  app/server/                     Python 原生服务、检测引擎与静态前端
  app/ui/config                   fnOS 桌面/统一网关入口
  cmd/                            fnOS 生命周期脚本
  config/                         package 用户最小权限与资源声明
tests/                            中文规则与合并策略测试
scripts/                          本地启动、测试和打包脚本
assets/                           可重新生成的 SVG 图标源文件
```

## 许可证

本项目代码使用 Apache License 2.0。OpenAI Privacy Filter 运行库与模型由其各自许可证约束，安装时从上游获取，不包含在 `.fpk` 中。
