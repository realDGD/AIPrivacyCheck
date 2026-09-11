# AI 脱敏器（AI Privacy Check）

面向飞牛 fnOS 的本地文本隐私闸门：先检测并把隐私字段替换为稳定占位符，再将脱敏文本交给外部 AI；AI 回复后，可在当前页面把原值精确放回。

当前版本：`0.4.1`（fnOS Native 原生应用）

## 已实现功能

- **四级插拔式检测与策略引擎**：
  1. **确定性多语言与中文规则（Tier 1，`BuiltInRuleDetector`）**：身份证号、手机号、国际电话（E.164）、固定电话、银行卡、护照、统一社会信用代码、车牌、姓名、地址、邮箱、账号/单号、出生日期、社交账号、IPv4、完整 IPv6、MAC、BIC/SWIFT、IBAN、US SSN、数据库连接串（`DATABASE_URI`）、私钥与各类 API Token/凭证。内置严格静态敏感度等级（PL4/PL3/PL2），不可被模型随意降级。
  2. **中文信息抽取引擎（Tier 2，`ChineseIEDetector`）**：针对中文姓名、复杂行政区划拓扑与建筑地址进行基于语言学特征和安全跨度对齐的抽取（`safe_sequential_span_alignment`，彻底杜绝同名多次出现时的偏移碰撞）；内置零依赖启发式抽取，支持可选适配 ModelScope `siamese-uie`。
  3. **通用 PII 实体抽取（Tier 3，`GLiNERDetector`）**：支持 ModelScope 官方模型 `gliner-pii-edge`（轻量 edge 版，~310MB）与 `gliner-pii-base`（高精度版，~850MB），利用模型原生字符偏移实现零偏移漂移的跨语言 PII 抽取。
  4. **深度语义隐私推理（Tier 4，`MemPrivacyDetector`）**：支持 ModelScope 官方模型 `memprivacy-1.7b-rl`（约 3.4GB）与 `memprivacy-4b-rl`（约 7.8GB），结合上下文对深层隐性隐私（健康状况、人际隐私、资产交易）进行逻辑判定，使用官方真实系统 Prompt 与容错结构化抽取。
- **PL1 - PL4 隐私分级策略与仲裁**：
  - **PL4（核心密码凭据）**：数据库连接串、私钥、API 令牌、系统密码。
  - **PL3（高敏合规凭证）**：身份证、护照、银行卡、医疗病历、财务资产、精确轨迹。
  - **PL2（可识别个人信息）**：姓名、手机、邮箱、地址、社交账号、IP、车牌。
  - **PL1（低敏偏好标签）**：个人公开偏好与低关联职业标签。
  - **层级仲裁机制**：校验位强规则 > 严格规则 > 专用实体抽取模型 > 生成式推理模型，高置信法定凭据永不被模型覆盖。
- **硬件探测与隔离运行时架构**：
  - **HardwareProbe**：直接执行 `nvidia-smi` 获取权威主机硬件与驱动信息，彻底解耦主服务 Python 与 PyTorch/Paddle 框架依赖。
  - **隔离 Runtime Profiles**：为 `torch-cpu`、`torch-cuda`、`paddle-cpu`、`paddle-cuda` 分配独立 Python 虚拟环境，互不污染。
  - **智能策略与安全降级**：根据模型所需框架及其 CUDA 就绪状态动态分派，未安装或异常时平滑回退至 CPU 或内置规则，服务永远不宕机。
- **模型共享目录与生命周期管理（ModelScope 官方源）**：
  - **fnOS 共享模型源**：支持通过 fnOS `data-share` 共享目录（`AI 脱敏器/models`）放置离线模型，应用自动扫描并一键导入。
  - **私有应用模型副本**：应用管理副本激活于 `${TRIM_PKGVAR}/data/models/`，与用户源文件隔离，保证运行稳定性。
  - **授权边界保护**：严格校验 `realpath` 规范路径，限制在共享模型目录、`TRIM_DATA_ACCESSIBLE_PATHS` 或应用私有目录下，杜绝越权访问。
  - **安全卸载保留策略**：支持三档卸载策略（默认 `keep` 保留全部模型与环境；`keep_runtime` 仅保留模型与环境；`delete` 彻底删除应用数据），且**绝不删除用户共享目录源文件**。
- **严格校验器矩阵**：中国身份证 18 位校验码与出生日期、银行卡 Luhn 算法、统一社会信用代码 GB 32100 校验码、IBAN Mod-97 校验、IPv4/IPv6 合法性、MAC 地址格式。
- **稳定可逆占位符**：中文类型前缀 + 序号 + SHA-256 局部指纹（如 `⟦姓名_01_B94F⟧`），同一原值全局一致，AI 回复后精确放回。
- **端到端隐私边界**：文本检测与还原全在本地或浏览器完成；无数据库、不记日志正文；加密保险箱在浏览器端使用 PBKDF2-SHA256 + AES-256-GCM 本地加解密导出。
- **原生 fnOS 体验**：
  - 由 fnOS 官方 `python312` 运行时启动 package 用户无特权守护进程，通过 Unix Stream Socket 直连 fnOS 统一网关。
  - `privilege` 加入 `video` 和 `render` 用户组以访问 GPU 设备节点。
  - 顶部导航栏集成“在新标签页打开 ↗”（桌面小窗与全屏工作流自由切换）。
- **多语言跨语种基准测试**：内置涵盖 11 种语言（中文、英语、德语、法语、西班牙语、俄语、日语、韩语、阿拉伯语、泰语、混合文本）与 20+ PII 类型的精确字符跨度基准测试套件。

## 工作流

```text
原始文本
  └─> 四级检测：多语言强规则 + 中文信息抽取 + GLiNER通用 + MemPrivacy深度推理
        └─> PL2 / PL3 / PL4 隐私策略过滤与优先级仲裁
              └─> 人工复核面板（启用/禁用/修改）
                    └─> ⟦姓名_01_B94F⟧ 等稳定占位符脱敏文本
                          └─> 外部 AI / 云端模型（只看到脱敏文本）
                                └─> 浏览器内按保险箱映射无损精确还原
```

## 本地开发与测试

要求：Python 3.9+，已安装 `uv`（作为标准包管理和环境工具）。

```bash
# 启动本地开发服务
./scripts/run_dev.sh
```

打开 `http://127.0.0.1:8976`。

运行完整测试矩阵（含单元测试、多语言基准测试、语法检查与 fnOS 配置验证）：

```bash
./scripts/test.sh
```

运行独立多语言精确跨度基准测试：

```bash
uv run python scripts/benchmark.py
```

模拟 fnOS Native 原生进程的生命周期（Unix Socket 启停、健康检查、PID 管理与资源清理）：

```bash
./scripts/test_native_lifecycle.sh
```

## 构建 fnOS 安装包

仓库遵循官方 [Native 应用案例](https://developer.fnnas.com/docs/examples/native/)、[运行时环境](https://developer.fnnas.com/docs/core-concepts/runtime/) 和 [统一网关](https://developer.fnnas.com/docs/core-concepts/gateway-registration/) 规范组织：

```bash
./scripts/build_fpk.sh
```

构建产物位于 `dist/ai-privacy-check_0.4.1_all.fpk`。安装包为纯净无架构绑定的原生包（`platform=all`），可安装于 x86_64 和 ARM64 fnOS。

在 fnOS 应用中心选择“手动安装”，上传 `.fpk` 即可。安装时系统会自动关联官方 Python 3.12 运行时。

## 模型存储与管理位置说明

- **应用管理模型副本**：位于 `${TRIM_PKGVAR}/data/models/`，由系统自动维护，不建议用户手动修改。
- **用户共享模型源目录**：位于 `AI 脱敏器/models`（通过 fnOS `data-share` 挂载），专供用户自行存放从魔搭下载的大模型文件。
- **典型使用流程**：
  1. 用户在 NAS 共享文件夹 `AI 脱敏器/models/` 中放入模型（如 `gliner-pii-edge` 或 `memprivacy-1.7b-rl`）；
  2. 打开应用“模型与硬件”面板，点击“扫描共享目录”；
  3. 系统自动校验完整性并提供“导入到应用”按钮；
  4. 应用执行暂存校验并原子激活私有副本；
  5. **卸载保护**：卸载应用时无论选择何种策略，默认均绝不删除用户手动放入共享目录的模型源文件。

## 模型管理与硬件加速

在 fnOS 桌面应用中，使用管理员账号打开“模型与硬件”面板：

1. **选择推理设备**：可在 `Auto`（优先 CUDA）、`CPU`、`NVIDIA CUDA` 之间切换。服务将安全探测显卡可用性，未就绪时自动提示。
2. **在线安装模型**：点击“从魔搭下载安装”，系统将从 ModelScope (魔搭社区) 官方获取指定版本的模型权重（GLiNER、MemPrivacy、SiameseUIE）。
3. **本地手动导入**：若 NAS 无法直接连通外网，可将下载好的模型目录存放于 NAS 共享文件夹中，在页面填入绝对路径（如 `/vol1/1000/models/gliner-pii-edge`），系统将执行完整性校验并原子替换生效。
4. **模型卸载与重载**：随时一键卸载模型，释放磁盘与显存。

## 隐私与安全承诺

- **零数据外流**：待处理的任何用户原始文本、脱敏映射或还原结果均不离开当前设备，不向任何第三方或云端发送请求。
- **无日志泄露**：Web 服务日志仅记录请求方法与静态路由，强制剔除 Query 参数与 Request Body，严禁记录任何 PII 明文。
- **最小特权**：应用以 fnOS 独立的受限 package 用户运行，禁止 root 权限；授权目录访问受 fnOS 安全沙盒约束。
- **内存安全**：会话映射默认仅存在于浏览器运行内存中，关闭或刷新页面后即刻焚毁。

## 许可证

本项目代码使用 Apache License 2.0 开源。相关依赖与可选神经网络模型遵循各自原始开源许可证。
