# 架构说明 (v0.6.0)

## 设计目标

1. **绝对本地化**：原文、实体与可逆映射不离开用户控制的 fnOS 存储与浏览器运行内存，严格杜绝数据向任何外网服务器泄露。
2. **轻量多级级联**：即使没有大模型、无外网或设备硬件资源受限，基础多语言与中文强规则（Tier 1）及中文信息抽取（Tier 2）也能即开即用、零额外内存驻留；神经网络模型（Tier 3 & Tier 4）按需加载。
3. **控制面与执行面彻底隔离 (Zero ML Control Plane)**：主控进程（Python 3.12）严格不直接 import 任何深度学习框架（`torch`, `transformers`, `gliner`, `modelscope`, `paddle`）。所有模型推理执行于独立的 Worker 子进程中，通过标准输入输出行式 JSONL 进行进程间通信（IPC）。
4. **硬件与运行时解耦**：物理 GPU 检测（`HardwareProbe`）与框架运行时探测（`RuntimeManager`）彻底分离，控制面零框架依赖，消除 CUDA 探测失真；智能利用 NAS 上的 NVIDIA GPU（CUDA）或安全降级至 CPU。
5. **统一且隔离的模型运行时 (Isolated PyTorch Runtimes)**：全系模型统一基于 PyTorch 体系（`torch-cpu` 与 `torch-cuda`），运行在各自独立的隔离虚拟环境中（`${DATA_DIR}/runtimes/{profile}/venv`），杜绝依赖冲突与主进程污染。
6. **配置原子持久化 (SettingsStore)**：通过 `${DATA_DIR}/settings.json` 保证设备偏好、槽位开关与激活模型选择在重启后不丢失。
7. **安全受限的原生集成**：依托 fnOS 官方 Python 3.12 运行时与统一网关 Unix Socket 机制，不暴露多余端口，权限采用 package 独立用户。
8. **用户可见模型共享与安全导入**：提供用户级共享目录（`ai-privacy-check/models`），支持本地模型离线导入与严格的 `realpath` 越权防护；卸载时提供三档数据保留选项，绝对不删除用户共享模型源文件。
9. **跨进程互斥与换代隔离 (v0.5.2)**：基于 `${DATA_DIR}/locks/{model_id}.lock` 的 `fcntl.flock` 保证跨进程模型操作严格互斥；通过 Worker 换代标记彻底杜绝孤儿 EOF 误杀新建 Worker 进程。
10. **模型标识白名单与路径边界防御 (v0.5.3)**：所有安装、导入、卸载及槽位激活 API 强制实施 ModelScope 官方目录白名单验证，杜绝路径遍历（如 `../runtimes`）与参数类型错乱。
11. **第三方 ML 工具可写沙盒收口 (v0.5.4)**：针对 fnOS package 用户环境无系统主目录权限（`/home/ai-privacy-check` 不可写）的特性，通过 `runtime_env.py` 和启动环境显式重定向 `HOME`、`XDG_CACHE_HOME`、`MODELSCOPE_HOME`、`MODELSCOPE_CACHE`、`HF_HOME` 和 `PIP_CACHE_DIR` 到 `${TRIM_PKGVAR}` 内部，保证 ModelScope、Hugging Face 与 PyTorch SDK 稳定初始化与缓存。
12. **模型权重与计算运行时解耦状态机 (v0.5.5)**：严格分离模型权重安装状态（`model_installed`）与运行时计算就绪状态（`runtime_ready` / `detector_ready`）。切换计算设备（如 CUDA ↔ CPU）若缺少对应运行时，系统保持模型权重为已安装状态，前端提供“安装 CPU/CUDA/推荐运行时”入口；补装运行时直接复用已有模型权重并跳过下载，模型卸载时完整保留运行时虚拟环境。
13. **事件驱动的运行时状态缓存失效 (v0.5.6)**：主服务通过 `RuntimeManager.invalidate_probe_cache()` 与 `DeviceManager.invalidate_runtime_state()` 实现事件驱动的探针缓存失效机制。当后台 installer 子进程部署完成运行时或本地导入模型后，主服务自动清除旧的未安装/未就绪探测缓存，执行即时刷新并重置检测器，彻底消除旧探测缓存导致前端显示“缺少运行时”的缺陷，无需停用启用或重启应用。
14. **安全副本与审查卡片双向联动 & 手工划词标注状态机 (v0.6.0)**：在前端引入脱敏文本与审查卡片的双向高亮聚焦联动机制。点击脱敏文本中的标签按钮可平滑滚动、置顶并高亮对应审查卡片；审查卡片点击或定位按钮反向聚焦并脉冲高亮脱敏文本标签。支持在脱敏预览中划词选中文本一键标记为隐私条目（支持通用标记与 16 种标准实体类型选择器）以及范围重选交互模式（隔离恢复当前实体原文并高亮展示，支持重新拖选、防重叠防交叉校验与 Esc/取消快照回滚）。在后端，进一步引入 RuntimeManager 与 DeviceManager 的并发纪元屏障（Generation Epoch Barrier），安装进程生命周期覆盖完成终态清理，彻底消除竞态与提前看到安装完成的问题。

## 系统架构拓扑

```text
fnOS 桌面 / 浏览器新标签页
  │ 统一网关凭证 (NAS 会话)
  ▼
fnOS 统一网关 (/app/ai-privacy-check)
  │ Unix Stream Socket: ai-privacy-check.sock
  ▼
fnOS Native Python 3.12 控制进程 (package 用户: AI 脱敏器, 零 ML 库侵入)
  ├── SettingsStore (原子配置存储)
  │     └── ${DATA_DIR}/settings.json (设备偏好 / 槽位启闭 / 激活模型)
  ├── HardwareProbe (物理硬件探测器)
  │     └── nvidia-smi 独立子进程探测 (GPU 型号 / 驱动版本 / 显存)，零框架依赖
  ├── RuntimeManager (隔离虚拟环境管理器)
  │     ├── torch-cpu / torch-cuda venv 隔离运行时 (uv 管理)
  │     └── 框架级独立验证探针 (cuda.is_available / version)
  ├── WorkerClient & RuntimeWorkerProcess (子进程 IPC 客户端)
  │     ├── GLiNER Worker (privacy/workers/gliner_worker.py)
  │     ├── SiameseUIE Worker (privacy/workers/siamese_uie_worker.py)
  │     ├── MemPrivacy Worker (privacy/workers/memprivacy_worker.py)
  │     └── ModelScope Downloader (privacy/workers/modelscope_downloader.py)
  ├── DetectorRegistry (插拔式检测器管理器)
  │     ├── BuiltInRuleDetector (Tier 1, 槽位 built_in, 始终就绪)
  │     │     ├── 20+ 种格式正则与跨语言上下文约束 (中文/英/德/法/西/俄/日/韩/阿/泰)
  │     │     └── 校验位/日期/Luhn/IBAN/USCC 算法校验器
  │     ├── ChineseIEDetector (Tier 2, 槽位 chinese_ie, 始终就绪)
  │     │     ├── 内置零依赖语言学启发式信息抽取 (百家姓/动词锚点/行政拓扑)
  │     │     ├── safe_sequential_span_alignment 游标防重定位
  │     │     └── 可选 ModelScope SiameseUIE 适配 (Torch runtime, Worker IPC)
  │     ├── GLiNERDetector (Tier 3, 槽位 general_pii, ModelScope: gliner-pii-edge / base, Torch runtime, Worker IPC)
  │     │     └── 原生 token-level span extraction, 零偏移漂移
  │     └── MemPrivacyDetector (Tier 4, 槽位 semantic_privacy, ModelScope: memprivacy-1.7b-rl / 4b-rl, Torch runtime, Worker IPC)
  │           ├── 官方 System Prompt + Qwen Chat 模板 + 贪婪解码 (temperature=0.0)
  │           ├── 容错 JSON 实体数组提取与清洗 (parse_memprivacy_json)
  │           └── 深度隐私逻辑推理与安全跨度对齐 (resolve_semantic_spans)
  ├── ModelLifecycleController & 安全导入
  │     ├── fnOS data-share 用户共享目录 (ai-privacy-check/models) 扫描
  │     ├── validate_import_source_path (realpath 穿透与目录越界防护)
  │     └── 临时暂存区 + os.replace 原子导入 + 合成冒烟测试验证
  └── 静态 Web UI
        ├── 人工复核与稳定占位符
        ├── PL1 - PL4 策略级联筛选
        ├── 实时脱敏与本地精确还原
        └── Web Crypto PBKDF2 + AES-256-GCM 本地加密保险箱
```

## 多级检测、PL策略与实体仲裁

所有检测器对**原始文本**进行纯读操作，检测输出标准半开区间 `[start, end)`。在 `merge_entities()` 阶段：

1. **同跨度同类型合并**：合并检测引擎来源（`sources`），取最高置信度。
2. **PL1 - PL4 敏感度分级与静态策略保护**：
   - **PL4（核心密码凭据）**：`DATABASE_URI` (125), `PRIVATE_KEY` (124), `API_TOKEN` (122), `PASSWORD` (121), `CARD_SECURITY_CODE` (119)
   - **PL3（高敏合规凭据）**：`US_SSN` (114), `IBAN` (113), `CN_USCC` (112), `CREDIT_CARD` (111), `CN_BANK_CARD` (110), `CN_ID_CARD` (115), `PASSPORT` (108)
   - **PL2（可识别个人信息）**：`CN_PHONE_NUMBER` (105), `EMAIL` (98), `CN_LICENSE_PLATE` (96), `IPV6_ADDRESS` (92), `IP_ADDRESS` (92), `CN_NAME` (65), `CN_ADDRESS` (70)
   - **PL1（低敏偏好标签）**：个人公开偏好与低关联职业标签。
   法定确定性实体具备绝对静态敏感度，不可被后续模型随意降级。
3. **重叠冲突仲裁（优先级阶梯）**：
   - **校验位优先原则**：通过校验位（如身份证校验码、银行卡 Luhn、企业税号 GB 32100、IBAN）验证的实体享有绝对优先权（`validated=True` 权重大于任何未校验候选）。
   - **优先级阶梯**：高危凭证密钥 (125) > 校验法定证件 (115) > 通信标识 (105) > 专用实体抽取 (90) > 生成式模型 (50)。
   - **强规则防覆盖**：确定性强规则结果永不被模型覆盖。

## 中文信息抽取（ChineseIEDetector）设计

针对中文无空格分词与同名多次出现的难点，`ChineseIEDetector` 引入：
- **分块全局字符偏移追踪**：长文本分块处理时严格维护绝对文本偏移，杜绝错位。
- **游标顺序对齐（`safe_sequential_span_alignment`）**：在基于词表或子串定位实体时，维护当前查找游标，杜绝 `str.find` 始终返回首个出现的经典缺陷，确保相同人名在不同语境下均能精准映射到各自的 span。

## 硬件与运行时架构 (HardwareProbe & RuntimeManager)

- **硬件探测完全解耦 (`HardwareProbe`)**：不再依赖 Python 内核或在主进程加载 PyTorch。通过调用 `nvidia-smi` 独立查询驱动、设备名称及显存，彻底解决未安装 PyTorch 时错误报告“未检测到 NVIDIA CUDA 环境”的问题。
- **隔离虚拟环境 (`RuntimeManager`)**：
  - 为 PyTorch 统一维护隔离的 venv（`${DATA_DIR}/runtimes/{profile}/venv`）。
  - 安装依赖时优先采用 `uv`，杜绝包依赖冲突。
  - 每个 runtime 在安装后通过独立的子进程探针验证 `cuda.is_available()`，并输出持久化的 `installed.json` 元数据。
- **设备多框架路由 (`DeviceManager`)**：
  - 提供 `resolve_for_framework("torch")`，匹配 runtime 的实际可用状态。
  - 支持 `auto / cpu / cuda` 设备策略，自动优雅降级。

## 模型生命周期管理与安全导入

- **在线下载**：一键从 ModelScope 拉取固定 revision 的模型权重。
- **用户共享目录 (`ai-privacy-check/models`)**：通过 fnOS `config/resource` 声明 `data-share`，用户可在 NAS 共享文件夹中直接放置下载好的模型。
- **本地安全导入边界 (`validate_import_source_path`)**：
  - 强制解析 `os.path.realpath`，抵御符号链接（symlink）越界逃逸与目录遍历（`../`）。
  - 导入源路径必须严格落在系统授权目录（`TRIM_DATA_SHARE_PATHS`、`TRIM_DATA_ACCESSIBLE_PATHS` 或应用数据目录）之内，严禁访问系统敏感目录（如 `/etc`、`/root`、`/var` 等）。
  - 导入过程写入 staging 临时区，校验 `config.json` 与权重完整性后使用原子操作替换。
- **卸载数据保留 (`wizard/uninstall` & `cmd/uninstall_callback`)**：
  - 提供 `keep`（保留全部模型与运行时）、`keep_runtime`（保留运行时环境）、`delete`（彻底删除私有数据）三档选择。
  - 无论用户选择何种卸载模式，均**绝对禁止**删除用户共享目录（`TRIM_DATA_SHARE_PATHS`）中的源文件。

## fnOS 原生适配规范

- **权限声明 (`config/privilege`)**：使用 package 受限用户运行，加入 `video` 与 `render` 组支持 GPU 加速。
- **资源共享声明 (`config/resource`)**：声明 `data-share` 共享文件夹 `ai-privacy-check/models`。
- **卸载向导 (`wizard/uninstall`)**：提供三档卸载数据保留选项。
- **统一网关集成**：通过 Unix Socket 提供静态与 API 服务，网关统一鉴权；桌面与新标签页双模态自适应。
- **展示名称**：正式命名为「AI 脱敏器」，技术代号保持 `ai-privacy-check`。
