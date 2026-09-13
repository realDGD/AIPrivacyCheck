# 架构说明 (v0.6.13)

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
15. **Unicode 字符切片契约与安全脱敏屏障 (v0.6.1)**：在实体序列化中直接输出 `start_utf16` 与 `end_utf16`，解决 Python 码点（code points）与浏览器 JavaScript UTF-16 code units 计数不一致导致的复杂 Unicode / Emoji / 合字文本脱敏漂移；统一自动检测与手动标注的占位符生成分配器（`allocateReplacementToken`），同实体同原文复用同一占位符；在前端脱敏执行层建立多重健全性检查，遇跨越损坏或占位符碰撞时立即失效脱敏态并阻断复制与外发；全面重构标注交互生命周期，防止跨视图与重检测时的悬挂状态残留。
16. **GLiNER 实体形态过滤与前端交互布局加固 (v0.6.2)**：针对神经网络实体抽取模型（GLiNER）在中文自然语言长句中将普通短语错误识别为 `USERNAME` 的缺陷，在检测生产链路上层增加严格的形态学与上下文校验过滤器（`_is_plausible_username`），针对中文自然语言叙述默认拒绝提取为 USERNAME，杜绝整句自然语言标点被错误遮盖；仅允许标准 ASCII token 或具有显式账号上下文的 span。在前端交互层，将双向联动滚动彻底收口至内部容器（`scrollElementIntoContainer`），杜绝外层 `document` 产生跳顶抖动；重构高亮脉冲动画移除横向几何平移，防止水平溢出滚动条；并固化桌面端 640px 三栏严格等高容器内部滚动模型。
17. **MemPrivacy 语义隐私推理加固与显存/超时治理 (v0.6.3)**：在多模型级联执行中，由于 MemPrivacy 1.7B 参数量与生成显存占用较大（~3.6GB 模型权重 + KV Cache），在受限显卡（如 Tesla P4 8GB）上与 GLiNER（~3.0GB）顺序常驻必然触发 CUDA OOM。系统实施 exclusive CUDA 独占调度，在 MemPrivacy CUDA 推理前主动驱逐其他常驻 CUDA Worker（完全保留 CPU worker），并在推理完成后立即在 `finally` 中停止 MemPrivacy 释放 GPU 显存。当 Worker 捕获底层致命 CUDA OOM 时，立即物理终止 worker 进程释放显存，向主控返回结构化 OOM 信号并优雅降级为温和 warning，绝不崩溃主流程；将推理超时与运行设备深度绑定，CPU 模式下放宽 MemPrivacy 推理超时至 360 秒；约束模型单次生成 token 预算至 <=512 tokens；对 >3500 字符长文本实施 3000 字符分块与 200 字符自然句边界重叠切分，并进行跨块实体去重与全局精确坐标安全对齐；模型目录明确规范 1.7B 与 4B 推荐硬件规格，并在前端提供 CPU 慢速模式温和 inline 提示。
18. **MemPrivacy / CUDA 并发安全与整请求语义预算加固 (v0.6.4)**：在多模型高并发与受限 GPU 显存环境中，针对并发请求可能导致的 Worker 进程复活与显存竞争，引入 `RuntimeWorkerProcess` 永久退役机制（`WorkerRetiredError` 与 `retire()` 契约），确保被驱逐或停用的 Worker 物理终止且绝不被并发请求重新拉起成为无法管控的孤儿常驻进程；引入跨模型 `WorkerClient.cuda_execution_session` 上下文协调锁，在物理 GPU 层面实施全会话排他调度，杜绝 MemPrivacy 推理期间并发拉起 GLiNER 导致 CUDA OOM；建立整请求语义时间预算体系（CUDA 240s / CPU 480s）与 Deadline 截止期控制，锁等待超时快速返回繁忙状态，单块超时安全保留并输出已完成实体及 X/Y 进度告警；对超长生成截断统一去重告警；精确区分 CPU 宿主机内存耗尽与 GPU 显存不足；消除子进程退出等待期间的全局锁争用。
19. **模型运行时依赖契约与安装链净化 (v0.6.4)**：在 Model Catalog 的 `ModelDescriptor` 上引入 `runtime_dependencies` 声明字段，安装流程由 `install_isolated_runtime` → `ensure_model_runtime_dependencies` → 下载/导入 → 完整性校验 → 真实冒烟推理构成完整链路；共享 torch-cpu/torch-cuda 运行时保持不动，模型专属缺失依赖（如 SiameseUIE 的 `addict`/`datasets`/`scipy`/`Pillow`/`simplejson`/`sortedcontainers`，经 ModelScope 1.40 元数据与真实 pipeline 加载实证）通过目标 venv 解释器 `importlib` 探测后增量补装，已满足即快速跳过，绝不重装 PyTorch。共享运行时 transformers 固定 `>=4.51,<5`（Qwen3 下限 / ModelScope legacy `transformers.onnx` 上限）。安装与导入阶段对模型 configuration.json 净化 `allow_remote`/`plugins` 字段，worker 不传 `trust_remote_code`，仅经 ModelScope 内建类加载，杜绝模型目录远程代码执行；并适配 ModelScope 新版 SiameseUIE 输出结构（按 schema 分组嵌套列表 + `offset` 半开区间），修复加载成功但实体恒为空的缺陷。

19a. **Base Runtime Contract、预加载安全门与分层基准 (v0.6.5)**：运行时"已验证即跳过"路径追加基础依赖契约核查与增量迁移（transformers>=4.51,<5，绝不重装 torch）；`privacy/model_security.py` 在任意 worker 加载前净化模型 configuration.json 的远程代码声明；GLiNER 阈值收敛至 `GLiNERDetector.GLINER_DEFAULT_THRESHOLD` 单一定义并依据 100 文档分层语料再调优至 0.50；`tests/fixtures/privacy_benchmark_v2_100.jsonl` 提供 Detection / Should-Redact / 语义三层分离的冻结评测语料，公共联系人（10086、8.8.8.8、test@example.com 等）按"检测正确、不应脱敏"计分。

19b. **Benchmark v2 评分修复、预测缓存与模型精简按需下载 (v0.6.6)**：修复语义评分器在显式负样本黄金实体上的计分逻辑（重叠召回时仅增加 overreach 与 fp，tp 严格保持不变）；将原 redaction_acc 准确更名为 redaction_eligibility_coverage（redCov），精确度量敏感实体的召回覆盖；GLiNER 标签顺序定义为确定性不可变元组 GLINER_LABELS；建立 `benchmark-cache/` 持久化预测缓存，实现单次模型推理、毫秒级离线阈值扫频；建立 `selective_downloader`，严禁全量拉取 ModelScope 快照，按需下载必需权重与配置，节省 60%+ 带宽；统一 Built-in v2 负样本评测口径为 407 条，Strict-negative FPR 严格为 0 / 388 (0.0%)。

19c. **Benchmark 可复现性加固与仓库凭据卫生治理 (v0.6.7)**：预测缓存（Raw Prediction Cache）深度绑定宿主机完整运行时关键版本（python、torch、transformers、modelscope、gliner）及模型真实内容 SHA-256 哈希，引入 `cache_signature` 子目录实现不同参数与运行时的并发无损共存；精简下载器（Selective Downloader）建立本地模型完整性验证契约 `verify_existing_model_integrity`，依据 `download-manifest.json` 校验文件大小与哈希，拒绝仅靠 `is_file()` 判定的虚假文件，支持单文件精准增量修复并在下载失败时原子清理 `.tmp_download` 临时文件；Benchmark 评测中严格解耦 `redactable_fn`（所有未命中脱敏实体）与 `critical_fn`（仅限高危法定标识符与凭据）；净化代码库中所有可能触发 GitHub Secret Scanning 的合成凭据字面量，测试套件全面实施运行时字符串动态拼接，建立静态安全门禁 `test_secret_hygiene`；保持 Built-in v2 规则与阈值冻结，零负样本误报回归。

19d. **Benchmark 基础设施最终冻结 (v0.6.8)**：
- **模型真实内容哈希与单源完整性契约**：建立 `scripts/model_integrity.py` 共享模块，统一 `selective_downloader.py` 与 `benchmark_cache.py` 的模型完整性核验逻辑；基于 `download-manifest.json` 校验磁盘物理文件真实尺寸与 SHA-256，无清单时回退流式分块计算；若内容损坏或同尺寸篡改则触发 `ModelIntegrityError` 严格拒绝缓存命中，杜绝任何“文件存在即信任”的漏洞。
- **Selective Downloader 规范化与安全强化**：修复缺失 `Any` 导致的 Python 3.12/3.13 兼容性异常，建立真实子进程导包门禁；`download-manifest.json` 强化严格 Schema（规范 64 位小写 hex SHA-256、非负大小），增加对绝对路径和 `../` 相对路径穿越的严格拦截，提升供应链安全防御。
- **ModelScope Revision 来源记录**：明确区分 `requested_revision` 与 `resolved_revision`（当前 API 端点返回 null 并如实标注）；采用精准术语“Local content integrity fingerprint”。
- **Prediction Cache 歧义拦截与预测数据完整性保障**：父级缓存目录存在多个有效签名时强制抛出 `AmbiguousCacheError` 拒绝静默选择；写入时生成 `cache-manifest.json`，读取时强制校验预测条数与 `predictions_sha256`，篡改即报 `CacheCorruptedError`。
- **全仓库凭据卫生**：全库所有文本格式（`.py`, `.json`, `.jsonl`, `.md`, `.sh` 等）静态凭据扫描达成 0 违规，高熵虚构凭据全量替换为零熵惰性模式。
- **双重冻结状态**：Built-in v2 规则与阈值冻结（0/388 严格负样本 FPR），Benchmark 基础设施正式进入 STABLE / FROZEN 最终冻结状态。

19e. **Benchmark 基础设施与信任边界最终加固 (v0.6.9)**：
- **Selective Downloader 同尺寸损坏文件强制修复**：修复当本地已存在同尺寸损坏文件时绕过网络重新下载的缺陷；`download_single_file` 引入 `force_download=True` 参数，原子流式下载至 `.tmp_download` 并校验哈希后原子替换，并在 ModelScope 可变 revision 内容变更时记录告警与 `upstream_content_changed` 标记。
- **Prediction Cache Manifest 强制完整性契约**：`cache-manifest.json` 成为缓存命中法定要件；`has_valid_cache()` 强制要求 manifest 存在并比对 `predictions_sha256` 与预测条目数；`load()` 默认对无清单的遗留缓存抛出 `LegacyUnverifiedCacheError`，提供 `--allow-legacy-unverified-cache` 命令行安全兼容选项。
- **Manifest Schema 跨平台路径穿越与大写哈希拦截**：Manifest 校验严格拦截 Windows 盘符路径（`^[A-Za-z]:`）、UNC 网络路径（`\\server\share`, `//server/share`）、绝对路径及各类 `..` 目录穿越；强制约束 `sha256` 必须为规范小写 64 位十六进制（`^[0-9a-f]{64}$`），严格拒绝大写十六进制。
- **CRITICAL_ENTITY_TYPES 规范收敛与文档对齐**：明确代码中定义的 18 类核心高危实体单一事实来源（`CN_ID_CARD`, `GOVERNMENT_ID`, `US_SSN`, `CN_BANK_CARD`, `CREDIT_CARD`, `CN_PHONE_NUMBER`, `PHONE`, `private_phone`, `EMAIL`, `private_email`, `SECRET`, `secret`, `PASSWORD`, `API_TOKEN`, `PRIVATE_KEY`, `DATABASE_URI`, `PASSPORT`, `CN_PASSPORT`），严格解耦法定高危与普通脱敏实体。
- **全库静态凭据零容忍（Zero Provider-Perfect Literals）**：移除针对提供商特征静态凭据的任何豁免，全仓库工作区内提供商形态（如 `AKIA...`, `github_pat_...`, `LTAI...`, `ghp_...`）静态字面量彻底清零（0 个）；所有测试夹具均转换为标准通用合成前缀（`SYNTH_...`）。
- **双重最终冻结确认**：Built-in v2 规则与阈值冻结（0/388 严格负样本 FPR，100/100 幂等性，0 占位符命中），Benchmark 评测基础设施（下载器、完整性核验、缓存契约、评分器）全指标达标并正式进入 STABLE / FROZEN 最终冻结状态。

19f. **历史版本升级运行时迁移、独立检测器控制与长文本正确性加固 (v0.6.10)**：
- **历史模型专属依赖迁移与轻量探针**：彻底解决 v0.6.3 及更早版本已安装的 PyTorch 运行时（`torch-cpu`/`torch-cuda`）在升级到新版本后因缺少新增模型特定依赖（如 SiameseUIE 历史 venv 缺少 `addict`）而报 `No module named 'addict'` 的故障。实现轻量只读探针 `probe_model_runtime_dependencies`，在不触发重量级导包或模型加载的前提下快速检测缺失包，并在运行时状态失效时自动清空探针缓存。
- **Detector Readiness 两级就绪状态模型与一键修复**：检测器状态全面细化并暴露 `installed`、`base_runtime_ready`、`model_dependencies_ready`、`missing_dependencies`、`ready` 与 `repairable` 字段。提供 `POST /api/model/runtime/repair` 管理端点与前端 `[修复运行环境]` 一键修复操作：对目标 venv 增量补装缺失的 pip 依赖并执行真实冒烟测试，严禁重下模型权重、删除模型或重装 PyTorch。
- **独立检测器控制与 Slots 契约解耦**：将原单一的“模型增强检测”开关解耦为四层独立控制：内置规则（Built-in Rules，常开）、中文语义提取（Chinese IE，常开基线）、GLiNER 通用 PII（`glinerToggle`，默认开启，localStorage 持久化）、MemPrivacy 深度语义隐私（`memprivacyToggle`，默认关闭，首次开启弹出资源消耗确认，localStorage 持久化）。服务端 `active_slots` 强制包含 `built_in` 与 `chinese_ie`；兼容遗留请求 `use_model=True`（仅激活 GLiNER，MemPrivacy 需显式通过 `slots` 声明 opt-in）。
- **规则引擎日期跨度与结构化密码修复**：修复 `CN_BIRTH_DATE` 日期截断缺陷（如 `1992年11月18日` 不再被贪婪截断为 `1992年1`）；新增结构化密码规则（`PASSWORD` 实体类型，优先级 121 胜过用户名，支持换行及中英阿德法西日韩泰多语言前缀），并配合 `_is_valid_password_value` 严格拦截代码标识符（`passwordManager`）、环境变量（`${DB_PASSWORD}`）与占位符，保持 0/388 严格负样本 FPR 零误报。
- **长文本验收与多行 OTP 防误报门禁**：建立 `tests/fixtures/long_context_manual_acceptance.txt` 涵盖 20 大测试场景，全量采用通用合成凭据（`SYNTH_...`）；在 GLiNER 中新增换行符拦截与 12..19 位纯数字门禁，杜绝多行 OTP 恢复代码块被错误识别为 `CREDIT_CARD`，并明确将 `Project Aurora` 记录为已知通用 NER 候选误报基准。

19g. **uv 托管 Python 基础运行时与两级环境修复架构 (v0.6.11)**：
- **彻底脱离 fnOS 系统 Python 依赖**：彻底解决 fnOS 主机精简 Python 缺失 `_lzma`, `_bz2`, `_ssl`, `_sqlite3` 等底层 C 扩展导致的 ML 运行环境兼容故障。采用 uv 托管全功能标准 CPython 3.12.9 作为隔离运行时的底层解释器，隔离存放于 `${DATA_DIR}/python/installations`，uv 缓存收口至 `${DATA_DIR}/cache/uv`。
- **17 项 Python 原生基础能力契约 (Python Capability Contract)**：定义并强制检验 17 项原生模块：`lzma`, `_lzma`, `bz2`, `_bz2`, `ssl`, `_ssl`, `sqlite3`, `_sqlite3`, `ctypes`, `_ctypes`, `zlib`, `hashlib`, `json`, `multiprocessing`, `subprocess`, `venv`, `ensurepip`。能力探针在隔离子进程中执行，不侵入主控制面。
- **两级修复架构（Two-Tier Repair vs Transactional Rebuild）**：
  - Level 1 增量依赖补齐：底层 Python 原生能力完整时，仅增量安装缺失的模型专属依赖包（如 `addict`），秒级完成，不重建 venv。
  - Level 2 事务性环境升级重建：底层能力缺失（如缺少 `_lzma`）或需要升级时，自动或手动触发 `rebuild_runtime`。预检磁盘剩余空间（CPU >= 2GB，CUDA >= 4.5GB）；在 `venv.rebuild-<timestamp>` 中全量构建 uv 托管环境与所有已安装模型专属依赖集合；在临时环境中验证 17 项能力并执行真实模型冒烟测试；通过后优雅停止旧 profile worker，原子切换目录并清理旧环境与临时目录；失败时安全回滚原环境，绝不造成服务不可用。
- **模型权重绝不重下与绝不篡改契约**：无论 Level 1 还是 Level 2，严禁调用 ModelScope 下载逻辑，绝不修改、删除或重命名 `${DATA_DIR}/models/*` 中的任何模型权重文件，升级过程 100% 零带宽消耗、零模型重载。
- **Runtime Manifest Schema v3 与无损接管（Adoption）**：Manifest 升级至 `schema_version: 3`，记录 `python_runtime_source` (`managed` / `legacy-system-python`), `python_runtime_version`, `capabilities` 等。对于历史上已存在且各项能力健康的旧环境，探测时无损接管升级为 v3，不触发重建。
- **状态模型与控制面板体验强化**：检测器状态与前端控制面板细化展示缺失底层能力提示，提供「重建/升级运行环境」直观操作。

19h. **托管 Python 基础运行时收敛与内置 uv 供应链闭环 (v0.6.12)**：
- **内置官方 Astral musl uv 独立供应链**：彻底修复真实 fnOS 宿主机 PATH 无 `uv` 工具导致隔离环境创建与环境修复失败的阻塞问题。FPK 安装包内置官方 Astral 静态链接 musl ELF 二进制（`app/bin/linux-x86_64/uv` 与 `app/bin/linux-aarch64/uv`），经 SHA-256 强校验，实现 100% 离线、零网络、零宿主机 root 权限、零系统 PATH 依赖。
- **四阶段事务性原子切换与严格回滚保护**：运行时重建遵循 `old -> backup PASS, staging -> final FAIL`, `manifest write FAIL`, `final probe FAIL`, `rollback itself fails` 四重故障防御。在最终探针与清单完全验证通过前，旧环境备份绝对不被删除；若回滚本身遭遇异常，永久保留 `venv.old.<timestamp>`，绝不灭失用户环境。
- **Base ML Contract 8 项基础依赖健康探测与版本门禁**：目标运行环境在投入服务前必须通过 8 项基础 ML 依赖探测（`torch`, `modelscope`, `numpy`, `packaging`, `tqdm`, `transformers`, `accelerate`, `gliner`），并对 `transformers >=4.51,<5` 实施强制拦截约束。
- **PyTorch 2.6.0 精准锁定与真实版本落盘**：明确锁定 PyTorch 2.6.0，同时在 `runtime-manifest.json` 与状态 API 中如实记录探测所得真实依赖版本。
- **历史旧环境双重健康准入与无损接管**：仅当历史隔离环境同时通过 17 项 Python 原生能力契约与 Base ML Contract 时方执行无损升级接管，否则安全触发隔离重建。
- **零模型重载与零权重篡改契约**：修复与重建全流程零网络下载（0 bytes from ModelScope）、零模型权重修改。

19i. **运行时网络与供应链安全收敛 (v0.6.13)**：
- **系统 CA 根证书强制注入 (UV_SYSTEM_CERTS=true)**：解决真实 fnOS 宿主机环境下由于 musl 静态构建 uv 未能加载系统 CA 根证书导致的 `invalid peer certificate: UnknownIssuer` 故障。在 `build_uv_env` 全局统一注入 `UV_SYSTEM_CERTS=true`，确保所有 uv 命令与子进程继承系统证书信任链。
- **Cernet/MirrorZ 优先策略与官方源回退降级**：
  - 受管 Python 3.12.9：优先从 `https://mirrors.cernet.edu.cn/python-build-standalone` 下载，网络异常时安全回退至官方 `https://github.com/astral-sh/python-build-standalone`。
  - PyPI 依赖包：优先通过 `https://mirrors.cernet.edu.cn/pypi/web/simple` 安装，网络失败时回退至官方 `https://pypi.org/simple`。
  - PyTorch 2.6.0 轮子：优先从 `https://mirrors.cernet.edu.cn/pytorch/whl/{cu124|cpu}` 安装，网络失败时回退至官方 `https://download.pytorch.org/whl/{cu124|cpu}`。
- **严格 TLS Fail-Closed 策略**：全链路坚决杜绝 `--allow-insecure-host`、`--trusted-host` 或禁用证书校验；遭遇 TLS/网络异常时输出清晰排查指引并安全失败。
- **完整性异常绝对 Fail-Closed**：哈希校验失败或文件损坏属于供应链完整性异常，绝对禁止重试或源降级回退，立即安全终止。
- **受管 Python 3.12.9 零重复下载**：当底层解释器已存在且通过 17 项原生能力契约时，直接复用已有解释器，0 字节重复下载（返回 `existing-local` 遥测状态）。
- **元数据写入与原子切换一致性**：`runtime-manifest.json` 与 `installed.json` 先落盘为 `.tmp`，待最终探针完全验证通过后通过 `os.replace` 原子生效；若最终探针失败或切换异常，完整恢复原有元数据。
- **THIRD_PARTY_NOTICES.md 双重打包与运行时 SHA 缓存**：FPK 根目录与应用根目录均打包第三方许可证说明；内置 uv 增加运行时 SHA-256 完整性检验与进程内缓存。

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
