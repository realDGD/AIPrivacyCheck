# 安全说明 (v0.6.14)

## 不保存的数据

- 用户粘贴的原始文本。
- 检测后的明文实体列表。
- AI 回复和恢复后的文本。
- 保险箱密码。

这些值仅瞬时存在于单次 HTTP 请求的内存或当前浏览器页面内存中。服务器访问日志主动剔除 Query 参数，且绝不记录请求正文。

## 会保存的数据

- 可选神经网络模型权重（来自 ModelScope 的 GLiNER、MemPrivacy 或 SiameseUIE 权重，存放于 `${DATA_DIR}/models/`）。
- 隔离运行时虚拟环境（统一基于 PyTorch，存放于 `${DATA_DIR}/runtimes/`）。
- 应用配置与偏好（持久化于 `${DATA_DIR}/settings.json`，原子文件写入）。
- 不含任何用户文本的模型安装状态与运行日志。

## 攻击面与缓解措施

- **网络边界与外网隔离**：业务处理全链路在本地完成，无任何外部遥测或云端传输。即使配置了大模型，也是运行在 NAS 本地推理。
- **远程访问与统一网关**：使用 fnOS 统一网关进行登录鉴权，不暴露额外宿主机端口。
- **管理操作安全隔离**：模型下载、本地导入、卸载及推理设备切换等管理端点均强制校验 fnOS 网关注入的管理员身份（`X-Trim-Isadmin: true`），非管理员或伪造请求一律被拦截。
- **第三方 SDK 运行沙盒与 HOME 重定向 (v0.5.4)**：全量拦截并重定向 ModelScope、Hugging Face 与 pip 缓存路径至 `${TRIM_PKGVAR}`，彻底杜绝子进程越权写入系统 `/home/ai-privacy-check` 或 `/root`。
- **模型标识白名单与目录遍历阻断 (v0.5.3)**：针对所有涉及 `model_id` 的管理接口（安装、导入、卸载、槽位切换、锁获取），实施严格正则格式匹配（`^[a-z0-9][a-z0-9._-]*$`）与官方 Catalog 白名单校验；杜绝构造 `../runtimes` 等恶意跨目录路径破坏主机虚拟环境或删除系统目录。
- **并发互斥与防竞态**：采用基于 `${DATA_DIR}/locks/{model_id}.lock` 的 `fcntl.flock` 机制，跨进程防止同一模型在安装、导入与卸载之间的竞态条件与目录损坏。
- **严格路径验证与逃逸防护**：本地模型导入通过 `validate_import_source_path` 实施严格检查：
  - 强制执行 `os.path.realpath` 展开全部符号链接，彻底防范 symlink 目录逃逸。
  - 源路径必须严格落在已声明共享目录（`TRIM_DATA_SHARE_PATHS`）、授权目录（`TRIM_DATA_ACCESSIBLE_PATHS`）或应用数据目录内。
  - 明确封禁系统敏感根目录（`/etc`、`/root`、`/var`、`/sys` 等）。
- **卸载隔离安全边界**：卸载向导与卸载脚本（`cmd/uninstall_callback`）严格校验输入路径并防范变量未定义导致的误删；严禁触碰或删除任何位于共享目录（`TRIM_DATA_SHARE_PATHS`）内的用户个人原始文件。
- **第三方模型与提示词合规**：独立自主编写 Apache-2.0 规范的语义提取提示词，避免衍生自非商业限制提示词文本；提供 `THIRD_PARTY_LICENSES.md` 明确声明各组件许可证。
- **GPU 资源与权限受控**：服务以无特权 package 用户运行，仅加入 `video` 和 `render` 组以便访问 GPU 节点，不请求额外 root 权限。
- **CSRF / XSS 防护**：所有状态变更 API 均要求 `application/json` 请求头；前端实体渲染通过 DOM 安全属性绑定，严格配置 Content-Security-Policy。
- **Unicode 编码漂移与脱敏逃逸防御 (v0.6.1)**：针对多语言、Emoji、复杂合字及生僻字场景，由服务端计算并输出精确的 UTF-16 code units 偏移（`start_utf16` 与 `end_utf16`），消除了 Python 与 JavaScript 字符串索引差异导致的跨度漂移与尾部字符残留；安全副本生成时实施占位符与原文冲突检查及跨越重叠校验，若发生异常立即破坏性失效脱敏副本并禁用复制与外发按钮，防止半脱敏或错误脱敏的明文泄露。
- **模型误报抑制与前端交互稳定性 (v0.6.2)**：针对神经网络实体抽取模型（GLiNER）在中文自然语言环境中过度提取短语为 `USERNAME` 的问题，实施了基于账号词汇特征与显式上下文判定的生产级过滤规则，防止中文日常会话整句被模型过度脱敏或破坏语义；前端交互与布局全面加固，双向定位收口至内部容器滚动，防止外层 document 滚动突兀跳跃导致的用户误操作。
- **显存耗尽与超时拒绝服务防御 (v0.6.3)**：为防止大参数量语义模型（MemPrivacy）在多模型级联或显存受限设备（如 8GB 显卡）上耗尽系统 GPU 显存，实施 exclusive CUDA 独占调度并在执行完成后立即释放显存；发生底层 CUDA OOM 时立即终止子进程释放显存并温和降级，杜绝显存泄漏导致的宿主机 GPU 瘫痪；对推理 token 生成预算实施 <=512 tokens 的严格上限与 >3500 字符长文本分块处理，杜绝长文本推理引发的 KV 缓存爆炸与长时间挂起。
- **并发 Worker 隔离与整请求预算拒绝服务防御 (v0.6.4)**：引入 Worker 永久退役机制（`WorkerRetiredError` 与 `retire()` 契约），彻底杜绝并发下已被停用的 Worker 进程被重新激活为孤儿进程盗用 GPU 显存；引入跨模型 `cuda_execution_session` 全局协调锁，在物理 GPU 层面排他互斥，杜绝并发导致 GLiNER 与 MemPrivacy 同时在 GPU 上运行引发 CUDA OOM；建立整请求语义时间预算体系（CUDA 240s / CPU 480s）与 Deadline 截止控制，锁等待超时快速返回繁忙状态，单块超时安全保留已完成分块与实体并输出 X/Y 进度告警，彻底避免长文本跨块累加导致十数小时的同步挂起拒绝服务。
- **模型目录远程代码执行防御 (v0.6.4)**：新版 ModelScope 会对携带 `allow_remote`/`plugins` 声明的模型 configuration.json 执行目录内任意 `.py` 代码并 pip 安装模型自带 requirements.txt（官方 iic 权重即携带 `allow_remote: true`）。本应用在模型下载与本地导入的暂存阶段对 configuration.json 实施净化（移除 `allow_remote`/`plugins` 字段，幂等且对已存在权重同样生效），推理 worker 坚决不传 `trust_remote_code`，所有目录模型仅经 ModelScope 内建 pipeline/model 类加载；共享运行时的 transformers 固定为 `>=4.51,<5` 兼容区间，杜绝依赖解析期被第三方声明拖入不可信版本。
- **模型专属依赖契约与供应链最小化 (v0.6.4)**：`ensure_model_runtime_dependencies` 仅安装 Catalog 中经实证声明的模型专属依赖（当前仅 SiameseUIE 需要 `addict`/`datasets`/`scipy`/`Pillow`/`simplejson`/`sortedcontainers`），通过目标 venv 解释器 `importlib` 探测实现幂等快速跳过，杜绝"遇错全量 pip freeze"式供应链扩散。
- **预加载模型远程代码安全门 (v0.6.5)**：新增 `privacy/model_security.py`，在任意 worker 加载模型前净化 configuration.json 中的 `allow_remote`/`plugins` 声明；v0.6.3 及更早版本安装的存量模型升级后首次使用即被自动净化。GLiNER / MemPrivacy / SiameseUIE 的 load 与 detect 路径、冒烟测试与安装期净化共用同一实现，杜绝旁路。
- **Base Runtime Contract 兼容性迁移 (v0.6.5)**：对"已验证即跳过"的历史 runtime 增加基础依赖契约核查；发现 transformers 5.x 等违约包时仅增量修复该包（>=4.51,<5），不重装 torch、不重建 venv；缺 torch 的损坏环境拒绝增量修复。
- **基准语料与凭证卫生 (v0.6.5)**：100 文档冻结语料中的全部凭证均为合成值（明显样例结构或运行时拼接），不包含任何真实秘密；vault-engine 评测仅在隔离目录中以库方式运行，禁用云端 provider。
- **模型精简按需下载与传输安全 (v0.6.6 - v0.6.9)**：严禁无限制全量拉取 ModelScope 社区仓库快照，仅由 `selective_downloader` 静态白名单枚举并单文件流式校验下载 PyTorch 必需文件，杜绝不可信仓库引入非必需可执行资产；自动过滤 ONNX 冗余文件与文档，减少 60%+ 网络流量暴露；v0.6.9 引入 `force_download=True`，彻底解决同尺寸损坏文件无法自动重新下载的缺陷，并在上游 mutable revision 变更时记录告警。
- **模型内容真实性指纹与缓存安全屏障 (v0.6.8 - v0.6.9)**：Prediction Cache 深度绑定 `scripts/model_integrity.py`，对模型磁盘文件进行 SHA-256 内容校验；损坏或篡改直接抛出 `ModelIntegrityError` 拒识伪造；父级目录多签名歧义强制抛出 `AmbiguousCacheError`；落盘与加载严格校验 `cache-manifest.json` 与 `predictions_sha256`（`CacheCorruptedError` / `LegacyUnverifiedCacheError`）。
- **精简下载器清单 Schema 与跨平台防路径穿越防御 (v0.6.8 - v0.6.9)**：强制要求 `download-manifest.json` 包含规范小写 64 位十六进制 SHA-256（`^[0-9a-f]{64}$`）和非负文件大小，严格过滤 Windows 盘符路径（`^[A-Za-z]:`）、UNC 网络路径（`\\server\share`, `//server/share`）、绝对路径与 `../` 路径穿越注入。
- **全仓库零提供商形态凭据卫生加固 (v0.6.9 - v0.6.10)**：全库静态凭据扫描门禁达成 0 违规，严禁存在任何提供商形态（`AKIA...`, `github_pat_...`, `LTAI...`, `ghp_...`）的静态字面量（即使包含 SAMPLE/EXAMPLE 或全 0 熵值亦被严格拦截），全量替换为通用 `SYNTH_...` 格式。新增 `tests/fixtures/long_context_manual_acceptance.txt` 严格遵循该契约。
- **高危实体（Critical Entities）解耦与定义边界**：明确核心法定与凭据高危实体（`CRITICAL_ENTITY_TYPES`）由 18 类明确定义组成：`CN_ID_CARD`, `GOVERNMENT_ID`, `US_SSN`, `CN_BANK_CARD`, `CREDIT_CARD`, `CN_PHONE_NUMBER`, `PHONE`, `private_phone`, `EMAIL`, `private_email`, `SECRET`, `secret`, `PASSWORD`, `API_TOKEN`, `PRIVATE_KEY`, `DATABASE_URI`, `PASSPORT`, `CN_PASSPORT`，评测时严格与通用脱敏实体（`redactable_fn`）解耦。
- **历史升级模型运行时修复安全门 (v0.6.10)**：`POST /api/model/runtime/repair` 强制校验管理员权限（`_is_admin()`），仅在沙盒 venv 目录执行 `uv pip install` 补装缺失依赖并执行只读冒烟测试，严禁重下模型、删除模型或重装 PyTorch，避免网络滥用与文件系统破坏。
- **结构化密码检测与严格负样本防护 (v0.6.10)**：实现 `_is_valid_password_value` 验证器，严格排除代码变量（`passwordManager`）、函数调用（`getPassword()`）、环境变量占位符（`${DB_PASSWORD}`）、隐藏占位符（`******`、`[已隐藏]`）以及中英自然语言描述，保持 0/388 严格负样本误报率；`PASSWORD` 实体赋予优先级 121，高于 `USERNAME`（83），彻底消除密码提取被截断为用户名的安全缺陷。
- **uv 托管 Python 基础运行时与沙盒安全门禁 (v0.6.11)**：
  - **私有存储安全隔离**：uv 安装的底层 CPython 解释器隔离部署于 `${DATA_DIR}/python/installations`，uv 缓存收口至 `${DATA_DIR}/cache/uv`，全量脱离系统 `/usr`、`/lib` 等特权敏感目录，严格限制在应用专属权限边界（package 用户）内运行。
  - **预检磁盘安全防护与拒绝服务防御**：`rebuild_runtime` 前置执行 `check_disk_space_for_rebuild`，强制要求 CPU 模式至少 2.0 GB、CUDA 模式至少 4.5 GB 可用磁盘空间，空间不足时快速失败，杜绝磁盘写满引发的宿主机或 NAS 系统级拒绝服务（DoS）。
  - **跨进程排他互斥锁防竞态**：基于 `fcntl.flock` 的 `runtime_operation_lock` 对目标 profile 加锁，拦截并发重建与并发修复冲突，保证隔离环境操作具备严格的 ACID 属性。
  - **事务性环境切换与零模型篡改安全**：在独立 staging 目录（`venv.rebuild-<timestamp>`）中构建并经多模型真实冒烟测试核验后原子替换；重构全过程禁止调用 ModelScope 网络下载，严格保持 `${DATA_DIR}/models/*` 权重文件哈希不变，免受网络劫持与供应链文件污染。
- **内置 uv 独立供应链与四阶段事务性回滚安全 (v0.6.12)**：
  - **静态二进制供应链完整性**：安装包内置的官方 Astral musl uv 独立二进制（x86_64 与 aarch64）在构建及运行阶段强制执行 SHA-256 完整性核验，杜绝非受控网络下载脚本（如 `curl ... | sh`）、非 root 权限越权及供应链投毒。
  - **四阶段事务性安全切换与防灭失保证**：切换流程严格分为 staging 构建、旧环境安全备份（`venv.old.<timestamp>`）、临时环境重命名生效、元数据与最终探针核验四阶段。只要任何一步未通过，立即触发事务回滚；若回滚本身遭遇异常，永久保留旧环境备份目录，绝不静默删除导致环境毁灭。
  - **Base ML Contract 强制版本门禁**：对新运行环境严格实施 8 项基础依赖健康核验与 `transformers >=4.51,<5` 安全门禁，阻断不兼容或畸形依赖投入生产服务。
- **TLS 与网络兼容性最终收敛安全契约 (v0.6.14)**：
  - **分层渐进式 TLS 信任与 Fail-Closed 底线**：针对不同系统 CA 证书环境，默认以 uv 原生 Mozilla CA 运行，仅在证书链不可信时渐进尝试系统 CA 与显式 CA bundle。严禁采用 `--allow-insecure-host`、`--trusted-host` 或禁用证书验证，杜绝网络中间人攻击。
  - **索引隔离与依赖混淆防范**：通过 `--index <torch_index> --default-index <pypi_index> --index-strategy first-index` 将 PyTorch 轮子严格限定于专用 wheel index，阻止恶意第三方在公开 PyPI 注册同名 wheel 实施依赖混淆投毒。
  - **包完整性损坏绝对 Fail-Closed**：哈希不匹配或文件损坏绝不自动换源或重试，直接终止并报警，防范投毒或篡改。
  - **本地磁盘/权限错误阻断**：本地存储满或无权限错误不触发无意义换源，明确告知管理员排查宿主机环境。
- **运行时网络与供应链安全收敛 (v0.6.13)**：
  - **系统 CA 根证书强制注入**：全量 uv 子进程与环境注入 `UV_SYSTEM_CERTS=true`，彻底解决 musl 静态构建未加载宿主机系统根证书导致的 PyPI/uv TLS UnknownIssuer 异常。
  - **严格 TLS Fail-Closed 策略**：全链路坚决杜绝 `--allow-insecure-host`、`--trusted-host` 或禁用证书校验；遭遇 TLS/网络异常时输出清晰排查指引并安全失败。
  - **完整性异常绝对 Fail-Closed**：哈希校验失败或文件损坏属于供应链完整性安全事件，绝对禁止重试或回退，立即快速终止。
  - **Cernet 镜像优先与官方源降级**：为受管 Python、PyPI 与 PyTorch 轮子三条供应链提供 Cernet 镜像与官方源降级能力，并在 manifest 和状态 API 中完整记录下载源与降级遥测。
  - **元数据写入与原子切换一致性**：`runtime-manifest.json` 与 `installed.json` 先落盘为 `.tmp`，待最终探针完全验证通过后通过 `os.replace` 原子生效；若最终探针失败或切换异常，完整恢复原有元数据。
  - **THIRD_PARTY_NOTICES.md 双重打包与运行时 SHA 缓存**：FPK 根目录与应用根目录均打包第三方许可证说明；内置 uv 增加运行时 SHA-256 完整性检验与进程内缓存。
- **资源耗尽保护**：单次处理正文限制为 2 MB，文本字符上限为 500,000 字符；模型推理采用进程级互斥锁保证串行，防止显存或内存击穿。

## 凭据卫生与 GitHub Secret Scanning 处置指引 (v0.6.14)

### 静态代码与测试凭据卫生策略

- **纯合成凭据契约与零容忍策略**：本仓库静态源码、配置文件与评测测试集严禁包含任何真实生产凭据，同时严禁包含任何第三方服务商特定形态（Provider-perfect shape）的静态字面量（即便为全 0、`SAMPLE` 或 `EXAMPLE` 亦一律禁止）。
- **防止扫描误报的模式碎片化 (Fragment Assembly)**：为防止 GitHub Secret Scanning 合作伙伴引擎将完整格式的合成测试向量误报为活跃凭据，测试套件中所有符合真实服务商格式的测试向量（如 Alibaba Cloud AccessKey ID/Secret、Google API Key、GitHub PAT、Slack Token）均采用运行时动态拼接（例如 `"ghp_" + "..."`、`"AIza" + "..."`）或通用合成命名（例如 `SYNTH_ACCESS_KEY_089_SAMPLE`），确保 Git blob 中不存储完整的活跃凭据形态。
- **仓库级静态凭据门禁**：通过 `tests/test_secret_hygiene.py` 实施持续静态检查，对全库文本文件执行正则扫描，杜绝第三方格式凭据回归进入静态源码或测试夹具。

### 历史 GitHub Secret Scanning 告警处置建议

针对历史提交曾触发的告警，处置人员请在 GitHub Security -> Secret scanning 界面进行人工审查关闭，无需且禁止执行凭据轮换（因为均为虚构字符串）：

1. **Alibaba Cloud AccessKey ID**（历史提交 `45f865a9`，`tests/test_credential_rules.py`）
   - **性质**：用于验证 LTAI 前缀及长度校验的纯合成测试向量，无真实阿里云账号关联。
   - **当前 HEAD 状态**：已重构为运行时字符串拼接（`"LTAI" + "..."`），当前主分支已无完整字面量。
   - **处置操作**：点击 **Close as** -> 选择 **"Used in tests"**（若无此选项则选 **"False positive"**）。
   - **处置备注模板**：
     ```text
     Synthetic test credential generated solely to validate secret detection.
     Never issued by or used with a real cloud account.
     Current HEAD no longer stores the complete credential literal.
     ```

2. **Google API Key**（历史提交 `622aac94`，`tests/test_builtin_hardening.py`）
   - **性质**：用于验证 AIza 前缀与 35-38 位规则的纯合成测试向量，无对应 Google Cloud 账户。
   - **当前 HEAD 状态**：已重构为运行时字符串拼接（`"AIza" + "..."`），当前主分支已无完整字面量。
   - **处置操作**：点击 **Close as** -> 选择 **"Used in tests"**（若无此选项则选 **"False positive"**）。
   - **处置备注模板**：
     ```text
     Synthetic test credential generated solely to validate secret detection.
     Never issued by or used with a real cloud account.
     Current HEAD no longer stores the complete credential literal.
     ```

3. **Alibaba Cloud AccessKey Secret**（历史/原基线 `tests/fixtures/privacy_benchmark_v2_100.jsonl` 中的 `case_089`）
   - **性质**：原测试用例包含形如 `OSS_SECRET=SampleOnly...` 的合成样例，被扫描器判定为疑似阿里云访问凭据。
   - **当前 HEAD 状态**：在 v0.6.7 中已彻底净化，替换为通用合成命名 `ACCESS_KEY=SYNTH_ACCESS_KEY_089_SAMPLE` 与 `SECRET=SYNTH_SECRET_KEY_089_SAMPLE001`，同时保持内置规则能精准识别为 `SECRET`。
   - **处置操作**：点击 **Close as** -> 选择 **"Used in tests"**（若无此选项则选 **"False positive"**）。
   - **处置备注模板**：
     ```text
     Synthetic test credential generated solely to validate secret detection.
     Never issued by or used with a real cloud account.
     Current HEAD no longer stores the complete credential literal.
     ```

### 为什么禁止重写 Git 历史

本项目明确**禁止**采用 `git filter-repo` 或 `BFG Repo-Cleaner` 强行重写 Git 提交历史并 force push：
1. 上述告警经审查已确认为纯合成测试代码，从未存在真实私钥或凭据泄露风险；
2. 历史提交 SHA-1 是 Benchmark 基准、技术架构文档以及对外版本发行的可审计性锚点，强行重写历史将破坏所有存量报告的溯源链条；
3. 在 GitHub 官方安全面板中正规标记为 `Used in tests` / `False positive` 是开源安全标准治理流程。

## 已知限制与使用建议

- 任何自动化隐私检测算法均存在极小概率的漏检或误检。高敏法律、财务或个人隐私数据在外发前，请务必利用“人工复核”界面进行最终核验。
- 浏览器映射默认保存在会话内存中，关闭或刷新页面即销毁。如需保存，请使用页面内置的 AES-256-GCM 加密导出功能。
- 在资源受限的 NAS 设备上，建议优先使用零内存占用的纯规则模式；在拥有独立 NVIDIA 显卡的 NAS 上，可启用 CUDA 设备加速。
