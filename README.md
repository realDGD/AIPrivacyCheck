# AI 脱敏器（AI Privacy Check）

面向飞牛 fnOS 的本地文本隐私闸门：先检测并把隐私字段替换为稳定占位符，再将脱敏文本交给外部 AI；AI 回复后，可在当前页面把原值精确放回。

当前版本：`0.6.12`（fnOS Native 原生应用）

## 已实现功能

- **托管 Python 基础运行时收敛与内置 uv 供应链闭环 (v0.6.12)**：
  - **内置官方 Astral musl uv 独立供应链**：彻底修复真实 fnOS 宿主机 PATH 无 `uv` 工具阻塞隔离环境创建与依赖修复的 P1 故障。FPK 安装包开箱内置官方 Astral 静态链接 musl ELF 二进制（`app/bin/linux-x86_64/uv` 与 `app/bin/linux-aarch64/uv`），经 SHA-256 强校验，实现 100% 离线、零网络、零宿主机 root 权限、零系统 PATH 依赖。
  - **四阶段事务性原子切换与严格回滚保护**：运行时重建遵循 `old -> backup PASS, staging -> final FAIL`, `manifest write FAIL`, `final probe FAIL`, `rollback itself fails` 四重故障防御。在最终探针与清单完全验证通过前，旧环境备份绝对不被删除；若回滚本身遭遇异常，永久保留 `venv.old.<timestamp>`，绝不灭失用户环境。
  - **Base ML Contract 8 项基础依赖健康探测与版本门禁**：目标运行环境在投入服务前必须通过 8 项基础 ML 依赖探测（`torch`, `modelscope`, `numpy`, `packaging`, `tqdm`, `transformers`, `accelerate`, `gliner`），并对 `transformers >=4.51,<5` 实施强制拦截约束。
  - **PyTorch 2.6.0 精准锁定与真实版本落盘**：明确锁定 PyTorch 2.6.0，同时在 `runtime-manifest.json` 与状态 API 中如实记录探测所得真实依赖版本。
  - **历史旧环境双重健康准入与无损接管**：仅当历史隔离环境同时通过 17 项 Python 原生能力契约与 Base ML Contract 时方执行无损升级接管，否则安全触发隔离重建。
  - **零模型重载与零权重篡改契约**：修复与重建全流程零网络下载（0 bytes from ModelScope）、零模型权重修改。
- **uv 托管 Python 基础运行时与两级环境修复架构 (v0.6.11)**：
  - **彻底脱离 fnOS 系统 Python 依赖**：使用 uv 托管全功能标准 CPython 3.12.9 作为隔离计算环境的底层解释器（安装存放于 `${DATA_DIR}/python/installations`，uv 缓存收口至 `${DATA_DIR}/cache/uv`），彻底解决宿主机系统 Python 缺少 `_lzma`, `_bz2`, `_ssl`, `_sqlite3` 等底层 C 扩展导致的运行环境兼容故障。
  - **17 项底层能力契约与健康探测**：定义并严格检验 17 项原生标准库模块能力（`_lzma`, `_bz2`, `_ssl`, `_sqlite3`, `ctypes`, `zlib`, `hashlib`, `json`, `multiprocessing`, `subprocess`, `venv`, `ensurepip` 等），能力探针在隔离子进程中执行，控制面零侵入。
  - **两级修复架构与事务性重建**：支持 Level 1 增量模型依赖修复（秒级补齐，不重建 venv）与 Level 2 事务性环境重建（预检磁盘剩余空间 CPU >= 2GB / CUDA >= 4.5GB、临时目录 `venv.rebuild-<timestamp>` 构建、多模型专属依赖聚合、真实冒烟测试核验、原子切换目录并清理旧环境，失败安全回滚原环境）。
  - **零模型重载与零权重篡改契约**：环境修复与重建全程严禁触发模型重新下载，严格保持 `${DATA_DIR}/models/*` 模型权重文件不变，升级过程 100% 零网络带宽消耗、零模型重载。
  - **Schema v3 运行时清单与旧环境无损接管**：旧环境经核验能力完整后自动无损接管升级为 `schema_version: 3` 清单，无需重建。前端控制面板提供「重建/升级运行环境」直观操作与底层能力缺失预警。
- **Runtime 迁移、独立检测器控制与长文本正确性加固 (v0.6.10)**：
  - **历史版本运行时依赖迁移与一键修复**：解决历史版本已安装的 PyTorch 运行时（`torch-cpu`/`torch-cuda`）在升级后缺失新增模型特定依赖契约（如 SiameseUIE 历史 venv 缺少 `addict`）的兼容性问题。通过 `probe_model_runtime_dependencies` 轻量探测缺失包，并在前端暴露 `[修复运行环境]` 按钮（API: `POST /api/model/runtime/repair`），仅对目标 venv 增量补装 pip 依赖并执行冒烟测试，严禁重新下载模型权重或删除模型。
  - **独立检测器控制与 Slots 契约解耦**：彻底拆分单一模型开关为分层控制：内置规则（Built-in Rules，常开）、中文语义提取（Chinese IE，常开基线）、GLiNER 通用 PII（`glinerToggle`，默认开启，localStorage 持久化）、MemPrivacy 深度语义隐私（`memprivacyToggle`，默认关闭，首次开启弹出资源消耗确认，localStorage 持久化）。服务端 `active_slots` 强制包含 `built_in` 与 `chinese_ie`，兼容遗留 `use_model=True` 请求（仅激活 GLiNER）。
  - **规则引擎日期跨度与结构化密码修复**：修复 `CN_BIRTH_DATE` 贪婪截断导致只保留部分日期的缺陷（`1992年11月18日` 不再截断为 `1992年1`）；新增结构化密码规则（`PASSWORD` 实体类型，优先级 121 高于用户名，支持换行及多语言前缀），配合 `_is_valid_password_value` 拦截变量与占位符，保持 0/388 严格负样本 FPR 零误报。
  - **长文本验收与多行 OTP 防误报门禁**：建立 `tests/fixtures/long_context_manual_acceptance.txt` 涵盖 20 大测试场景，严格执行全库零提供商形态凭据规范（`SYNTH_...`）；GLiNER 增加多行换行与 12..19 位长度门禁，杜绝多行 OTP 恢复代码块误报为 `CREDIT_CARD`，并记录 `Project Aurora` 为已知候选通用实体。
- **Benchmark 基础设施与信任边界最终加固 (v0.6.9)**：
  - **Selective Downloader 同尺寸损坏文件强制修复**：修复当本地已存在同尺寸损坏文件时绕过网络重新下载的缺陷；`download_single_file` 引入 `force_download=True` 参数，原子流式下载至 `.tmp_download` 校验哈希并原子替换，并在 ModelScope 可变 revision 内容变更时记录告警与 `upstream_content_changed` 标记。
  - **Prediction Cache Manifest 强制完整性契约**：`cache-manifest.json` 成为缓存有效性法定要件；`has_valid_cache()` 强制要求 manifest 存在并比对 `predictions_sha256` 与预测条目数；`load()` 默认对无清单的遗留缓存抛出 `LegacyUnverifiedCacheError`，提供 `--allow-legacy-unverified-cache` 命令行安全兼容选项。
  - **Manifest Schema 跨平台防路径穿越与大写哈希拦截**：Manifest 校验严格拦截 Windows 盘符路径（`^[A-Za-z]:`）、UNC 网络路径（`\\server\share`, `//server/share`）、绝对路径及各类 `..` 目录穿越；强制约束 `sha256` 必须为规范小写 64 位十六进制（`^[0-9a-f]{64}$`），严格拒绝大写十六进制。
  - **CRITICAL_ENTITY_TYPES 规范收敛与文档对齐**：明确代码中定义的 18 类核心高危实体单一事实来源（`CN_ID_CARD`, `GOVERNMENT_ID`, `US_SSN`, `CN_BANK_CARD`, `CREDIT_CARD`, `CN_PHONE_NUMBER`, `PHONE`, `private_phone`, `EMAIL`, `private_email`, `SECRET`, `secret`, `PASSWORD`, `API_TOKEN`, `PRIVATE_KEY`, `DATABASE_URI`, `PASSPORT`, `CN_PASSPORT`），严格解耦法定高危与普通脱敏实体。
  - **全库静态凭据零容忍（Zero Provider-Perfect Literals）**：移除针对提供商特征静态凭据的任何豁免，全仓库工作区内提供商形态（如 `AKIA...`, `github_pat_...`, `LTAI...`, `ghp_...`）静态字面量彻底清零（0 个）；所有测试夹具均转换为标准通用合成前缀（`SYNTH_...`）。
  - **双重最终冻结确认**：Built-in v2 规则与阈值冻结（0/388 严格负样本 FPR，100/100 幂等性，0 占位符命中），Benchmark 评测基础设施（下载器、完整性核验、缓存契约、评分器）全指标达标并正式进入 STABLE / FROZEN 最终冻结状态。
- **Benchmark 评测基础设施最终冻结 (v0.6.8)**：
  - **Prediction Cache 模型指纹真实性验证**：`scripts/benchmark_cache.py` 严格校验模型磁盘真实内容哈希，优先比对 `download-manifest.json` 记录的 SHA-256 校验和与文件大小，无清单时回退至流式计算；若检测到同尺寸内容篡改或文件损坏直接抛出 `ModelIntegrityError` 拒绝虚假缓存复用。
  - **单源模型完整性验证模块**：创建 `scripts/model_integrity.py`，作为 `selective_downloader.py` 和 `benchmark_cache.py` 共享的单一真实来源（Single Source of Truth），统一文件校验、大小比对及复合内容哈希生成。
  - **Selective Downloader 兼容性与安全加固**：修复 Python 3.12/3.13 下缺少 `Any` 导入的兼容性缺陷，引入真实子进程测试门禁；加强 `download-manifest.json` schema 规范（64位小写 hex SHA-256、非负大小校验），建立防路径穿越安全防御（绝对路径与 `../` 严格拦截）。
  - **ModelScope Revision 真实来源记录**：明确记录 `requested_revision` 与 `resolved_revision: null`，如实标注本地内容指纹契约（Local content integrity fingerprint）。
  - **Prediction Cache 歧义隔离与数据校验**：父级缓存目录存在多个有效签名时强制报错 `AmbiguousCacheError`，写入时生成并落盘 `cache-manifest.json`，读取时强制校验预测条目数与 `predictions_sha256` 完整性，篡改即报 `CacheCorruptedError`。
  - **全仓库凭据卫生 0 违规**：全仓库所有文本文件（`.py`, `.json`, `.jsonl`, `.md`, `.sh` 等）静态凭据扫描 0 活跃/扫描器形态违规，高熵虚构凭据统一替换为零熵惰性虚构凭据。
  - **Built-in v2 与评测基础设施双重冻结**：Built-in v2 规则与阈值冻结（0/388 严格负样本 FPR），Benchmark 基础设施正式进入 STABLE / FROZEN 最终冻结状态。
- **Benchmark 可复现性与仓库凭据卫生加固 (v0.6.7)**：
  - **预测缓存运行时与内容哈希绑定**：`scripts/benchmark_cache.py` 将持久化 Raw Prediction Cache 严格绑定宿主机关键环境（`python`, `torch`, `transformers`, `modelscope`, `gliner`）与模型真实文件的 SHA-256 内容哈希（优先从 `download-manifest.json` 提取，无清单时回退流式哈希，杜绝 `mtime` 虚假失效）；引入 `cache_signature` 16 进制目录哈希，使不同配置与运行时的合法缓存能够安全并存。
  - **已有模型下载完整性校验契约**：`scripts/selective_downloader.py` 修复对已有模型“只要文件存在就判 complete”的重大缺陷，新增 `verify_existing_model_integrity` 严格比对文件尺寸与 SHA-256 校验和；单文件损坏精确报错，`--download-missing` 仅下载损坏文件；单文件下载增加 `try ... finally` 确保 `.tmp_download` 临时文件在异常或中断时原子清理。
  - **解耦 Redactable FN 与 Critical FN**：修正 Benchmark 评测中将所有 `should_redact=true` 漏检统称为“Critical FN”的命名与统计缺陷，明确区分为 `Redactable FN`（全部应脱敏实体的漏检）与 `Critical FN`（集中定义的法定高危直接标识符与敏感凭据漏检），阈值扫频表格同步升级；全面支持使用已有缓存纯离线重评分（无需 GPU 或模型推理）。
  - **清理 GitHub Secret Scanning 触发字面量**：彻底清除 `case_089` 中的提供商完整格式字面量，重构为安全合规的合成凭据；测试套件中的敏感前缀（`ghp_`, `gho_`, `ghs_`, `xoxr-`, `xoxs-`, `AIza`, `LTAI`）全量采用运行时动态拼接（Fragment Assembly），消除了静态扫描触发点。
  - **仓库级凭据卫生门禁**：新增 `tests/test_secret_hygiene.py`，持续守护静态测试夹具与脚本目录，阻断任何完整形式的第三方测试凭据进入 Git blob；在 `docs/security.md` 中给出历史告警的标准关闭指南（"Used in tests" / "False positive"）及禁止重写 Git 历史的合规依据。
  - **保持 Built-in v2 规则冻结**：Built-in v2 规则与校验器完全冻结，严格负样本保持 0/388 (0.0%) 零误报基准。
- **Benchmark v2 评分修复与评测基准可复现性加固 (v0.6.6)**：
  - **语义评分与显式负样本逻辑修复**：修复 `score_semantic()` 在显式负样本（`sensitive=false`）黄金实体上的计分逻辑——模型重叠检出时仅增加 `overreach` 与 `fp`，绝对不再错误累加 `tp`；重叠与类型判断严格限定在 `sensitive=true` 实体集合内。
  - **检测与脱敏指标严谨重命名**：将原 `redaction_acc`（脱敏准确率）更名为 `redaction_eligibility_coverage`（脱敏资格覆盖率，命令行输出 `redCov`），准确度量在所有标注为 `should_redact=true` 的法定/敏感实体中被系统成功发现的比例；保留 `redaction_acc` 作为向后兼容别名。公共实体（如 10086、8.8.8.8 等 `should_redact=false`）在 Layer A 中正确计入 Detection TP（检出正确），绝不再误记为 FP。
  - **GLiNER 标签顺序确定性**：将 GLiNER 实体标签列表由无序的 `list(set(...))` 升级为冻结的有序元组 `GLiNERDetector.GLINER_LABELS`，彻底杜绝 Python hash 随机化导致的轻微推理波动；单一推理基准与生产环境完全保持一致。
  - **GLiNER 阈值单次推理验证与离线重评分**：GLiNER 在基准测试中以最低阈值（0.30）执行单次推理，并由可持久化的 Raw Prediction Cache 支持毫秒级离线阈值扫频（0.35 - 0.65）；实证表明在 0.50 阈值下消除自然叙述下用户名混淆且维持极佳 Critical FN，确认为最优生产阈值并再次冻结。
  - **Raw Prediction Cache 评测缓存**：建立 `benchmark-cache/` 持久化预测缓存，深度绑定语料哈希、模型哈希、运行时版本与参数签名，未来仅调整评分逻辑时无需反复加载或重新运行耗时模型。
  - **模型按需精简下载（Selective Downloader）**：严禁全量下载 ModelScope 快照，仅下载推理必需的 PyTorch 模型文件（`safetensors` / `bin`、配置、分词器），自动过滤 ONNX、README 与 Git 元数据，节省 60%+ 带宽；基准测试脚本强制要求显式 `--download-missing` 参数。
  - **Built-in v2 负样本基准冻结核验**：统一全库 negative 语料统计口径为 407 条（388 条严格负样本 + 5 条 RFC-2606 保留文档域名 + 14 条高置信虚构样本），Built-in 规则在严格负样本上的假阳性率严格保证为 `Strict-negative FPR = 0 / 388 (0.0%)`。
  - **历史模型评分状态标注**：历史模型（MemPrivacy、OpenAI Privacy Filter、AIguard、Qwen 等）在 v0.6.6 前的评测得分明确标注为“Historical / pre-v0.6.6 scorer”，其中涉及负样本重叠的 MemPrivacy 语义评分标记为“INVALIDATED FOR FINAL COMPARISON”，杜绝不同口径评分混淆。
- **Base Runtime Contract 与预加载安全门 (v0.6.5)**：
  - **Base Runtime Contract**：运行时"已验证即跳过"路径新增基础依赖契约核查（modelscope / torch / transformers>=4.51,<5 / accelerate / gliner / numpy / packaging / tqdm）。历史遗留 runtime（如 transformers 5.16.1）即使 torch Probe 通过也会被识别，并仅对违约包做增量修复（pip install "transformers>=4.51,<5"），绝不重装 torch、不重建 venv；缺 torch 的损坏环境拒绝增量修复并提示重建。
  - **预加载模型远程代码安全门**：新增 `privacy/model_security.py`，在任何 worker 加载模型前对 configuration.json 净化 `allow_remote`/`plugins` 声明；GLiNER / MemPrivacy / SiameseUIE 的 load+detect、冒烟测试与安装期净化共用同一实现。v0.6.3 及更早版本安装的存量模型在升级后首次使用时即被自动净化。
  - **GLiNER 阈值单一来源与再调优**：`GLiNERDetector.GLINER_DEFAULT_THRESHOLD` 成为唯一阈值定义（生产与基准共享）；在 100 文档分层语料上重扫 0.35-0.65，Detection F1 于 0.50 达峰（P 55.2 / R 30.9 / F1 39.6），PII-free FPR 与 USERNAME↔PERSON 混淆与 0.55 持平，生产阈值据此调整为 0.50。
- **Benchmark v2 分层评测体系 (v0.6.5)**：
  - **100 文档冻结语料**（`tests/fixtures/privacy_benchmark_v2_100.jsonl`，确定性生成器 + 固定 seed）：55 中文 / 35 英文 / 10 结构化混合（JSON/YAML/SQL/Python/Shell/URI/日志/邮件/配置/Markdown）；14 篇超过 3000 字符；≥10 篇准标识符组合（quasi-identifier）；≥15 条语义案例（含"去过医院但未说明原因"类显式负样本）；≥10 条合成凭证/代码案例（不含任何真实秘密）；≥10 条 Unicode 压力案例（Emoji/ZWJ/SIP CJK/组合变音/阿拉伯语 RTL）。
  - **Detection 与 Should Redact 完全分层**：每个实体独立标注 `should_redact` 与 `context_class`；公共联系人（10086 / 8.8.8.8 / test@example.com / 公开办公地址）按"正确检测、不应脱敏"计分，绝不再误记为检测 FP；PII-free FPR 仅统计零检测 gold 文档。
  - **语义层独立评分**：`semantic_privacy` 层按跨度覆盖单独计分，模型不因 taxonomy 范围（如不含糖尿病/离婚/负债）而被记为 PII F1 失败。
  - **vault-engine 隔离基准 harness**（`scripts/benchmark_vault_engine.py`）：以库方式接入固定 commit 的 vault-engine，进程内本地模型 provider（无 Ollama、无云端端点、无 pip 安装），内置 Vault 往返一致性（要求 100%）、稳定 token 与碰撞检查、20 篇最难文档稳定性重测。
- **MemPrivacy / CUDA 并发安全与整请求语义预算加固 (v0.6.4)**：
- **模型运行时依赖契约与 SiameseUIE 安装链修复 (v0.6.4)**：
  - **模型专属运行时依赖契约（Model-specific Runtime Dependency Contract）**：在 Model Catalog 的 `ModelDescriptor` 上新增 `runtime_dependencies` 声明字段，并新增 `ensure_model_runtime_dependencies` 安装阶段：先复现真实 fnOS 故障（共享 torch-cuda 运行时 Probe 通过但 SiameseUIE Pipeline 报 `No module named 'addict'`），经实证确认 ModelScope 1.40 将 `addict`/`datasets`/`scipy`/`Pillow`/`simplejson`/`sortedcontainers` 全部移入 extras 而非核心依赖后，按模型声明、增量补装缺失依赖（`importlib` 探测已满足项即快速跳过，绝不重装 PyTorch 或重建 venv）。现有已安装运行时升级后同样自动补齐。
  - **共享运行时 transformers 兼容区间钉扎**：基础运行时固定 `transformers>=4.51,<5` —— 下限来自 MemPrivacy/Qwen3 权重（`Qwen3ForCausalLM` 需要 >=4.51），上限来自 ModelScope 旧式 NLP pipeline 依赖的 `transformers.onnx` 模块（transformers 5.x 已移除）。
  - **拒绝模型目录远程代码执行**：安装/导入时对 `configuration.json` 执行净化，移除 `allow_remote`/`plugins` 声明（官方 iic 模型携带 `allow_remote: true` 会令新版 ModelScope pip 安装模型自带 requirements.txt 并执行目录内任意 .py）；worker 坚决不传 `trust_remote_code`，仅经 ModelScope 内建 pipeline/model 类加载。
  - **SiameseUIE 输出结构适配修复**：适配 ModelScope 新版真实输出形状（`{"output": [[{"type","span","offset"}]]}` 按 schema 分组嵌套列表 + 半开区间 `offset`），修复 pipeline 加载成功但实体抽取恒为空的问题；真实端到端冒烟（下载官方权重 → 生产 worker 推理）已验证通过。
- **高确定性凭证规则扩展与误报治理 (v0.6.4)**：
  - **新增 JDBC 连接串规则**：覆盖 `jdbc:mysql:`/`jdbc:postgresql:`/`jdbc:oracle:`/`jdbc:sqlserver:`/`jdbc:mariadb:`（含 loadbalance/replication/sequential 等 inner qualifier 变体），并为既有 `xxx://` 规则添加 `(?<!jdbc:)` 防双杀；URI 内嵌 `user:password@host` 的 overlap 回归受保护（EMAIL 等小规则不得吞并）。
  - **新增云厂商与协作平台凭证规则（全部查证一手官方文档）**：阿里云 AccessKey ID（`LTAI` 前缀）、腾讯云 SecretId（`AKID`/`IKID` 36 位）、Slack 官方在册前缀（`xoxb-`/`xoxp-`/`xapp-`/`xwfp-`；`xoxa-`/`xoxr-` 因现行官方文档未定义而刻意排除）、GitHub fine-grained PAT（`github_pat_` + 22 + `_` + 59 官方结构；`ghp_`/`gho_`/`ghu_`/`ghs_`/`ghr_` 已有规则继续保留）。
  - **JWT 结构校验器**：三段 Base64URL 之外新增 header 解码校验（必须为含字符串 `alg` 的 JSON 对象，`typ` 可选），不做签名验证；垃圾三段串不再误报。
  - **中文上下文误报修复**：移除误伤「我的密码是…」「我的用户名是…」的 CJK 前置 lookbehind；用户名值支持中文（有显式标签约束）；病历号/工号/学号等 ID 规则支持空格分隔；安全码支持「安全码」中文标签；阿拉伯语人名模式支持单词与「،」分隔。
- **Benchmark v2 中文上下文隐私种子与模型基准能力 (v0.6.4)**：
  - **61 条中文 Contextual Privacy Seed**（`tests/fixtures/contextual_privacy_seed.jsonl`，25% 负样本/占位符）：覆盖用户名↔人名混淆、公开 vs 私有（10086/8.8.8.8/test@example.com/公开办公地址）、医疗/财务/关系语义样本、Emoji/ZWJ/SIP CJK/阿拉伯语 RTL/JSON/YAML/SQL/URI 等 Unicode 压力样本；schema 校验（span↔文本一致、UTF-16 换算一致性、重复 ID、overlap 语义）由单元测试把关。
  - **指标体系**：Exact Span P/R/F1、PII-free FPR、上下文误报分类、字符泄漏（Character leakage）、过度脱敏（Over-redaction）、逐模型延迟（`scripts/benchmark_v2.py` 内置评估 + `scripts/benchmark_scoring.py` 共享评分）。
  - **模型基准 Harness**（`scripts/benchmark_models.py`）：对已安装模型运行相同数据集，不自动下载（未安装明确 SKIP）、不触碰生产服务；GLiNER 适配器镜像生产标签集/阈值/USERNAME 过滤器，RANER 走 ModelScope NER pipeline，MemPrivacy 复用生产 worker 提示词与 JSON 解析；记录墙钟、进程峰值 RSS，CUDA 显存仅在真机报告。
  - **基于基准的 GLiNER 阈值调优**：种子集上阈值 0.40→0.55 在召回不变（14.5%）的前提下将模型假阳性下降约 5 倍（6/15 → 1/15 量级），生产阈值据此调整。
- **MemPrivacy / CUDA 并发安全与整请求语义预算加固 (v0.6.4)**：
  - **Worker 永久退役契约与防复活安全**：在 `RuntimeWorkerProcess` 引入 `self._retired` 与 `retire()` 方法及 `WorkerRetiredError` 契约。当 Worker 因换代、驱逐或主动停止被移除时，严格调用 `retire()` 并永久终止进程；任何已停退役 Worker 绝不允许被后续并发请求重新唤醒，彻底根除高并发下孤儿 Worker 驻留后台窃取 GPU 显存的问题。
  - **跨模型 CUDA 执行全局协调器**：引入 `WorkerClient.cuda_execution_session` 上下文协调锁，在多模型级联（MemPrivacy、GLiNER、SiameseUIE 及冒烟测试）中统一排他管理物理 GPU 访问；在 MemPrivacy 独占会话期间，阻止并发 GPU 请求拉起 GLiNER，彻底杜绝导致 CUDA OOM 的时序竞态。
  - **整请求语义推理时间预算体系**：建立整请求语义推理预算与 Deadline 截止时间体系（CUDA 请求总预算 240 秒，CPU 请求总预算 480 秒）。在锁等待阶段自动扣除消耗时间，若系统繁忙超时则快速返回友好状态；在长文本分块处理循环中动态扣减剩余时间配额，彻底根治大文本长达十数小时的同步阻塞。
  - **单块超时保护与部分结果安全保留**：多分块长文本遇到单块超时或预算耗尽时，已成功推理完成的分块实体被完整保留并返回，并在前端与日志中明确提示完成度（如“MemPrivacy 仅完成 1/2 个语义分块，结果可能不完整”），兼顾用户体验与隐私覆盖率。
  - **超长生成截断统一去重告警**：对多个分块发生的生成上限截断统一计数汇总，输出单条精简告警，杜绝大量冗余告警刷屏。
  - **精准故障分类（CPU 内存 vs CUDA 显存）**：严格区分 CPU 宿主机内存耗尽（`MemoryError` / `bad_alloc`）与 GPU 显存不足（`CUDA out of memory`），避免 CPU 内存压力被误报为显存问题。
  - **消除 Worker 终止过程的锁争用**：Worker 终止过程移至 `WorkerClient._lock` 外部执行，避免子进程退出等待阻塞其他线程的常规状态查询。
- **MemPrivacy 语义隐私推理加固与显存/超时治理 (v0.6.3)**：
  - **Exclusive CUDA 独占显存调度**：在显存受限的 GPU 设备（如 Tesla P4 8GB）上运行 MemPrivacy 推理前，自动驱逐并终止其他处于常驻状态的 CUDA worker（如 GLiNER），推理完成后即刻通过 `finally` 释放 MemPrivacy 进程及其占用的全部 GPU 显存，彻底根治顺序推理累积导致的 CUDA OOM。
  - **CUDA OOM 优雅降级与自愈释放**：捕获 Worker 进程报告的致命 CUDA OOM（`OutOfMemoryError` / `CUDA error: out of memory`），立即物理终止 worker 释放显存并将状态置为 FAILED，同时以温和 warning 降级提醒用户（“MemPrivacy 可用显存不足，已终止语义模型并释放显存，其他检测结果不受影响”），绝不阻断基础规则或普通模型的检测结果。
  - **设备感知自适应超时**：将推理超时与计算设备解耦，针对 CPU 模式推理耗时特点将 MemPrivacy 超时由 180 秒放宽至 360 秒（启动超时 150 秒），CUDA 模式保持 180 秒超时（启动超时 120 秒），根除 CPU 环境慢速推理被 180 秒硬超时中断的缺陷。
  - **有界生成 Token 预算控制**：根据文本长度动态分配有界生成预算（短文本 256 tokens、中长文本 384 tokens、长文本 512 tokens，严格上限 <=512 tokens），彻底消除 2048 tokens 默认过度分配带来的 KV 缓存膨胀与推理超时风险。
  - **长文本平滑重叠分块与全局坐标映射**：对超过 3500 字符的长文本自动采用 3000 字符分块与 200 字符平滑重叠切分，并优先在自然语句边界切分；分块抽取结果通过全局坐标准确回贴并进行跨块实体去重，杜绝长文本内存耗尽与跨块坐标错位。
  - **模型目录规范与 CPU 慢速模式温和提示**：明确标注 MemPrivacy 1.7B 在 GPU（>=6GB）与 CPU 下的运行特性，4B 模型明确标注需要最低 >=12GB（推荐 16GB+）显存并警示 <=8GB 显卡上的极高 OOM 风险；在 Web 前端为处于 CPU 模式下的语义隐私槽位提供温和 inline 提示，禁止弹窗阻断。
- **GLiNER 中文误报抑制与前端交互布局加固 (v0.6.2)**：
  - **降低 GLiNER 中文 USERNAME 误报**：增加严格的用户名形态学与上下文环境过滤（`_is_plausible_username`），针对中文自然语言叙述默认拒绝提取为 USERNAME，杜绝整句自然语言与标点被错误遮盖；仅允许标准 ASCII token 或具有显式账号上下文的合规 span（reduce GLiNER username false positives in Chinese narrative text）。
  - **修复 marker 右键菜单生命周期**：在脱敏预览 `mouseup` 事件中严格限定仅左键生效（`event.button === 0`），彻底修复安全副本 marker 右键点击弹出菜单后、松开按键菜单立即误关闭的交互缺陷（fix marker context-menu lifecycle）。
  - **长文本检测工作区三栏等高与内部滚动**：桌面端 `#maskView` 三栏采用 640px 严格等高模型，输入框与脱敏预览区高度受控内部纵向滚动，杜绝因文本变长导致各栏失衡或向外无节制拉伸；Restore 恢复视图保持原生自适应弹性（stabilize long-text detection workspace）。
  - **双向联动容器级定位**：彻底废除引发外层 `document` 突兀跳动的 `scrollIntoView`，封装容器级相对滚动 `scrollElementIntoContainer`，并结合 `focus({ preventScroll: true })`，确保 marker 与卡片联动时平滑仅在内部容器滚动，页面整体垂直滚动位置保持不动（keep linked-entity navigation inside panel scroll containers）。
  - **审查卡片高亮横向溢出消除**：重构 `@keyframes entity-link-pulse` 移除所有 `translateX` 几何平移，配合 `.entity-list` 显式声明 `overflow-x: hidden`，彻底解决动画期间瞬间横向滚动条闪烁问题（remove highlight-induced horizontal overflow）。
  - **范围重选视觉降噪**：去除误导性的虚线拖拽边框（dashed outline），改用高亮黄色背景与轻量内嵌投影，消除用户“可拖拉拉伸”的误解（simplify range reselection highlight）。
- **Unicode 字符契约加固与安全副本保障 (v0.6.1)**：
  - **统一 Unicode / UTF-16 契约**：后端实体序列化直接输出精确的 `start_utf16` 与 `end_utf16`，前端统一使用 UTF-16 code units 字符切片，彻底根除包含 Emoji、生僻字、合字及复杂多语言文本时的占位符偏移错位与尾部字符残留缺陷。
  - **占位符统一分配与同值复用**：自动检测与手动划词标注统一使用 `allocateReplacementToken`，相同文本与实体类型严格复用同一个脱敏占位符与保险箱映射，彻底消除同值多 token 冲突。
  - **安全副本校验与故障失效阻断**：安全副本生成时执行严格的占位符跨越重叠与原文碰撞校验，一旦检测到异常立即失效脱敏状态并禁用“复制脱敏文本”与“复制 Prompt”，绝不让损坏的脱敏文本流出。
  - **交互状态机与重选健全性**：重新检测、清空、切换视图时全局重置重选（retargeting）、悬浮操作栏（popover）与划词选区；支持在已禁用（`enabled: false`）的实体范围上重新划词标注；模型安装与导入流程在异常分支下依然可靠重置检测器。
- **四级插拔式检测与策略引擎**：
  1. **确定性多语言与中文规则（Tier 1，`BuiltInRuleDetector`）**：身份证号、手机号、国际电话（E.164）、固定电话、银行卡、护照、统一社会信用代码、车牌、姓名、地址、邮箱、账号/单号、出生日期、社交账号、IPv4、完整 IPv6、MAC、BIC/SWIFT、IBAN、US SSN、数据库连接串（`DATABASE_URI`）、私钥与各类 API Token/凭证。内置严格静态敏感度等级（PL4/PL3/PL2），不可被模型随意降级。
  2. **中文信息抽取引擎（Tier 2，`ChineseIEDetector`）**：针对中文姓名、复杂行政区划拓扑与建筑地址进行基于语言学特征和安全跨度对齐的抽取（`safe_sequential_span_alignment`，彻底杜绝同名多次出现时的偏移碰撞）；内置零依赖启发式抽取，支持可选适配 ModelScope 社区模型 `siamese-uie`（PyTorch 架构）。
  3. **通用 PII 实体抽取（Tier 3，`GLiNERDetector`）**：支持 ModelScope 社区模型 `gliner-pii-edge`（轻量 edge 版，~310MB）与 `gliner-pii-base`（高精度版，~850MB），利用模型原生字符偏移实现零偏移漂移的跨语言 PII 抽取。
  4. **深度语义隐私推理（Tier 4，`MemPrivacyDetector`）**：支持 ModelScope 社区模型 `memprivacy-1.7b-rl`（约 3.4GB）与 `memprivacy-4b-rl`（约 7.8GB），结合上下文对深层隐性隐私（健康状况、人际隐私、资产交易）进行逻辑判定，使用全自主编写的 Apache-2.0 规范系统 Prompt 与容错结构化抽取。
- **PL1 - PL4 隐私分级策略与仲裁**：
  - **PL4（核心密码凭据）**：数据库连接串、私钥、API 令牌、系统密码。
  - **PL3（高敏合规凭证）**：身份证、护照、银行卡、医疗病历、财务资产、精确轨迹。
  - **PL2（可识别个人信息）**：姓名、手机、邮箱、地址、社交账号、IP、车牌。
  - **PL1（低敏偏好标签）**：个人公开偏好与低关联职业标签。
  - **层级仲裁机制**：校验位强规则 > 严格规则 > 专用实体抽取模型 > 生成式推理模型，高置信法定凭据永不被模型覆盖。
- **脱敏预览与审查卡片双向联动 & 手工划词标注 (v0.6.0)**：
  - **双向高亮聚焦联动**：脱敏后呈现交互式脱敏文本与审查卡片列表；点击脱敏文本中的标签按钮可平滑滚动、置顶并高亮对应审查卡片；审查卡片点击或点击卡片内 `↔` 按钮反向聚焦并脉冲高亮脱敏文本标签，完整适配 `prefers-reduced-motion`。
  - **划词标注隐私条目**：在脱敏预览文本中选中漏网或自定义隐私文本，即可触发紧凑悬浮操作栏，支持一键创建通用隐私条目（`⟦隐私条目_01_XXXX⟧`）或通过 16 种标准实体类型选择器指定类型；支持右键/操作菜单，提供“重新选择范围”、“删除条目”、“定位”等全套管理功能。
  - **范围重选交互模式**：支持对误标或多选的条目点击“重新选择范围”进入局部重选模式，系统仅将目标实体临时恢复为原文并高亮框选，其余区域保持占位并柔和弱化；支持用户重新拖拽划选、首尾空白自动微调、防重叠防交叉实体校验，并支持 `Esc` 或取消一键无损快照回滚。
- **控制面与执行面彻底隔离架构（Zero ML Control Plane）**：
  - **主控进程零 ML 依赖**：fnOS Native 主进程（Python 3.12）严格不 import 任何重量级 ML 库（`torch`, `transformers`, `gliner`, `modelscope`, `paddle`），保证控制面极速响应与低内存占用。
  - **JSONL IPC 隔离 Worker 与高可靠生命周期**：所有神经网络模型均由独立 Python Worker 子进程通过标准输入输出行式 JSONL 通信（`privacy/workers/`）。v0.5.2 引入换代竞争隔离机制（彻底杜绝孤儿 EOF 误杀新建 Worker 进程）、基于队列的真实超时中断、后台守护式 stderr 消耗（彻底杜绝 OS 管道死锁）、跨进程单模型互斥锁（`fcntl.flock`）与状态机就绪握手，并支持故障单次自动崩溃重启。
  - **HardwareProbe 与 RuntimeManager**：直接通过 `nvidia-smi` 独立子进程探测主机 GPU，运行时环境统一至 PyTorch 体系（`torch-cpu` / `torch-cuda`），支持 Auto 模式平滑降级与显式 CUDA 缺失规范语义（`actual_device="none"`、`ready=False`，无静默回退伪警告）。
  - **配置持久化与设备切换清理**：通过 `settings.json` 原子事务持久化计算设备偏好，设备切换或模型切换时主动终止旧 Worker 释放内存/显存。
- **模型共享目录与生命周期管理（ModelScope 社区源）**：
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

构建产物位于 `dist/ai-privacy-check_0.6.12_all.fpk`。安装包为纯净无架构绑定的原生包（`platform=all`），可安装于 x86_64 和 ARM64 fnOS。

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
2. **模型权重与隔离运行时分离 (Model Weights vs Runtime Separation)**：
   - **模型权重只需安装一次**：无论是在线从 ModelScope 下载还是从 NAS 本地共享目录导入，模型权重落地后独立保存于应用私有模型目录中。
   - **计算运行时按需安装**：CPU 运行时（`torch-cpu`）与 NVIDIA CUDA 运行时（`torch-cuda`）各自处于独立的隔离虚拟环境中，互不干扰。
   - **设备切换与按需补装**：
     - 若已在 CUDA 环境安装过模型（如 `gliner-pii-edge`），将设备切换为“强制 CPU”时，模型权重状态依然为 `已安装`。
     - 若此时 CPU 运行时尚未安装，模型卡会明确提示“权重已就绪 (缺少运行时)”，并显示 **“安装 CPU 运行时”** 按钮。
     - 点击后仅在后台部署 CPU 隔离环境，**直接复用已有模型权重，跳过重新下载**；Runtime 安装/更新完成后服务会自动使运行时探测缓存失效并刷新状态，无需重启应用，完成后自动就绪并恢复增强检测。
     - 卸载模型时仅清理对应的模型权重文件，保留已部署的隔离运行时环境，避免后续重复安装。
3. **在线安装模型**：点击“从魔搭下载安装”，系统将从 ModelScope (魔搭社区) 获取指定版本的模型权重（GLiNER、MemPrivacy、SiameseUIE）。
4. **本地手动导入**：若 NAS 无法直接连通外网，可将下载好的模型目录存放于 NAS 共享文件夹中，在页面填入绝对路径（如 `/vol1/1000/models/gliner-pii-edge`），系统将执行完整性校验并原子替换生效。
5. **模型卸载与重载**：随时一键卸载模型，释放磁盘与显存。卸载模型不影响已安装的计算运行时。

## 已知限制与后续规划

- **MemPrivacy 早期版本资产兼容性 (TODO)**：v0.5.1 及更早版本在 `${TRIM_PKGVAR}/data/models/memprivacy-*` 中保留的历史 prompt 模板在 v0.5.2+ 升级后不会被自动覆写。若历史安装环境出现提示词版本不一致，建议在“模型与硬件”面板点击“重新安装”或“导入”以刷新生效。后续版本将增加内置资产的自动平滑迁移。

## 隐私与安全承诺

- **零数据外流**：待处理的任何用户原始文本、脱敏映射或还原结果均不离开当前设备，不向任何第三方或云端发送请求。
- **无日志泄露**：Web 服务日志仅记录请求方法与静态路由，强制剔除 Query 参数与 Request Body，严禁记录任何 PII 明文。
- **最小特权**：应用以 fnOS 独立的受限 package 用户运行，禁止 root 权限；授权目录访问受 fnOS 安全沙盒约束。
- **内存安全**：会话映射默认仅存在于浏览器运行内存中，关闭或刷新页面后即刻焚毁。

## 许可证

本项目代码使用 Apache License 2.0 开源。相关依赖与可选神经网络模型遵循各自原始开源许可证。
