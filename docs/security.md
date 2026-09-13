# 安全说明 (v0.6.6)

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
- **模型精简按需下载与传输安全 (v0.6.6)**：严禁无限制全量拉取 ModelScope 社区仓库快照，仅由 `selective_downloader` 静态白名单枚举并单文件流式校验下载 PyTorch 必需文件，杜绝不可信仓库引入非必需可执行资产；自动过滤 ONNX 冗余文件与文档，减少 60%+ 网络流量暴露。
- **资源耗尽保护**：单次处理正文限制为 2 MB，文本字符上限为 500,000 字符；模型推理采用进程级互斥锁保证串行，防止显存或内存击穿。

## 已知限制与使用建议

- 任何自动化隐私检测算法均存在极小概率的漏检或误检。高敏法律、财务或个人隐私数据在外发前，请务必利用“人工复核”界面进行最终核验。
- 浏览器映射默认保存在会话内存中，关闭或刷新页面即销毁。如需保存，请使用页面内置的 AES-256-GCM 加密导出功能。
- 在资源受限的 NAS 设备上，建议优先使用零内存占用的纯规则模式；在拥有独立 NVIDIA 显卡的 NAS 上，可启用 CUDA 设备加速。
