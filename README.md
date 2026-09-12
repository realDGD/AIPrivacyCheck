# AI 脱敏器（AI Privacy Check）

面向飞牛 fnOS 的本地文本隐私闸门：先检测并把隐私字段替换为稳定占位符，再将脱敏文本交给外部 AI；AI 回复后，可在当前页面把原值精确放回。

当前版本：`0.6.4`（fnOS Native 原生应用）

## 已实现功能

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

构建产物位于 `dist/ai-privacy-check_0.6.4_all.fpk`。安装包为纯净无架构绑定的原生包（`platform=all`），可安装于 x86_64 和 ARM64 fnOS。

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
