# 架构说明 (v0.3.0)

## 设计目标

1. **绝对本地化**：原文、实体与可逆映射不离开用户控制的 fnOS 存储与浏览器运行内存，严格杜绝数据向任何外网服务器泄露。
2. **轻量多级级联**：即使没有大模型、无外网或设备硬件资源受限，基础多语言与中文强规则（Tier 1）及中文信息抽取（Tier 2）也能即开即用、零额外内存驻留；神经网络模型（Tier 3）按需加载。
3. **硬件自适应加速**：智能利用 NAS 上的 NVIDIA GPU（CUDA）或降级 CPU 运行，具备惰性探测与安全回退机制。
4. **低确定性人工复核**：对人名、地址、模糊代号提供可视化复核与实时修改能力。
5. **安全受限的原生集成**：依托 fnOS 官方 Python 3.12 运行时与统一网关 Unix Socket 机制，不暴露多余端口，权限采用 package 独立用户。

## 系统架构拓扑

```text
fnOS 桌面 / 浏览器新标签页
  │ 统一网关凭证 (NAS 会话)
  ▼
fnOS 统一网关 (/app/ai-privacy-check)
  │ Unix Stream Socket: ai-privacy-check.sock
  ▼
fnOS Native Python 3.12 进程 (package 用户)
  ├── MultilingualRuleDetector (Tier 1)
  │     ├── 20+ 种格式正则与跨语言上下文约束 (中文/英/德/法/西/俄/日/韩/阿/泰)
  │     └── 校验位/日期/Luhn/IBAN/USCC 算法校验器
  ├── ChineseIEDetector (Tier 2)
  │     ├── 内置零依赖语言学启发式信息抽取 (百家姓/动词锚点/行政拓扑)
  │     ├── safe_sequential_span_alignment 游标防重定位
  │     └── 可选 PaddleNLP Taskflow UIE 适配器
  ├── OpenAIModelDetector (Tier 3, 可选、惰性加载)
  │     └── opf.OPF Token Classification 模型
  ├── DeviceManager
  │     └── auto / cpu / cuda 惰性探测与智能降级
  └── 静态 Web UI
        ├── 人工复核与稳定占位符
        ├── 实时脱敏与本地精确还原
        └── Web Crypto PBKDF2 + AES-256-GCM 本地加密保险箱
```

## 三级检测与实体仲裁

所有检测器输出标准半开区间 `[start, end)`。在 `merge_entities()` 阶段：

1. **同跨度同类型合并**：合并检测引擎来源（`sources`），取最高置信度。
2. **重叠冲突仲裁（优先级阶梯）**：
   - **高危凭证/密钥**（Priority 125~121）：`DATABASE_URI` (125), `PRIVATE_KEY` (124), `API_TOKEN` (122), `PASSWORD` (121)
   - **强校验法定证件**（Priority 114~108）：`US_SSN` (114), `CREDIT_CARD` / `CN_BANK_CARD` (111), `CN_ID_CARD` (110), `CN_USCC` (108)
   - **高置信通信标识**（Priority 105~100）：`PHONE` / `CN_PHONE_NUMBER` (105), `EMAIL` (103), `IPV6_ADDRESS` (102), `IP_ADDRESS` (100)
   - **业务/语义实体**（Priority 95~85）：`CN_ADDRESS` (92), `CN_NAME` (90), `MEDICAL_RECORD_ID` (90), `INSURANCE_ID` (89), `EMPLOYEE_ID` (86), `STUDENT_ID` (85)
   - **神经网络通用模型输出**（Priority 50）：`OpenAIModelDetector` 输出作为兜底补充。
3. **校验位优先原则**：通过校验位（如身份证校验码、银行卡 Luhn、企业税号 GB 32100、IBAN）验证的实体享有绝对优先权，杜绝普通文本模型误分类覆盖。

## 中文信息抽取（ChineseIEDetector）设计

针对中文无空格分词与同名多次出现的难点，`ChineseIEDetector` 引入：
- **分块全局字符偏移追踪**：长文本分块处理时严格维护绝对文本偏移，杜绝错位。
- **游标顺序对齐（`safe_sequential_span_alignment`）**：在基于词表或子串定位实体时，维护当前查找游标，杜绝 `str.find` 始终返回首个出现的经典缺陷，确保相同人名在不同语境下均能精准映射到各自的 span。

## 推理设备管理（DeviceManager）

- **惰性加载**：启动和纯规则扫描期间不触发 `import torch`，保持应用轻量。
- **平滑降级**：当选择 `auto` 时，优先检查 `torch.cuda.is_available()`；若显卡驱动未就绪或显存不足，自动切至 `cpu` 并输出诊断日志，服务不中断。

## 模型生命周期管理

- **在线下载**：一键拉取固定 revision 的模型与依赖，支持自动匹配 PyTorch CUDA/CPU wheel。
- **本地手动导入**：支持从 fnOS 授权目录导入预下载的模型文件。导入时写入临时暂存区，执行完整性校验（`config.json`、Safetensors 或模型权重）后使用 `os.replace` 原子替换，防止意外中断导致文件损坏。
- **卸载与状态管理**：支持一键清空模型文件并释放显存，通过 `status` 实时向 UI 反馈。

## fnOS 原生适配规范

- **权限声明 (`config/privilege`)**：使用 package 受限用户运行，加入 `video` 与 `render` 组支持 GPU 加速。
- **路径授权 (`manifest`)**：设置 `disable_authorization_path = false`，允许用户授权共享文件夹路径用于本地模型导入。
- **统一网关集成**：通过 Unix Socket 提供静态与 API 服务，网关统一鉴权；桌面与新标签页双模态自适应。
