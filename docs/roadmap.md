# 路线图

## 0.2：目标设备验证

- 在 fnOS x86_64 与 ARM64 设备验证安装、升级、卸载和统一网关生命周期。
- 记录 CPU、内存、首次加载时间与典型文本吞吐。
- 验证 fnOS `python312` 运行时及 PyTorch wheel 在两种架构上的兼容性。
- 根据实测决定是否提供量化 ONNX/Transformers.js 模式。

## 0.3：中文模型增强

- 建立完全合成的中文 PII 评测集，按实体类型统计 precision/recall/F1。
- 评估 PaddleNLP UIE、小型中文 NER 与 Privacy Filter 微调三条路线。
- 增加机构、项目名、病历号、合同号等可配置领域策略。
- 为姓名、地址提供更清晰的低置信度复核提示。

## 后续

- TXT/Markdown/JSON/CSV 文件导入与安全副本导出。
- 多用户隔离的、服务器端加密会话保险箱（默认仍不持久化）。
- 管理员可导入离线模型包，满足 fnOS 无外网环境。
- 评估 MemPrivacy 作为 Agent 记忆上云策略模块，而不是替换 PII span detector。
