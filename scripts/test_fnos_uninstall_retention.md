# fnOS 卸载数据保留与共享模型安全验证方案

本文档说明 fnOS 原生应用「AI 脱敏器」(ai-privacy-check) 在卸载生命周期中的数据保留机制与安全边界保护原则。

---

## 1. 卸载保留向导规格 (wizard/uninstall)

在 fnOS 应用中心执行卸载时，向导提供三个清晰级别供管理员选择（由环境变量 `wizard_data_action` 传递给 `cmd/uninstall_callback`）：

| 选项标识 (`wizard_data_action`) | 界面文案 | 处理逻辑 | 适用场景 |
| :--- | :--- | :--- | :--- |
| **`keep`** (默认推荐) | **保留全部数据** | 保留 `$TRIM_PKGVAR/data` 下的模型权重、隔离运行时环境、应用配置与映射状态。 | 临时卸载或重装升级，避免重复耗时下载数 GB 模型。 |
| **`keep_runtime`** | **保留模型与环境** | 仅保留 `$TRIM_PKGVAR/data/models` 与 `$TRIM_PKGVAR/data/runtimes`，彻底清理 `$TRIM_PKGETC` 及临时缓存日志。 | 彻底重置应用配置但保留已部署的模型大文件。 |
| **`delete`** | **彻底删除应用数据** | 完全移除 `$TRIM_PKGVAR` 下的应用私有目录与 `$TRIM_PKGETC`。 | 彻底清理磁盘空间。 |

---

## 2. 核心法定不变式：用户共享目录永不触碰

无论用户在卸载向导中选择上述哪一种选项（即使选择 `delete` 彻底删除）：

> [!IMPORTANT]
> **绝对不可侵害原则**：用户通过 fnOS 文件中心管理或放置在共享目录（通过 `TRIM_DATA_SHARE_PATHS` 注入，如 `AI 脱敏器/models`）的任何源模型文件与目录，**严禁执行任何写操作、重命名或删除操作**。

`cmd/uninstall_callback` 的安全限制：
1. 校验 `$TRIM_PKGVAR` 和 `$TRIM_PKGETC` 路径有效性与名称包含 `ai-privacy-check`，防止意外误删系统根路径或父目录。
2. 仅清理 `$TRIM_PKGVAR/data` 私有工作区。
3. 决不向 `TRIM_DATA_SHARE_PATHS` 发出任何删除指令。

---

## 3. 自动化测试与验证

本项目在自动化单元测试集 `tests/test_runtime_hardening.py` 的 `UninstallCallbackTests` 中包含了三档保留机制及共享目录隔离性的完整测试用例：

- `test_keep_preserves_everything`: 验证选择 `keep` 时完整保留模型、运行时和配置。
- `test_keep_runtime_preserves_models_and_runtimes_cleans_cache_and_etc`: 验证选择 `keep_runtime` 时清理配置与缓存但保留大文件。
- `test_delete_cleans_pkgvar_but_preserves_shared`: 验证选择 `delete` 时完全清理私有目录，同时验证用户共享目录文件 `user_saved_model.safetensors` 完好无损。

执行验证指令：
```bash
uv run python -m unittest tests/test_runtime_hardening.py -v
```
