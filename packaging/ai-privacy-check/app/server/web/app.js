"use strict";

const APP_PREFIX = location.pathname.startsWith("/app/ai-privacy-check") ? "/app/ai-privacy-check" : "";
const encoder = new TextEncoder();
const decoder = new TextDecoder();

const state = {
  source: "",
  entities: [],
  vault: [],
  model: null,
  modelPoll: null,
};

const $ = (id) => document.getElementById(id);
const elements = {
  sourceText: $("sourceText"),
  sourceCounter: $("sourceCounter"),
  detectButton: $("detectButton"),
  useModelToggle: $("useModelToggle"),
  modelInlineStatus: $("modelInlineStatus") || $("modelToggleLabel"),
  entityList: $("entityList") || $("entityTableBody"),
  entityEmpty: $("entityEmpty") || { hidden: false },
  entityCount: $("entityCount") || $("entityCountTag"),
  reviewFooter: $("reviewFooter") || { hidden: false },
  redactedText: $("redactedText") || $("maskedText"),
  redactedCounter: $("redactedCounter") || { textContent: "" },
  copyRedactedButton: $("copyRedactedButton") || $("copyMaskedButton"),
  copyPromptButton: $("copyPromptButton") || { disabled: false, addEventListener: () => {} },
  vaultSummary: $("vaultSummary") || { textContent: "" },
  exportVaultButton: $("exportVaultButton"),
  detectionNotice: $("detectionNotice") || { hidden: true },
  replyText: $("replyText") || $("aiReplyText"),
  replyCounter: $("replyCounter") || { textContent: "" },
  restoredText: $("restoredText"),
  restoredCounter: $("restoredCounter") || { textContent: "" },
  restoreButton: $("restoreButton"),
  copyRestoredButton: $("copyRestoredButton"),
  restoreReport: $("restoreReport") || { hidden: true },
  activeVaultBadge: $("activeVaultBadge") || { textContent: "" },
  modelStatusBadge: $("modelStatusBadge"),
  modelDetail: $("modelDetail"),
  installModelButton: $("installModelButton"),
  installLog: $("installLog"),
  exportDialog: $("exportDialog"),
  openNewTabButton: $("openNewTabButton"),
  deviceSelect: $("deviceSelect"),
  deviceStatusBadge: $("deviceStatusBadge"),
  deviceDetail: $("deviceDetail"),
  modelActiveDevice: $("modelActiveDevice"),
  uninstallModelButton: $("uninstallModelButton"),
  reloadModelButton: $("reloadModelButton"),
  importModelType: $("importModelType"),
  importSourcePath: $("importSourcePath"),
  confirmImportButton: $("confirmImportButton"),
  fillSampleButton: $("fillSampleButton"),
};

async function api(path, options = {}) {
  const response = await fetch(`${APP_PREFIX}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      "X-Requested-With": "AIPrivacyCheck",
      ...(options.headers || {}),
    },
  });
  const payload = await response.json().catch(() => ({ error: "服务返回了无法解析的响应" }));
  if (!response.ok) throw new Error(payload.error || `请求失败 (${response.status})`);
  return payload;
}

function setCounter(textarea, counter) {
  counter.textContent = `${textarea.value.length.toLocaleString("zh-CN")} 字`;
}

let toastTimer;
function toast(message) {
  const node = $("toast");
  node.textContent = message;
  node.classList.add("is-visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => node.classList.remove("is-visible"), 2400);
}

function switchView(name) {
  document.querySelectorAll(".view-tab").forEach((tab) => tab.classList.toggle("is-active", tab.dataset.view === name));
  document.querySelectorAll(".view").forEach((view) => {
    const active = view.id === `${name}View`;
    view.hidden = !active;
    view.classList.toggle("is-active", active);
  });
}

function showNotice(messages, error = false) {
  const values = Array.isArray(messages) ? messages.filter(Boolean) : [messages].filter(Boolean);
  elements.detectionNotice.hidden = values.length === 0;
  elements.detectionNotice.textContent = values.join(" ");
  elements.detectionNotice.classList.toggle("is-error", error);
}

function bytesToBase64(bytes) {
  let binary = "";
  bytes.forEach((byte) => { binary += String.fromCharCode(byte); });
  return btoa(binary);
}

function base64ToBytes(value) {
  const binary = atob(value);
  return Uint8Array.from(binary, (char) => char.charCodeAt(0));
}

async function shortFingerprint(value) {
  if (!crypto.subtle) return Math.random().toString(16).slice(2, 6).toUpperCase();
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", encoder.encode(value)));
  return Array.from(digest.slice(0, 2), (byte) => byte.toString(16).padStart(2, "0")).join("").toUpperCase();
}

async function prepareEntities(entities) {
  const typeCounts = new Map();
  const tokens = new Map();
  for (const entity of entities) {
    const key = `${entity.type}\u0000${entity.text}`;
    if (!tokens.has(key)) {
      const next = (typeCounts.get(entity.type) || 0) + 1;
      typeCounts.set(entity.type, next);
      const fingerprint = await shortFingerprint(key);
      const cleanLabel = entity.label.replace(/[\s/]+/g, "_");
      tokens.set(key, `⟦${cleanLabel}_${String(next).padStart(2, "0")}_${fingerprint}⟧`);
    }
    entity.enabled = true;
    entity.replacement = tokens.get(key);
  }
  return entities;
}

function maskPreview(value) {
  if (value.length <= 3) return `${value[0] || ""}${"•".repeat(Math.max(1, value.length - 1))}`;
  return `${value.slice(0, 2)}${"•".repeat(Math.min(8, value.length - 3))}${value.slice(-1)}`;
}

function renderEntities() {
  elements.entityList.replaceChildren();
  elements.entityEmpty.hidden = state.entities.length > 0;
  elements.reviewFooter.hidden = state.entities.length === 0;
  elements.entityCount.textContent = `${state.entities.filter((item) => item.enabled).length} / ${state.entities.length} 项`;

  state.entities.forEach((entity, index) => {
    const card = document.createElement("label");
    card.className = "entity-card";

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = entity.enabled;
    checkbox.addEventListener("change", () => {
      entity.enabled = checkbox.checked;
      elements.entityCount.textContent = `${state.entities.filter((item) => item.enabled).length} / ${state.entities.length} 项`;
      generateRedacted();
    });

    const main = document.createElement("div");
    main.className = "entity-main";
    const top = document.createElement("div");
    top.className = "entity-top";
    const type = document.createElement("span");
    type.className = "entity-type";
    type.textContent = entity.label;
    const score = document.createElement("span");
    score.className = "entity-score";
    score.textContent = entity.validated ? "已校验" : `${Math.round(entity.confidence * 100)}%`;
    top.append(type, score);

    const value = document.createElement("span");
    value.className = "entity-value";
    value.textContent = maskPreview(entity.text);
    value.title = "点击原始输入框可核对全文";

    const replacement = document.createElement("input");
    replacement.className = "entity-replacement";
    replacement.value = entity.replacement;
    replacement.setAttribute("aria-label", `${entity.label}的替换占位符`);
    replacement.addEventListener("input", () => {
      state.entities[index].replacement = replacement.value.trim();
      generateRedacted();
    });

    const sources = document.createElement("div");
    sources.className = "source-tags";
    sources.textContent = entity.sources.includes("openai_privacy_filter") ? "本地多语言规则 + 本地模型" : "本地多语言规则";
    main.append(top, value, replacement, sources);
    card.append(checkbox, main);
    elements.entityList.append(card);
  });
}

function generateRedacted() {
  if (!state.source) return;
  const enabled = state.entities.filter((entity) => entity.enabled && entity.replacement);
  const uniqueTokens = new Map();
  enabled.forEach((entity) => uniqueTokens.set(entity.replacement, entity));
  if (uniqueTokens.size !== enabled.reduce((set, entity) => set.add(`${entity.type}\u0000${entity.text}`), new Set()).size) {
    showNotice("存在重复占位符，请修改后再复制。", true);
    return;
  }

  let result = state.source;
  [...enabled].sort((a, b) => b.start - a.start).forEach((entity) => {
    result = result.slice(0, entity.start) + entity.replacement + result.slice(entity.end);
  });
  state.vault = [];
  const seen = new Set();
  enabled.forEach((entity) => {
    const key = `${entity.replacement}\u0000${entity.text}`;
    if (seen.has(key)) return;
    seen.add(key);
    state.vault.push({ token: entity.replacement, value: entity.text, type: entity.type, label: entity.label });
  });
  elements.redactedText.value = result;
  setCounter(elements.redactedText, elements.redactedCounter);
  elements.copyRedactedButton.disabled = !result;
  elements.copyPromptButton.disabled = !result;
  elements.exportVaultButton.disabled = state.vault.length === 0 || !crypto.subtle;
  elements.vaultSummary.textContent = `${state.vault.length} 个加密映射仅保留在当前页面`;
  elements.activeVaultBadge.textContent = `当前会话：${state.vault.length} 个映射`;
}

async function detect() {
  const text = elements.sourceText.value;
  if (!text.trim()) {
    showNotice("请先粘贴需要处理的文本。", true);
    elements.sourceText.focus();
    return;
  }
  elements.detectButton.disabled = true;
  elements.detectButton.textContent = "正在本地检测…";
  showNotice([]);
  try {
    const result = await api("/api/detect", {
      method: "POST",
      body: JSON.stringify({ text, use_model: elements.useModelToggle.checked }),
    });
    state.source = text;
    state.entities = await prepareEntities(result.entities);
    renderEntities();
    generateRedacted();
    const summary = state.entities.length
      ? `检测完成：发现 ${state.entities.length} 项，耗时 ${result.processing_ms} ms。请逐项复核后再发送。`
      : "未发现明确的隐私字段。自动检测可能漏检，请人工检查原文。";
    showNotice([summary, ...(result.warnings || [])]);
  } catch (error) {
    showNotice(error.message, true);
  } finally {
    elements.detectButton.disabled = false;
    elements.detectButton.innerHTML = '<span aria-hidden="true">⌕</span> 检测隐私字段';
  }
}

async function copyText(value, message) {
  if (!value) return;
  try {
    await navigator.clipboard.writeText(value);
    toast(message);
  } catch {
    const helper = document.createElement("textarea");
    helper.value = value;
    helper.setAttribute("readonly", "");
    document.body.append(helper);
    helper.select();
    document.execCommand("copy");
    helper.remove();
    toast(message);
  }
}

function restoreText() {
  if (!state.vault.length) {
    elements.restoreReport.hidden = false;
    elements.restoreReport.className = "restore-report is-warning";
    elements.restoreReport.textContent = "当前没有可用映射。请先完成脱敏，或在下方导入加密保险箱。";
    return;
  }
  let restored = elements.replyText.value;
  if (!restored.trim()) {
    elements.replyText.focus();
    toast("请先粘贴 AI 回复");
    return;
  }
  let replaced = 0;
  const missing = [];
  [...state.vault].sort((a, b) => b.token.length - a.token.length).forEach((entry) => {
    const occurrences = restored.split(entry.token).length - 1;
    if (occurrences) {
      restored = restored.split(entry.token).join(entry.value);
      replaced += occurrences;
    } else {
      missing.push(entry.token);
    }
  });
  const unknown = restored.match(/⟦[^⟦⟧\n]{2,80}⟧/g) || [];
  elements.restoredText.value = restored;
  setCounter(elements.restoredText, elements.restoredCounter);
  elements.copyRestoredButton.disabled = false;
  elements.restoreReport.hidden = false;
  const notes = [`已恢复 ${replaced} 处占位符。`];
  if (unknown.length) notes.push(`仍有 ${unknown.length} 个未知占位符未恢复。`);
  if (missing.length) notes.push(`${missing.length} 个保险箱映射未在回复中出现。`);
  elements.restoreReport.className = `restore-report${unknown.length ? " is-warning" : ""}`;
  elements.restoreReport.textContent = notes.join(" ");
}

async function deriveVaultKey(password, salt, iterations) {
  const material = await crypto.subtle.importKey("raw", encoder.encode(password), "PBKDF2", false, ["deriveKey"]);
  return crypto.subtle.deriveKey(
    { name: "PBKDF2", hash: "SHA-256", salt, iterations },
    material,
    { name: "AES-GCM", length: 256 },
    false,
    ["encrypt", "decrypt"],
  );
}

async function encryptVault(password) {
  const salt = crypto.getRandomValues(new Uint8Array(16));
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const iterations = 310000;
  const key = await deriveVaultKey(password, salt, iterations);
  const plaintext = encoder.encode(JSON.stringify({ version: 1, created_at: new Date().toISOString(), entries: state.vault }));
  const ciphertext = new Uint8Array(await crypto.subtle.encrypt(
    { name: "AES-GCM", iv, additionalData: encoder.encode("AIPrivacyCheckVault-v1") },
    key,
    plaintext,
  ));
  return {
    format: "AIPrivacyCheckVault",
    version: 1,
    kdf: { name: "PBKDF2", hash: "SHA-256", iterations, salt: bytesToBase64(salt) },
    cipher: { name: "AES-256-GCM", iv: bytesToBase64(iv), data: bytesToBase64(ciphertext) },
  };
}

async function decryptVault(container, password) {
  if (container.format !== "AIPrivacyCheckVault" || container.version !== 1) throw new Error("不是受支持的保险箱文件");
  const salt = base64ToBytes(container.kdf.salt);
  const iv = base64ToBytes(container.cipher.iv);
  const data = base64ToBytes(container.cipher.data);
  const key = await deriveVaultKey(password, salt, Number(container.kdf.iterations));
  const plaintext = await crypto.subtle.decrypt(
    { name: "AES-GCM", iv, additionalData: encoder.encode("AIPrivacyCheckVault-v1") },
    key,
    data,
  );
  const payload = JSON.parse(decoder.decode(plaintext));
  if (!Array.isArray(payload.entries) || payload.entries.some((entry) => typeof entry.token !== "string" || typeof entry.value !== "string")) {
    throw new Error("保险箱内容无效");
  }
  return payload.entries;
}

async function exportVault() {
  const password = $("exportPassword").value;
  const confirm = $("exportPasswordConfirm").value;
  if (password.length < 10) { toast("密码至少需要 10 个字符"); return; }
  if (password !== confirm) { toast("两次输入的密码不一致"); return; }
  try {
    $("confirmExportButton").disabled = true;
    const container = await encryptVault(password);
    const blob = new Blob([JSON.stringify(container, null, 2)], { type: "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `privacy-vault-${new Date().toISOString().replace(/[:.]/g, "-")}.aipvault.json`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
    elements.exportDialog.close();
    $("exportPassword").value = "";
    $("exportPasswordConfirm").value = "";
    toast("加密保险箱已下载");
  } catch (error) {
    toast(`导出失败：${error.message}`);
  } finally {
    $("confirmExportButton").disabled = false;
  }
}

async function importVault() {
  const file = $("vaultFile").files[0];
  const password = $("vaultPassword").value;
  if (!file || !password) { toast("请选择保险箱文件并输入密码"); return; }
  try {
    const container = JSON.parse(await file.text());
    state.vault = await decryptVault(container, password);
    elements.activeVaultBadge.textContent = `当前会话：${state.vault.length} 个映射`;
    elements.vaultSummary.textContent = `${state.vault.length} 个映射来自加密保险箱`;
    $("vaultPassword").value = "";
    toast(`已导入 ${state.vault.length} 个映射`);
  } catch {
    toast("无法解锁：密码错误或文件已损坏");
  }
}

function updateModelUI(data) {
  state.model = data;
  const opf = (data.models && data.models.privacy_filter) || (data.model) || {};
  const status = opf.state || "packages_missing";
  const labels = {
    ready: "已就绪",
    installing: "安装中",
    packages_missing: "未安装",
    model_missing: "待下载权重",
    error: "安装失败",
  };
  elements.modelStatusBadge.textContent = labels[status] || status || "未知状态";
  elements.modelStatusBadge.className = `status-badge${status === "ready" ? " is-ready" : status === "error" ? " is-error" : ""}`;
  elements.useModelToggle.disabled = status !== "ready";
  if (status !== "ready") elements.useModelToggle.checked = false;
  elements.modelInlineStatus.textContent = status === "ready" ? "增强模型已就绪" : status === "installing" ? "正在后台安装" : "未安装（中文语义与多语言规则始终可用）";

  // Device status
  const dev = data.device || {};
  if (elements.deviceStatusBadge) {
    elements.deviceStatusBadge.textContent = dev.actual_device === "cuda" ? "NVIDIA CUDA 加速" : "CPU 运行模式";
    elements.deviceStatusBadge.className = `status-badge${dev.actual_device === "cuda" ? " is-ready" : ""}`;
  }
  if (elements.deviceDetail) {
    elements.deviceDetail.textContent = dev.cuda_available
      ? `检测到 GPU: ${dev.cuda_device_name || "NVIDIA 显卡"} (驱动就绪，目标: ${dev.requested_device})`
      : `未检测到 NVIDIA CUDA 环境 (使用 CPU 推理，PyTorch: ${dev.torch_version || "未加载"})`;
  }
  if (elements.deviceSelect && dev.requested_device) {
    elements.deviceSelect.value = dev.requested_device;
  }
  if (elements.modelActiveDevice) {
    elements.modelActiveDevice.textContent = (dev.actual_device || "CPU").toUpperCase();
  }

  // Model details
  const installDetail = (data.install_states && data.install_states["privacy-filter-install"] && data.install_states["privacy-filter-install"].detail)
    || (opf.install && opf.install.detail);
  elements.modelDetail.textContent = status === "ready"
    ? "OpenAI Privacy Filter 权重完整，已保存在 fnOS 数据目录中，离线推理保护中。"
    : installDetail || "中文规则与中文语义信息抽取开箱即用；可选模型增强可进一步提升长难句泛化度。";
  elements.installModelButton.disabled = !data.is_admin || data.installing || status === "ready";
  elements.installModelButton.textContent = data.installing ? "正在处理中..." : status === "ready" ? "模型已就绪" : "在线安装增强模型";
  if (elements.uninstallModelButton) {
    elements.uninstallModelButton.style.display = status === "ready" ? "inline-block" : "none";
  }
  elements.installLog.textContent = data.log_tail && data.log_tail.length ? data.log_tail.join("\n") : "暂无日志";

  clearTimeout(state.modelPoll);
  if (data.installing) state.modelPoll = setTimeout(refreshModelStatus, 3000);
}

async function refreshModelStatus() {
  try {
    updateModelUI(await api("/api/model/status"));
  } catch (error) {
    elements.modelInlineStatus.textContent = "无法读取模型状态";
    elements.modelDetail.textContent = error.message;
  }
}

async function installModel() {
  const confirmed = window.confirm("在线安装将从 GitHub 和 Hugging Face 下载运行库与约 2.8GB 权重。继续吗？");
  if (!confirmed) return;
  elements.installModelButton.disabled = true;
  try {
    const result = await api("/api/model/install", { method: "POST", body: JSON.stringify({ model: "privacy-filter" }) });
    updateModelUI({ ...result.model, is_admin: true });
    $("installLogPanel").open = true;
    toast("模型已开始在后台下载与安装");
  } catch (error) {
    toast(error.message);
    await refreshModelStatus();
  }
}

async function importModel() {
  const modelType = elements.importModelType ? elements.importModelType.value : "privacy-filter";
  const sourcePath = (elements.importSourcePath ? elements.importSourcePath.value : "").trim();
  if (!sourcePath) {
    toast("请输入已授权的模型源目录路径");
    return;
  }
  elements.confirmImportButton.disabled = true;
  try {
    const result = await api("/api/model/import", {
      method: "POST",
      body: JSON.stringify({ model: modelType, source_path: sourcePath }),
    });
    toast(result.message || "模型导入成功");
    updateModelUI(result.status);
    if (elements.importSourcePath) elements.importSourcePath.value = "";
  } catch (error) {
    toast(`导入失败: ${error.message}`);
  } finally {
    elements.confirmImportButton.disabled = false;
  }
}

async function uninstallModel() {
  if (!window.confirm("确定要卸载当前存储在应用目录内的该模型副本吗？不会影响您原始导入的文件。")) return;
  try {
    const result = await api("/api/model/uninstall", {
      method: "POST",
      body: JSON.stringify({ model: "privacy-filter" }),
    });
    toast(result.message || "已卸载");
    updateModelUI(result.status);
  } catch (error) {
    toast(`卸载失败: ${error.message}`);
  }
}

async function reloadModel() {
  try {
    const result = await api("/api/model/reload", { method: "POST", body: "{}" });
    toast("模型已成功重载");
    updateModelUI(result.status);
  } catch (error) {
    toast(`重载失败: ${error.message}`);
  }
}

async function changeDevice() {
  if (!elements.deviceSelect) return;
  const target = elements.deviceSelect.value;
  try {
    await api("/api/device/select", {
      method: "POST",
      body: JSON.stringify({ device: target }),
    });
    toast(`计算设备已切换为: ${target}`);
    await refreshModelStatus();
  } catch (error) {
    toast(`切换失败: ${error.message}`);
  }
}

function clearAll() {
  state.source = "";
  state.entities = [];
  state.vault = [];
  [elements.sourceText, elements.redactedText, elements.replyText, elements.restoredText].forEach((node) => { node.value = ""; });
  renderEntities();
  [
    [elements.sourceText, elements.sourceCounter],
    [elements.redactedText, elements.redactedCounter],
    [elements.replyText, elements.replyCounter],
    [elements.restoredText, elements.restoredCounter],
  ].forEach(([node, counter]) => setCounter(node, counter));
  elements.copyRedactedButton.disabled = true;
  elements.copyPromptButton.disabled = true;
  elements.copyRestoredButton.disabled = true;
  elements.exportVaultButton.disabled = true;
  elements.vaultSummary.textContent = "尚未生成映射";
  elements.activeVaultBadge.textContent = "当前会话：0 个映射";
  elements.restoreReport.hidden = true;
  showNotice([]);
  toast("当前会话已清空");
}

if (elements.openNewTabButton) {
  elements.openNewTabButton.addEventListener("click", () => {
    // 1. Try fnOS Web SDK if embedded
    if (window.parent && window.parent !== window && window.parent.fnos && typeof window.parent.fnos.openURL === "function") {
      try {
        window.parent.fnos.openURL("/app/ai-privacy-check/");
        return;
      } catch {}
    }
    // 2. Fallback to window.open with gateway URL
    const targetUrl = window.location.pathname.startsWith("/app/ai-privacy-check")
      ? window.location.pathname
      : "/app/ai-privacy-check/";
    window.open(targetUrl, "_blank");
  });
}

if (elements.deviceSelect) elements.deviceSelect.addEventListener("change", changeDevice);
if (elements.confirmImportButton) elements.confirmImportButton.addEventListener("click", importModel);
if (elements.uninstallModelButton) elements.uninstallModelButton.addEventListener("click", uninstallModel);
if (elements.reloadModelButton) elements.reloadModelButton.addEventListener("click", reloadModel);
if (elements.fillSampleButton) {
  elements.fillSampleButton.addEventListener("click", () => {
    elements.sourceText.value = "请寄给上海市浦东新区世纪大道100号的收件人张伟先生，联系手机 13800138000，邮箱 zhangwei@example.com，身份证号 11010519491231002X。另请备份数据库 postgresql://appuser:SecretPass123@db.internal:5432/crm。";
    setCounter(elements.sourceText, elements.sourceCounter);
  });
}

document.querySelectorAll(".view-tab").forEach((tab) => tab.addEventListener("click", () => switchView(tab.dataset.view)));
if (elements.sourceText) elements.sourceText.addEventListener("input", () => setCounter(elements.sourceText, elements.sourceCounter));
if (elements.replyText) elements.replyText.addEventListener("input", () => setCounter(elements.replyText, elements.replyCounter));
if (elements.detectButton) elements.detectButton.addEventListener("click", detect);
if (elements.restoreButton) elements.restoreButton.addEventListener("click", restoreText);
if (elements.copyRedactedButton && elements.copyRedactedButton.addEventListener) {
  elements.copyRedactedButton.addEventListener("click", () => copyText(elements.redactedText.value, "脱敏文本已复制"));
}
if (elements.copyPromptButton && elements.copyPromptButton.addEventListener) {
  elements.copyPromptButton.addEventListener("click", () => copyText(
    `请保留所有形如 ⟦类型_序号_校验码⟧ 的占位符原样，不要翻译、改写、删除或合并。\n\n${elements.redactedText.value}`,
    "带占位符说明的文本已复制",
  ));
}
if (elements.copyRestoredButton && elements.copyRestoredButton.addEventListener) {
  elements.copyRestoredButton.addEventListener("click", () => copyText(elements.restoredText.value, "恢复结果已复制"));
}
if ($("selectAllButton")) $("selectAllButton").addEventListener("click", () => { state.entities.forEach((entity) => { entity.enabled = true; }); renderEntities(); generateRedacted(); });
if ($("selectNoneButton")) $("selectNoneButton").addEventListener("click", () => { state.entities.forEach((entity) => { entity.enabled = false; }); renderEntities(); generateRedacted(); });
if ($("goModelButton")) $("goModelButton").addEventListener("click", () => switchView("model"));
if ($("clearAllButton")) $("clearAllButton").addEventListener("click", clearAll);
if (elements.exportVaultButton) elements.exportVaultButton.addEventListener("click", () => elements.exportDialog.showModal());
if ($("confirmExportButton")) $("confirmExportButton").addEventListener("click", exportVault);
if ($("importVaultButton")) $("importVaultButton").addEventListener("click", importVault);
if (elements.installModelButton) elements.installModelButton.addEventListener("click", installModel);

renderEntities();
refreshModelStatus();
