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
  pendingSelection: null,
  retargeting: null,
};

const MANUAL_ENTITY_OPTIONS = [
  { type: "GENERIC_PRIVACY", label: "隐私条目", privacyLevel: "PL2" },
  { type: "CN_NAME", label: "姓名", privacyLevel: "PL2" },
  { type: "PHONE", label: "电话号码", privacyLevel: "PL2" },
  { type: "EMAIL", label: "电子邮箱", privacyLevel: "PL2" },
  { type: "CN_ADDRESS", label: "地址", privacyLevel: "PL2" },
  { type: "CN_ID_CARD", label: "身份证", privacyLevel: "PL3" },
  { type: "PASSPORT", label: "护照", privacyLevel: "PL3" },
  { type: "CN_BANK_CARD", label: "银行卡号", privacyLevel: "PL3" },
  { type: "USERNAME", label: "用户名", privacyLevel: "PL2" },
  { type: "PASSWORD", label: "密码", privacyLevel: "PL4" },
  { type: "API_TOKEN", label: "API Token", privacyLevel: "PL4" },
  { type: "IP_ADDRESS", label: "IP 地址", privacyLevel: "PL2" },
  { type: "MEDICAL", label: "医疗信息", privacyLevel: "PL3" },
  { type: "FINANCIAL", label: "财务信息", privacyLevel: "PL3" },
  { type: "ORGANIZATION", label: "组织机构", privacyLevel: "PL2" },
  { type: "OTHER_PRIVACY", label: "其他隐私", privacyLevel: "PL2" },
];

const $ = (id) => document.getElementById(id);
const elements = {
  sourceText: $("sourceText"),
  sourceCounter: $("sourceCounter"),
  detectButton: $("detectButton"),
  useModelToggle: $("useModelToggle"),
  policySelect: $("policySelect"),
  slotListContainer: $("slotListContainer"),
  modelInlineStatus: $("modelInlineStatus") || $("modelToggleLabel"),
  entityList: $("entityList") || $("entityTableBody"),
  entityEmpty: $("entityEmpty") || { hidden: false },
  entityCount: $("entityCount") || $("entityCountTag"),
  reviewFooter: $("reviewFooter") || { hidden: false },
  redactedPreview: $("redactedPreview"),
  selectionPopover: $("selectionPopover"),
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
  installLog: $("installLog"),
  exportDialog: $("exportDialog"),
  openNewTabButton: $("openNewTabButton"),
  deviceSelect: $("deviceSelect"),
  deviceStatusBadge: $("deviceStatusBadge"),
  deviceDetail: $("deviceDetail"),
  deviceTelemetry: $("deviceTelemetry"),
  scanSharedButton: $("scanSharedButton"),
  sharedModelsList: $("sharedModelsList"),
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
  if (name !== "detect") {
    if (state.retargeting) {
      cancelEntityRetarget(false);
    }
    resetAnnotationInteractionState();
  }
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

let entitySequence = 0;

async function allocateReplacementToken(entityType, label, text, entities = []) {
  const existing = entities.find((e) => e.type === entityType && e.text === text && e.replacement);
  if (existing) {
    return existing.replacement;
  }
  const distinctTexts = new Set(
    entities
      .filter((e) => e.type === entityType && e.replacement)
      .map((e) => e.text)
  );
  const next = distinctTexts.size + 1;
  const key = `${entityType}\u0000${text}`;
  const fingerprint = await shortFingerprint(key);
  const cleanLabel = (label || entityType).replace(/[\s/]+/g, "_");
  return `⟦${cleanLabel}_${String(next).padStart(2, "0")}_${fingerprint}⟧`;
}

async function prepareEntities(entities) {
  const prepared = [];
  for (const entity of entities) {
    entitySequence += 1;
    entity.id = entity.id || `ent_${entitySequence}_${Math.random().toString(36).slice(2, 7)}`;
    if (typeof entity.start_utf16 === "number" && typeof entity.end_utf16 === "number") {
      entity.start = entity.start_utf16;
      entity.end = entity.end_utf16;
    }
    entity.enabled = true;
    entity.replacement = await allocateReplacementToken(entity.type, entity.label, entity.text, prepared);
    prepared.push(entity);
  }
  return prepared;
}

function maskPreview(value) {
  if (value.length <= 3) return `${value[0] || ""}${"•".repeat(Math.max(1, value.length - 1))}`;
  return `${value.slice(0, 2)}${"•".repeat(Math.min(8, value.length - 3))}${value.slice(-1)}`;
}

function buildRedactedSegments(source, entities, excludedEntity = null) {
  const ordered = entities
    .map((entity, entityIndex) => ({ entity, entityIndex }))
    .filter(({ entity }) => entity !== excludedEntity && entity.enabled && entity.replacement)
    .sort((left, right) => left.entity.start - right.entity.start || left.entity.end - right.entity.end);
  const segments = [];
  let cursor = 0;

  ordered.forEach(({ entity, entityIndex }) => {
    const start = Number(entity.start);
    const end = Number(entity.end);
    if (!Number.isInteger(start) || !Number.isInteger(end) || start < cursor || end <= start || end > source.length) return;
    if (start > cursor) {
      segments.push({ kind: "text", text: source.slice(cursor, start), sourceStart: cursor, sourceEnd: start });
    }
    const tokenSegment = {
      kind: "token",
      text: entity.replacement,
      entityIndex,
      label: entity.label,
      sourceStart: start,
      sourceEnd: end,
    };
    if (entity.id) {
      tokenSegment.entityId = entity.id;
    }
    segments.push(tokenSegment);
    cursor = end;
  });

  if (cursor < source.length) {
    segments.push({ kind: "text", text: source.slice(cursor), sourceStart: cursor, sourceEnd: source.length });
  }
  return segments;
}

function normalizeSourceSelection(source, start, end, entities, excludedEntity = null) {
  let sourceStart = Math.max(0, Math.min(Number(start), Number(end)));
  let sourceEnd = Math.min(source.length, Math.max(Number(start), Number(end)));
  if (!Number.isInteger(sourceStart) || !Number.isInteger(sourceEnd) || sourceEnd <= sourceStart) return null;

  let text = source.slice(sourceStart, sourceEnd);
  const leadingWhitespace = text.length - text.trimStart().length;
  const trailingWhitespace = text.length - text.trimEnd().length;
  sourceStart += leadingWhitespace;
  sourceEnd -= trailingWhitespace;
  text = source.slice(sourceStart, sourceEnd);
  if (!text) return null;

  const overlapsEntity = entities.some((entity) => (
    entity !== excludedEntity
    && entity.enabled !== false
    && Number.isFinite(Number(entity.start))
    && Number.isFinite(Number(entity.end))
    && sourceStart < Number(entity.end)
    && sourceEnd > Number(entity.start)
  ));
  if (overlapsEntity) return null;
  return { start: sourceStart, end: sourceEnd, text };
}

function scrollElementIntoContainer(container, target, behavior = "smooth") {
  if (!container || !target) return;
  if (typeof container.getBoundingClientRect === "function" && typeof target.getBoundingClientRect === "function") {
    const containerRect = container.getBoundingClientRect();
    const targetRect = target.getBoundingClientRect();
    const targetHeight = typeof targetRect.height === "number" ? targetRect.height : (typeof targetRect.bottom === "number" && typeof targetRect.top === "number" ? targetRect.bottom - targetRect.top : (target.clientHeight || 0));
    const containerHeight = typeof containerRect.height === "number" ? containerRect.height : (container.clientHeight || 0);
    const delta = targetRect.top - containerRect.top - (containerHeight - targetHeight) / 2;
    if (typeof container.scrollTo === "function") {
      container.scrollTo({
        top: Math.max(0, container.scrollTop + delta),
        behavior,
      });
    }
  } else if (typeof container.scrollTo === "function" && typeof target.offsetTop === "number") {
    const targetHeight = target.offsetHeight || 0;
    const containerHeight = container.clientHeight || 0;
    container.scrollTo({
      top: Math.max(0, target.offsetTop - (containerHeight - targetHeight) / 2),
      behavior,
    });
  }
}

function activateLinkedElement(container, selector, moveFirst = false) {
  if (!container || typeof container.querySelector !== "function") return false;
  const target = container.querySelector(selector);
  if (!target || !target.classList) return false;

  const reduceMotion = typeof window !== "undefined"
    && typeof window.matchMedia === "function"
    && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const behavior = reduceMotion ? "auto" : "smooth";

  if (moveFirst && typeof container.prepend === "function") {
    container.prepend(target);
    if (typeof container.scrollTo === "function") {
      container.scrollTo({ top: 0, behavior });
    }
  } else if (typeof scrollElementIntoContainer === "function") {
    scrollElementIntoContainer(container, target, behavior);
  }

  target.classList.remove("is-link-target");
  void target.offsetWidth;
  target.classList.add("is-link-target");
  if (typeof target.addEventListener === "function") {
    target.addEventListener("animationend", () => target.classList.remove("is-link-target"), { once: true });
  }

  if (typeof target.focus === "function") {
    try {
      target.focus({ preventScroll: true });
    } catch (_) {
      target.focus();
    }
  }
  return true;
}

function focusReviewEntity(target) {
  const selector = typeof target === "string" && isNaN(Number(target))
    ? `[data-entity-id="${target}"]`
    : `[data-entity-id="${target}"], [data-entity-index="${target}"]`;
  return activateLinkedElement(elements.entityList, selector, true);
}

function focusRedactedEntity(target) {
  const selector = typeof target === "string" && isNaN(Number(target))
    ? `[data-entity-id="${target}"]`
    : `[data-entity-id="${target}"], [data-entity-index="${target}"]`;
  const found = activateLinkedElement(elements.redactedPreview, selector);
  if (!found) toast("该项目当前未替换，启用后即可在安全副本中定位。");
  return found;
}

function updateRedactedActionAvailability() {
  const hasResult = Boolean(elements.redactedText.value);
  const editingRange = Boolean(state.retargeting);
  elements.copyRedactedButton.disabled = editingRange || !hasResult;
  elements.copyPromptButton.disabled = editingRange || !hasResult;
  elements.exportVaultButton.disabled = editingRange || state.vault.length === 0 || !crypto.subtle;
}

function compactRect(rect) {
  return {
    left: rect.left,
    right: rect.right,
    top: rect.top,
    bottom: rect.bottom,
    width: rect.width,
    height: rect.height,
  };
}

function pointRect(x, y) {
  return { left: x, right: x, top: y, bottom: y, width: 0, height: 0 };
}

function positionSelectionPopover(anchorRect) {
  if (!elements.selectionPopover) return;
  const popover = elements.selectionPopover;
  popover.hidden = false;
  popover.style.visibility = "hidden";
  requestAnimationFrame(() => {
    const margin = 8;
    const gap = 9;
    const width = popover.offsetWidth;
    const height = popover.offsetHeight;
    const preferredLeft = anchorRect.left + (anchorRect.width / 2) - (width / 2);
    const left = Math.max(margin, Math.min(window.innerWidth - width - margin, preferredLeft));
    const above = anchorRect.top - height - gap;
    const top = above >= margin
      ? above
      : Math.min(window.innerHeight - height - margin, anchorRect.bottom + gap);
    popover.style.left = `${Math.round(left)}px`;
    popover.style.top = `${Math.round(Math.max(margin, top))}px`;
    popover.style.visibility = "visible";
    const firstButton = popover.querySelector("button");
    if (firstButton && typeof firstButton.focus === "function") {
      firstButton.focus({ preventScroll: true });
    }
  });
}

function clearBrowserSelection() {
  const selection = typeof window !== "undefined" ? window.getSelection() : null;
  if (selection && typeof selection.removeAllRanges === "function") selection.removeAllRanges();
}

function hideSelectionPopover(clearSelection = false) {
  if (elements.selectionPopover) {
    elements.selectionPopover.hidden = true;
    elements.selectionPopover.replaceChildren();
    elements.selectionPopover.classList.remove("is-type-picker");
  }
  if (clearSelection) clearBrowserSelection();
}

function resetAnnotationInteractionState(options = {}) {
  const { preserveSelection = false } = options;
  state.retargeting = null;
  state.pendingSelection = null;
  if (!preserveSelection && typeof clearBrowserSelection === "function") clearBrowserSelection();
  hideSelectionPopover(true);
  if (elements.redactedPreview && elements.redactedPreview.classList) {
    elements.redactedPreview.classList.remove("is-retargeting");
  }
  if (elements.entityList && elements.entityList.classList) {
    elements.entityList.classList.remove("is-retargeting");
  }
  if (typeof updateRedactedActionAvailability === "function") {
    updateRedactedActionAvailability();
  }
}

function renderSelectionPopover(actions, anchorRect, title = "", typePicker = false) {
  if (!elements.selectionPopover) return;
  const popover = elements.selectionPopover;
  popover.replaceChildren();
  popover.classList.toggle("is-type-picker", typePicker);
  if (title) {
    const heading = document.createElement("div");
    heading.className = "selection-popover-title";
    heading.textContent = title;
    popover.append(heading);
  }
  const actionRow = document.createElement("div");
  actionRow.className = "selection-popover-actions";
  actions.forEach((action) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `selection-action${action.primary ? " is-primary" : ""}${action.danger ? " is-danger" : ""}`;
    button.textContent = action.label;
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      Promise.resolve(action.run()).catch((error) => toast(error.message || "操作失败"));
    });
    actionRow.append(button);
  });
  popover.append(actionRow);
  positionSelectionPopover(anchorRect);
}

function sourceOffsetFromBoundary(node, offset) {
  if (!node) return null;
  const element = node.nodeType === 1 ? node : node.parentElement;
  const sourceSpan = element && element.closest
    ? element.closest(".redacted-plain, .redacted-retarget-source")
    : null;
  if (!sourceSpan || !elements.redactedPreview.contains(sourceSpan)) return null;
  const prefix = document.createRange();
  prefix.selectNodeContents(sourceSpan);
  try {
    prefix.setEnd(node, offset);
  } catch {
    return null;
  }
  return Number(sourceSpan.dataset.sourceStart) + prefix.toString().length;
}

function readPreviewSelection() {
  const selection = window.getSelection();
  if (!selection || selection.rangeCount === 0 || selection.isCollapsed) return null;
  const range = selection.getRangeAt(0);
  if (!elements.redactedPreview.contains(range.startContainer) || !elements.redactedPreview.contains(range.endContainer)) return null;

  const startInToken = Boolean(
    (range.startContainer.nodeType === 1 ? range.startContainer : range.startContainer.parentElement)?.closest(".redacted-token")
  );
  const endInToken = Boolean(
    (range.endContainer.nodeType === 1 ? range.endContainer : range.endContainer.parentElement)?.closest(".redacted-token")
  );
  if (startInToken || endInToken) {
    return { error: "crossing" };
  }

  const start = sourceOffsetFromBoundary(range.startContainer, range.startOffset);
  const end = sourceOffsetFromBoundary(range.endContainer, range.endOffset);
  if (start === null || end === null) return { error: "crossing" };

  const sourceStart = Math.min(start, end);
  const sourceEnd = Math.max(start, end);
  const excludedEntity = state.retargeting ? state.retargeting.entity : null;
  const crossesEntity = state.entities.some((entity) => (
    entity !== excludedEntity
    && entity.enabled !== false
    && Number.isFinite(Number(entity.start))
    && Number.isFinite(Number(entity.end))
    && sourceStart < Number(entity.start)
    && sourceEnd > Number(entity.end)
  ));
  if (crossesEntity) {
    return { error: "crossing" };
  }

  const normalized = normalizeSourceSelection(state.source, start, end, state.entities, excludedEntity);
  if (!normalized) return { error: "overlap" };
  return { ...normalized, anchorRect: compactRect(range.getBoundingClientRect()) };
}

function showManualTypePicker(anchorRect) {
  if (!elements.selectionPopover || !state.pendingSelection) return;
  const popover = elements.selectionPopover;
  popover.replaceChildren();
  popover.classList.add("is-type-picker");

  const heading = document.createElement("div");
  heading.className = "selection-popover-title";
  heading.textContent = "选择隐私类型";
  const hint = document.createElement("div");
  hint.className = "selection-popover-hint";
  hint.textContent = "不选择直接确认将设为“隐私条目”";
  const options = document.createElement("div");
  options.className = "selection-type-options";
  let selectedOption = null;
  MANUAL_ENTITY_OPTIONS.forEach((option) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "selection-type-option";
    button.textContent = option.label;
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      selectedOption = option;
      options.querySelectorAll(".selection-type-option").forEach((item) => item.classList.toggle("is-selected", item === button));
    });
    options.append(button);
  });

  const actions = document.createElement("div");
  actions.className = "selection-popover-actions selection-popover-confirm";
  const cancel = document.createElement("button");
  cancel.type = "button";
  cancel.className = "selection-action";
  cancel.textContent = "取消";
  cancel.addEventListener("click", () => {
    state.pendingSelection = null;
    hideSelectionPopover(true);
  });
  const confirm = document.createElement("button");
  confirm.type = "button";
  confirm.className = "selection-action is-primary";
  confirm.textContent = "确认";
  confirm.addEventListener("click", () => createManualEntity(state.pendingSelection, selectedOption || MANUAL_ENTITY_OPTIONS[0]));
  actions.append(cancel, confirm);
  popover.append(heading, hint, options, actions);
  positionSelectionPopover(anchorRect);
}

function showMarkSelectionActions(selection) {
  state.pendingSelection = selection;
  renderSelectionPopover([
    { label: "设为隐私条目", primary: true, run: () => showManualTypePicker(selection.anchorRect) },
    {
      label: "取消",
      run: () => {
        state.pendingSelection = null;
        hideSelectionPopover(true);
      },
    },
  ], selection.anchorRect, `已选择 ${selection.text.length} 个字符`);
}

async function createManualEntity(selection, option) {
  if (!selection) return;
  const chosen = option || MANUAL_ENTITY_OPTIONS[0];
  entitySequence += 1;
  const replacement = await allocateReplacementToken(chosen.type, chosen.label, selection.text, state.entities);
  const entity = {
    id: `manual_${entitySequence}_${Math.random().toString(36).slice(2, 7)}`,
    type: chosen.type,
    label: chosen.label,
    text: selection.text,
    start: selection.start,
    end: selection.end,
    confidence: 1,
    validated: true,
    manual: true,
    source: "manual",
    sources: ["manual_review"],
    privacy_level: chosen.privacyLevel || "PL2",
    enabled: true,
    replacement,
  };
  state.entities.push(entity);
  state.entities.sort((left, right) => left.start - right.start || left.end - right.end);
  resetAnnotationInteractionState();
  renderEntities();
  generateRedacted();
  focusReviewEntity(entity.id);
  toast(`已添加“${chosen.label}”人工复核项目`);
}

function resolveEntity(target) {
  if (!target && target !== 0) return null;
  if (typeof target === "object") return target;
  if (typeof target === "number") return state.entities[target] || null;
  if (typeof target === "string") {
    const byId = state.entities.find((e) => e.id === target);
    if (byId) return byId;
    const num = Number(target);
    if (!isNaN(num)) return state.entities[num] || null;
  }
  return null;
}

function showEntityActions(target, anchorRect, isCard = false) {
  const entity = resolveEntity(target);
  if (!entity) return;
  if (state.retargeting) {
    toast("请先完成或取消当前的范围重选。");
    return;
  }
  state.pendingSelection = null;
  const actions = [
    { label: "重新选择范围", primary: true, run: () => startEntityRetarget(entity) },
    { label: "删除条目", danger: true, run: () => deleteEntity(entity) },
    {
      label: isCard ? "定位安全副本" : "定位人工复核",
      run: () => {
        hideSelectionPopover();
        if (isCard) {
          focusRedactedEntity(entity.id || target);
        } else {
          focusReviewEntity(entity.id || target);
        }
      },
    },
    { label: "取消", run: () => hideSelectionPopover() },
  ];
  renderSelectionPopover(actions, anchorRect, entity.label);
}

function deleteEntity(target) {
  const entity = resolveEntity(target);
  if (!entity) return;
  const index = state.entities.indexOf(entity);
  if (index >= 0) {
    state.entities.splice(index, 1);
  }
  state.pendingSelection = null;
  hideSelectionPopover(true);
  renderEntities();
  generateRedacted();
  toast(`已删除“${entity.label}”条目，原文已恢复`);
}

function startEntityRetarget(target) {
  const entity = resolveEntity(target);
  if (!entity) return;
  hideSelectionPopover(true);
  state.pendingSelection = null;
  state.retargeting = {
    entity,
    snapshot: {
      start: entity.start,
      end: entity.end,
      text: entity.text,
      enabled: entity.enabled,
      replacement: entity.replacement,
    },
  };
  renderEntities();
  renderRedactedPreview();
  updateRedactedActionAvailability();
  const original = elements.redactedPreview.querySelector(".redacted-retarget-source");
  if (original) {
    const reduceMotion = typeof window !== "undefined"
      && typeof window.matchMedia === "function"
      && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (typeof scrollElementIntoContainer === "function") {
      scrollElementIntoContainer(elements.redactedPreview, original, reduceMotion ? "auto" : "smooth");
    }
    if (typeof original.focus === "function") {
      try {
        original.focus({ preventScroll: true });
      } catch (_) {
        original.focus();
      }
    }
  }
  toast(`请重新划选“${entity.label}”的准确范围，Esc 可取消`);
}

function cancelEntityRetarget(announce = true) {
  if (!state.retargeting) return;
  const { entity, snapshot } = state.retargeting;
  if (entity && snapshot) {
    entity.start = snapshot.start;
    entity.end = snapshot.end;
    entity.text = snapshot.text;
    entity.enabled = snapshot.enabled;
    entity.replacement = snapshot.replacement;
  }
  resetAnnotationInteractionState();
  renderEntities();
  generateRedacted();
  if (announce) toast("已取消范围重选");
}

async function applyEntityRetarget(selection) {
  if (!state.retargeting || !selection) return;
  const entity = state.retargeting.entity;
  const overlaps = state.entities.some((other) => (
    other !== entity
    && other.enabled !== false
    && Number.isFinite(Number(other.start))
    && Number.isFinite(Number(other.end))
    && selection.start < Number(other.end)
    && selection.end > Number(other.start)
  ));
  if (overlaps) {
    toast("所选范围与现有隐私条目重叠，请重新选择。");
    return;
  }
  entity.start = selection.start;
  entity.end = selection.end;
  entity.text = selection.text;
  entity.replacement = await allocateReplacementToken(
    entity.type,
    entity.label,
    entity.text,
    state.entities.filter((e) => e !== entity)
  );
  resetAnnotationInteractionState();
  state.entities.sort((left, right) => left.start - right.start || left.end - right.end);
  renderEntities();
  generateRedacted();
  focusReviewEntity(entity.id);
  toast(`“${entity.label}”范围已更新`);
}

function handlePreviewSelection() {
  const browserSelection = window.getSelection();
  if (!browserSelection || browserSelection.isCollapsed || !browserSelection.toString().trim()) {
    hideSelectionPopover();
    return;
  }
  const selection = readPreviewSelection();
  if (!selection) {
    hideSelectionPopover();
    return;
  }
  if (selection.error === "crossing") {
    hideSelectionPopover();
    toast("不能跨已有隐私占位符划选，请只选择普通文本。");
    return;
  }
  if (selection.error === "overlap") {
    hideSelectionPopover();
    toast("所选范围与现有隐私条目重叠，请重新选择。");
    return;
  }
  state.pendingSelection = selection;
  if (state.retargeting) {
    renderSelectionPopover([
      { label: "确认新范围", primary: true, run: () => applyEntityRetarget(state.pendingSelection) },
      { label: "取消", run: () => cancelEntityRetarget() },
    ], selection.anchorRect, `新范围：${selection.text.length} 个字符`);
    return;
  }
  showMarkSelectionActions(selection);
}

function createSourceSpan(text, sourceStart, sourceEnd, className = "redacted-plain") {
  const span = document.createElement("span");
  span.className = className;
  span.dataset.sourceStart = String(sourceStart);
  span.dataset.sourceEnd = String(sourceEnd);
  span.textContent = text;
  return span;
}

function appendTextSegment(fragment, segment, retargetEntity) {
  if (!retargetEntity || retargetEntity.start < segment.sourceStart || retargetEntity.end > segment.sourceEnd) {
    fragment.append(createSourceSpan(segment.text, segment.sourceStart, segment.sourceEnd));
    return;
  }
  if (retargetEntity.start > segment.sourceStart) {
    fragment.append(createSourceSpan(
      state.source.slice(segment.sourceStart, retargetEntity.start),
      segment.sourceStart,
      retargetEntity.start,
    ));
  }
  const original = createSourceSpan(
    state.source.slice(retargetEntity.start, retargetEntity.end),
    retargetEntity.start,
    retargetEntity.end,
    "redacted-retarget-source",
  );
  original.tabIndex = 0;
  original.setAttribute("aria-label", `${retargetEntity.label}当前范围，重新圈选正确文字`);
  fragment.append(original);
  if (retargetEntity.end < segment.sourceEnd) {
    fragment.append(createSourceSpan(
      state.source.slice(retargetEntity.end, segment.sourceEnd),
      retargetEntity.end,
      segment.sourceEnd,
    ));
  }
}

function renderRedactedPreview() {
  if (!elements.redactedPreview) return;
  elements.redactedPreview.replaceChildren();
  const retargetEntity = state.retargeting ? state.retargeting.entity : null;
  elements.redactedPreview.classList.toggle("is-retargeting", Boolean(retargetEntity));
  if (!state.source) {
    const empty = document.createElement("span");
    empty.className = "redacted-preview-empty";
    empty.textContent = "复核命中项并生成脱敏文本后，结果会出现在这里。";
    elements.redactedPreview.append(empty);
    return;
  }

  const fragment = document.createDocumentFragment();
  if (retargetEntity) {
    const instruction = document.createElement("div");
    instruction.className = "redacted-retarget-instruction";
    const instructionText = document.createElement("span");
    instructionText.textContent = `重新划选“${retargetEntity.label}”的准确范围`;
    const cancel = document.createElement("button");
    cancel.type = "button";
    cancel.textContent = "取消重选";
    cancel.addEventListener("click", () => cancelEntityRetarget());
    instruction.append(instructionText, cancel);
    fragment.append(instruction);
  }
  buildRedactedSegments(state.source, state.entities, retargetEntity).forEach((segment) => {
    if (segment.kind === "text") {
      appendTextSegment(fragment, segment, retargetEntity);
      return;
    }
    const token = document.createElement("button");
    token.type = "button";
    token.className = "redacted-token";
    token.dataset.entityIndex = String(segment.entityIndex);
    if (segment.entityId) token.dataset.entityId = segment.entityId;
    token.textContent = segment.text;
    token.title = "定位到对应的人工复核项目";
    token.setAttribute("aria-label", `定位到${segment.label}的人工复核项目`);
    token.addEventListener("click", () => focusReviewEntity(segment.entityId || segment.entityIndex));
    token.addEventListener("contextmenu", (event) => {
      event.preventDefault();
      showEntityActions(segment.entityId || segment.entityIndex, pointRect(event.clientX, event.clientY), false);
    });
    token.disabled = Boolean(retargetEntity);
    fragment.append(token);
  });
  elements.redactedPreview.append(fragment);
}

function renderEntities() {
  elements.entityList.replaceChildren();
  elements.entityList.classList.toggle("is-retargeting", Boolean(state.retargeting));
  elements.entityEmpty.hidden = state.entities.length > 0;
  elements.reviewFooter.hidden = state.entities.length === 0;
  elements.entityCount.textContent = `${state.entities.filter((item) => item.enabled).length} / ${state.entities.length} 项`;

  const SOURCE_NAMES = {
    multilingual_rules: "基础规则",
    chinese_ie: "中文语义",
    gliner_pii: "GLiNER通用",
    memprivacy: "MemPrivacy深度",
  };

  state.entities.forEach((entity, index) => {
    const card = document.createElement("article");
    card.className = "entity-card";
    card.dataset.entityIndex = String(index);
    if (entity.id) card.dataset.entityId = entity.id;
    card.tabIndex = 0;
    card.classList.toggle("is-retargeting", Boolean(state.retargeting && state.retargeting.entity === entity));
    card.setAttribute("aria-label", `${entity.label}复核项目；按回车定位到安全副本中的占位符`);

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = entity.enabled;
    checkbox.disabled = Boolean(state.retargeting);
    checkbox.setAttribute("aria-label", `${entity.label}是否替换`);
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

    const pl = entity.privacy_level || "PL2";
    const plBadge = document.createElement("span");
    plBadge.className = `pl-tag pl-${pl.toLowerCase()}`;
    plBadge.textContent = pl;

    const score = document.createElement("span");
    score.className = "entity-score";
    score.textContent = entity.manual ? "人工" : (entity.validated ? "已校验" : `${Math.round(entity.confidence * 100)}%`);

    const linkButton = document.createElement("button");
    linkButton.type = "button";
    linkButton.className = "entity-link-button";
    linkButton.textContent = "↔";
    linkButton.disabled = Boolean(state.retargeting);
    linkButton.title = "定位到安全副本中的占位符";
    linkButton.setAttribute("aria-label", `定位到安全副本中的${entity.label}占位符`);
    linkButton.addEventListener("click", (event) => {
      event.stopPropagation();
      focusRedactedEntity(entity.id || index);
    });
    const menuButton = document.createElement("button");
    menuButton.type = "button";
    menuButton.className = "entity-menu-button";
    menuButton.textContent = "⋯";
    menuButton.title = "项目操作";
    menuButton.disabled = Boolean(state.retargeting);
    menuButton.setAttribute("aria-label", `${entity.label}项目操作`);
    menuButton.setAttribute("aria-haspopup", "menu");
    menuButton.addEventListener("click", (event) => {
      event.stopPropagation();
      showEntityActions(entity.id || index, compactRect(menuButton.getBoundingClientRect()), true);
    });
    top.append(type, plBadge, score, linkButton, menuButton);

    const value = document.createElement("span");
    value.className = "entity-value";
    value.textContent = maskPreview(entity.text);
    value.title = "点击原始输入框可核对全文";

    const replacement = document.createElement("input");
    replacement.className = "entity-replacement";
    replacement.value = entity.replacement;
    replacement.disabled = Boolean(state.retargeting);
    replacement.setAttribute("aria-label", `${entity.label}的替换占位符`);
    replacement.addEventListener("input", () => {
      state.entities[index].replacement = replacement.value.trim();
      generateRedacted();
    });

    const sources = document.createElement("div");
    sources.className = "source-tags";
    const readableSources = (entity.sources || []).map((s) => SOURCE_NAMES[s] || s).join(" + ") || "本地规则";
    sources.textContent = `引擎: ${readableSources}`;
    main.append(top, value, replacement, sources);
    card.append(checkbox, main);
    card.addEventListener("click", (event) => {
      if (event.target.closest && event.target.closest("input, button")) return;
      if (state.retargeting) return;
      focusRedactedEntity(entity.id || index);
    });
    card.addEventListener("contextmenu", (event) => {
      if (event.target.closest && event.target.closest("input")) return;
      event.preventDefault();
      showEntityActions(entity.id || index, pointRect(event.clientX, event.clientY), true);
    });
    card.addEventListener("keydown", (event) => {
      if (event.target !== card || (event.key !== "Enter" && event.key !== " ")) return;
      event.preventDefault();
      focusRedactedEntity(entity.id || index);
    });
    elements.entityList.append(card);
  });
}

function invalidateRedactedState(hasConflict = false) {
  elements.redactedText.value = "";
  state.vault = [];
  renderRedactedPreview();
  setCounter(elements.redactedText, elements.redactedCounter);
  elements.copyRedactedButton.disabled = true;
  elements.copyPromptButton.disabled = true;
  elements.exportVaultButton.disabled = true;
  elements.vaultSummary.textContent = hasConflict ? "当前脱敏结果无效，映射已作废" : "尚未生成映射";
  elements.activeVaultBadge.textContent = "当前会话：0 个映射";
}

function generateRedacted() {
  if (!state.source) {
    invalidateRedactedState(false);
    return;
  }
  const enabled = state.entities.filter((entity) => entity.enabled && entity.replacement);

  // Invariant 1: same replacement token must map to the exact same original text
  const tokenToValue = new Map();
  for (const entity of enabled) {
    if (tokenToValue.has(entity.replacement)) {
      const existingVal = tokenToValue.get(entity.replacement);
      if (existingVal !== entity.text) {
        showNotice("当前脱敏结果无效：同一占位符对应了不同原文内容，请修复冲突后再复制。", true);
        invalidateRedactedState(true);
        return;
      }
    } else {
      tokenToValue.set(entity.replacement, entity.text);
    }
  }

  // Invariant 2: enabled entities must not overlap in character spans
  const sorted = [...enabled].sort((a, b) => a.start - b.start || a.end - b.end);
  for (let i = 0; i < sorted.length - 1; i++) {
    if (sorted[i].end > sorted[i + 1].start) {
      showNotice("当前脱敏结果无效：存在重叠的隐私条目范围，请调整后再复制。", true);
      invalidateRedactedState(true);
      return;
    }
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
  renderRedactedPreview();
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
    const policyLevel = elements.policySelect ? elements.policySelect.value : "PL2";
    const result = await api("/api/detect", {
      method: "POST",
      body: JSON.stringify({
        text,
        use_model: elements.useModelToggle ? elements.useModelToggle.checked : false,
        policy_level: policyLevel,
      }),
    });
    resetAnnotationInteractionState();
    state.source = text;
    state.entities = await prepareEntities(result.entities);
    renderEntities();
    generateRedacted();
    const summary = state.entities.length
      ? `检测完成 [${result.policy_level || policyLevel}]：发现 ${state.entities.length} 项，耗时 ${result.processing_ms} ms。请逐项复核后再发送。`
      : `未发现符合 [${result.policy_level || policyLevel}] 策略的隐私字段。自动检测可能漏检，请人工检查原文。`;
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

function deriveModelRuntimeState(slot, deviceData = {}, catalogModel = null) {
  const det = (slot && slot.detector) || {};
  const isInstalled = Boolean(det.installed);
  const isReady = Boolean(det.ready);
  const reqDevice = (deviceData.requested_device || "auto").toLowerCase();
  const hw = deviceData.hardware || {};
  const hasNvidia = Boolean(hw.nvidia_available);
  const runtimes = deviceData.runtimes || {};
  const tCpu = runtimes.torch_cpu || {};
  const tCuda = runtimes.torch_cuda || {};
  const cpuReady = Boolean(tCpu.installed && tCpu.verified);
  const cudaReady = Boolean(tCuda.installed && tCuda.verified && tCuda.cuda_available);

  const supportsCpu = catalogModel ? catalogModel.supports_cpu !== false : true;
  const supportsCuda = catalogModel ? catalogModel.supports_cuda !== false : true;

  let installButtonText = null;
  let runtimeMissing = false;
  let missingRuntimeName = null;

  if (!isInstalled) {
    installButtonText = "从魔搭下载安装";
  } else if (!isReady) {
    runtimeMissing = true;
    if (reqDevice === "cpu") {
      missingRuntimeName = "CPU";
      installButtonText = "安装 CPU 运行时";
    } else if (reqDevice === "cuda") {
      missingRuntimeName = "CUDA";
      installButtonText = "安装 CUDA 运行时";
    } else {
      // auto
      if (hasNvidia && supportsCuda && !cudaReady) {
        missingRuntimeName = "CUDA (推荐)";
        installButtonText = "安装推荐运行时";
      } else if (!cpuReady) {
        missingRuntimeName = "CPU";
        installButtonText = "安装推荐运行时";
      } else {
        missingRuntimeName = "隔离";
        installButtonText = "安装推荐运行时";
      }
    }
  }

  return {
    isInstalled,
    isReady,
    runtimeMissing,
    missingRuntimeName,
    installButtonText,
    showUninstall: isInstalled,
  };
}

function updateModelUI(data) {
  state.model = data;
  const reg = data.registry || {};
  const slots = reg.slots || {};

  // Device & Runtime Telemetry
  const dev = data.device || {};
  const hw = dev.hardware || {};
  const runtimes = dev.runtimes || {};
  const modelDevices = dev.model_devices || {};

  // Check overall model readiness and installation for optional enhanced detectors
  const glinerSlot = slots.general_pii || {};
  const memSlot = slots.semantic_privacy || {};
  const glinerInstalled = Boolean(glinerSlot.detector && glinerSlot.detector.installed);
  const memInstalled = Boolean(memSlot.detector && memSlot.detector.installed);
  const anyModelInstalled = glinerInstalled || memInstalled;

  const glinerReady = Boolean(glinerSlot.detector && glinerSlot.detector.ready);
  const memReady = Boolean(memSlot.detector && memSlot.detector.ready);
  const anyModelReady = glinerReady || memReady;

  if (elements.useModelToggle) {
    elements.useModelToggle.disabled = !anyModelReady;
    if (!anyModelReady) elements.useModelToggle.checked = false;
  }
  if (elements.modelInlineStatus) {
    if (data.installing) {
      let installMsg = "正在后台安装模型或运行环境…";
      const states = data.install_states || {};
      for (const mId in states) {
        if (states[mId] && states[mId].detail) {
          installMsg = states[mId].detail;
          break;
        }
      }
      elements.modelInlineStatus.textContent = installMsg;
    } else if (anyModelReady) {
      elements.modelInlineStatus.textContent = "增强模型已就绪 (GLiNER / MemPrivacy)";
    } else if (anyModelInstalled) {
      const req = (dev.requested_device || "auto").toLowerCase();
      if (req === "cpu") {
        elements.modelInlineStatus.textContent = "增强模型已安装，但 CPU 运行时未就绪。";
      } else if (req === "cuda") {
        elements.modelInlineStatus.textContent = "增强模型已安装，但 CUDA 运行时未就绪。";
      } else {
        elements.modelInlineStatus.textContent = "增强模型已安装，但计算运行时未就绪。";
      }
    } else {
      elements.modelInlineStatus.textContent = "未安装增强模型（基础规则与中文语义始终可用）";
    }
  }

  if (elements.deviceStatusBadge) {
    elements.deviceStatusBadge.textContent = dev.actual_device === "cuda" ? "NVIDIA CUDA 加速" : "CPU 运行模式";
    elements.deviceStatusBadge.className = `status-badge${dev.actual_device === "cuda" ? " is-ready" : ""}`;
  }

  if (elements.deviceSelect && dev.requested_device) {
    elements.deviceSelect.value = dev.requested_device;
  }

  if (elements.deviceTelemetry) {
    elements.deviceTelemetry.replaceChildren();

    // 1. Hardware section
    const hwCard = document.createElement("div");
    hwCard.style.padding = "8px 12px";
    hwCard.style.borderRadius = "6px";
    hwCard.style.background = "var(--surface)";
    hwCard.style.border = "1px solid var(--line)";

    const hwTitle = document.createElement("div");
    hwTitle.style.fontWeight = "600";
    hwTitle.style.marginBottom = "4px";
    if (hw.nvidia_available && hw.gpus && hw.gpus.length > 0) {
      const gpu = hw.gpus[0];
      hwTitle.textContent = `✅ NVIDIA 显卡硬件: ${gpu.name}`;
      hwTitle.style.color = "var(--brand-green, #137333)";

      const hwDesc = document.createElement("div");
      hwDesc.style.color = "var(--text-muted)";
      hwDesc.style.fontSize = "12px";
      hwDesc.textContent = `驱动版本: ${hw.driver_version || "未知"} · 显存: ${gpu.memory_total_mb ? gpu.memory_total_mb + " MB" : "未知"} (共 ${hw.gpu_count} 块)`;
      hwCard.append(hwTitle, hwDesc);
    } else {
      hwTitle.textContent = "○ NVIDIA 显卡硬件: 未检测到";
      hwTitle.style.color = "var(--text-muted)";
      const hwDesc = document.createElement("div");
      hwDesc.style.color = "var(--text-muted)";
      hwDesc.style.fontSize = "12px";
      hwDesc.textContent = hw.reason || "未找到 nvidia-smi 驱动或无可用英伟达 GPU。";
      hwCard.append(hwTitle, hwDesc);
    }

    // 2. PyTorch Runtime section (GLiNER, SiameseUIE, MemPrivacy)
    const torchCard = document.createElement("div");
    torchCard.style.padding = "8px 12px";
    torchCard.style.borderRadius = "6px";
    torchCard.style.background = "var(--surface)";
    torchCard.style.border = "1px solid var(--line)";

    const torchTitle = document.createElement("div");
    torchTitle.style.fontWeight = "600";
    torchTitle.style.marginBottom = "4px";

    const tCuda = runtimes.torch_cuda || {};
    const tCpu = runtimes.torch_cpu || {};
    const cudaOk = tCuda.installed && tCuda.verified && tCuda.cuda_available;
    const cpuOk = tCpu.installed && tCpu.verified;

    if (cudaOk && cpuOk) {
      torchTitle.textContent = "✅ PyTorch 运行时: CUDA 与 CPU 均已就绪";
      torchTitle.style.color = "var(--brand-green, #137333)";
    } else if (cudaOk) {
      torchTitle.textContent = "✅ PyTorch CUDA 运行时: 已就绪 (CPU 运行时未安装)";
      torchTitle.style.color = "var(--brand-green, #137333)";
    } else if (cpuOk) {
      torchTitle.textContent = "○ PyTorch CPU 运行时: 已就绪 (CUDA 未就绪)";
      torchTitle.style.color = "var(--text)";
    } else {
      torchTitle.textContent = "○ PyTorch 运行时: 未就绪 (按需安装)";
      torchTitle.style.color = "var(--text-muted)";
    }
    const torchDesc = document.createElement("div");
    torchDesc.style.color = "var(--text-muted)";
    torchDesc.style.fontSize = "12px";
    const torchDevInfo = modelDevices.torch || {};
    torchDesc.textContent = `服务模型: GLiNER, SiameseUIE, MemPrivacy · 当前分配设备: ${torchDevInfo.device ? torchDevInfo.device.toUpperCase() : "CPU"}`;
    torchCard.append(torchTitle, torchDesc);

    elements.deviceTelemetry.append(hwCard, torchCard);
  } else if (elements.deviceDetail) {
    elements.deviceDetail.textContent = hw.nvidia_available
      ? `检测到 GPU: ${hw.driver_version ? "驱动 " + hw.driver_version : "就绪"} (目标: ${dev.requested_device})`
      : `未检测到 NVIDIA CUDA 环境 (使用 CPU 推理)`;
  }

  // Render Slots in #slotListContainer
  if (elements.slotListContainer) {
    elements.slotListContainer.replaceChildren();

    const slotOrder = ["built_in", "chinese_ie", "general_pii", "semantic_privacy"];
    slotOrder.forEach((slotId) => {
      const slot = slots[slotId];
      if (!slot) return;

      const item = document.createElement("div");
      item.className = "slot-item";

      const header = document.createElement("div");
      header.className = "slot-item-header";

      const title = document.createElement("div");
      title.className = "slot-item-title";
      title.textContent = slot.name;

      const badge = document.createElement("span");
      badge.className = "status-badge";

      const det = slot.detector || {};
      const isReady = det.ready;
      const isInstalled = det.installed;
      const installState = data.install_states && data.install_states[slot.active_model];

      const activeDescriptor = (slot.available_models || []).find((m) => m.id === slot.active_model) || null;
      const runtimeState = deriveModelRuntimeState(slot, dev, activeDescriptor);

      if (slotId === "built_in") {
        badge.textContent = "系统内置 (始终运行)";
        badge.className = "status-badge is-ready";
      } else if (slotId === "chinese_ie") {
        badge.textContent = isInstalled ? (isReady ? "已加载 SiameseUIE" : "权重已就绪 (缺少运行时)") : "系统内置语法 (始终就绪)";
        badge.className = (isInstalled && isReady) || !isInstalled ? "status-badge is-ready" : "status-badge";
      } else if (installState && installState.state === "installing") {
        badge.textContent = "正在安装";
        badge.className = "status-badge";
      } else if (isReady) {
        badge.textContent = "已就绪";
        badge.className = "status-badge is-ready";
      } else if (isInstalled) {
        badge.textContent = "权重已就绪 (缺少运行时)";
        badge.className = "status-badge";
      } else {
        badge.textContent = "未安装";
        badge.className = "status-badge";
      }
      header.append(title, badge);

      const desc = document.createElement("div");
      desc.className = "slot-item-desc";
      desc.textContent = slot.description;

      const controls = document.createElement("div");
      controls.className = "slot-item-controls";

      if (slot.available_models && slot.available_models.length > 0) {
        const select = document.createElement("select");
        select.style.padding = "4px 8px";
        select.style.borderRadius = "6px";
        select.style.border = "1px solid var(--line)";
        select.style.fontSize = "12px";

        slot.available_models.forEach((m) => {
          const opt = document.createElement("option");
          opt.value = m.id;
          opt.textContent = `${m.display_name} (${m.size_hint}, ${m.license})`;
          if (m.id === slot.active_model) opt.selected = true;
          select.append(opt);
        });

        select.addEventListener("change", async () => {
          try {
            await api("/api/model/select", {
              method: "POST",
              body: JSON.stringify({ slot: slotId, model: select.value }),
            });
            toast(`已切换激活模型为 ${select.value}`);
            await refreshModelStatus();
          } catch (err) {
            toast(`切换失败: ${err.message}`);
          }
        });
        controls.append(select);
      }

      const actions = document.createElement("div");
      actions.style.display = "flex";
      actions.style.gap = "8px";
      actions.style.alignItems = "center";

      if (slotId !== "built_in") {
        if (runtimeState.installButtonText && slot.active_model) {
          const installBtn = document.createElement("button");
          installBtn.className = "button button-primary button-small";
          installBtn.textContent = runtimeState.installButtonText;
          installBtn.disabled = !data.is_admin || data.installing;
          installBtn.addEventListener("click", () => installModel(slot.active_model, isInstalled));
          actions.append(installBtn);
        }
        if (runtimeState.showUninstall && slot.active_model) {
          const uninstallBtn = document.createElement("button");
          uninstallBtn.className = "button button-ghost button-small";
          uninstallBtn.textContent = "卸载模型";
          uninstallBtn.disabled = !data.is_admin || data.installing;
          uninstallBtn.addEventListener("click", () => uninstallModel(slot.active_model));
          actions.append(uninstallBtn);
        }
      }

      controls.append(actions);
      item.append(header, desc, controls);
      elements.slotListContainer.append(item);
    });
  }

  // Render Shared Models
  if (elements.sharedModelsList) {
    elements.sharedModelsList.replaceChildren();
    const candidates = data.shared_candidates || [];
    const sharedDirs = data.shared_dirs || [];

    if (candidates.length === 0) {
      const emptyHint = document.createElement("p");
      emptyHint.style.fontSize = "12px";
      emptyHint.style.color = "var(--text-muted)";
      const dirText = sharedDirs.length > 0 ? sharedDirs.join(" 或 ") : "AI 脱敏器/models";
      emptyHint.textContent = `未在共享目录 (${dirText}) 中扫描到匹配的模型文件夹。将模型文件夹放置后点击“扫描共享目录”即可一键导入。`;
      elements.sharedModelsList.append(emptyHint);
    } else {
      candidates.forEach((cand) => {
        const row = document.createElement("div");
        row.style.display = "flex";
        row.style.alignItems = "center";
        row.style.justifyContent = "space-between";
        row.style.padding = "8px 12px";
        row.style.borderRadius = "6px";
        row.style.background = "var(--surface)";
        row.style.border = "1px solid var(--line)";

        const info = document.createElement("div");
        const titleStrong = document.createElement("strong");
        titleStrong.textContent = cand.display_name || "";
        const folderSmall = document.createElement("small");
        folderSmall.style.color = "var(--text-muted)";
        folderSmall.textContent = ` (${cand.folder_name || ""} · ${cand.approx_size || ""})`;
        const statusDiv = document.createElement("div");
        statusDiv.style.fontSize = "11px";
        statusDiv.style.color = cand.valid ? "var(--brand-green, #137333)" : "var(--danger, #d93025)";
        statusDiv.textContent = cand.valid ? "格式校验通过" : (cand.reason || "校验失败");
        info.append(titleStrong, " ", folderSmall, statusDiv);

        const btn = document.createElement("button");
        btn.className = "button button-secondary button-small";
        btn.textContent = "导入到应用";
        btn.disabled = !cand.valid;
        btn.addEventListener("click", async () => {
          btn.disabled = true;
          btn.textContent = "正在导入…";
          try {
            const res = await api("/api/model/shared/import", {
              method: "POST",
              body: JSON.stringify({ model: cand.model_id, source_path: cand.source_path }),
            });
            toast(res.message || "模型导入成功！");
            await refreshModelStatus();
          } catch (err) {
            toast(`导入失败: ${err.message}`);
          } finally {
            btn.disabled = false;
            btn.textContent = "导入到应用";
          }
        });

        row.append(info, btn);
        elements.sharedModelsList.append(row);
      });
    }
  }

  if (elements.installLog) {
    elements.installLog.textContent = data.log_tail && data.log_tail.length ? data.log_tail.join("\n") : "暂无日志";
  }

  clearTimeout(state.modelPoll);
  if (data.installing) state.modelPoll = setTimeout(refreshModelStatus, 3000);
}

async function refreshModelStatus() {
  try {
    updateModelUI(await api("/api/model/status"));
  } catch (error) {
    if (elements.modelInlineStatus) elements.modelInlineStatus.textContent = "无法读取模型状态";
  }
}

async function installModel(modelId = "gliner-pii-edge", isWeightsAlreadyInstalled = false) {
  const confirmMsg = isWeightsAlreadyInstalled
    ? `检测到模型权重已就绪，将为您安装并配置当前设备所需的隔离运行环境。继续吗？`
    : `在线安装将从 ModelScope (魔搭社区) 下载运行库与模型权重。继续吗？`;
  const confirmed = window.confirm(confirmMsg);
  if (!confirmed) return;
  try {
    const result = await api("/api/model/install", { method: "POST", body: JSON.stringify({ model: modelId }) });
    updateModelUI({ ...result.status, is_admin: true });
    if ($("installLogPanel")) $("installLogPanel").open = true;
    toast(isWeightsAlreadyInstalled ? "已开始在后台配置隔离运行环境…" : "模型已开始在后台从 ModelScope 下载与安装");
  } catch (error) {
    toast(error.message);
    await refreshModelStatus();
  }
}

async function importModel() {
  const modelType = elements.importModelType ? elements.importModelType.value : "gliner-pii-edge";
  const sourcePath = (elements.importSourcePath ? elements.importSourcePath.value : "").trim();
  if (!sourcePath) {
    toast("请输入已授权的模型源目录路径");
    return;
  }
  if (elements.confirmImportButton) elements.confirmImportButton.disabled = true;
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
    if (elements.confirmImportButton) elements.confirmImportButton.disabled = false;
  }
}

async function uninstallModel(modelId = "gliner-pii-edge") {
  if (!window.confirm(`确定要卸载应用目录内的该模型 (${modelId}) 吗？`)) return;
  try {
    const result = await api("/api/model/uninstall", {
      method: "POST",
      body: JSON.stringify({ model: modelId }),
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
    toast("全部模型已成功重载");
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
  resetAnnotationInteractionState();
  state.source = "";
  state.entities = [];
  state.vault = [];
  [elements.sourceText, elements.redactedText, elements.replyText, elements.restoredText].forEach((node) => { node.value = ""; });
  renderEntities();
  renderRedactedPreview();
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
if (elements.scanSharedButton) {
  elements.scanSharedButton.addEventListener("click", async () => {
    toast("正在扫描共享模型目录…");
    await refreshModelStatus();
  });
}
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

if (elements.redactedPreview) {
  elements.redactedPreview.addEventListener("mouseup", (event) => {
    if (event.button !== 0) return;
    setTimeout(handlePreviewSelection, 20);
  });
  elements.redactedPreview.addEventListener("touchend", () => {
    setTimeout(handlePreviewSelection, 20);
  });
  elements.redactedPreview.addEventListener("keyup", (event) => {
    if (event.key === "Shift" || event.key.startsWith("Arrow")) {
      setTimeout(handlePreviewSelection, 20);
    }
  });
}

document.addEventListener("mousedown", (event) => {
  if (!elements.selectionPopover || elements.selectionPopover.hidden) return;
  if (elements.selectionPopover.contains(event.target)) return;
  if (event.target.closest && event.target.closest(".entity-menu-button, .redacted-token")) return;
  hideSelectionPopover();
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    if (state.retargeting) {
      cancelEntityRetarget();
    } else if (elements.selectionPopover && !elements.selectionPopover.hidden) {
      hideSelectionPopover(true);
    }
  }
});

window.addEventListener("resize", () => {
  if (elements.selectionPopover && !elements.selectionPopover.hidden) {
    hideSelectionPopover();
  }
});

if (elements.redactedPreview) {
  elements.redactedPreview.addEventListener("scroll", () => {
    if (elements.selectionPopover && !elements.selectionPopover.hidden) {
      hideSelectionPopover(true);
    }
  }, { passive: true });
}

window.addEventListener("scroll", () => {
  if (elements.selectionPopover && !elements.selectionPopover.hidden) {
    hideSelectionPopover(true);
  }
}, { passive: true });

renderEntities();
renderRedactedPreview();
refreshModelStatus();

if (typeof window !== "undefined") {
  window.deriveModelRuntimeState = deriveModelRuntimeState;
}
if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    deriveModelRuntimeState,
    allocateReplacementToken,
    resetAnnotationInteractionState,
    invalidateRedactedState,
    prepareEntities,
    generateRedacted,
    normalizeSourceSelection,
    buildRedactedSegments,
    buildRetargetSegments,
    createManualEntity,
    scrollElementIntoContainer,
  };
}
