"""Safe exact span resolution for semantic/generative models.

Guarantees:
1. Every emitted Entity has strictly verified character offsets [start, end)
   such that text[start:end] == target_text.
2. Under NO circumstances performs global or blind string replacement.
3. If target_text appears multiple times:
   - Uses cursor-based sequential scanning and surrounding context matching.
   - If ambiguous and cannot be securely pinpointed: skips entity and emits a warning.
"""

from typing import Iterable, List, Optional, Tuple

from .entities import Entity
from .taxonomy import resolve_privacy_level


def resolve_semantic_spans(
    full_text: str,
    predictions: Iterable[Tuple[str, str, float, Optional[str], Optional[str], Optional[str]]],
    source_name: str,
) -> Tuple[List[Entity], List[str]]:
    """Resolve raw semantic extractions (entity_type, text_snippet, confidence, semantic_type, model_pl, context_hint)
    into validated non-overlapping Entity objects with exact half-open character offsets.
    """
    entities: List[Entity] = []
    warnings: List[str] = []
    occupied_spans: List[Tuple[int, int]] = []
    cursor = 0

    def is_occupied(s: int, e: int) -> bool:
        return any(os < e and s < oe for os, oe in occupied_spans)

    for item in predictions:
        if isinstance(item, dict):
            etype = item.get("entity_type") or item.get("type") or "PII"
            target_text = item.get("text") or item.get("target_text") or ""
            conf = float(item.get("confidence", 0.8))
            sem_type = item.get("semantic_type")
            model_pl = item.get("privacy_level") or item.get("model_pl")
            context_hint = item.get("context_hint") or item.get("context_snippet")
        elif len(item) == 3:
            etype, target_text, conf = item
            sem_type, model_pl, context_hint = None, None, None
        elif len(item) == 4:
            etype, target_text, conf, sem_type = item
            model_pl, context_hint = None, None
        elif len(item) == 5:
            etype, target_text, conf, sem_type, model_pl = item
            context_hint = None
        else:
            etype, target_text, conf, sem_type, model_pl, context_hint = item

        if not target_text or not isinstance(target_text, str):
            continue

        target_text = target_text.strip()
        if not target_text:
            continue

        # Count total occurrences in full text
        occurrences: List[int] = []
        pos = 0
        while True:
            idx = full_text.find(target_text, pos)
            if idx == -1:
                break
            occurrences.append(idx)
            pos = idx + 1

        if not occurrences:
            warnings.append(f"语义模型提取的实体 '{target_text}' 在原文中未找到对应片段，已忽略。")
            continue

        matched_start: Optional[int] = None

        if len(occurrences) == 1:
            cand_s = occurrences[0]
            cand_e = cand_s + len(target_text)
            if not is_occupied(cand_s, cand_e):
                matched_start = cand_s
            else:
                warnings.append(f"实体 '{target_text}' 所在唯一区间已由更高优先级规则占用。")
                continue
        else:
            # Multiple occurrences: attempt cursor-based and context-guided disambiguation
            cands_after_cursor = [idx for idx in occurrences if idx >= cursor and not is_occupied(idx, idx + len(target_text))]
            if context_hint:
                # 1. Direct exact placement if context_hint is in full_text and contains target_text
                hint_pos = full_text.find(context_hint)
                if hint_pos != -1 and target_text in context_hint:
                    cand_pos = hint_pos + context_hint.find(target_text)
                    if cand_pos in occurrences and not is_occupied(cand_pos, cand_pos + len(target_text)):
                        matched_start = cand_pos

                # 2. Heuristic overlap if not directly placed
                if matched_start is None:
                    best_idx = None
                    best_score = -1
                    for idx in (cands_after_cursor or [i for i in occurrences if not is_occupied(i, i + len(target_text))]):
                        window_start = max(0, idx - 12)
                        window_end = min(len(full_text), idx + len(target_text) + 12)
                        local_window = full_text[window_start:window_end]
                        score = sum(1 for c in context_hint if c in local_window)
                        if score > best_score:
                            best_score = score
                            best_idx = idx
                    if best_idx is not None and best_score >= 2:
                        matched_start = best_idx
            elif cands_after_cursor:
                matched_start = cands_after_cursor[0]
            else:
                # Ambiguous: cannot safely determine which occurrence the model meant
                warnings.append(f"实体 '{target_text}' 在文本中出现 {len(occurrences)} 次且上下文无法精确唯一定位，已安全跳过以防误脱敏。")
                continue

        if matched_start is not None:
            matched_end = matched_start + len(target_text)
            # Strict sanity invariant
            if full_text[matched_start:matched_end] == target_text:
                pl = resolve_privacy_level(etype, semantic_type=sem_type, model_level=model_pl)
                entities.append(
                    Entity(
                        entity_type=etype,
                        start=matched_start,
                        end=matched_end,
                        text=target_text,
                        confidence=conf,
                        sources=(source_name,),
                        validated=False,
                        privacy_level=pl,
                        semantic_type=sem_type,
                    )
                )
                occupied_spans.append((matched_start, matched_end))
                cursor = matched_end
            else:
                warnings.append(f"实体 '{target_text}' 偏移校验不一致，已跳过。")

    return entities, warnings
