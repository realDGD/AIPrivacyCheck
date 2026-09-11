"""Merge deterministic and model spans into a non-overlapping review list."""

from typing import Dict, Iterable, List, Tuple

from .entities import Entity


def _overlaps(left: Entity, right: Entity) -> bool:
    return left.start < right.end and right.start < left.end


def _quality(entity: Entity) -> Tuple[int, int, float, int]:
    return (
        1 if entity.validated else 0,
        entity.priority,
        entity.confidence,
        entity.end - entity.start,
    )


def merge_entities(entities: Iterable[Entity]) -> List[Entity]:
    """Deduplicate identical spans and resolve overlaps conservatively."""

    exact: Dict[Tuple[int, int, str], Entity] = {}
    for entity in entities:
        key = (entity.start, entity.end, entity.entity_type)
        current = exact.get(key)
        if current is None:
            exact[key] = entity
        else:
            exact[key] = current.with_sources(entity.sources, entity.confidence)

    selected: List[Entity] = []
    for candidate in sorted(
        exact.values(),
        key=lambda item: (_quality(item), -(item.start), item.end - item.start),
        reverse=True,
    ):
        conflicts = [existing for existing in selected if _overlaps(candidate, existing)]
        if not conflicts:
            selected.append(candidate)
            continue
        if all(_quality(candidate) > _quality(existing) for existing in conflicts):
            selected = [existing for existing in selected if existing not in conflicts]
            selected.append(candidate)

    return sorted(selected, key=lambda item: (item.start, item.end))
