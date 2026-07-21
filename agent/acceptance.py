"""根据初始意图和工具结果计算验收进度。"""
from __future__ import annotations

from typing import Any

from memory import normalize_hull_number


REQUIREMENT_LABELS = {
    "complete_track_scope": "完整轨迹范围",
    "exact_hull_classification": "稳定舷号精确查库",
    "registry_catalog": "完整先验库",
    "keyframe_evidence": "剩余轨迹正式关键帧",
    "registry_image_classification": "剩余轨迹与库参考图匹配",
    "gray_verification": "灰区视觉核验",
    "registry_coverage": "先验库向量覆盖",
    "segment_verification": "目标船片段核验",
    "relation_result": "在库/未在库关系列表",
    "deduplicated_count": "跨轨迹去重数量",
    "target_match": "目标匹配证据",
    "registry_query": "先验库查询结果",
}


def _track_id(value: Any) -> str:
    return str(value) if value is not None else ""


def _result_values(scope: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        value
        for key, value in scope.items()
        if key != "acceptance" and isinstance(value, dict)
    ]


def build_acceptance_progress(
    intent: dict[str, Any],
    scope: dict[str, Any],
) -> dict[str, Any]:
    """生成供 PlanAgent 和 ReflectAgent 共用的结构化验收进度。"""
    results = _result_values(scope)
    relation = str(intent.get("registryRelation") or "any")
    operation = str(intent.get("operation") or "list")
    target_kind = str(intent.get("targetKind") or "all")
    target_scope = str(intent.get("targetScope") or "track_memory")

    tracks: dict[str, dict[str, Any]] = {}
    track_totals: list[int] = []
    terminal_track_page = False
    track_result_seen = False
    for result in results:
        rows = result.get("tracks")
        if not isinstance(rows, list) or "trackIds" not in result:
            continue
        track_result_seen = True
        for row in rows:
            if isinstance(row, dict) and row.get("trackId") is not None:
                tracks[_track_id(row.get("trackId"))] = row
        total = result.get("totalTrackCount")
        if isinstance(total, int) and total >= 0:
            track_totals.append(total)
        if result.get("hasMore") is False:
            terminal_track_page = True

    expected_track_count = max(track_totals) if track_totals else None
    track_scope_complete = bool(track_result_seen) and terminal_track_page
    if expected_track_count is not None:
        track_scope_complete = track_scope_complete and len(tracks) >= expected_track_count

    registry_results = [
        result
        for result in results
        if isinstance(result.get("registryItems"), list)
        and "registryReferences" in result
        and "hullNumber" not in result
    ]
    registry_loaded = bool(registry_results)
    registry_items: dict[str, dict[str, Any]] = {}
    registry_references: dict[str, dict[str, Any]] = {}
    unsearchable_registry_ids: set[str] = set()
    for result in registry_results:
        for item in result.get("registryItems") or []:
            if isinstance(item, dict) and item.get("registryId") is not None:
                registry_items[str(item["registryId"])] = item
        for reference in result.get("registryReferences") or []:
            if isinstance(reference, dict) and reference.get("referenceId") is not None:
                registry_references[str(reference["referenceId"])] = reference
        unsearchable_registry_ids.update(
            str(value) for value in result.get("unsearchableRegistryIds") or []
        )
    registry_complete = registry_loaded and not unsearchable_registry_ids

    matched_hulls: set[str] = set()
    unmatched_hulls: set[str] = set()
    exact_result_seen = False
    for result in results:
        if "matchedHullNumbers" not in result and "unmatchedHullNumbers" not in result:
            continue
        exact_result_seen = True
        matched_hulls.update(
            normalize_hull_number(value) for value in result.get("matchedHullNumbers") or []
            if normalize_hull_number(value)
        )
        unmatched_hulls.update(
            normalize_hull_number(value) for value in result.get("unmatchedHullNumbers") or []
            if normalize_hull_number(value)
        )

    confirmed_hulls = {
        normalize_hull_number(track.get("finalHullNumber"))
        for track in tracks.values()
        if track.get("finalMatchType") == "confirmed"
        and normalize_hull_number(track.get("finalHullNumber"))
    }
    checked_hulls = matched_hulls | unmatched_hulls
    exact_hull_complete = not confirmed_hulls or (
        exact_result_seen and confirmed_hulls.issubset(checked_hulls)
    )
    exact_in_track_ids = {
        track_id
        for track_id, track in tracks.items()
        if track.get("finalMatchType") == "confirmed"
        and normalize_hull_number(track.get("finalHullNumber")) in matched_hulls
    }
    remaining_track_ids = set(tracks) - exact_in_track_ids

    frame_track_ids: set[str] = set()
    unsearchable_track_ids: set[str] = set()
    for result in results:
        grouped = result.get("keyframesByTrack")
        if isinstance(grouped, dict):
            for track_id, group in grouped.items():
                if isinstance(group, dict) and group.get("keyframes"):
                    frame_track_ids.add(_track_id(track_id))
        for frame in result.get("keyframes") or []:
            if isinstance(frame, dict) and frame.get("trackId") is not None:
                frame_track_ids.add(_track_id(frame.get("trackId")))
        unsearchable_track_ids.update(
            _track_id(value) for value in result.get("unsearchableTrackIds") or []
        )

    image_matches_by_track: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        if result.get("matchMode") != "image_to_image":
            continue
        for match in result.get("matches") or []:
            if not isinstance(match, dict) or match.get("matchedTrackId") is None:
                continue
            image_matches_by_track.setdefault(
                _track_id(match.get("matchedTrackId")), []
            ).append(match)

    image_in_track_ids: set[str] = set()
    image_out_track_ids: set[str] = set()
    uncertain_track_ids: set[str] = set()
    if registry_loaded and not registry_items:
        image_out_track_ids.update(remaining_track_ids)
    else:
        for track_id in remaining_track_ids:
            candidates = image_matches_by_track.get(track_id, [])
            bands = [str(item.get("scoreBand") or "") for item in candidates]
            if "match" in bands:
                image_in_track_ids.add(track_id)
            elif "uncertain" in bands:
                uncertain_track_ids.add(track_id)
            elif candidates and all(band == "mismatch" for band in bands) and registry_complete:
                image_out_track_ids.add(track_id)

    in_registry_track_ids = exact_in_track_ids | image_in_track_ids
    out_of_registry_track_ids = image_out_track_ids
    resolved_track_ids = in_registry_track_ids | out_of_registry_track_ids
    unresolved_track_ids = set(tracks) - resolved_track_ids

    requirements: list[str] = []
    pending: list[str] = []
    next_action: str | None = None
    acceptance_satisfied = False

    if relation in {"in", "out"} and target_scope != "registry":
        requirements = [
            "complete_track_scope",
            "exact_hull_classification",
            "registry_image_classification",
            "relation_result",
        ]
        if not track_scope_complete:
            pending.append("complete_track_scope")
            next_action = "下一轮继续调用 getTrack，按 nextOffset 读取剩余轨迹，直到 hasMore=false 且轨迹覆盖完整。"
        elif not exact_hull_complete:
            pending.append("exact_hull_classification")
            next_action = "下一轮调用 matchHull，输入 acceptance.confirmedHullNumbers，完成所有稳定舷号的精确查库。"
        elif remaining_track_ids and not registry_loaded:
            pending.append("registry_catalog")
            next_action = "下一轮调用 listRegistry 读取完整先验库及可用参考图。"
        elif registry_items and remaining_track_ids:
            missing_frames = remaining_track_ids - frame_track_ids - unsearchable_track_ids
            if missing_frames:
                pending.append("keyframe_evidence")
                next_action = "下一轮调用 getFrames，输入 acceptance.remainingTrackIds，读取未由舷号精确命中的轨迹关键帧。"
            elif uncertain_track_ids:
                pending.append("gray_verification")
                next_action = "下一轮仅对 acceptance.uncertainTrackIds 调用 verifyTarget；关键帧仍不确定时再读取对应 getClip。"
            elif unsearchable_registry_ids:
                pending.append("registry_coverage")
                next_action = "停止并说明先验库存在不可检索库项，当前不能可靠判定未在库船。"
            elif unresolved_track_ids & unsearchable_track_ids:
                pending.append("segment_verification")
                next_action = "下一轮对 acceptance.unsearchableTrackIds 逐条调用 getClip，并结合库参考图调用 verifyTarget。"
            elif unresolved_track_ids:
                pending.append("registry_image_classification")
                next_action = "下一轮调用 matchImage，将剩余轨迹正式关键帧与先验库参考图逐轨迹匹配并保留全部轨迹结果。"
        acceptance_satisfied = (
            track_scope_complete
            and exact_hull_complete
            and not pending
            and not unresolved_track_ids
        )
        if acceptance_satisfied:
            next_action = "停止并按验收关系返回已经完成分类的轨迹列表。"
    elif operation == "count":
        requirements = ["deduplicated_count"]
        count_results = [result for result in results if "upperCount" in result]
        acceptance_satisfied = bool(count_results)
        if not acceptance_satisfied:
            pending.append("deduplicated_count")
            next_action = "下一轮读取目标轨迹关键帧并调用 dedupTracks，得到可审计数量。"
    elif target_kind == "description":
        requirements = ["target_match"]
        match_results = [
            result
            for result in results
            if result.get("matchMode") in {"text_to_image", "text_to_registry"}
            or result.get("targetType") in {"description", "description_registry"}
        ]
        acceptance_satisfied = bool(match_results)
        if not acceptance_satisfied:
            pending.append("target_match")
            next_action = "下一轮获取正式关键帧并调用 matchText；只有灰区结果才调用 verifyTarget。"
    elif target_scope == "registry":
        requirements = ["registry_query"]
        acceptance_satisfied = any("registryItems" in result for result in results)
        if not acceptance_satisfied:
            pending.append("registry_query")
            next_action = "下一轮根据舷号调用 getRegistry，或在完整库查询时调用 listRegistry。"
    else:
        requirements = ["complete_track_scope"]
        acceptance_satisfied = track_scope_complete
        if not acceptance_satisfied:
            pending.append("complete_track_scope")
            next_action = "下一轮继续调用 getTrack，直到查询范围读取完整。"

    return {
        "expectedOutcome": intent.get("expectedOutcome"),
        "successCriteria": intent.get("successCriteria"),
        "requirements": requirements,
        "pendingRequirements": pending,
        "pendingRequirementLabels": [REQUIREMENT_LABELS.get(value, value) for value in pending],
        "acceptanceSatisfied": acceptance_satisfied,
        "nextAction": next_action,
        "trackScopeComplete": track_scope_complete,
        "trackCount": len(tracks),
        "expectedTrackCount": expected_track_count,
        "trackIds": sorted(tracks),
        "confirmedHullNumbers": sorted(confirmed_hulls),
        "checkedHullNumbers": sorted(checked_hulls),
        "exactHullComplete": exact_hull_complete,
        "exactInRegistryTrackIds": sorted(exact_in_track_ids),
        "remainingTrackIds": sorted(remaining_track_ids),
        "registryLoaded": registry_loaded,
        "registryCount": len(registry_items),
        "registryComplete": registry_complete,
        "registryReferenceIds": sorted(registry_references),
        "unsearchableRegistryIds": sorted(unsearchable_registry_ids),
        "frameTrackIds": sorted(frame_track_ids),
        "unsearchableTrackIds": sorted(unsearchable_track_ids),
        "inRegistryTrackIds": sorted(in_registry_track_ids),
        "outOfRegistryTrackIds": sorted(out_of_registry_track_ids),
        "uncertainTrackIds": sorted(uncertain_track_ids),
        "unresolvedTrackIds": sorted(unresolved_track_ids),
    }

def compact_acceptance(progress: dict[str, Any]) -> dict[str, Any]:
    """压缩长编号列表，仅保留验收决策所需信息。"""
    list_fields = {
        "trackIds",
        "confirmedHullNumbers",
        "checkedHullNumbers",
        "exactInRegistryTrackIds",
        "remainingTrackIds",
        "registryReferenceIds",
        "unsearchableRegistryIds",
        "frameTrackIds",
        "unsearchableTrackIds",
        "inRegistryTrackIds",
        "outOfRegistryTrackIds",
        "uncertainTrackIds",
        "unresolvedTrackIds",
    }
    compact: dict[str, Any] = {}
    for key, value in progress.items():
        if key in list_fields and isinstance(value, list):
            compact[f"{key}Count"] = len(value)
            compact[f"{key}Sample"] = value[:8]
        else:
            compact[key] = value
    return compact

