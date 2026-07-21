"""SeaAgent 原子工具服务。"""
from __future__ import annotations
import hashlib
import json
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Iterable
import cv2
import numpy as np
from config import load_config
from memory import MemoryRepository, normalize_hull_number
from services import AgentLLMService, QwenMultimodalEmbedder
from vector_store import VectorCatalog

class ToolService:
    def __init__(self, config: dict[str, Any] | None = None, repository: MemoryRepository | None = None, embedder: QwenMultimodalEmbedder | None = None, llm: AgentLLMService | None = None, vectors: VectorCatalog | None = None):
        self.config = config or load_config()
        self.repository = repository or MemoryRepository(self.config)
        self.embedder = embedder or QwenMultimodalEmbedder(self.config)
        self.llm = llm or AgentLLMService(self.config)
        self.vectors = vectors or VectorCatalog(self.config)
        self.settings = self.config["pipeline"]["retrieval"]

    def getTrack(self, timeRange: tuple[float, float] | None = None, hullNumber: str | None = None, finalMatchType: str | None = None, offset: int = 0, limit: int = 0) -> dict[str, Any]:
        tracks = self.repository.find_tracks(timeRange, hullNumber, finalMatchType)
        start = max(0, int(offset or 0))
        page_size = max(0, min(200, int(limit or 0)))
        selected = tracks[start:start + page_size] if page_size else tracks[start:]
        next_offset = start + len(selected)
        return {"ok": True, "queryScope": list(timeRange) if timeRange else None, "queryHullNumber": normalize_hull_number(hullNumber), "queryFinalMatchType": finalMatchType, "trackIds": [item["trackId"] for item in selected], "tracks": selected, "totalTrackCount": len(tracks), "returnedTrackCount": len(selected), "offset": start, "limit": page_size, "hasMore": next_offset < len(tracks), "nextOffset": next_offset if next_offset < len(tracks) else None}

    def getFrames(self, trackIds: Iterable[str | int]) -> dict[str, Any]:
        ids = [str(value) for value in dict.fromkeys(trackIds)]
        all_frames = self.repository.get_keyframes(ids, embedded_only=False)
        vector_ids = [int(frame["keyframeVectorId"]) for frames in all_frames.values() for frame in frames if frame.get("isEmbedded") and frame.get("keyframeVectorId") is not None]
        available_vectors = self.vectors.keyframes.get_many(vector_ids)
        grouped, discarded, missing = {}, [], []
        for track_id in ids:
            valid = [frame for frame in all_frames.get(track_id, []) if frame["isEmbedded"] and frame.get("keyframeVectorId") is not None and int(frame["keyframeVectorId"]) in available_vectors]
            invalid = [frame["keyframeId"] for frame in all_frames.get(track_id, []) if frame not in valid]
            grouped[track_id] = {"keyframeIds": [frame["keyframeId"] for frame in valid], "keyframes": valid}
            discarded.extend(invalid)
            if not valid:
                missing.append(track_id)
        frames = [frame for group in grouped.values() for frame in group["keyframes"]]
        return {"ok": True, "keyframeIds": [frame["keyframeId"] for frame in frames], "keyframes": frames, "keyframesByTrack": grouped, "discardedKeyframeIds": discarded, "unsearchableTrackIds": missing}

    def getClip(self, trackId: str | int, timeRange: tuple[float, float] | None = None) -> dict[str, Any]:
        track = self.repository.get_track(trackId)
        if not track:
            return {"ok": False, "error": "track_not_found", "trackId": str(trackId)}
        start, end = track["startTime"], track["endTime"]
        monitor_scope = (max(start, timeRange[0]), min(end, timeRange[1])) if timeRange else (start, end)
        if monitor_scope[0] > monitor_scope[1]:
            return {"ok": True, "found": False, "trackId": str(trackId)}
        trajectory_value = track.get("trajectoryPath")
        if not trajectory_value:
            return {"ok": False, "error": "trajectory_not_found", "trackId": str(trackId)}
        trajectory_path = Path(trajectory_value)
        if not trajectory_path.is_file():
            return {"ok": False, "error": "trajectory_not_found", "trackId": str(trackId)}
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        source_value = trajectory.get("sourceVideoPath")
        source_path = Path(source_value) if source_value else None
        all_boxes = trajectory.get("boxes", [])
        if timeRange and any("observedAt" in item for item in all_boxes):
            boxes = [item for item in all_boxes if monitor_scope[0] <= float(item.get("observedAt") or 0) <= monitor_scope[1]]
        elif timeRange:
            video_start, video_end = track.get("videoStartTime", start), track.get("videoEndTime", end)
            boxes = [item for item in all_boxes if video_start <= float(item["timestamp"]) <= video_end]
        else:
            boxes = all_boxes
        if source_path is None or not source_path.is_file() or not boxes:
            return {"ok": False, "error": "source_evidence_unavailable", "trackId": str(trackId)}
        box_map = {int(item["frameIndex"]): item["bbox"] for item in boxes}
        evidence = self.config["pipeline"].get("evidence", {})
        canvas_size = (
            self._even_size(evidence.get("clip_width", 640)),
            self._even_size(evidence.get("clip_height", 360)),
        )
        clip_fps = max(1.0, float(evidence.get("clip_fps", 15)))
        clip_crf = max(0, min(51, int(evidence.get("clip_crf", 31))))
        poster_quality = max(1, min(100, int(evidence.get("poster_quality", 75))))
        output_dir = Path(self.config["paths"]["clip_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        video_start_time = min(float(item["timestamp"]) for item in boxes)
        video_end_time = max(float(item["timestamp"]) for item in boxes)
        cache_key = (
            f"evidence-v2|{trackId}|{video_start_time:.3f}|{video_end_time:.3f}|"
            f"{trajectory_path.stat().st_mtime_ns}|{canvas_size[0]}x{canvas_size[1]}|"
            f"{clip_fps:.3f}|{clip_crf}|{poster_quality}"
        )
        segment_id = f"segment-{hashlib.sha1(cache_key.encode('utf-8')).hexdigest()[:12]}"
        output_path = output_dir / f"{segment_id}.mp4"
        poster_path = output_dir / f"{segment_id}.jpg"
        if output_path.is_file() and output_path.stat().st_size > 0:
            self._ensure_clip_poster(output_path, poster_path, poster_quality)
            return {"ok": True, "found": True, "trackId": str(trackId), "shipSegmentId": segment_id, "segmentPath": str(output_path), "posterPath": str(poster_path) if poster_path.is_file() else None, "codec": "cached", "startTime": monitor_scope[0], "endTime": monitor_scope[1], "videoStartTime": video_start_time, "videoEndTime": video_end_time}
        source_fps = max(1.0, float(trajectory.get("sourceFps") or 25))
        output_fps = min(source_fps, clip_fps)
        temporary_path = output_dir / f"{segment_id}.raw.mp4"
        temporary_path.unlink(missing_ok=True)
        writer, codec = self._open_video_writer(temporary_path, output_fps, canvas_size)
        capture = cv2.VideoCapture(str(source_path))
        if writer is None or not capture.isOpened():
            capture.release()
            if writer is not None:
                writer.release()
            temporary_path.unlink(missing_ok=True)
            return {"ok": False, "error": "segment_codec_unavailable", "trackId": str(trackId)}
        written = 0
        last_written_frame: int | None = None
        frame_gap = max(1, int(round(source_fps / output_fps)))
        poster_frame: np.ndarray | None = None
        poster_area = -1
        try:
            first, last = min(box_map), max(box_map)
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, first))
            for frame_index in range(first, last + 1):
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index not in box_map:
                    continue
                if last_written_frame is not None and frame_index - last_written_frame < frame_gap:
                    continue
                x1, y1, x2, y2 = self._clamp_box(box_map[frame_index], frame.shape)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                canvas = self._fit_crop_to_canvas(crop, canvas_size)
                writer.write(canvas)
                last_written_frame = frame_index
                crop_area = crop.shape[0] * crop.shape[1]
                if crop_area > poster_area:
                    poster_frame = canvas.copy()
                    poster_area = crop_area
                written += 1
        finally:
            capture.release()
            writer.release()
        if written == 0:
            temporary_path.unlink(missing_ok=True)
            return {"ok": False, "error": "empty_target_segment", "trackId": str(trackId)}
        codec = self._compress_evidence_clip(temporary_path, output_path, codec, clip_crf)
        self._write_poster(poster_path, poster_frame, poster_quality)
        return {"ok": True, "found": True, "trackId": str(trackId), "shipSegmentId": segment_id, "segmentPath": str(output_path), "posterPath": str(poster_path) if poster_path.is_file() else None, "codec": codec, "startTime": monitor_scope[0], "endTime": monitor_scope[1], "videoStartTime": video_start_time, "videoEndTime": video_end_time}

    def getRegistry(self, hullNumber: str) -> dict[str, Any]:
        items = self.repository.registry_by_hull(hullNumber)
        references = [reference for item in items for reference in item.get("references", [])]
        vector_ids = [int(item["registryVectorId"]) for item in references if item.get("isEmbedded") and item.get("registryVectorId") is not None]
        available_vectors = self.vectors.registry.get_many(vector_ids)
        valid = [item for item in references if item["isEmbedded"] and item.get("registryVectorId") is not None and int(item["registryVectorId"]) in available_vectors]
        discarded = [item["referenceId"] for item in references if item not in valid]
        return {"ok": True, "found": bool(items), "searchable": bool(valid), "hullNumber": normalize_hull_number(hullNumber), "registryIds": [item["registryId"] for item in items], "registryItems": items, "registryReferenceIds": [item["referenceId"] for item in valid], "registryReferences": valid, "discardedReferenceIds": discarded}

    def matchHull(self, hullNumberArray: Iterable[str | None]) -> dict[str, Any]:
        hull_numbers = list(dict.fromkeys(normalize_hull_number(value) for value in hullNumberArray if normalize_hull_number(value)))
        matches = {hull: self.repository.registry_by_hull(hull) for hull in hull_numbers}
        return {"ok": True, "exactMatches": {hull: items for hull, items in matches.items() if items}, "matchedHullNumbers": [hull for hull, items in matches.items() if items], "unmatchedHullNumbers": [hull for hull, items in matches.items() if not items]}

    def listRegistry(self) -> dict[str, Any]:
        items = self.repository.list_registry()
        references = [reference for item in items for reference in item.get("references", [])]
        vector_ids = [int(item["registryVectorId"]) for item in references if item.get("isEmbedded") and item.get("registryVectorId") is not None]
        available_vectors = self.vectors.registry.get_many(vector_ids)
        valid = [item for item in references if item["isEmbedded"] and item.get("registryVectorId") is not None and int(item["registryVectorId"]) in available_vectors]
        discarded = [item["referenceId"] for item in references if item not in valid]
        searchable_ids = {item["registryId"] for item in valid}
        return {"ok": True, "registryItems": items, "registryReferenceIds": [item["referenceId"] for item in valid], "registryReferences": valid, "discardedReferenceIds": discarded, "unsearchableRegistryIds": [item["registryId"] for item in items if item["registryId"] not in searchable_ids]}

    def matchText(self, description: str, galleryImages: list[dict[str, Any]] | dict[str, Any] | None, topK: int | None = None) -> dict[str, Any]:
        if not description.strip():
            return {"ok": False, "error": "description_required", "matches": []}
        galleryImages = self._flatten_image_records(galleryImages)
        registry_images = [item for item in galleryImages if item.get("registryId") is not None and item.get("registryVectorId") is not None and item.get("isEmbedded")]
        if registry_images:
            query = self.embedder.encode_text(description, self.config["prompts"]["text_retrieval_instruction"])
            vectors = self.vectors.registry.get_many(item["registryVectorId"] for item in registry_images)
            grouped: dict[str, list[tuple[float, dict[str, Any]]]] = {}
            for image in registry_images:
                vector = vectors.get(int(image["registryVectorId"]))
                if vector is not None:
                    grouped.setdefault(str(image["registryId"]), []).append((float(np.dot(query, vector)), image))
            matches = []
            for registry_id, scored in grouped.items():
                ranked = sorted(scored, key=lambda item: item[0], reverse=True)
                if not ranked:
                    continue
                # 先验库多视角：以最高相似参考图为准，避免弱视角把强匹配平均稀释
                best_score = float(ranked[0][0])
                support = ranked[:3]
                mean_top3 = float(np.mean([item[0] for item in support]))
                matches.append({
                    "matchedRegistryId": registry_id,
                    "embeddingScore": round(best_score, 6),
                    "maxEmbeddingScore": round(best_score, 6),
                    "meanTop3Score": round(mean_top3, 6),
                    "scoreBand": self._band(best_score, "text"),
                    "matchedRegistryReferenceIds": [item[1]["referenceId"] for item in support],
                    "supportReferenceCount": len(support),
                })
            matches.sort(key=lambda item: item["embeddingScore"], reverse=True)
            # 附加 rank gap，便于后续灰区判断
            if len(matches) >= 2:
                matches[0]["rankGap"] = round(float(matches[0]["embeddingScore"]) - float(matches[1]["embeddingScore"]), 6)
            elif matches:
                matches[0]["rankGap"] = round(float(matches[0]["embeddingScore"]), 6)
            return {"ok": True, "matchMode": "text_to_registry", "matches": matches[:topK] if topK else matches}
        images = [item for item in galleryImages if item.get("isEmbedded") and item.get("keyframeVectorId") is not None]
        if not images:
            return {"ok": True, "matchMode": "text_to_image", "matches": [], "missingKeyframeIds": []}
        query = self.embedder.encode_text(description, self.config["prompts"]["text_retrieval_instruction"])
        vectors = self.vectors.keyframes.get_many(item["keyframeVectorId"] for item in images)
        missing = [item["keyframeId"] for item in images if int(item["keyframeVectorId"]) not in vectors]
        grouped: dict[str, list[tuple[float, dict[str, Any]]]] = {}
        for image in images:
            vector = vectors.get(int(image["keyframeVectorId"]))
            if vector is not None:
                grouped.setdefault(str(image["trackId"]), []).append((float(np.dot(query, vector)), image))
        matches = []
        for track_id, scored in grouped.items():
            ranked = sorted(scored, key=lambda item: item[0], reverse=True)
            if not ranked:
                continue
            top = ranked[:2]
            best_score = float(top[0][0])
            mean_top2 = float(np.mean([item[0] for item in top]))
            # 轨迹描述检索：主分用最高关键帧，避免次优帧把分数拉低
            matches.append({
                "matchedTrackId": track_id,
                "embeddingScore": round(best_score, 6),
                "maxEmbeddingScore": round(best_score, 6),
                "meanTop2Score": round(mean_top2, 6),
                "scoreBand": self._band(best_score, "text"),
                "matchedKeyframeIds": [item[1]["keyframeId"] for item in top],
            })
        matches.sort(key=lambda item: item["embeddingScore"], reverse=True)
        return {"ok": True, "matchMode": "text_to_image", "matches": matches[:topK] if topK else matches, "missingKeyframeIds": missing}

    def matchImage(self, queryImages: list[dict[str, Any]] | dict[str, Any] | None, galleryImages: list[dict[str, Any]] | dict[str, Any] | None, topK: int | None = None) -> dict[str, Any]:
        queryImages = self._flatten_image_records(queryImages)
        galleryImages = self._flatten_image_records(galleryImages)
        if not queryImages or not galleryImages:
            return {"ok": True, "matchMode": "image_to_image", "queryType": None, "matches": [], "missingKeyframeIds": [], "missingRegistryReferenceIds": []}
        query_keyframes = [item for item in queryImages if item.get("trackId") is not None and item.get("keyframeVectorId") is not None and item.get("isEmbedded")]
        query_references = [item for item in queryImages if item.get("registryId") is not None and item.get("registryVectorId") is not None and item.get("isEmbedded")]
        gallery_keyframes = [item for item in galleryImages if item.get("trackId") is not None and item.get("keyframeVectorId") is not None and item.get("isEmbedded")]
        gallery_references = [item for item in galleryImages if item.get("registryId") is not None and item.get("registryVectorId") is not None and item.get("isEmbedded")]
        query_is_track = bool(query_keyframes) and bool(gallery_references) and not query_references and not gallery_keyframes
        query_is_registry = bool(query_references) and bool(gallery_keyframes) and not query_keyframes and not gallery_references
        if not query_is_track and not query_is_registry:
            return {"ok": False, "error": "one_keyframe_side_and_one_registry_side_required", "matchMode": "image_to_image", "matches": []}
        keyframes = query_keyframes if query_is_track else gallery_keyframes
        references = gallery_references if query_is_track else query_references
        frame_vectors = self.vectors.keyframes.get_many(item["keyframeVectorId"] for item in keyframes)
        reference_vectors = self.vectors.registry.get_many(item["registryVectorId"] for item in references)
        missing_keyframes = [item["keyframeId"] for item in keyframes if int(item["keyframeVectorId"]) not in frame_vectors]
        missing_references = [item["referenceId"] for item in references if int(item["registryVectorId"]) not in reference_vectors]
        frames_by_track = self._group(keyframes, "trackId")
        refs_by_registry = self._group(references, "registryId")
        matches = []
        for track_id, track_frames in frames_by_track.items():
            for registry_id, registry_refs in refs_by_registry.items():
                frame_scores = []
                for frame in track_frames:
                    frame_vector = frame_vectors.get(int(frame["keyframeVectorId"]))
                    candidates = [(float(np.dot(frame_vector, reference_vectors[int(reference["registryVectorId"])])), reference) for reference in registry_refs if frame_vector is not None and int(reference["registryVectorId"]) in reference_vectors]
                    if candidates:
                        score, reference = max(candidates, key=lambda item: item[0])
                        frame_scores.append((score, frame, reference))
                top = sorted(frame_scores, key=lambda item: item[0], reverse=True)[:2]
                if not top:
                    continue
                score = float(np.mean([item[0] for item in top]))
                matched_keyframes = [item[1]["keyframeId"] for item in top]
                matched_references = list(dict.fromkeys(item[2]["referenceId"] for item in top))
                matches.append({"matchedTrackId": track_id, "matchedRegistryId": registry_id, "embeddingScore": round(score, 6), "scoreBand": self._band(score, "image"), "queryKeyframeIds": matched_keyframes if query_is_track else [], "queryRegistryReferenceIds": matched_references if query_is_registry else [], "matchedKeyframeIds": matched_keyframes, "matchedRegistryReferenceIds": matched_references})
        matches.sort(key=lambda item: item["embeddingScore"], reverse=True)
        if topK:
            owner_key = "matchedTrackId" if query_is_track else "matchedRegistryId"
            owner_ids = frames_by_track if query_is_track else refs_by_registry
            matches = [item for owner_id in owner_ids for item in [entry for entry in matches if entry[owner_key] == owner_id][:topK]]
            matches.sort(key=lambda item: item["embeddingScore"], reverse=True)
        return {"ok": True, "matchMode": "image_to_image", "queryType": "track" if query_is_track else "registry", "matches": matches, "missingKeyframeIds": missing_keyframes, "missingRegistryReferenceIds": missing_references}

    def verifyTarget(self, description: str | None = None, registryReferenceIds: list[str] | None = None, keyframeIds: list[str] | None = None, shipSegmentIds: list[str] | None = None) -> dict[str, Any]:
        """支持三类核验：
        1) 文字描述 vs 轨迹关键帧/片段
        2) 库参考图 vs 轨迹关键帧/片段
        3) 文字描述 vs 库参考图（先验库描述灰区）
        """
        has_description = bool(description and str(description).strip())
        has_registry = bool(registryReferenceIds)
        has_keyframes = bool(keyframeIds)
        has_segments = bool(shipSegmentIds)
        text_to_registry = has_description and has_registry and not has_keyframes and not has_segments
        text_to_track = has_description and not has_registry and (has_keyframes != has_segments)
        registry_to_track = has_registry and not has_description and (has_keyframes != has_segments)
        if not (text_to_registry or text_to_track or registry_to_track):
            return {"ok": False, "error": "unsupported_verify_combination", "decision": "uncertain"}

        references = self.repository.references_by_ids(registryReferenceIds or [])
        reference_paths = [item["imagePath"] for item in references if Path(item["imagePath"]).exists()][:3]
        if has_keyframes:
            frames = self.repository.keyframes_by_ids(keyframeIds or [])
            evidence = [item["keyframePath"] for item in frames if Path(item["keyframePath"]).exists()][:6 if has_description else 3]
        elif has_segments:
            evidence = self._sample_segments(shipSegmentIds or [], 6 if has_description else 3)
        else:
            # 文字描述核验先验库：库参考图本身作为待审图像
            evidence = reference_paths[:3]

        if text_to_registry:
            if not evidence:
                return {"ok": True, "targetType": "description_registry", "description": description, "registryReferenceIds": registryReferenceIds or [], "decision": "uncertain", "facts": ["先验库参考图不可用"], "keyframeIds": [], "shipSegmentIds": []}
            result = self.llm.verify(description, [], evidence)
            return {"ok": True, "targetType": "description_registry", "description": description, "registryReferenceIds": registryReferenceIds or [], "decision": result["decision"], "facts": result["facts"], "keyframeIds": [], "shipSegmentIds": []}

        if not evidence or (registry_to_track and not reference_paths):
            return {"ok": True, "targetType": "description" if has_description else "registry", "description": description, "registryReferenceIds": registryReferenceIds or [], "decision": "uncertain", "facts": ["视觉证据不足"], "keyframeIds": keyframeIds or [], "shipSegmentIds": shipSegmentIds or []}
        result = self.llm.verify(description if has_description else None, reference_paths, evidence)
        return {"ok": True, "targetType": "description" if has_description else "registry", "description": description, "registryReferenceIds": registryReferenceIds or [], "decision": result["decision"], "facts": result["facts"], "keyframeIds": keyframeIds or [], "shipSegmentIds": shipSegmentIds or []}


    def showEvidence(self, keyframeIds: list[str] | None = None, shipSegmentIds: list[str] | None = None, registryReferenceIds: list[str] | None = None) -> dict[str, Any]:
        return {
            "ok": True,
            "displayId": f"display-{uuid.uuid4().hex[:12]}",
            "shownKeyframeIds": list(dict.fromkeys(keyframeIds or [])),
            "shownShipSegmentIds": list(dict.fromkeys(shipSegmentIds or [])),
            "shownRegistryReferenceIds": self._representative_registry_reference_ids(registryReferenceIds or []),
        }

    def dedupTracks(self, tracks: list[dict[str, Any]], keyframesByTrack: dict[str, Any]) -> dict[str, Any]:
        candidates = {}
        for track in tracks:
            track_id = str(track["trackId"])
            group = keyframesByTrack.get(track_id, {})
            frames = group.get("keyframes", group if isinstance(group, list) else [])
            candidates[track_id] = [frame for frame in frames if frame.get("isEmbedded") and frame.get("keyframeVectorId") is not None]
        track_map = {str(item["trackId"]): item for item in tracks}
        vector_ids = [frame["keyframeVectorId"] for frames in candidates.values() for frame in frames]
        vectors = self.vectors.keyframes.get_many(vector_ids)
        selected, missing = {}, []
        for track_id, frames in candidates.items():
            available = [frame for frame in frames if int(frame["keyframeVectorId"]) in vectors]
            selected[track_id] = sorted(available, key=lambda item: item["retentionScore"], reverse=True)[:3]
            if not selected[track_id]:
                missing.append(track_id)
        pair_scores: dict[tuple[str, str], float] = {}
        track_ids = list(track_map)
        for index, left_id in enumerate(track_ids):
            for right_id in track_ids[index + 1:]:
                if self._overlap(track_map[left_id], track_map[right_id]) or not selected[left_id] or not selected[right_id]:
                    continue
                scores = []
                for left in selected[left_id]:
                    for right in selected[right_id]:
                        left_vector = vectors.get(int(left["keyframeVectorId"]))
                        right_vector = vectors.get(int(right["keyframeVectorId"]))
                        if left_vector is not None and right_vector is not None:
                            scores.append(float(np.dot(left_vector, right_vector)))
                if scores:
                    pair_scores[tuple(sorted((left_id, right_id)))] = float(np.mean(sorted(scores, reverse=True)[:3]))
        high = float(self.settings["dedup_high"])
        low = float(self.settings["dedup_low"])
        high_groups = self._complete_link(track_ids, track_map, pair_scores, high)
        low_groups = self._complete_link(track_ids, track_map, pair_scores, low)
        gray = [{"trackIds": list(pair), "embeddingScore": round(score, 6)} for pair, score in pair_scores.items() if low < score < high]
        status = "incomplete" if missing else "stable" if len(high_groups) == len(low_groups) else "sensitive"
        scores = [{"trackIds": list(pair), "embeddingScore": round(score, 6)} for pair, score in sorted(pair_scores.items())]
        return {"ok": True, "trackCount": len(tracks), "highThresholdShipCount": len(high_groups), "lowThresholdShipCount": len(low_groups), "countStability": status, "highGroups": high_groups, "lowGroups": low_groups, "pairScores": scores, "grayPairs": gray, "unsearchableTrackIds": missing}

    @classmethod
    def _flatten_image_records(cls, value: Any) -> list[dict[str, Any]]:
        """兼容工具结果中的 references、keyframes 和按轨迹分组的关键帧。"""
        records: list[dict[str, Any]] = []

        def visit(item: Any) -> None:
            if isinstance(item, (list, tuple)):
                for child in item:
                    visit(child)
                return
            if not isinstance(item, dict):
                return
            if isinstance(item.get("registryReferences"), list):
                visit(item["registryReferences"])
                return
            if isinstance(item.get("references"), list):
                visit(item["references"])
                return
            if isinstance(item.get("keyframes"), list):
                visit(item["keyframes"])
                return
            if isinstance(item.get("keyframesByTrack"), dict):
                visit(list(item["keyframesByTrack"].values()))
                return
            records.append(item)

        visit(value)
        return records

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        tool = getattr(self, name, None)
        if not callable(tool) or name.startswith("_") or name == "execute":
            return {"ok": False, "error": "tool_not_allowed", "tool": name}
        try:
            return tool(**arguments)
        except Exception as error:
            return {"ok": False, "error": str(error), "tool": name}

    def _band(self, score: float, mode: str) -> str:
        match = float(self.settings[f"{mode}_match"])
        exclude = float(self.settings[f"{mode}_exclude"])
        return "match" if score >= match else "mismatch" if score <= exclude else "uncertain"

    @staticmethod
    def _group(items: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            grouped.setdefault(str(item[key]), []).append(item)
        return grouped

    def _representative_registry_reference_ids(self, reference_ids: list[str]) -> list[str]:
        ordered_ids = list(dict.fromkeys(str(value) for value in reference_ids if value))
        reference_map = {item["referenceId"]: item for item in self.repository.references_by_ids(ordered_ids)}
        hull_by_registry = {
            str(item["registryId"]): normalize_hull_number(item.get("hullNumber")) or str(item["registryId"])
            for item in self.repository.list_registry()
        }
        selected, seen = [], set()
        for reference_id in ordered_ids:
            reference = reference_map.get(reference_id)
            registry_id = str(reference.get("registryId")) if reference else reference_id
            owner = hull_by_registry.get(registry_id, registry_id)
            if owner in seen:
                continue
            seen.add(owner)
            selected.append(reference_id)
        return selected

    @staticmethod
    def _clamp_box(box: list[int], shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        height, width = shape[:2]
        x1, y1, x2, y2 = map(int, box)
        return max(0, min(x1, width - 1)), max(0, min(y1, height - 1)), max(1, min(x2, width)), max(1, min(y2, height))

    @staticmethod
    def _open_video_writer(path: Path, fps: float, size: tuple[int, int]) -> tuple[Any | None, str | None]:
        for codec in ("avc1", "H264", "mp4v"):
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
            if writer.isOpened():
                return writer, codec
            writer.release()
            path.unlink(missing_ok=True)
        return None, None

    @staticmethod
    def _even_size(value: int) -> int:
        value = max(2, int(value))
        return value if value % 2 == 0 else value + 1

    @staticmethod
    def _fit_crop_to_canvas(crop: np.ndarray, canvas_size: tuple[int, int]) -> np.ndarray:
        canvas_width, canvas_height = canvas_size
        content_width = max(2, int(canvas_width * 0.94))
        content_height = max(2, int(canvas_height * 0.94))
        crop_height, crop_width = crop.shape[:2]
        scale = min(content_width / max(1, crop_width), content_height / max(1, crop_height))
        resized_width = max(1, min(content_width, int(round(crop_width * scale))))
        resized_height = max(1, min(content_height, int(round(crop_height * scale))))
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        resized = cv2.resize(crop, (resized_width, resized_height), interpolation=interpolation)
        canvas = np.full((canvas_height, canvas_width, 3), (5, 14, 20), dtype=np.uint8)
        offset_x = (canvas_width - resized_width) // 2
        offset_y = (canvas_height - resized_height) // 2
        canvas[offset_y:offset_y + resized_height, offset_x:offset_x + resized_width] = resized
        return canvas

    @staticmethod
    def _write_poster(path: Path, frame: np.ndarray | None, quality: int) -> bool:
        if frame is None or frame.size == 0:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        return bool(cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)]))

    @classmethod
    def _ensure_clip_poster(cls, clip_path: Path, poster_path: Path, quality: int) -> bool:
        if poster_path.is_file() and poster_path.stat().st_size > 0:
            return True
        capture = cv2.VideoCapture(str(clip_path))
        try:
            ok, frame = capture.read()
        finally:
            capture.release()
        return cls._write_poster(poster_path, frame if ok else None, quality)

    @staticmethod
    def _compress_evidence_clip(source_path: Path, output_path: Path, codec: str | None, crf: int) -> str | None:
        ffmpeg = shutil.which("ffmpeg")
        converted = output_path.with_name(f"{output_path.stem}.compressing.mp4")
        if ffmpeg:
            try:
                result = subprocess.run([
                    ffmpeg, "-y", "-loglevel", "error", "-i", str(source_path), "-an",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(converted),
                ], capture_output=True, timeout=120, check=False)
                if result.returncode == 0 and converted.is_file() and converted.stat().st_size > 0:
                    converted.replace(output_path)
                    source_path.unlink(missing_ok=True)
                    return "h264"
            except (OSError, subprocess.SubprocessError):
                pass
        converted.unlink(missing_ok=True)
        source_path.replace(output_path)
        return codec

    def _sample_segments(self, segment_ids: list[str], limit: int) -> list[np.ndarray]:
        samples = []
        for segment_id in segment_ids:
            path = Path(self.config["paths"]["clip_dir"]) / f"{segment_id}.mp4"
            capture = cv2.VideoCapture(str(path))
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            positions = np.linspace(0, max(0, total - 1), min(limit - len(samples), max(1, total)), dtype=int)
            for position in positions:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
                ok, frame = capture.read()
                if ok:
                    samples.append(frame)
            capture.release()
            if len(samples) >= limit:
                break
        return samples[:limit]

    @staticmethod
    def _overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
        return not (float(left["endTime"]) < float(right["startTime"]) or float(right["endTime"]) < float(left["startTime"]))

    def _complete_link(self, track_ids: list[str], tracks: dict[str, dict[str, Any]], scores: dict[tuple[str, str], float], threshold: float) -> list[list[str]]:
        groups = [{track_id} for track_id in track_ids]
        def score(left: str, right: str) -> float | None:
            return scores.get(tuple(sorted((left, right))))
        while True:
            best: tuple[float, int, int] | None = None
            for left_index in range(len(groups)):
                for right_index in range(left_index + 1, len(groups)):
                    cross = [(left, right) for left in groups[left_index] for right in groups[right_index]]
                    values = [score(left, right) for left, right in cross]
                    if any(self._overlap(tracks[left], tracks[right]) for left, right in cross) or any(value is None for value in values):
                        continue
                    minimum = min(float(value) for value in values)
                    if minimum >= threshold and (best is None or minimum > best[0]):
                        best = (minimum, left_index, right_index)
            if best is None:
                break
            _, left_index, right_index = best
            groups[left_index] |= groups[right_index]
            groups.pop(right_index)
        return [sorted(group) for group in groups]
