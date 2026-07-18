"""智能体、记忆和证据接口。"""
from __future__ import annotations
from pathlib import Path
from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from agent import AgentController
from web.models import AgentQuery
router = APIRouter(tags=["agent-memory"])

@router.post("/api/agent/query")
async def query_agent(body: AgentQuery, request: Request):
    controller = AgentController(request.app.state.config, request.app.state.repository, request.app.state.tool_service, request.app.state.llm, request.app.state.embedder, request.app.state.vectors)
    return await run_in_threadpool(controller.answer, body.question)

@router.get("/api/memory/tracks")
async def list_tracks(request: Request, start: float | None = None, end: float | None = None):
    time_range = (start, end) if start is not None and end is not None else None
    tracks = request.app.state.repository.find_tracks(time_range)
    return {"total": len(tracks), "tracks": tracks}

@router.get("/api/memory/tracks/{track_id}/frames")
async def track_frames(track_id: str, request: Request):
    return request.app.state.tool_service.getFrames([track_id])

@router.get("/api/evidence/keyframes/{keyframe_id}")
async def keyframe_file(keyframe_id: str, request: Request):
    item = request.app.state.repository.get_keyframe(keyframe_id)
    if not item or not Path(item["keyframePath"]).exists():
        raise HTTPException(404, "关键帧不存在")
    return FileResponse(item["keyframePath"])

@router.get("/api/evidence/clips/{segment_id}")
async def clip_file(segment_id: str, request: Request):
    if Path(segment_id).name != segment_id:
        raise HTTPException(400, "片段编号非法")
    path = Path(request.app.state.config["paths"]["clip_dir"]) / f"{segment_id}.mp4"
    if not path.exists():
        raise HTTPException(404, "目标船片段不存在")
    return FileResponse(path, media_type="video/mp4")

@router.get("/api/evidence/registry/{reference_id}")
async def registry_file(reference_id: str, request: Request):
    items = request.app.state.repository.references_by_ids([reference_id])
    if not items or not Path(items[0]["imagePath"]).exists():
        raise HTTPException(404, "先验库参考图不存在")
    return FileResponse(items[0]["imagePath"])

@router.get("/api/config")
async def public_config(request: Request):
    config = request.app.state.config
    return {"app": config.get("app", {}), "models": {"recognition": config["llm"]["model"], "embedding": config["embedding"]["model"], "embeddingDimension": config["embedding"]["dimension"]}, "pipeline": {"candidateEveryNFrames": config["pipeline"]["candidate_every_n_frames"], "keyframePoolSize": config["pipeline"]["keyframe_pool_size"], "maxRounds": config["pipeline"]["agent"]["max_rounds"], "displayLimit": config["pipeline"]["agent"]["display_limit"]}}
