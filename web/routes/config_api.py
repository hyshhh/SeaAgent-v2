"""运行参数设置接口。"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from config.settings import public_settings, reset_settings, update_settings
from web.models import ApiResponse

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("")
async def get_settings(request: Request):
    return public_settings(request.app.state.config)


@router.put("", response_model=ApiResponse)
async def save_settings(body: dict[str, Any], request: Request):
    try:
        data = update_settings(request.app.state.config, body)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return ApiResponse(success=True, message="运行参数已保存，新任务开始时生效", data=data)


@router.post("/reset", response_model=ApiResponse)
async def restore_settings(request: Request):
    data = reset_settings(request.app.state.config)
    return ApiResponse(success=True, message="运行参数已恢复默认值", data=data)
