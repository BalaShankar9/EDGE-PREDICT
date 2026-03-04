"""API key authentication for pipeline endpoints."""
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from sharpedge.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(api_key: str = Security(api_key_header)) -> str:
    if not settings.api_key:
        raise HTTPException(status_code=500, detail="API key not configured")
    if api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key
