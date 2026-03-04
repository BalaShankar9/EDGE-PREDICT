"""Run the FastAPI server: python -m sharpedge.api"""
import uvicorn
from sharpedge.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "sharpedge.api.app:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
