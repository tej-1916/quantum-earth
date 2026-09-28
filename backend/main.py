"""Quantum Earth Lab V2 - FastAPI Backend Application."""

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from backend.core.config import settings
from backend.api import health, gibs, cmr, dataset, models, quantum

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS if settings.CORS_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Security: Limit maximum payload size
@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > settings.MAX_UPLOAD_SIZE_BYTES:
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content={
                "error": "PAYLOAD_TOO_LARGE",
                "detail": f"Request size exceeds limit of {settings.MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB.",
            },
        )
    return await call_next(request)


# Register API Routers
app.include_router(health.router, prefix="/api")
app.include_router(gibs.router, prefix="/api")
app.include_router(cmr.router, prefix="/api")
app.include_router(dataset.router, prefix="/api")
app.include_router(models.router, prefix="/api")
app.include_router(quantum.router, prefix="/api/models/quantum")
app.include_router(quantum.router, prefix="/api/quantum")


@app.get("/")
async def root():
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online",
        "documentation": "/docs",
        "health": "/api/health",
        "scientific_integrity": "Active - No fabricated predictions or mock performance figures.",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host=settings.HOST, port=settings.PORT, reload=True)
