"""
FastAPI v3 Endpoint - Production-ready contract classification API.

"""

import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, status, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

import structlog

from contractmap import get_classifier, ClassificationResult
from contractmap.classifier import ClassifierSettings

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

class ContractDescription(BaseModel):
    """Input contract description with validation."""
    description: str = Field(..., min_length=1, max_length=10000, description="Contract description text")

    @field_validator('description')
    @classmethod
    def sanitize_description(cls, v: str) -> str:
        cleaned = ''.join(c for c in v if c == '\n' or c == '\t' or ord(c) >= 32)
        cleaned = ' '.join(cleaned.split())
        if not cleaned:
            raise ValueError("Description cannot be empty after sanitization")
        return cleaned


class ClassificationResponse(BaseModel):
    """Full classification response with metadata."""
    AI_label: str
    method: str
    confidence: float
    heuristic_score: int
    latency_ms: float
    tokens_used: int = 0
    cache_hit: bool = False
    runner_up: Optional[str] = None
    runner_up_score: int = 0


class HealthResponse(BaseModel):
    status: str
    version: str


class ReadinessResponse(BaseModel):
    status: str
    checks: dict


class StatsResponse(BaseModel):
    categories_loaded: int
    scoring_config: dict
    llm_config: dict
    cache_size: Optional[int] = None


class SimpleRateLimiter:
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = {}

    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        if client_ip not in self._requests:
            self._requests[client_ip] = []

        self._requests[client_ip] = [
            t for t in self._requests[client_ip]
            if now - t < self.window_seconds
        ]

        if len(self._requests[client_ip]) >= self.max_requests:
            return False

        self._requests[client_ip].append(now)
        return True


rate_limiter = SimpleRateLimiter(
    max_requests=int(os.getenv("RATE_LIMIT_REQUESTS", "100")),
    window_seconds=int(os.getenv("RATE_LIMIT_WINDOW", "60"))
)


async def check_rate_limit(request: Request) -> None:
    """Dependency to check rate limit."""
    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.is_allowed(client_ip):
        logger.warning("rate_limit_exceeded", client_ip=client_ip)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later."
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown."""
    logger.info("api_starting", version="0.2.0")

    try:
        classifier = get_classifier()
        stats = classifier.get_stats()
        logger.info("classifier_initialized", **stats)
    except Exception as e:
        logger.error("classifier_init_failed", error=str(e), exc_info=True)
        raise

    yield

    # Shutdown
    logger.info("api_shutting_down")



# FastAPI App
app = FastAPI(
    title="CCS Contract Map API",
    description="Hybrid keyword + LLM classifier for CCS contract categories",
    version="0.2.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Logging Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing."""
    start = time.perf_counter()
    request_id = request.headers.get("X-Request-ID", "none")

    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        client_ip=request.client.host if request.client else "unknown"
    )

    try:
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info("request_completed",
                    status_code=response.status_code,
                    duration_ms=round(duration_ms, 2))
        return response
    except Exception as e:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.error("request_failed", error=str(e), duration_ms=round(duration_ms, 2), exc_info=True)
        raise



# Health Endpoints
@app.get("/healthz", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Kubernetes liveness probe - basic health check."""
    return HealthResponse(status="healthy", version="0.2.0")


@app.get("/readyz", response_model=ReadinessResponse, tags=["Health"])
async def readiness_check():
    """Kubernetes readiness probe - checks dependencies."""
    checks = {"api": "ok"}
    try:
        classifier = get_classifier()
        if classifier.heuristic._categories:
            checks["classifier"] = "ok"
        else:
            checks["classifier"] = "no_categories_loaded"
    except Exception as e:
        checks["classifier"] = f"error: {type(e).__name__}"

    all_healthy = all(v == "ok" for v in checks.values())
    status_code = 200 if all_healthy else 503

    return JSONResponse(
        status_code=status_code,
        content=ReadinessResponse(
            status="ready" if all_healthy else "not_ready",
            checks=checks
        ).model_dump()
    )


@app.get("/stats", response_model=StatsResponse, tags=["Monitoring"])
async def get_stats():
    """Get classifier statistics."""
    classifier = get_classifier()
    return classifier.get_stats()



# Classification Endpoints
@app.post(
    "/api/v1/classify",
    response_model=ClassificationResponse,
    tags=["Classification"],
    dependencies=[Depends(check_rate_limit)]
)
async def classify_contract(body: ContractDescription):
    classifier = get_classifier()

    try:
        result: ClassificationResult = await classifier.classify(body.description)

        logger.info("classification_completed",
                    category=result.category,
                    method=result.method,
                    confidence=result.confidence,
                    latency_ms=result.latency_ms)

        return ClassificationResponse(
            AI_label=result.category,
            method=result.method,
            confidence=result.confidence,
            heuristic_score=result.heuristic_score,
            latency_ms=result.latency_ms,
            tokens_used=result.tokens_used,
            cache_hit=result.cache_hit,
            runner_up=result.runner_up,
            runner_up_score=result.runner_up_score,
        )

    except Exception as e:
        logger.error("classification_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Classification failed. Please try again."
        )


@app.post(
    "/api/v1/classify/batch",
    response_model=list[ClassificationResponse],
    tags=["Classification"],
    dependencies=[Depends(check_rate_limit)]
)
async def classify_batch(body: list[ContractDescription]):
    if len(body) > 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch size exceeds maximum of 50"
        )

    classifier = get_classifier()
    descriptions = [item.description for item in body]

    try:
        results = await classifier.classify_batch(descriptions)

        return [
            ClassificationResponse(
                AI_label=r.category,
                method=r.method,
                confidence=r.confidence,
                heuristic_score=r.heuristic_score,
                latency_ms=r.latency_ms,
                tokens_used=r.tokens_used,
                cache_hit=r.cache_hit,
                runner_up=r.runner_up,
                runner_up_score=r.runner_up_score,
            )
            for r in results
        ]

    except Exception as e:
        logger.error("batch_classification_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch classification failed. Please try again."
        )



# Backward Compatibility Endpoint
@app.post("/v0.2.0/map", tags=["Legacy"])
async def legacy_map(body: ContractDescription):
    classifier = get_classifier()
    result = await classifier.classify(body.description)
    return {"AI_label": result.category}