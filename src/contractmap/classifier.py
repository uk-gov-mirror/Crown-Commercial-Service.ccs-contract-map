"""
Unified Contract Classifier - Hybrid Keyword + LLM with Cost Optimization.

This module replaces classification_v1.py, v2.py, v5.py, v2_mix_v5.py
with a single, configurable, production-ready classifier.
"""

import json
import re
import logging
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from functools import lru_cache

import structlog
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Local imports
from .calibration import HeuristicScoreCalibrator
from .semantic_cache import SemanticCache, HybridCache

# Configure structured logging
logger = structlog.get_logger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class ClassifierSettings(BaseSettings):
    """All classifier settings via environment variables."""
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Azure OpenAI
    azure_openai_endpoint: str = Field(..., alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_key: str = Field(..., alias="AZURE_OPENAI_KEY")
    deployment_name: str = Field(..., alias="DEPLOYMENT_NAME")
    azure_openai_api_version: str = Field("2024-02-01", alias="AZURE_OPENAI_API_VERSION")

    # Classifier behavior
    keyword_threshold: int = Field(3, alias="KEYWORD_THRESHOLD")
    keyword_margin: int = Field(5, alias="KEYWORD_MARGIN")
    max_tokens: int = Field(50, alias="MAX_TOKENS")  # Increased from 6 for reliability
    temperature: float = Field(0.0, alias="TEMPERATURE")
    llm_timeout: float = Field(30.0, alias="LLM_TIMEOUT")

    # Paths
    semantic_anchors_path: Path = Field(Path("semantic_anchors.json"), alias="SEMANTIC_ANCHORS_PATH")
    system_prompt_path: Path = Field(Path("prompts/system_prompt_v2.md"), alias="SYSTEM_PROMPT_PATH")

    # Cost optimization
    enable_llm_cache: bool = Field(True, alias="ENABLE_LLM_CACHE")
    cache_ttl_seconds: int = Field(3600, alias="CACHE_TTL_SECONDS")  # 1 hour
    max_description_length: int = Field(10000, alias="MAX_DESCRIPTION_LENGTH")

    # Semantic cache settings
    enable_semantic_cache: bool = Field(True, alias="ENABLE_SEMANTIC_CACHE")
    semantic_cache_model: str = Field("all-MiniLM-L6-v2", alias="SEMANTIC_CACHE_MODEL")
    semantic_cache_threshold: float = Field(0.92, alias="SEMANTIC_CACHE_THRESHOLD")
    semantic_cache_max_size: int = Field(10000, alias="SEMANTIC_CACHE_MAX_SIZE")
    semantic_cache_persist_path: Optional[Path] = Field(None, alias="SEMANTIC_CACHE_PERSIST_PATH")

    # Scoring weights (configurable)
    rm_keyword_points: int = Field(50, alias="RM_KEYWORD_POINTS")
    primary_keyword_points: int = Field(5, alias="PRIMARY_KEYWORD_POINTS")
    secondary_keyword_points: int = Field(1, alias="SECONDARY_KEYWORD_POINTS")


@dataclass(frozen=True)
class ScoringConfig:
    """Immutable scoring configuration."""
    rm_keyword_points: int
    primary_keyword_points: int
    secondary_keyword_points: int
    threshold: int
    margin: int


@dataclass
class ClassificationResult:
    """Result of classification with full metadata."""
    category: str
    method: str  # "heuristic" | "llm" | "cache"
    confidence: float  # 0.0 - 1.0
    heuristic_score: int
    latency_ms: float
    tokens_used: int = 0
    cache_hit: bool = False
    runner_up: Optional[str] = None
    runner_up_score: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "method": self.method,
            "confidence": round(self.confidence, 4),
            "heuristic_score": self.heuristic_score,
            "latency_ms": round(self.latency_ms, 2),
            "tokens_used": self.tokens_used,
            "cache_hit": self.cache_hit,
            "runner_up": self.runner_up,
            "runner_up_score": self.runner_up_score,
        }


# =============================================================================
# Heuristic Classifier (Async, No Thread Pool)
# =============================================================================

class AsyncHeuristicClassifier:
    """
    Fully async heuristic classifier using pre-compiled regex.
    No blocking I/O, no thread pool needed.
    """

    def __init__(self, registry_path: Path, scoring: ScoringConfig):
        self.registry_path = registry_path
        self.scoring = scoring
        self._registry: Dict = {}
        self._compiled: Dict[str, Tuple[List[Tuple[re.Pattern, int]], List[re.Pattern]]] = {}
        self._categories: List[str] = []
        self._load_and_compile()

    def _load_and_compile(self) -> None:
        """Load and pre-compile all regex patterns at startup."""
        with open(self.registry_path, "r", encoding="utf-8") as f:
            self._registry = json.load(f)

        self._categories = sorted(self._registry.keys())

        for category, (primary_list, secondary_list) in self._registry.items():
            compiled_primary = []
            for word in primary_list:
                pattern = re.compile(rf"\b{re.escape(word.lower())}\b")
                # RM codes get higher points
                points = self.scoring.rm_keyword_points if word.lower().startswith("rm") else self.scoring.primary_keyword_points
                compiled_primary.append((pattern, points))

            compiled_secondary = [
                re.compile(rf"\b{re.escape(word.lower())}\b") for word in secondary_list
            ]

            self._compiled[category] = (compiled_primary, compiled_secondary)

        logger.info("heuristic_classifier_loaded", categories=len(self._categories))

    async def classify(self, description: str) -> Tuple[Optional[str], int, Optional[str], int]:
        """
        Classify using keyword matching.
        Returns: (best_category, best_score, runner_up_category, runner_up_score)
        """
        if not description:
            return None, 0, None, 0

        start = time.perf_counter()
        desc_lower = description.lower()
        category_scores: Dict[str, int] = {}

        # Score all categories (pure CPU, very fast - no await needed)
        for category, (primary_patterns, secondary_patterns) in self._compiled.items():
            score = 0

            # Primary keywords (weighted)
            for pattern, points in primary_patterns:
                if pattern.search(desc_lower):
                    score += points

            # Secondary keywords (1 point each)
            for pattern in secondary_patterns:
                if pattern.search(desc_lower):
                    score += self.scoring.secondary_keyword_points

            category_scores[category] = score

        # Find winner and runner-up
        sorted_results = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)

        if not sorted_results or sorted_results[0][1] == 0:
            return None, 0, None, 0

        best_cat, best_score = sorted_results[0]
        runner_up_cat, runner_up_score = sorted_results[1] if len(sorted_results) > 1 else (None, 0)

        latency = (time.perf_counter() - start) * 1000
        logger.debug("heuristic_classified",
                     best_cat=best_cat, best_score=best_score,
                     runner_up=runner_up_cat, runner_up_score=runner_up_score,
                     latency_ms=latency)

        return best_cat, best_score, runner_up_cat, runner_up_score

    def should_use_llm(self, best_score: int, runner_up_score: int) -> bool:
        """Determine if LLM fallback is needed based on thresholds."""
        if best_score < self.scoring.threshold:
            return True
        if (best_score - runner_up_score) < self.scoring.margin:
            return True
        return False


# =============================================================================
# LLM Classifier with Cost Optimization
# =============================================================================

class LLMCache:
    """Simple in-memory cache for LLM responses with TTL."""

    def __init__(self, ttl_seconds: int = 3600):
        self._cache: Dict[str, Tuple[ClassificationResult, float]] = {}
        self._ttl = ttl_seconds

    def _make_key(self, description: str, prompt_hash: str) -> str:
        """Create cache key from description hash + prompt hash."""
        desc_hash = hashlib.sha256(description.encode()).hexdigest()[:16]
        return f"{prompt_hash}:{desc_hash}"

    def get(self, description: str, prompt_hash: str) -> Optional[ClassificationResult]:
        key = self._make_key(description, prompt_hash)
        if key in self._cache:
            result, timestamp = self._cache[key]
            if time.time() - timestamp < self._ttl:
                result.cache_hit = True
                logger.info("llm_cache_hit", key=key[:32])
                return result
            else:
                del self._cache[key]
        return None

    def set(self, description: str, prompt_hash: str, result: ClassificationResult) -> None:
        key = self._make_key(description, prompt_hash)
        self._cache[key] = (result, time.time())
        logger.info("llm_cache_set", key=key[:32], cache_size=len(self._cache))

    def clear(self) -> None:
        self._cache.clear()


class OptimizedLLMClassifier:
    """
    LLM classifier with cost optimization:
    - Response caching (semantic + exact)
    - Reduced token usage
    - Structured output validation
    - Timeout and retry logic
    """

    VALID_CATEGORIES = frozenset([
        "Financial Services", "Fleet", "HR & Workforce Services",
        "Professional and Business Services", "Travel, Accommodation and Venues",
        "Construction", "Energy", "Facilities Management",
        "Cloud and Hosting", "Digital and Technology Services",
        "Network Services", "Software", "Hardware",
        "Outside New Taxonomy"
    ])

    def __init__(
        self,
        settings: ClassifierSettings,
        system_prompt: str,
        cache: Optional[LLMCache] = None
    ):
        self.settings = settings
        self.system_prompt = system_prompt
        self._prompt_hash = hashlib.sha256(system_prompt.encode()).hexdigest()[:16]

        # Build cache hierarchy
        exact_cache = cache or LLMCache(settings.cache_ttl_seconds)
        
        # Initialize semantic cache if enabled
        semantic_cache = None
        if settings.enable_semantic_cache:
            try:
                semantic_cache = SemanticCache(
                    model_name=settings.semantic_cache_model,
                    similarity_threshold=settings.semantic_cache_threshold,
                    max_size=settings.semantic_cache_max_size,
                    ttl_seconds=settings.cache_ttl_seconds,
                    persist_path=settings.semantic_cache_persist_path,
                )
            except Exception as e:
                logger.warning("semantic_cache_init_failed", error=str(e))

        self.cache = HybridCache(exact_cache=exact_cache, semantic_cache=semantic_cache)

        # Build LLM with optimized settings
        self.llm = AzureChatOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            openai_api_key=settings.azure_openai_key,
            azure_deployment=settings.deployment_name,
            openai_api_version=settings.azure_openai_api_version,
            temperature=settings.temperature,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            seed=42,
            max_tokens=settings.max_tokens,
            request_timeout=settings.llm_timeout,
            max_retries=2,
        )

        logger.info("llm_classifier_initialized",
                    deployment=settings.deployment_name,
                    max_tokens=settings.max_tokens,
                    exact_cache_enabled=settings.enable_llm_cache,
                    semantic_cache_enabled=settings.enable_semantic_cache)

    def _truncate_description(self, description: str) -> str:
        """Truncate description to max length to control token costs."""
        if len(description) > self.settings.max_description_length:
            truncated = description[:self.settings.max_description_length]
            logger.warning("description_truncated", original_len=len(description), truncated_len=len(truncated))
            return truncated
        return description

    def _validate_category(self, response: str) -> str:
        """Validate LLM response against known taxonomy."""
        cleaned = response.strip().strip('"\'').strip()
        if cleaned in self.VALID_CATEGORIES:
            return cleaned
        # Fuzzy match for common variations
        for cat in self.VALID_CATEGORIES:
            if cat.lower() == cleaned.lower():
                return cat
        logger.warning("llm_invalid_category", response=cleaned, valid=list(self.VALID_CATEGORIES))
        return "Outside New Taxonomy"

    async def classify(self, description: str, heuristic_score: int) -> ClassificationResult:
        """Classify with caching and validation."""
        start = time.perf_counter()
        description = self._truncate_description(description)

        # Check cache first
        if self.settings.enable_llm_cache:
            cached = self.cache.get(description, self._prompt_hash)
            if cached:
                cached.latency_ms = (time.perf_counter() - start) * 1000
                return cached

        # Build messages
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=description),
        ]

        # Call LLM
        try:
            response = await self.llm.ainvoke(messages)
            raw_category = response.content.strip()
            category = self._validate_category(raw_category)

            # Estimate tokens (rough approximation)
            tokens_used = len(self.system_prompt.split()) + len(description.split()) + len(raw_category.split())

            result = ClassificationResult(
                category=category,
                method="llm",
                confidence=0.85,  # Heuristic - could be calibrated
                heuristic_score=heuristic_score,
                latency_ms=(time.perf_counter() - start) * 1000,
                tokens_used=tokens_used,
            )

            # Cache result
            if self.settings.enable_llm_cache:
                self.cache.set(description, self._prompt_hash, result)

            logger.info("llm_classified", category=category, tokens=tokens_used, latency_ms=result.latency_ms)
            return result

        except Exception as e:
            logger.error("llm_classification_failed", error=str(e), exc_info=True)
            # Return fallback
            return ClassificationResult(
                category="Outside New Taxonomy",
                method="llm_fallback",
                confidence=0.0,
                heuristic_score=heuristic_score,
                latency_ms=(time.perf_counter() - start) * 1000,
            )


# =============================================================================
# Unified Classifier
# =============================================================================

class ContractClassifier:
    """
    Main classifier combining heuristic + LLM with full observability.
    """

    def __init__(self, settings: Optional[ClassifierSettings] = None):
        self.settings = settings or ClassifierSettings()
        self.scoring = ScoringConfig(
            rm_keyword_points=self.settings.rm_keyword_points,
            primary_keyword_points=self.settings.primary_keyword_points,
            secondary_keyword_points=self.settings.secondary_keyword_points,
            threshold=self.settings.keyword_threshold,
            margin=self.settings.keyword_margin,
        )

        # Load system prompt
        with open(self.settings.system_prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read()

        # Initialize calibrator (loads from disk if available)
        self.calibrator = self._load_calibrator()

        # Initialize components
        self.heuristic = AsyncHeuristicClassifier(
            self.settings.semantic_anchors_path,
            self.scoring
        )
        self.llm_cache = LLMCache(self.settings.cache_ttl_seconds) if self.settings.enable_llm_cache else None
        self.llm = OptimizedLLMClassifier(self.settings, self.system_prompt, self.llm_cache)

        logger.info("contract_classifier_initialized",
                    threshold=self.scoring.threshold,
                    margin=self.scoring.margin,
                    categories=len(self.heuristic._categories),
                    calibrator_loaded=self.calibrator.is_fitted)

    def _load_calibrator(self) -> HeuristicScoreCalibrator:
        """Load calibrator from disk if available."""
        calibrator_path = Path("calibrator.pkl")
        if calibrator_path.exists():
            try:
                return HeuristicScoreCalibrator.load(calibrator_path)
            except Exception as e:
                logger.warning("failed_to_load_calibrator", error=str(e))
        return HeuristicScoreCalibrator()

    def _calibrate_heuristic_confidence(self, score: int) -> float:
        """Calibrate heuristic score to confidence using fitted calibrator."""
        if self.calibrator.is_fitted:
            return self.calibrator.calibrate(score)
        # Fallback: simple mapping
        return min(0.95, max(0.05, score / 100.0))

    async def classify(self, description: str) -> ClassificationResult:
        """
        Main classification entry point.
        Flow: Heuristic -> (if needed) LLM -> Validate -> Return
        """
        total_start = time.perf_counter()

        # Step 1: Heuristic classification (always runs, very fast)
        best_cat, best_score, runner_up, runner_up_score = await self.heuristic.classify(description)

        # Step 2: Check if heuristic is confident enough
        if best_cat and not self.heuristic.should_use_llm(best_score, runner_up_score):
            return ClassificationResult(
                category=best_cat,
                method="heuristic",
                confidence=self._calibrate_heuristic_confidence(best_score),
                heuristic_score=best_score,
                latency_ms=(time.perf_counter() - total_start) * 1000,
                runner_up=runner_up,
                runner_up_score=runner_up_score,
            )

        # Step 3: Fallback to LLM
        logger.info("falling_back_to_llm",
                    heuristic_cat=best_cat,
                    heuristic_score=best_score,
                    runner_up=runner_up,
                    margin=best_score - runner_up_score,
                    required_margin=self.scoring.margin)

        llm_result = await self.llm.classify(description, best_score)
        llm_result.latency_ms = (time.perf_counter() - total_start) * 1000
        llm_result.runner_up = runner_up
        llm_result.runner_up_score = runner_up_score

        return llm_result

    async def classify_batch(self, descriptions: List[str]) -> List[ClassificationResult]:
        """Classify multiple descriptions concurrently."""
        import asyncio
        tasks = [self.classify(desc) for desc in descriptions]
        return await asyncio.gather(*tasks)

    def get_stats(self) -> Dict[str, Any]:
        """Get classifier statistics."""
        stats = {
            "categories_loaded": len(self.heuristic._categories),
            "scoring_config": {
                "threshold": self.scoring.threshold,
                "margin": self.scoring.margin,
                "rm_points": self.scoring.rm_keyword_points,
                "primary_points": self.scoring.primary_keyword_points,
                "secondary_points": self.scoring.secondary_keyword_points,
            },
            "llm_config": {
                "max_tokens": self.settings.max_tokens,
                "temperature": self.settings.temperature,
                "exact_cache_enabled": self.settings.enable_llm_cache,
                "semantic_cache_enabled": self.settings.enable_semantic_cache,
                "cache_ttl": self.settings.cache_ttl_seconds,
            },
        }
        # Cache stats from LLM classifier
        if hasattr(self.llm, 'cache') and hasattr(self.llm.cache, 'stats'):
            stats["cache"] = self.llm.cache.stats()
        elif self.llm_cache:
            stats["cache_size"] = len(self.llm_cache._cache)
        return stats


# =============================================================================
# Factory Function (for FastAPI dependency injection)
# =============================================================================

_classifier_instance: Optional[ContractClassifier] = None


def get_classifier() -> ContractClassifier:
    """Get or create singleton classifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = ContractClassifier()
    return _classifier_instance


def reset_classifier() -> None:
    """Reset singleton (useful for testing)."""
    global _classifier_instance
    _classifier_instance = None