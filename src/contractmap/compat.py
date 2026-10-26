"""
Compatibility layer for legacy code.

This module provides the old `keywords_and_llm` function interface
used by evaluation scripts, backed by the new unified classifier.
"""

import asyncio
from pathlib import Path
from typing import Optional, Tuple

from .classifier import (
    ContractClassifier,
    ClassificationResult,
    get_classifier,
    reset_classifier,
)


# Global classifier instance for backward compatibility
_compat_classifier: Optional[ContractClassifier] = None


def _get_compat_classifier(system_prompt_file_location: Optional[Path] = None) -> ContractClassifier:
    global _compat_classifier
    
    # If custom prompt provided, we can't use singleton - create new instance
    if system_prompt_file_location:
        from .classifier import ClassifierSettings
        settings = ClassifierSettings()
        with open(system_prompt_file_location, "r", encoding="utf-8") as f:
            custom_prompt = f.read()
        
        # Create a temporary classifier with custom prompt
        classifier = ContractClassifier(settings)
        classifier.system_prompt = custom_prompt
        classifier.llm.system_prompt = custom_prompt
        classifier.llm._prompt_hash = __import__('hashlib').sha256(custom_prompt.encode()).hexdigest()[:16]
        return classifier
    
    # Use singleton
    if _compat_classifier is None:
        _compat_classifier = get_classifier()
    return _compat_classifier


async def keywords_and_llm(
    description: str,
    threshold: int,
    margin: int,
    system_prompt_file_location: Optional[Path] = None,
) -> Tuple[str, str, int]:
    classifier = _get_compat_classifier(system_prompt_file_location)
    
    # Override threshold/margin for this call if different from defaults
    original_threshold = classifier.scoring.threshold
    original_margin = classifier.scoring.margin
    
    if threshold != original_threshold or margin != original_margin:
        # Create temporary scoring config
        from .classifier import ScoringConfig
        temp_scoring = ScoringConfig(
            rm_keyword_points=classifier.scoring.rm_keyword_points,
            primary_keyword_points=classifier.scoring.primary_keyword_points,
            secondary_keyword_points=classifier.scoring.secondary_keyword_points,
            threshold=threshold,
            margin=margin,
        )
        classifier.heuristic.scoring = temp_scoring
    
    try:
        result: ClassificationResult = await classifier.classify(description)
        
        # Map method to reason string
        method_to_reason = {
            "heuristic": f"Heuristic Match (Score: {result.heuristic_score})",
            "llm": "LLM used",
            "cache": f"Cached LLM (Score: {result.heuristic_score})",
            "llm_fallback": "LLM Fallback",
        }
        reason = method_to_reason.get(result.method, result.method)
        
        return result.category, reason, result.heuristic_score
        
    finally:
        # Restore original scoring
        if threshold != original_threshold or margin != original_margin:
            from .classifier import ScoringConfig
            classifier.heuristic.scoring = ScoringConfig(
                rm_keyword_points=classifier.scoring.rm_keyword_points,
                primary_keyword_points=classifier.scoring.primary_keyword_points,
                secondary_keyword_points=classifier.scoring.secondary_keyword_points,
                threshold=original_threshold,
                margin=original_margin,
            )


# Also expose the old HeuristicClassifier for any direct usage
from .classifier import AsyncHeuristicClassifier as HeuristicClassifier


__all__ = [
    "keywords_and_llm",
    "HeuristicClassifier",
    "get_classifier",
    "reset_classifier",
]