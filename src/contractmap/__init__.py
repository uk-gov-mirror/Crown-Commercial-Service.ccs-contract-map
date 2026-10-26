"""
CCS Contract Map - Hybrid Keyword + LLM Classifier for CCS Categories.
"""

from .classifier import (
    ContractClassifier,
    ClassificationResult,
    ClassifierSettings,
    get_classifier,
    reset_classifier,
)

# Backward compatibility
from .compat import (
    keywords_and_llm,
    HeuristicClassifier,
)

# Calibration
from .calibration import (
    HeuristicScoreCalibrator,
    ConfidenceCalibrator,
    create_calibrator_from_evaluation,
)

# Semantic Cache
from .semantic_cache import (
    SemanticCache,
    HybridCache,
)

__version__ = "0.2.0"
__all__ = [
    "ContractClassifier",
    "ClassificationResult",
    "ClassifierSettings",
    "get_classifier",
    "reset_classifier",
    # Backward compatibility
    "keywords_and_llm",
    "HeuristicClassifier",
    # Calibration
    "HeuristicScoreCalibrator",
    "ConfidenceCalibrator",
    "create_calibrator_from_evaluation",
    # Semantic Cache
    "SemanticCache",
    "HybridCache",
]