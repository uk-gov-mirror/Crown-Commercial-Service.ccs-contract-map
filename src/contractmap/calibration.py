"""
Confidence Calibration for Contract Classifier.

Provides Platt scaling and Isotonic regression for calibrated confidence scores.
"""

import json
import pickle
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class ConfidenceCalibrator:
    """
    Calibrates raw classifier scores to well-calibrated probabilities.
    
    Supports:
    - Platt Scaling (sigmoid calibration) - good for small datasets
    - Isotonic Regression - good for larger datasets, non-parametric
    """
    
    def __init__(self, method: str = "isotonic"):
        """
        Initialize calibrator.
        
        Args:
            method: "platt" for sigmoid calibration, "isotonic" for isotonic regression
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for calibration. Install with: pip install scikit-learn")
        
        self.method = method
        self.calibrator = None
        self.is_fitted = False
        
        if method == "platt":
            # Platt scaling uses logistic regression on scores
            self.base_estimator = LogisticRegression(solver='lbfgs', max_iter=1000)
            self.calibrator = CalibratedClassifierCV(
                self.base_estimator, 
                method='sigmoid', 
                cv=3
            )
        elif method == "isotonic":
            # Isotonic regression - non-parametric, needs more data
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
        else:
            raise ValueError(f"Unknown calibration method: {method}")
    
    def fit(self, scores: List[float], labels: List[int]) -> "ConfidenceCalibrator":
        """
        Fit calibrator on validation data.
        
        Args:
            scores: Raw heuristic scores (or confidence scores)
            labels: Binary labels (1 for correct, 0 for incorrect)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for calibration")
        
        X = np.array(scores).reshape(-1, 1)
        y = np.array(labels)
        
        if self.method == "platt":
            self.calibrator.fit(X, y)
        else:  # isotonic
            # IsotonicRegression expects 1D arrays
            self.calibrator.fit(X.ravel(), y)
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, scores: List[float]) -> np.ndarray:
        """
        Predict calibrated probabilities.
        
        Args:
            scores: Raw scores to calibrate
            
        Returns:
            Calibrated probabilities (same length as scores)
        """
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")
        
        X = np.array(scores).reshape(-1, 1)
        
        if self.method == "platt":
            # Returns (n_samples, 2) - prob for class 0 and 1
            probas = self.calibrator.predict_proba(X)
            return probas[:, 1]  # Return probability of class 1 (correct)
        else:  # isotonic
            return self.calibrator.predict(X.ravel())
    
    def save(self, path: Path) -> None:
        """Save calibrator to disk."""
        with open(path, "wb") as f:
            pickle.dump({
                "method": self.method,
                "calibrator": self.calibrator,
                "is_fitted": self.is_fitted,
            }, f)
    
    @classmethod
    def load(cls, path: Path) -> "ConfidenceCalibrator":
        """Load calibrator from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        calibrator = cls(method=data["method"])
        calibrator.calibrator = data["calibrator"]
        calibrator.is_fitted = data["is_fitted"]
        return calibrator


class HeuristicScoreCalibrator:
    """
    Specialized calibrator for heuristic scores.
    
    Maps heuristic scores to calibrated confidence using
    isotonic regression on (score, is_correct) pairs.
    """
    
    def __init__(self):
        self.calibrator = None
        self.is_fitted = False
    
    def fit_from_evaluation_results(self, results_path: Path) -> "HeuristicScoreCalibrator":
        """
        Fit calibrator from evaluation results CSV.
        
        Expected columns: heuristic_score, is_correct (or AI_Prediction == Category)
        """
        import pandas as pd
        
        df = pd.read_csv(results_path)
        
        # Determine correctness
        if "is_correct" in df.columns:
            labels = df["is_correct"].astype(int).tolist()
        elif "AI_Prediction" in df.columns and "Category" in df.columns:
            labels = (df["AI_Prediction"] == df["Category"]).astype(int).tolist()
        else:
            raise ValueError("CSV must contain 'is_correct' or both 'AI_Prediction' and 'Category' columns")
        
        scores = df["Heuristic_Score"].tolist() if "Heuristic_Score" in df.columns else df["heuristic_score"].tolist()
        
        return self.fit(scores, labels)
    
    def fit(self, scores: List[float], labels: List[int]) -> "HeuristicScoreCalibrator":
        """Fit isotonic regression on scores vs correctness."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for calibration")
        
        X = np.array(scores).reshape(-1, 1)
        y = np.array(labels)
        
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(X.ravel(), y)
        self.is_fitted = True
        return self
    
    def calibrate(self, score: float) -> float:
        """Calibrate a single heuristic score to confidence."""
        if not self.is_fitted:
            # Fallback: simple linear mapping
            return min(0.95, max(0.05, score / 100.0))
        
        return float(self.calibrator.predict(np.array([score]).reshape(-1, 1).ravel())[0])
    
    def calibrate_batch(self, scores: List[float]) -> List[float]:
        """Calibrate multiple scores."""
        if not self.is_fitted:
            return [min(0.95, max(0.05, s / 100.0)) for s in scores]
        
        return self.calibrator.predict(np.array(scores)).tolist()
    
    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({
                "calibrator": self.calibrator,
                "is_fitted": self.is_fitted,
            }, f)
    
    @classmethod
    def load(cls, path: Path) -> "HeuristicScoreCalibrator":
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        cal = cls()
        cal.calibrator = data["calibrator"]
        cal.is_fitted = data["is_fitted"]
        return cal


def create_calibrator_from_evaluation(
    results_dir: Path,
    method: str = "isotonic",
    output_path: Optional[Path] = None
) -> HeuristicScoreCalibrator:
    """
    Create and fit calibrator from latest evaluation results.
    
    Args:
        results_dir: Directory containing evaluation CSVs
        method: Calibration method
        output_path: Optional path to save calibrator
        
    Returns:
        Fitted HeuristicScoreCalibrator
    """
    import glob
    
    # Find latest evaluation CSV
    csv_files = glob.glob(str(results_dir / "eval_keywords_llm_t*_m*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No evaluation CSV files found in {results_dir}")
    
    # Use the one with most recent modification time
    latest_csv = max(csv_files, key=lambda f: Path(f).stat().st_mtime)
    print(f"Using evaluation results from: {latest_csv}")
    
    calibrator = HeuristicScoreCalibrator()
    calibrator.fit_from_evaluation_results(Path(latest_csv))
    
    if output_path:
        calibrator.save(output_path)
        print(f"Calibrator saved to: {output_path}")
    
    return calibrator