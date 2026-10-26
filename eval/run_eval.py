import argparse
import asyncio
import ast
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
PROMPTS_DIR = REPO_ROOT / "prompts"
DEFAULT_INPUT_FILE = REPO_ROOT / "new_AI_results_for_Jasmine.xlsx"
DEFAULT_SYSTEM_PROMPT = "system_prompt_v2.md"

# Ensure imports work when running this file directly from repo root.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _available_prompt_files() -> list[Path]:
    """List available prompt files in the prompts directory."""
    files = sorted(PROMPTS_DIR.glob("*.md"))
    return [path for path in files if path.is_file()]


def _resolve_prompt_file(prompt_name: str) -> Path:
    """Resolve prompt file path from name."""
    prompt_file = PROMPTS_DIR / prompt_name

    if not prompt_file.exists():
        available = ", ".join(path.name for path in _available_prompt_files())
        raise FileNotFoundError(
            f"Prompt '{prompt_name}' not found in '{PROMPTS_DIR}'. "
            f"Available prompts: {available}"
        )

    return prompt_file


def _get_mlflow_module() -> Any | None:
    """Import Azure MLflow module (azureml-mlflow extends standard mlflow)."""
    try:
        import importlib

        return importlib.import_module("mlflow")
    except ModuleNotFoundError:
        return None


def _load_truthset(truthset_path: Path) -> pd.DataFrame:
    if not truthset_path.exists():
        raise FileNotFoundError(f"Truth set file not found: {truthset_path}")
    file_suffix = truthset_path.suffix.lower()
    if file_suffix == ".csv":
        df = pd.read_csv(truthset_path)
    elif file_suffix == ".tsv":
        df = pd.read_csv(truthset_path, sep="\t")
    elif file_suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(truthset_path)
    else:
        raise ValueError(
            f"Unsupported file format: {file_suffix}. "
            f"Supported formats: .csv, .tsv, .xlsx, .xls"
        )
    required_cols = {"Description", "Category", "Title"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(
            f"Truth set must include columns {sorted(required_cols)}. Missing: {sorted(missing)}"
        )

    # Keep only necessary columns and drop rows with missing values
    df = (
        df[list(required_cols)]
        .dropna(subset=["Description", "Category"])
        .reset_index(drop=True)
    )

    return df


def _parse_contract_description(description_raw: Any) -> str:
    """Safely parse the Description dictionary string."""
    try:
        # Handle cases where it might already be a dict or needs parsing
        if isinstance(description_raw, str):
            description_dict = ast.literal_eval(description_raw)
        else:
            description_dict = description_raw

        return description_dict.get("description", "")
    except (ValueError, SyntaxError, AttributeError) as e:
        print(f"Warning: Could not parse description: {e}")
        return str(description_raw)


async def evaluate_truthset(
    truthset_path: Path,
    output_path: Path | None,
    threshold: int,
    margin: int,
    prompt_name: str,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str | None = None,
    mlflow_run_name: str | None = None,
) -> None:
    """
    Run classification test with MLflow tracking.

    Args:
        truthset_path: Path to input Excel file
        output_path: Path to save output results (CSV)
        threshold: Keyword threshold for classification_v2_mix_v5
        margin: Margin parameter for classification_v2_mix_v5
        prompt_name: System prompt file name for LLM fallback
        mlflow_tracking_uri: Azure MLflow tracking URI
        mlflow_experiment_name: MLflow experiment name
        mlflow_run_name: MLflow run name
    """
    mlflow_module = _get_mlflow_module()
    if mlflow_module is None:
        raise ModuleNotFoundError(
            "Azure MLflow is not installed. Install with 'pip install azureml-mlflow' "
            "and authenticate using 'az login'."
        )

    # Resolve prompt file
    prompt_file = _resolve_prompt_file(prompt_name)

    # Import classification function (using new unified classifier with compat layer)
    from contractmap import keywords_and_llm

    # Load and normalize data
    df = _load_truthset(truthset_path=truthset_path)

    # Set default output path
    if output_path is None:
        output_path = (
            REPO_ROOT / f"data/results/eval_keywords_llm_t{threshold}_m{margin}.csv"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure MLflow
    tracking_uri = mlflow_tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ValueError(
            "MLflow tracking URI is required. "
            "Set --mlflow-tracking-uri or MLFLOW_TRACKING_URI."
        )
    experiment_name = mlflow_experiment_name or os.getenv(
        "MLFLOW_EXPERIMENT_NAME", "ContractMap-Evaluation"
    )
    mlflow_module.set_tracking_uri(tracking_uri)
    mlflow_module.set_experiment(experiment_name)
    run_context = mlflow_module.start_run(run_name=mlflow_run_name)

    with run_context:
        # Log parameters
        mlflow_module.log_param("truthset_path", str(truthset_path.resolve()))
        mlflow_module.log_param("num_samples", len(df))
        mlflow_module.log_param("threshold", threshold)
        mlflow_module.log_param("margin", margin)
        mlflow_module.log_param(
            "classifier", "contractmap.keywords_and_llm (unified)"
        )
        mlflow_module.log_param("prompt_name", prompt_name)
        mlflow_module.log_param("prompt_path", str(prompt_file.resolve()))

        # Log prompt file as artifact
        mlflow_module.log_artifact(str(prompt_file.resolve()), artifact_path="prompts")

        output_labels = []
        heuristic_scores = []
        wrong_results = {}
        start_time = time.perf_counter()

        print(f"Starting analysis on {len(df)} rows...")

        for index, row in df.iterrows():
            print(f"--- Processing Row: {index} ---")

            # Parse contract description
            clean_desc = _parse_contract_description(row["Description"])

            # Build the combined text for the classifier
            # Combine title and description for maximum keyword context
            contract_text = f"{row['Title']} : {clean_desc}"

            # Run classification with system prompt for LLM fallback
            result, reason, heuristic_score = await keywords_and_llm(
                description=contract_text,
                threshold=threshold,
                margin=margin,
                system_prompt_file_location=prompt_file,
            )

            # Track wrong results
            if result != row["Category"]:
                wrong_results[str(index)] = {
                    "description": row["Description"],
                    "AI_label": result,
                    "Actual_label": row["Category"],
                    "reason": reason,
                    "heuristic_score": heuristic_score,
                }

            output_labels.append(result)
            heuristic_scores.append(heuristic_score)

            print(f"AI Prediction: {result}")
            print(f"Actual Label:  {row['Category']}")

        elapsed_seconds = time.perf_counter() - start_time

        # Update the DataFrame with predictions
        df["AI_Prediction"] = output_labels
        df["Heuristic_Score"] = heuristic_scores

        # Accuracy Calculations
        accuracy_series = df["AI_Prediction"] == df["Category"]
        correct_count = accuracy_series.sum()
        total_count = len(df)
        accuracy_pct = (correct_count / total_count) * 100

        # Comprehensive metrics
        macro_f1 = f1_score(df["Category"], df["AI_Prediction"], average="macro", zero_division=0)
        macro_precision = precision_score(df["Category"], df["AI_Prediction"], average="macro", zero_division=0)
        macro_recall = recall_score(df["Category"], df["AI_Prediction"], average="macro", zero_division=0)
        weighted_f1 = f1_score(df["Category"], df["AI_Prediction"], average="weighted", zero_division=0)

        # Per-class metrics
        class_report = classification_report(df["Category"], df["AI_Prediction"], output_dict=True, zero_division=0)
        conf_matrix = confusion_matrix(df["Category"], df["AI_Prediction"])

        # Method distribution
        heuristic_count = sum(1 for r in output_labels if "heuristic" in str(r).lower() or "Heuristic" in str(r))
        llm_count = total_count - heuristic_count

        print("--- Test Results ---")
        print(f"Total Analyzed: {total_count}")
        print(f"Correct Matches: {correct_count}")
        print(f"Accuracy: {accuracy_pct:.2f}%")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Macro Precision: {macro_precision:.4f}")
        print(f"Macro Recall: {macro_recall:.4f}")
        print(f"Weighted F1: {weighted_f1:.4f}")
        print(f"Heuristic decisions: {heuristic_count} ({heuristic_count/total_count*100:.1f}%)")
        print(f"LLM fallbacks: {llm_count} ({llm_count/total_count*100:.1f}%)")
        print(f"Wrong predictions: {len(wrong_results)}")

        # Save results
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

        # Save detailed metrics
        metrics_path = output_path.parent / f"{output_path.stem}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({
                "accuracy_percent": accuracy_pct,
                "macro_f1": macro_f1,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "weighted_f1": weighted_f1,
                "total_samples": total_count,
                "correct_predictions": int(correct_count),
                "wrong_predictions": len(wrong_results),
                "heuristic_decisions": heuristic_count,
                "llm_fallbacks": llm_count,
                "per_class_metrics": class_report,
                "confusion_matrix": conf_matrix.tolist(),
                "evaluation_duration_seconds": elapsed_seconds,
            }, f, indent=2)
        print(f"Detailed metrics saved to {metrics_path}")

        # Log metrics to MLflow
        mlflow_module.log_metric("accuracy_percent", accuracy_pct)
        mlflow_module.log_metric("macro_f1", macro_f1)
        mlflow_module.log_metric("macro_precision", macro_precision)
        mlflow_module.log_metric("macro_recall", macro_recall)
        mlflow_module.log_metric("weighted_f1", weighted_f1)
        mlflow_module.log_metric("correct_predictions", correct_count)
        mlflow_module.log_metric("wrong_predictions", len(wrong_results))
        mlflow_module.log_metric("heuristic_decisions", heuristic_count)
        mlflow_module.log_metric("llm_fallbacks", llm_count)
        mlflow_module.log_metric("evaluation_duration_seconds", elapsed_seconds)

        # Log artifacts
        mlflow_module.log_artifact(str(output_path.resolve()), artifact_path="results")
        mlflow_module.log_artifact(str(metrics_path.resolve()), artifact_path="results")

        # Log wrong results as JSON if any
        if wrong_results:
            wrong_results_path = (
                output_path.parent / f"{output_path.stem}_wrong_results.json"
            )
            with open(wrong_results_path, "w") as f:
                json.dump(wrong_results, f, indent=2)
            mlflow_module.log_artifact(
                str(wrong_results_path.resolve()), artifact_path="results"
            )
            print(f"Wrong results saved to {wrong_results_path}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run contract mapping evaluation with keywords_and_llm classifier and MLflow tracking."
    )
    parser.add_argument(
        "--truthset",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="Path to truthset Excel file with columns: Title, Description, Category.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path. Defaults to data/results/eval_keywords_llm_t{threshold}_m{margin}.csv.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=10,
        help="Keyword threshold parameter for classification (default: 10).",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=0,
        help="Margin parameter for classification (default: 0).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help=f"System prompt file name from prompts/ for LLM fallback (default: {DEFAULT_SYSTEM_PROMPT}).",
    )
    parser.add_argument(
        "--list-prompts",
        action="store_true",
        help="List available prompt files in prompts/ and exit.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help="Azure MLflow tracking URI (azureml:// scheme). Defaults to MLFLOW_TRACKING_URI env var.",
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        type=str,
        default=None,
        help="Optional Azure MLflow experiment name. Defaults to MLFLOW_EXPERIMENT_NAME or ContractMap-KeywordLLM-Evaluation.",
    )
    parser.add_argument(
        "--mlflow-run-name",
        type=str,
        default=None,
        help="Optional Azure MLflow run name.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.list_prompts:
        print("Available prompts:")
        for prompt in _available_prompt_files():
            print(f"- {prompt.name}")
        return

    asyncio.run(
        evaluate_truthset(
            truthset_path=args.truthset,
            output_path=args.output,
            threshold=args.threshold,
            margin=args.margin,
            prompt_name=args.prompt,
            mlflow_tracking_uri=args.mlflow_tracking_uri,
            mlflow_experiment_name=args.mlflow_experiment_name,
            mlflow_run_name=args.mlflow_run_name,
        )
    )


if __name__ == "__main__":
    main()
