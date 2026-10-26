# ccs-contract-map 


This repository provides a tool to automatically label contract descriptions using the CCS categories. It leverages a Large Language Model (LLM) to classify contract descriptions into predefined categories accurately and consistently.

## Project Structure

```
ccs-contract-map/
├── src/                           # Source code
│   ├── core/                      # Core classification modules
│   │   ├── classification_v1.py   # Version 1 (SystemMessage/HumanMessage)
│   │   └── classification_v2.py   # Version 2 (string concatenation)
│   └── api/                       # FastAPI endpoints
│       ├── v1_endpoint.py         # API for version 1
│       └── v2_endpoint.py         # API for version 2
├── prompts/                       # System prompts
│   ├── system_prompts.py          # Python prompt definitions
│   ├── new_system_prompt.txt      # Text-based prompt
│   └── contractmap_prompt_with_descriptions.txt
├── evaluation/                    # Evaluation scripts
│   ├── run_evaluation.py          # Unified evaluation CLI (v1 or v2)
│   └── prompt_engineering_experiment.ipynb
├── utils/                         # Utility modules
│   └── file_io/                  # File I/O utilities
│       └── file_to_string.py
├── data/                          # Data files
│   ├── input/                    # Input datasets
│   └── results/                  # Evaluation results
└── tests/                        # Unit tests
```

## How It Works

Contractmap is a hybrid keyword + LLM system. When a contract description is passed in, the following happens:
1. Keyword Matching: the description is searched for the occurrence of keywords that are relevant to each category. If enough keywords are found for a given category (as defined by the threshold value), and there is not a similar number of keywords for another category (with "similar" defined by the margin value), that category is chosen as the match for the description.
2. LLM (optional): if there is no single category that achieves a high enough keyword matching score, the description is passed to an LLM, which uses its understanding of natural language to assign the contract to a GCA category.
Once this process finishes, the system returns a single CCS category name.

## How To Install Locally

### Python

The system and its unit tests are all written in python. To install it, first ensure that Python is installed on your system, and then follow these steps:

1. Run the starter script:
```
python start.py --setup
```

### TypeScript

Functional tests are written in TypeScript and run through Playwright. To install the TypeScript environment, first ensure that node.js is installed on your system, and then run:
```
npm install
```

## Developer Tooling (Pre-commit, Ruff, pytest, playwright)

This project uses:

- [pre-commit](https://pre-commit.com/) for running checks automatically before each commit.
- [Ruff](https://docs.astral.sh/ruff/) for fast linting.
- [pytest](https://docs.pytest.org/) for unit testing.
- [playwright](https://playwright.dev/) for functional testing.

### Set up pre-commit hooks

Install hooks locally:

```bash
pre-commit install
```

Run all hooks manually across the repository:

```bash
pre-commit run --all-files
```

### Run Ruff and tests manually

Run Ruff:

```bash
ruff check .
```

Run unit tests:

```bash
pytest -q
```

Run functional tests:

```bash
npx playwright test
```

## Evaluation

This section describes how to evaluate the contract mapping model using ground truth data and MLflow tracking.

### Prerequisites

1. Install the required dependencies:
```bash
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

2. Set up environment variables in `.env`:
```bash
# Azure OpenAI (required for keyword extraction and LLM classification)
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_KEY=<your-api-key>
DEPLOYMENT_NAME=<your-deployment-name>
AZURE_OPENAI_API_VERSION=2024-02-01

# Azure MLflow (required for evaluation tracking)
MLFLOW_TRACKING_URI=azureml://...
MLFLOW_EXPERIMENT_NAME=ContractMap-Evaluation
```

3. Authenticate with Azure:
```bash
az login
```

### Step 1: Process the Truthset

Before running evaluation, prepare your ground truth data using the `process_truthset.py` script. This script:
- Cleans and normalizes category names (e.g., converts '&' to 'and', standardizes plural forms)
- Removes rows with missing required data
- Splits data into train and test sets with stratified sampling

**Usage:**

```bash
python src/utils/process_truthset.py \
  --input data/input/your_truthset.xlsx \
  --train-ratio 0.8
```

**Parameters:**
- `--input`: Path to input Excel file with columns: `Title`, `Description`, `Category` (required)
- `--train-ratio`: Proportion of data for training set, between 0.0 and 1.0 (required)

**Example:**

```bash
# Split into 80% train, 20% test
python src/utils/process_truthset.py \
  --input data/input/your_truthset.xlsx \
  --train-ratio 0.8
```

This will:
1. Clean and normalize the input file (overwrites original)
2. Create two TSV files:
   - `your_truthset_train.tsv` (80% of data)
   - `your_truthset_test.tsv` (20% of data)

### Step 2: Run Evaluation with One Configuration

Use the `run_eval.py` script to evaluate the model on your test set. This script:
- Runs the hybrid keyword + LLM classifier on each contract description
- Calculates accuracy metrics
- Logs results to Azure MLflow
- Saves predictions and wrong results to CSV/JSON files

**Usage:**

```bash
python eval/run_eval.py \
  --truthset data/input/your_test_set.tsv \
  --threshold 10 \
  --margin 0 \
  --prompt system_prompt_v2.md \
  --mlflow-tracking-uri "azureml://..." \
  --mlflow-experiment-name "ContractMap-Evaluation" \
  --mlflow-run-name "test-run-001"
```

**Parameters:**
- `--truthset`: Path to truthset file (CSV, TSV, or Excel) with `Title`, `Description`, `Category` columns
- `--output`: Optional output CSV path (defaults to `data/results/eval_keywords_llm_t{threshold}_m{margin}.csv`)
- `--threshold`: Keyword threshold for classification (default: 10)
- `--margin`: Margin parameter for classification (default: 0)
- `--prompt`: System prompt file name from `prompts/` directory (default: `system_prompt_v2.md`)
- `--list-prompts`: List available prompt files and exit
- `--mlflow-tracking-uri`: Azure MLflow tracking URI (or set `MLFLOW_TRACKING_URI` env var)
- `--mlflow-experiment-name`: MLflow experiment name (optional)
- `--mlflow-run-name`: MLflow run name (optional)

**Example:**

```bash
# List available prompts
python eval/run_eval.py --list-prompts

# Run evaluation with default parameters
python eval/run_eval.py \
  --truthset data/input/your_truthset_test.tsv \
  --threshold 10 \
  --margin 0 \
  --prompt system_prompt_v2.md

# Run evaluation with custom parameters and MLflow tracking
python eval/run_eval.py \
  --truthset data/input/your_truthset_test.tsv \
  --output data/results/my_evaluation.csv \
  --threshold 5 \
  --margin 1 \
  --prompt keyword_system_prompt_v2.md \
  --mlflow-experiment-name "ContractMap-PromptOptimization" \
  --mlflow-run-name "threshold-5-margin-1"
```

Files created:
- `data/results/eval_keywords_llm_t10_m0.csv`: Predictions for all samples
- `data/results/eval_keywords_llm_t10_m0_wrong_results.json`: Detailed information about incorrect predictions

**MLflow Tracking:**

The evaluation script logs the following to MLflow:
- **Parameters**: truthset path, threshold, margin, prompt name, classifier version
- **Metrics**: accuracy percentage, correct/wrong prediction counts, evaluation duration
- **Artifacts**: results CSV, wrong results JSON, system prompt file

View results in Azure ML Studio or query programmatically.

### Step 3: Run Evaluation across Multiple Configurations

To find the optimal configuration, use the `eval/optimise_parameters.sh` file, which runs a grid search across multiple combinations of the threshold and margin parameters. You can run the file directly through the terminal:

```bash
bash eval/optimise_parameters.sh
```

### Step 4: Visualize Results

After running multiple evaluations, use the `visualise_experiments.py` script to visualize the optimization results. This script fetches all runs from an MLflow experiment and generates plots showing how accuracy varies with threshold and margin parameters.

**Usage:**

```bash
python eval/visualise_experiments.py \
  --mlflow-tracking-uri "azureml://..." \
  --mlflow-experiment-name "ContractMap-KeywordLLM-Evaluation" \
  --plot-type surface
```

**Parameters:**
- `--mlflow-tracking-uri`: Azure MLflow tracking URI (or set `MLFLOW_TRACKING_URI` env var)
- `--mlflow-experiment-name`: MLflow experiment name (default: `ContractMap-KeywordLLM-Evaluation`)
- `--plot-type`: Plot type - `surface` for 3D surface plot or `scatterplot` for 2D scatter plot (default: `surface`)
- `--output`: Optional output path to save the plot (e.g., `optimization_surface.png`)
- `--no-show`: Don't display the plot interactively (useful for headless environments)

**Examples:**

```bash
# Generate 3D surface plot (default)
python eval/visualise_experiments.py

# Generate 2D scatter plot
python eval/visualise_experiments.py --plot-type scatterplot

# Save plot to file without displaying it
python eval/visualise_experiments.py \
  --plot-type surface \
  --output data/results/optimization_surface.png \
  --no-show

# Visualize results from a specific experiment
python eval/visualise_experiments.py \
  --mlflow-experiment-name "ContractMap-PromptOptimization" \
  --plot-type scatterplot \
  --output data/results/prompt_optimization.png
```

## DVC Pipeline

This project uses [DVC (Data Version Control)](https://dvc.org/) to orchestrate a reproducible machine learning pipeline for training and evaluating the contract classification model. The pipeline automates data processing, keyword extraction, hyperparameter optimization, and results visualization.

### Pipeline Overview

The pipeline consists of four stages:

1. **process_data**: Cleans and normalizes the truthset data, then splits it into train and test sets
2. **extract_keywords**: Extracts semantic anchors (keywords) from the training data using LLM
3. **optimize_params**: Runs a grid search across threshold and margin parameters, logging results to MLflow
4. **visualize**: Generates visualization plots of the optimization results

### Pipeline Architecture

```
new_AI_results_for_Jasmine.xlsx
  ├── [1] process_data
  │   ├── Output: new_AI_results_for_Jasmine_train.tsv
  │   └── Output: new_AI_results_for_Jasmine_test.tsv
  │
  ├── [2] extract_keywords (depends on train.tsv)
  │   └── Output: semantic_anchors2.json
  │
  └── [3] optimize_params (depends on semantic_anchors2.json, train.tsv)
      ├── Outputs: data/results/eval_keywords_llm_t*_m*.csv
      ├── MLflow: Logs 21 runs with threshold, margin, accuracy
      └── Output: data/results/optimization_complete.txt
          │
          └── [4] visualize (depends on completion marker)
              └── Output: data/results/optimization_surface.png
```

### Prerequisites

Ensure you have:
1. DVC installed (included in `requirements.txt`)
2. Environment variables configured in `.env` (see [Prerequisites](#prerequisites) above — Azure OpenAI is required for keyword extraction; Azure MLflow is required for evaluation tracking)
3. Authenticated with Azure: `az login`

### Running the Full Pipeline

To execute the entire pipeline from start to finish:

```bash
# Run all stages
dvc repro

# View the pipeline dependency graph
dvc dag
```

The pipeline will:
1. Process the input Excel file and create train/test splits
2. Extract semantic keywords from training data
3. Run 21 evaluations (7 thresholds × 3 margins) and log to MLflow
4. Generate a 3D optimization surface plot

**Expected runtime**: 1-2 hours (depends on LLM API response times)

### Running Individual Stages

You can run specific stages independently:

```bash
# Run only the data processing stage
dvc repro process_data

# Run up to and including the keyword extraction stage
dvc repro extract_keywords

# Run only the visualization stage (requires optimization results)
dvc repro visualize
```

### Configuration

All pipeline parameters are centralized in `params.yaml`:

```yaml
# Data Processing
data:
  input_excel: "data/input/evaluation_data.xlsx"
  train_tsv: "data/input/evaluation_data_train.tsv"
  test_tsv: "data/input/evaluation_data_test.tsv"
  train_split_ratio: 0.8
  semantic_anchors_json: "semantic_anchors2.json"
  RMnumber_framework_mapping: "data/input/GCA_RMnumber_framework_mapping.csv"  # CSV mapping RM/framework numbers to categories

# Grid Search Parameters
gridsearch:
  thresholds: [1, 2, 3, 5, 10, 25, 50]
  margins: [1, 5, 10]

# MLflow Configuration
mlflow:
  experiment_name: "ContractMap-Evaluation"

# Evaluation Settings
evaluation:
  prompt_name: "system_prompt_v2.md"

# Output Paths
outputs:
  results_dir: "data/results"
  completion_marker: "data/results/optimization_complete.txt"
  plot_output: "data/results/optimization_surface.png"
```

**To modify parameters:**

1. Edit `params.yaml` with your desired values
2. Run `dvc repro` to re-execute affected stages

DVC automatically detects which stages need to re-run based on changed parameters or dependencies.

### Pipeline Outputs

After running the full pipeline, you'll have:

- **Train/Test Data**: 
  - `new_AI_results_for_Jasmine_train.tsv` (80% of data)
  - `new_AI_results_for_Jasmine_test.tsv` (20% of data)

- **Semantic Anchors**: 
  - `semantic_anchors2.json` (keyword registry by category)

- **Evaluation Results**: 
  - `data/results/eval_keywords_llm_t{threshold}_m{margin}.csv` (21 files)
  - `data/results/eval_keywords_llm_t{threshold}_m{margin}_wrong_results.json` (21 files)

- **MLflow Runs**: 
  - 21 runs logged to Azure ML with parameters, metrics, and artifacts

- **Visualization**: 
  - `data/results/optimization_surface.png` (3D plot of optimization results)

### Checking Pipeline Status

```bash
# Show pipeline status (which stages need to run)
dvc status

# Show detailed pipeline information
dvc status --show-json

# Visualize the pipeline dependency graph (ASCII art)
dvc dag
```

### Re-running the Pipeline

DVC uses content-based caching to avoid re-running stages unnecessarily:

```bash
# Force re-run all stages (ignores cache)
dvc repro --force

# Re-run from a specific stage onward
dvc repro --downstream optimize_params

# Re-run a single stage only
dvc repro --single-item visualize
```

### Reproducing Results

To reproduce results from a specific commit:

```bash
# Checkout a specific commit
git checkout <commit-hash>

# Reproduce the pipeline at that point in history
dvc repro

# Compare results with current version
dvc metrics diff
```

### Manual Execution (Alternative)

If you prefer to run stages manually without DVC:

```bash
# 1. Process data
python src/utils/process_truthset.py \
  --input data/input/evaluation_data.xlsx \
  --train-ratio 0.8

# 2. Extract keywords
python src/utils/semantic_anchors_tool.py \
  --input-train-tsv data/input/evaluation_data_train.tsv \
  --output-json semantic_anchors2.json \
  --rm-mapping-csv data/input/GCA_RMnumber_framework_mapping.csv \
  --max-categories 3  # optional: max categories a word can appear in before being filtered (default: 3)

# 3. Run grid search
python eval/run_gridsearch.py \
  --params-file params.yaml \
  --truthset data/input/evaluation_data.xlsx

# 4. Visualize results
python eval/visualise_experiments.py \
  --mlflow-experiment-name ContractMap-Evaluation \
  --plot-type surface \
  --output data/results/optimization_surface.png \
  --no-show
```

### Troubleshooting

**Issue: `dvc: command not found`**
```bash
# Use the venv path directly
/path/to/venv/bin/dvc repro
```

**Issue: MLflow authentication errors**
```bash
# Re-authenticate with Azure
az login

# Verify your credentials
az account show
```

**Issue: Pipeline stages not re-running when expected**
```bash
# Check what changed
dvc status

# Force re-run
dvc repro --force
```

**Issue: Out of memory during keyword extraction**
```bash
# Process categories in smaller batches by modifying semantic_anchors_tool.py
# or run the stage with increased memory limits
```
