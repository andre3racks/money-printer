# Project Plan: Trading Algorithm Evolution Engine

## Overview
This project builds a continuous evolution engine for Python-based trading algorithms. It leverages **vectorBT** for data acquisition and backtest evaluation, and an **LLM** to generate and iteratively improve trading logic and hyperparameters. 

The architecture is built around a continuous loop: **Fetch Data -> Generate/Evolve Strategy (LLM) -> Evaluate (vectorBT) -> Record (JSON Leaderboard) -> Select Fittest Ancestor -> Repeat**.

---

## 1. Environment & Setup
* Initialize a Python environment with `uv`.
* Install dependencies: `vectorbt`, `yfinance`, `pandas`, `requests` (for LLM API/URL scraping), and any required LLM SDKs (start with google gemini).
* Set up directory structure:
  * `data/`: Local cache for historical data.
  * `algorithms/`: Directory to store generated Python snippet files.
  * `core/`: Application logic (data, evaluation, LLM client, loop engine).

## 2. Data Ingestion Module (`data_ingest.py`)
* Abstracted data fetcher capable of extension.
* Use `vectorbt`'s built-in Yahoo Finance wrapper (`yf`) to pull 2 years of data at 1-hour resolution for target ticker(s).
* For a working demo, use BTC. But make the implementation abstract so any ticker, or bucket of tickers, can be used.
* **Caching Mechanism**: Save data to disk (e.g., Parquet for fast I/O).
* Check cache freshness: only re-fetch if the data is older than 1 week.

## 3. Algorithm Evolution Module (`algorithm_evolution.py`)
* **LLM Integration**: Build an interface to prompt the chosen LLM (start with google, but build it so models can be swapped).
* **Context Preparation**:
  * Create a system prompt to specify the exact task.
  * Pass the VectorBT documentation URL so the LLM has grounding.
  * If evolving, include the Python code of the "ancestor" algorithm.
  * Include a summary of the ancestor's performance.
* **Prompt Instructions**: Request the LLM to write a Python trading strategy using VectorBT and explicitly define the hyperparameters to be tuned. The LLM must output valid Python code.
* Parse the LLM's response to extract and save the Python logic to the `algorithms/` directory.

## 4. Evaluation Engine (`evaluation.py`)
* Load the LLM-generated Python strategy dynamically.
* **Data Splitting**:
  * Extract a 6-month in-sample window from the 2-year dataset for hyperparameter tuning. Use the top 5 combinations of hyperparameters.
  * Define 3 out-of-sample (eval) windows of 6 months each.
* **Backtesting**: Execute the vectorBT backtest using the defined hyperparameters over these windows.
* Calculate and return key metrics (e.g., Sharpe Ratio, Total Return, Max Drawdown) to assess the strategy's fitness.
* Run it through a fitness function of your recommendation to output a single fitness number to power algorithm precedence.
* Cull algorithms that performed poorly (make this configurable).

## 5. Record & Leaderboard System (`record.py`)
* Store the generated Python algorithm to disk (e.g., `algorithms/strategy_{id}.py`).
* **Leaderboard Data Structure**: Maintain an in-memory JSON list of objects representing evaluated strategies.
  * Example Object: `{ "id": "uuid", "file_path": "algorithms/...", "parent_id": "...", "metrics": {"sharpe": 1.5, ...} }`
* **Sorting & Persistence**: Sort the JSON list in memory based on the primary fitness metric (e.g., Sharpe Ratio) and write the updated list to `leaderboard.json` on disk after each iteration.

## 6. The Evolutionary Loop Engine (`main.py`)
* Construct the main loop that continuously drives the evolution:
  1. Initialize or load data (`data_ingest`).
  2. Load the leaderboard (`leaderboard.json`).
  3. Select the "best" algorithm as the ancestor (or start from scratch if empty).
  4. Generate a new algorithm via LLM (`algorithm_evolution`).
  5. Test the algorithm (`evaluation`).
  6. Update and save the leaderboard (`record`).
  7. Loop back to step 3.
