# goal
A trading algorithm evolution engine in python.

# dependencies
* **vectorBT**: for evaluation, reporting, and data acquisition.
* **LLM**: for trading strategy evolution and generating Python logic.

# data ingest
2 years of data @ 1 hr resolution, via vectorBT's built-in yf (Yahoo Finance) wrapper. 
* Save to disk (e.g., Parquet or CSV) for local caching.
* Only refresh if the local cache is > 1 week old. 
* Make the approach abstracted (Strategy/Factory pattern) so we can easily swap or add new data sources later without breaking core logic.

# algorithm evolution
* Invoke LLM to generate trading logic (Python code).
* **Context building**: Use a summary of the ancestor algorithm's performance and logic, if available, to guide improvements.
* **Grounding**: Ground the LLM by providing vectorBT documentation (via URL, though we may need to scrape/summarize key VectorBT API concepts if the LLM cannot browse URLs effectively, or use an LLM with built-in browsing capabilities).
* **Output Requirements**: Ensure the LLM explicitly defines the hyperparameters that need tuning as part of its generated strategy code.

# evaluation
* **In-sample Tuning**: 6 months of the 2-year data are kept in scope for hyperparameter tuning.
* **Out-of-sample Testing**: 3 distinct, rolling, or sequential windows of 6 months for evaluation to ensure robustness.
* Perform all evaluation using vectorBT's backtesting engine.
* Capture key metrics (e.g., Total Return, Sharpe Ratio, Max Drawdown).

# record
* Record a snapshot of the generated algorithm (the Python snippet) to disk (e.g., inside an `algorithms/` directory).
* Reference that algo Python snippet via an entry in a leaderboard-like data structure.
* **Leaderboard Structure**: An in-memory JSON list of objects containing strategy metadata, paths to the Python snippets, and evaluation metrics. This JSON list can easily be sorted in memory and serialized/saved to a `.json` file on disk for persistence.

# loop
* After evaluation, consult the ranking data structure (the JSON leaderboard) to select the "fittest" or best-performing algorithm.
* Use this top algorithm as the ancestor to evolve the next iteration, forming a continuous improvement loop.
