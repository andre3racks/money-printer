import os
import re
from pathlib import Path

from core.llm import get_llm_provider

ALGO_DIR = Path("algorithms")
ALGO_DIR.mkdir(parents=True, exist_ok=True)

# System prompt for algorithm generation
SYSTEM_PROMPT = """You are an expert quantitative developer and algorithmic trader.
Your task is to write a Python trading strategy using the `vectorbt` library.
You MUST output valid, executable Python code. 

The strategy should be encapsulated in a function named `run_strategy` with the following signature:
`def run_strategy(data: pd.DataFrame, **kwargs) -> tuple[pd.Series, pd.Series]:`

- The `data` parameter will be a pandas DataFrame containing OHLCV data (from Yahoo Finance).
- You MUST explicitly define the hyperparameters to be tuned as keyword arguments with default values in the `run_strategy` function signature. For example: `def run_strategy(data: pd.DataFrame, fast_window: int = 10, slow_window: int = 50) -> tuple[pd.Series, pd.Series]:`
- You MUST define a global dictionary named `HYPERPARAMETERS` outside of the `run_strategy` function that provides a list of options for each hyperparameter to be used for tuning combinations. For example: `HYPERPARAMETERS = {"fast_window": [10, 15, 20], "slow_window": [50, 100, 200]}`
- The keys in `HYPERPARAMETERS` must exactly match the keyword arguments in `run_strategy`.
- The function MUST return a tuple of two pandas Series: `(entries, exits)`.
- `entries` is a boolean Series indicating where to enter a long position (True for enter).
- `exits` is a boolean Series indicating where to exit a long position (True for exit).
- You should use `vectorbt`'s built-in indicators (e.g., `vbt.MA`, `vbt.RSI`, `vbt.MACD`, `vbt.BBANDS`) to generate signals.

VectorBT Documentation: https://vectorbt.dev/api/

Do not include any explanations or markdown formatting outside of the Python code block. Only provide the raw Python code.
"""

def extract_python_code(text: str) -> str:
    """Extracts Python code from a markdown-formatted string."""
    pattern = r"```python\n(.*?)\n```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def generate_algorithm(ancestor_code: str = None, ancestor_performance: str = None, strategy_name: str = None) -> str:
    """
    Prompts the LLM to generate a new trading algorithm.
    If ancestor_code and ancestor_performance are provided, it asks the LLM to evolve the strategy.
    If strategy_name is provided, it reads the strategy description from strategies/{strategy_name}.md and incorporates it into the prompt.
    Returns the generated Python code as a string.
    """
    provider = get_llm_provider()
    
    prompt = "Write a new trading strategy using vectorbt.\n"
    
    if strategy_name:
        strategy_file = Path(f"strategies/{strategy_name}.md")
        if strategy_file.exists():
            with open(strategy_file, "r") as f:
                strategy_description = f.read()
            prompt += f"\nHere are the specific instructions for the trading strategy:\n{strategy_description}\n"
        
    if ancestor_code and ancestor_performance:
        prompt = f"""
Evolve the following ancestor trading strategy to improve its performance.

Ancestor Performance:
{ancestor_performance}

Ancestor Code:
```python
{ancestor_code}
```

Write an improved version of this strategy using vectorbt.
"""
    
    response_text = provider.generate_code(prompt, SYSTEM_PROMPT)
    
    code = extract_python_code(response_text)
    return code

def save_algorithm(code: str, algorithm_id: str, strategy_name: str = None) -> str:
    """Saves the generated algorithm code to a file."""
    if strategy_name:
        algo_dir = Path("algorithms") / strategy_name
    else:
        algo_dir = Path("algorithms")
        
    algo_dir.mkdir(parents=True, exist_ok=True)
    file_path = algo_dir / f"strategy_{algorithm_id}.py"
    
    # Ensure necessary imports are present
    imports = "import pandas as pd\nimport vectorbt as vbt\nimport numpy as np\n\n"
    if "import pandas" not in code:
        code = imports + code
        
    with open(file_path, "w") as f:
        f.write(code)
    
    return str(file_path)

if __name__ == "__main__":
    # Test generation (requires GEMINI_API_KEY env var)
    try:
        code = generate_algorithm()
        print("Generated Code:")
        print(code)
    except Exception as e:
        print(f"Error generating algorithm: {e}")
