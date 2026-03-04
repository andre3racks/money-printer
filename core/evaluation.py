import pandas as pd
import vectorbt as vbt
from typing import Dict, Any
import traceback

def evaluate_strategy_code(code: str, data: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluates a given strategy's raw code over backtest periods.
    Instead of writing to disk and loading as a module, we execute the code directly.
    """
    try:
        # Create a local dictionary to store the execution results
        local_env = {}
        
        # Add required libraries to the local environment so the code can run
        local_env['pd'] = pd
        local_env['vbt'] = vbt
        
        # Execute the generated Python code
        exec(code, globals(), local_env)
        
        # Find the function we need
        if 'run_strategy' not in local_env:
            return {"success": False, "error": "Code does not define a 'run_strategy' function."}
            
        run_strategy = local_env['run_strategy']
        
        # 1. Run Strategy with default parameters
        portfolio = run_strategy(data)
        
        if not isinstance(portfolio, vbt.Portfolio):
            return {"success": False, "error": f"run_strategy returned {type(portfolio)}, expected vbt.Portfolio."}
            
        # 2. Extract Metrics
        stats = portfolio.stats()
        
        # Extract key metrics
        # Use .get() and default to 0.0 in case the stat isn't available
        metrics = {
            "Total Return [%]": float(stats.get('Total Return [%]', 0.0) if not pd.isna(stats.get('Total Return [%]')) else 0.0),
            "Sharpe Ratio": float(stats.get('Sharpe Ratio', 0.0) if not pd.isna(stats.get('Sharpe Ratio')) else 0.0),
            "Max Drawdown [%]": float(stats.get('Max Drawdown [%]', 0.0) if not pd.isna(stats.get('Max Drawdown [%]')) else 0.0),
            "Win Rate [%]": float(stats.get('Win Rate [%]', 0.0) if not pd.isna(stats.get('Win Rate [%]')) else 0.0),
        }
        
        # 3. Calculate Fitness
        # Using a combination of Total Return, Sharpe, and Drawdown
        # Penalize negative Sharpe and Drawdown
        fitness = (metrics["Sharpe Ratio"] * 0.5) + (metrics["Total Return [%]"] * 0.3) - (abs(metrics["Max Drawdown [%]"]) * 0.2)
        
        metrics["fitness"] = fitness
        
        return {
            "success": True,
            "metrics": metrics,
            "stats_summary": str(stats)
        }
        
    except Exception as e:
        error_msg = f"Error evaluating strategy:\n{traceback.format_exc()}"
        return {"success": False, "error": error_msg}

if __name__ == "__main__":
    print("Testing evaluate_strategy_code...")
