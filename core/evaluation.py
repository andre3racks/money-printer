import pandas as pd
import vectorbt as vbt
from typing import Dict, Any, Tuple, Callable
import traceback
import itertools
from datetime import timedelta

def calculate_fitness(metrics: Dict[str, float]) -> float:
    """Calculate fitness score from backtest metrics."""
    return (metrics.get("Sharpe Ratio", 0.0) * 0.5) + \
           (metrics.get("Total Return [%]", 0.0) * 0.3) - \
           (abs(metrics.get("Max Drawdown [%]", 0.0)) * 0.2)

def validate_metrics(stats: pd.Series) -> bool:
    """Validates that a backtest run produced meaningful results."""
    # Check if there were any trades
    total_trades = stats.get('Total Trades', 0)
    if pd.isna(total_trades) or total_trades < 5:
        return False
        
    # Check for infinite/NaN Sharpe Ratio
    sharpe = stats.get('Sharpe Ratio', 0.0)
    if pd.isna(sharpe) or sharpe == float('inf') or sharpe == -float('inf'):
        return False
        
    # Check for extreme drawdowns or returns that might indicate an error
    mdd = stats.get('Max Drawdown [%]', 0.0)
    if pd.isna(mdd) or mdd < -99.9:
        return False
        
    return True

def run_strategy_and_get_metrics(run_strategy_fn: Callable, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    """Runs a strategy on data with params, constructs portfolio and returns metrics."""
    entries, exits = run_strategy_fn(data, **params)
    portfolio = vbt.Portfolio.from_signals(data['Close'], entries, exits, freq='1h')
    stats = portfolio.stats()
    
    is_valid = validate_metrics(stats)
    
    metrics = {
        "Total Return [%]": float(stats.get('Total Return [%]', 0.0) if not pd.isna(stats.get('Total Return [%]')) else 0.0),
        "Sharpe Ratio": float(stats.get('Sharpe Ratio', 0.0) if not pd.isna(stats.get('Sharpe Ratio')) else 0.0),
        "Max Drawdown [%]": float(stats.get('Max Drawdown [%]', 0.0) if not pd.isna(stats.get('Max Drawdown [%]')) else 0.0),
        "Win Rate [%]": float(stats.get('Win Rate [%]', 0.0) if not pd.isna(stats.get('Win Rate [%]')) else 0.0),
    }
    
    # If not valid, heavily penalize fitness so it's not selected
    if not is_valid:
        metrics["fitness"] = -1000.0
    else:
        metrics["fitness"] = calculate_fitness(metrics)
        
    metrics["stats_summary"] = str(stats)
    metrics["is_valid"] = is_valid
    return metrics

def tune_hyperparameters(run_strategy_fn: Callable, data_is: pd.DataFrame, hyperparameters: Dict[str, list]) -> Tuple[Dict[str, Any], float]:
    """Tunes hyperparameters on in-sample data."""
    best_params = {}
    best_fitness = -float('inf')
    
    if not hyperparameters:
        return best_params, best_fitness
        
    keys = list(hyperparameters.keys())
    values = list(hyperparameters.values())
    combinations = list(itertools.product(*values))
    
    for combo in combinations:
        params = dict(zip(keys, combo))
        try:
            metrics = run_strategy_and_get_metrics(run_strategy_fn, data_is, params)
            fitness = metrics["fitness"]
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_params = params
        except Exception as e:
            print(f"Failed combination {params}: {e}")
            continue
            
    return best_params, best_fitness

def evaluate_strategy_code(code: str, data: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluates a given strategy's raw code over backtest periods.
    Splits data into In-Sample (IS) for parameter tuning and Out-Of-Sample (OOS) for evaluation.
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
        
        # Find the function and hyperparameters we need
        if 'run_strategy' not in local_env:
            return {"success": False, "error": "Code does not define a 'run_strategy' function."}
            
        run_strategy = local_env['run_strategy']
        
        hyperparameters = local_env.get('HYPERPARAMETERS', {})
        
        # If we have less than ~8 months of data, just use a simple percentage or fail.
        # Assuming we have 2 years of data, 6 months is ~180 days.
        total_days = (data.index.max() - data.index.min()).days
        if total_days < 200:
            return {"success": False, "error": f"Insufficient data for IS/OOS split. Need at least 200 days, got {total_days}."}
        
        # Use pandas filtering to be safe, since vectorbt Data.from_data might have issues with column alignment 
        # depending on how the initial download is structured
        split_date = data.index.min() + timedelta(days=180)
        
        data_is = data[data.index <= split_date].copy()
        data_oos = data[data.index > split_date].copy()
        
        best_params, best_fitness = tune_hyperparameters(run_strategy, data_is, hyperparameters)

        # Run OOS with best parameters (or defaults if best_params is empty)
        oos_metrics = run_strategy_and_get_metrics(run_strategy, data_oos, best_params)
        
        metrics = oos_metrics.copy()
        metrics["best_params"] = best_params
        
        stats_summary = metrics.pop("stats_summary")
        
        return {
            "success": True,
            "metrics": metrics,
            "stats_summary": stats_summary
        }
        
    except Exception as e:
        error_msg = f"Error evaluating strategy:\n{traceback.format_exc()}"
        return {"success": False, "error": error_msg}

if __name__ == "__main__":
    print("Testing evaluate_strategy_code...")
