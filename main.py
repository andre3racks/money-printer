import argparse
import time
import uuid
import json
import logging
from core.data_ingest import fetch_data
from core.algorithm_evolution import generate_algorithm, save_algorithm
from core.evaluation import evaluate_strategy_code
from core.record import load_leaderboard, save_leaderboard, update_leaderboard, get_best_ancestor
from google.genai.errors import APIError

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_evolution_loop(ticker: str, interval: str, iterations: int):
    logging.info(f"Starting Evolutionary Loop Engine for {ticker} at {interval} interval")
    
    # 1. Initialize/Load Data
    logging.info("Fetching data...")
    data = fetch_data(ticker, interval=interval)
    
    if data is None or data.empty:
        logging.error("Failed to fetch data. Exiting.")
        return
        
    logging.info(f"Data shape: {data.shape}")
    
    # 2. Load Leaderboard
    leaderboard = load_leaderboard()
    logging.info(f"Loaded leaderboard with {len(leaderboard)} entries.")
    
    for i in range(iterations):
        logging.info(f"--- Iteration {i+1}/{iterations} ---")
        
        # 3. Select best ancestor
        best_ancestor = get_best_ancestor(leaderboard)
        
        ancestor_code = None
        ancestor_performance = None
        parent_id = None
        
        if best_ancestor:
            parent_id = best_ancestor.get("id")
            file_path = best_ancestor.get("file_path")
            metrics = best_ancestor.get("metrics")
            
            logging.info(f"Selected ancestor ID: {parent_id} with fitness: {metrics.get('fitness', 0):.4f}")
            
            try:
                with open(file_path, "r") as f:
                    ancestor_code = f.read()
                ancestor_performance = json.dumps(metrics, indent=2)
            except FileNotFoundError:
                logging.warning(f"Ancestor file {file_path} not found. Starting from scratch.")
        else:
            logging.info("No ancestors found in leaderboard. Starting from scratch.")
            
        # 4. Generate new algorithm
        logging.info("Generating new algorithm via LLM...")
        try:
            new_code = generate_algorithm(ancestor_code, ancestor_performance)
        except APIError as e:
            logging.error(f"Gemini API Error: {e}")
            logging.info("Sleeping for 10 seconds before continuing...")
            time.sleep(10)
            continue
            
        new_id = str(uuid.uuid4())
        
        # 5. Evaluate
        logging.info(f"Evaluating new algorithm {new_id}...")
        eval_result = evaluate_strategy_code(new_code, data)
        
        if not eval_result.get("success"):
            logging.warning(f"Evaluation failed: {eval_result.get('error')}")
            continue
            
        metrics = eval_result.get("metrics")
        logging.info(f"Evaluation success! Fitness: {metrics.get('fitness', 0):.4f}, Sharpe: {metrics.get('Sharpe Ratio', 0):.4f}")
        
        # 6. Save code and update leaderboard
        file_path = save_algorithm(new_code, new_id)
        
        new_entry = {
            "id": new_id,
            "file_path": str(file_path),
            "parent_id": parent_id,
            "metrics": metrics,
            "timestamp": time.time(),
            "ticker": ticker,
            "interval": interval
        }
        
        leaderboard = update_leaderboard(leaderboard, new_entry)
        save_leaderboard(leaderboard)
        
        logging.info(f"Saved algorithm and updated leaderboard. Current top fitness: {leaderboard[0]['metrics']['fitness']:.4f}")
        
        # Be nice to APIs
        time.sleep(2)
        
    logging.info("Evolution loop completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trading Algorithm Evolution Engine")
    parser.add_argument("--ticker", type=str, default="BTC-USD", help="Ticker symbol to fetch data for")
    parser.add_argument("--interval", type=str, default="1h", help="Data interval (e.g., 1d, 1h)")
    parser.add_argument("--iterations", type=int, default=5, help="Number of evolution iterations to run")
    
    args = parser.parse_args()
    
    run_evolution_loop(args.ticker, args.interval, args.iterations)
