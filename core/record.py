import json
from pathlib import Path
from typing import Dict, Any, List
import os

def get_leaderboard_file(strategy_name: str = None) -> Path:
    """Returns the correct leaderboard file path based on the strategy name."""
    leaderboard_dir = Path("leaderboards")
    
    if strategy_name:
        return leaderboard_dir / f"leaderboard_{strategy_name}.json"
    return leaderboard_dir / "leaderboard.json"

def load_leaderboard(strategy_name: str = None) -> List[Dict[str, Any]]:
    """Loads the leaderboard from disk. Fails hard if corrupted."""
    file_path = get_leaderboard_file(strategy_name)
    if not file_path.exists():
        return []
    
    with open(file_path, "r") as f:
        return json.load(f)

def save_leaderboard(leaderboard: List[Dict[str, Any]], strategy_name: str = None):
    """Saves the leaderboard to disk."""
    file_path = get_leaderboard_file(strategy_name)
    with open(file_path, "w") as f:
        json.dump(leaderboard, f, indent=4)

def update_leaderboard(leaderboard: List[Dict[str, Any]], new_entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Updates the leaderboard with a new entry and sorts it by fitness.
    No culling implemented for now.
    """
    leaderboard.append(new_entry)
    
    # Sort by fitness (descending)
    leaderboard.sort(key=lambda x: x.get('metrics', {}).get('fitness', -float('inf')), reverse=True)
            
    return leaderboard

def get_best_ancestor(leaderboard: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Returns the top performing algorithm from the leaderboard."""
    if not leaderboard:
        return None
    return leaderboard[0]

if __name__ == "__main__":
    # Test record
    lb = load_leaderboard()
    print(f"Loaded {len(lb)} entries.")
