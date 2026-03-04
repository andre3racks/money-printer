import json
from pathlib import Path
from typing import Dict, Any, List
import os

LEADERBOARD_FILE = Path("leaderboard.json")

def load_leaderboard() -> List[Dict[str, Any]]:
    """Loads the leaderboard from disk. Fails hard if corrupted."""
    if not LEADERBOARD_FILE.exists():
        return []
    
    with open(LEADERBOARD_FILE, "r") as f:
        return json.load(f)

def save_leaderboard(leaderboard: List[Dict[str, Any]]):
    """Saves the leaderboard to disk."""
    with open(LEADERBOARD_FILE, "w") as f:
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
