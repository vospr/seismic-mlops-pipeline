"""
Utility functions for MLOps pipeline.
"""
from pathlib import Path
import json
from typing import Dict, Any, Optional


def load_json(file_path: str) -> Dict[str, Any]:
    """Load JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], file_path: str):
    """Save data to JSON file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent
