"""Loads config.yaml once and shares it across agents.

The agents in this package are imported as flat siblings (``from planner_agent
import PlannerAgent``), so this module follows the same convention.
"""

from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_FILE = PROJECT_ROOT / "config.yaml"

# Fallbacks so an agent still runs if config.yaml is missing or trimmed down.
_DEFAULTS: Dict[str, Any] = {
    "llm": {"model": "mistral:7b", "judge_model": "mistral:7b"},
    "languages": {
        "default": "en",
        "supported": ["en", "de"],
        "names": {"en": "English", "de": "German"},
        "spacy_models": {"en": "en_core_web_sm", "de": "de_core_news_sm"},
    },
    "embedding": {
        "model": "paraphrase-multilingual-MiniLM-L12-v2",
        "dimension": 384,
        "legacy_model": "all-MiniLM-L6-v2",
    },
    "retrieval": {
        "index_name": "policy-rag-index-multilingual",
        "cloud": "aws",
        "region": "us-east-1",
        "alpha": 0.5,
        "top_k": 5,
    },
    "evaluation": {
        "judge_scale": 5,
        "k_values": [1, 3, 5],
        "pass_threshold": 4,
        "eval_set": "queries/multilingual_eval_set.json",
        "report_json": "results/eval_report.json",
        "report_markdown": "results/eval_report.md",
    },
}

_cache: Dict[str, Any] = {}


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively overlay ``override`` onto ``base`` without mutating either."""
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_config(reload: bool = False) -> Dict[str, Any]:
    """Returns the merged config, reading config.yaml at most once."""
    global _cache
    if _cache and not reload:
        return _cache

    file_config: Dict[str, Any] = {}
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            file_config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"⚠️ {CONFIG_FILE} not found. Using built-in defaults.")
    except yaml.YAMLError as e:
        print(f"⚠️ Could not parse {CONFIG_FILE} ({e}). Using built-in defaults.")

    _cache = _merge(_DEFAULTS, file_config)
    return _cache


def get(path: str, default: Any = None) -> Any:
    """Reads a dotted key, e.g. ``get("retrieval.alpha")``."""
    node: Any = get_config()
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def resolve_path(relative: str) -> Path:
    """Turns a config path like ``results/eval_report.json`` into an absolute one."""
    return PROJECT_ROOT / relative
