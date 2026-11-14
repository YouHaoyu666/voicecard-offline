"""Eval suite switch via EVAL_SUITE: unset = original Chinese/Singapore set (data/eval -> results/); en = English multi-accent set (data/eval_en -> results_en/); pub = public standard test sets (data/pub -> results_pub/, prep_pub.py)."""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
SUITE = os.environ.get("EVAL_SUITE", "")
EVAL = os.path.join(ROOT, "data", {"en": "eval_en", "pub": "pub"}.get(SUITE, "eval"))
RESULTS = os.path.join(ROOT, {"en": "results_en", "pub": "results_pub"}.get(SUITE, "results"))
