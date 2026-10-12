"""Run Experiment 1's own analysis code (exp1/analyze.py, unmodified) on Experiment 3's results.

usage: python analyze3.py      (CPU math env is enough: numpy only)
Reads exp3/res/*.jsonl (files starting with '_' are skipped, as in exp1), writes exp3/RESULTS_tables.md and
exp3/results_summary.json. Nothing in exp1/ is modified.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'exp1'))
import analyze  # noqa: E402  (exp1/analyze.py)

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')  # analyze.py prints arrows; Windows console default is cp1252
    analyze.HERE = HERE  # glob res/ and write outputs relative to exp3
    analyze.main()
