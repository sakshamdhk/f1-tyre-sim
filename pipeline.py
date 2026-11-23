"""Rebuild the whole project from scratch: data -> models -> analysis -> plots.

    python pipeline.py              # everything (about 30-40 min; the download is cached)
    python pipeline.py --from evaluate

Steps:
  collect    download 2023-2024 races from OpenF1 (skips races already in data/raw)
  dataset    clean laps, targets, features, folds, pit truth
  evaluate   train every model under both test protocols
  analysis   metrics, pit-stop evaluation, importance, temperature effect, learning curve
  plots      all figures into plots/
"""
import argparse
import subprocess
import sys
import time

STEPS = ["collect", "dataset", "evaluate", "analysis", "plots"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", choices=STEPS, default="collect")
    start = STEPS.index(ap.parse_args().start)
    for step in STEPS[start:]:
        t0 = time.time()
        print(f"\n===== {step}", flush=True)
        subprocess.run([sys.executable, "-W", "ignore", "-m", f"f1sim.{step}"], check=True)
        print(f"===== {step} done in {time.time() - t0:.0f} s", flush=True)
