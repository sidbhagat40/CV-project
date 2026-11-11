"""Run (or print) every job of an experiment grid. Jobs whose output already exists are skipped, so the
command can be re-run after interruptions.

  python scripts/run_experiment.py --exp configs/experiments/e1_diversity.yaml --dry-run
  python scripts/run_experiment.py --exp configs/experiments/e1_diversity.yaml --only pool pretrain
"""

import argparse
import os
import shlex
import subprocess
import time

from xdssl.experiments import expand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="*", choices=["pool", "pretrain", "finetune"], default=None)
    ap.add_argument("--filter", default=None, help="substring that job names must contain")
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument("--paths", default=None)
    a = ap.parse_args()
    if a.paths:
        os.environ["XDSSL_PATHS"] = a.paths  # inherited by every job subprocess

    jobs = [j for j in expand(a.exp, a.paths) if (a.only is None or j.kind in a.only) and (not a.filter or a.filter in j.name)]
    seen, failed = set(), []
    for j in jobs:
        if j.output in seen:
            continue
        seen.add(j.output)
        status = "done" if j.output.exists() else "todo"
        print(f"[{status}] {j.kind:8s} {j.name}")
        if status == "done":
            continue
        if a.dry_run:
            print("   ", " ".join(shlex.quote(c) for c in j.cmd))
            continue
        t0 = time.time()
        rc = subprocess.run(j.cmd).returncode
        print(f"    -> exit {rc} after {(time.time() - t0) / 60:.1f} min")
        if rc != 0:
            failed.append(j.name)
            if a.stop_on_error:
                break
    if failed:
        print("FAILED:", *failed, sep="\n  ")


if __name__ == "__main__":
    main()
