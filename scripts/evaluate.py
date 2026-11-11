"""Re-evaluate a fine-tuned model.pt on an evaluation suite (e.g. after adding a new target dataset) without
retraining. Results go to <run>/results_<suite>.json; results.json is only replaced with --update.

  python scripts/evaluate.py --model runs/finetune/<run>/model.pt --eval-config configs/eval/seg_targets.yaml
"""

import argparse
from pathlib import Path

from xdssl.engine.evaluate import evaluate_domains
from xdssl.engine.finetune import PRIMARY_METRIC, load_finetuned
from xdssl.metrics.robustness import summarize_domains
from xdssl.utils.config import load_config, resolve_path
from xdssl.utils.misc import get_device, load_json, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval-config", required=True)
    ap.add_argument("--update", action="store_true", help="merge new domains into results.json")
    ap.add_argument("--paths", default=None)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    device = get_device()
    model, cfg = load_finetuned(a.model, device)
    ecfg = load_config(a.eval_config, a.overrides, a.paths).eval
    res = evaluate_domains(model, cfg.task, ecfg, device, cfg.optim.amp)
    run_dir = resolve_path(a.model).parent
    save_json({"domains": res, "summary": summarize_domains(res, PRIMARY_METRIC[cfg.task], ecfg.in_domain)},
              run_dir / f"results_{Path(a.eval_config).stem}.json")
    if a.update and (run_dir / "results.json").exists():
        full = load_json(run_dir / "results.json")
        full["domains"].update(res)
        full["summary"] = summarize_domains(full["domains"], PRIMARY_METRIC[cfg.task], full["summary"].get("in_domain", ecfg.in_domain))
        save_json(full, run_dir / "results.json")
    for k, v in res.items():
        print(k, {m: v[m] for m in ("mIoU", "AP", "AP50") if m in v})


if __name__ == "__main__":
    main()
