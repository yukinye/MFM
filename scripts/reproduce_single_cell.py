"""Reproduce the OT-MFM rows of the single-cell trajectory inference experiments (Tables 3, 4, 6).

Setup follows Appendix D.3 of the paper (arXiv:2405.14780v2):
  - leave-one-out interpolation: EB leaves out t=1,2,3; Cite/Multi leave out t=1,2
  - 5 independent runs (seeds 42-46), W1 averaged over left-out marginals
  - Adam lr 1e-4 for the geopath net, AdamW lr 1e-3 / wd 1e-5 for the flow net,
    90/10 train/val split, Euler with 100 steps at test time (parser defaults)
  - per-dataset hyperparameters from configs/single_cell/*

Every (config, seed, left-out timepoint) is run as an independent job in its own
working dir, so jobs can run in parallel (JOBS at a time) and finished jobs are
skipped on re-run. 7 configs x left-out timepoints x 5 seeds = 75 jobs.

Missing datasets are downloaded into <repo>/data/ before training starts
(EB from TrajectoryNet, Cite/Multi from Mendeley Data) and checked by sha256.

Usage:
  conda activate mfm
  python scripts/reproduce_single_cell.py
  python scripts/reproduce_single_cell.py --only 5dims --dry-run
  python scripts/reproduce_single_cell.py --summary-only
"""

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"
MENDELEY = "https://data.mendeley.com/public-files/datasets/hhny5ff7yj/files"
# data_name -> (file name, download URL, sha256)
DATASETS = {
    "eb": (
        "eb_velocity_v5.npz",
        "https://github.com/KrishnaswamyLab/TrajectoryNet/raw/master/data/eb_velocity_v5.npz",
        "17623d16dd4fe7b679aad80916d130cb2863a4cd4ee742f139a3c2eed91ba230",
    ),
    "cite": (
        "op_cite_inputs_0.h5ad",
        f"{MENDELEY}/1862acf5-6294-4eb1-8644-d1c6d25e4126/file_downloaded",
        "fa1d117df3d6c23e0b80a997a259bcac3a79ad27d581e1a6654e107c39885e4c",
    ),
    "multi": (
        "op_train_multi_targets_0.h5ad",
        f"{MENDELEY}/5f4b6e5b-f122-4f5a-8ede-0d188c5cf00c/file_downloaded",
        "a16c28ef861503111bb2f4c6ab6b1d5ed3df07d49eb0baa7378515fc098af4d2",
    ),
}

# Number of runs executed in parallel on the single GPU.
JOBS = 8

# OT-MFM only: (table, config, paper W1 mean, paper W1 std)
EXPERIMENTS = [
    ("Table 3 (100D)", "100dims/ot-mfm_cite.yaml", 41.784, 1.020),
    ("Table 3 (100D)", "100dims/ot-mfm_multi.yaml", 50.906, 4.627),
    ("Table 6 (50D)", "50dims/ot-mfm_cite.yaml", 36.394, 1.886),
    ("Table 6 (50D)", "50dims/ot-mfm_multi.yaml", 45.16, 4.96),
    ("Table 4 (5D)", "5dims/ot-mfm_cite.yaml", 0.724, 0.070),
    ("Table 4 (5D)", "5dims/ot-mfm_eb.yaml", 0.713, 0.039),
    ("Table 4 (5D)", "5dims/ot-mfm_multi.yaml", 0.890, 0.123),
]

EMD_RE = re.compile(r"test_EMD\s+([0-9.eE+-]+)")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_data(data_names):
    """Download any missing dataset into DATA_DIR and verify its checksum."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in sorted(data_names):
        file_name, url, digest = DATASETS[name]
        path = DATA_DIR / file_name
        if path.exists():
            continue
        part = path.with_name(path.name + ".part")
        print(f"Downloading {name} -> {path}", flush=True)
        # -C - resumes an interrupted download of the .part file
        subprocess.run(["curl", "-L", "--fail", "-C", "-", "-o", str(part), url], check=True)
        if sha256(part) != digest:
            part.unlink()
            sys.exit(f"Checksum mismatch for {file_name}; the partial download was removed, re-run to retry.")
        part.rename(path)


def build_jobs(seeds, only, left_out=None):
    jobs = []
    for table, cfg_rel, _, _ in EXPERIMENTS:
        if only and not any(o in cfg_rel for o in only):
            continue
        cfg = yaml.safe_load(open(REPO / "configs/single_cell" / cfg_rel))
        for seed in seeds:
            for i, t in enumerate(cfg["t_exclude"]):
                if left_out and t not in left_out:
                    continue
                job_cfg = dict(cfg, seeds=[seed], t_exclude=[t])
                if "gammas" in cfg:
                    job_cfg["gammas"] = [cfg["gammas"][i]]
                name = cfg_rel.replace(".yaml", "").replace("/", "_")
                jobs.append(
                    {
                        "exp": cfg_rel,
                        "name": f"{name}/seed{seed}_t{t}",
                        "cfg": job_cfg,
                        "seed": seed,
                        "t": t,
                    }
                )
    # High-dim jobs are the slowest; start them first for better packing.
    jobs.sort(key=lambda j: -j["cfg"]["dim"])
    return jobs


def run_job(job, out_root, threads, extra_args):
    job_dir = out_root / job["name"]
    result_path = job_dir / "result.json"
    if result_path.exists():
        return json.load(open(result_path))

    (job_dir / "data").mkdir(parents=True, exist_ok=True)
    data_file = DATASETS[job["cfg"]["data_name"]][0]
    link = job_dir / "data" / data_file
    if not link.exists():
        link.symlink_to(DATA_DIR / data_file)
    cfg_path = job_dir / "config.yaml"
    yaml.safe_dump(job["cfg"], open(cfg_path, "w"))

    env = dict(
        os.environ,
        WANDB_MODE=os.environ.get("WANDB_MODE", "offline"),
        WANDB_SILENT="true",
        PYTHONPATH=os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")])),
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
        OMP_NUM_THREADS=str(threads),
        MKL_NUM_THREADS=str(threads),
    )
    cmd = [
        sys.executable, "-m", "mfm.train.main",
        "--config_path", str(cfg_path),
        "--working_dir", str(job_dir),
        *extra_args,
    ]
    start = time.time()
    with open(job_dir / "train.log", "w") as log:
        proc = subprocess.run(cmd, cwd=job_dir, env=env, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.time() - start

    matches = EMD_RE.findall(open(job_dir / "train.log").read())
    if proc.returncode != 0 or not matches:
        raise RuntimeError(f"{job['name']} failed (exit {proc.returncode}), see {job_dir}/train.log")
    result = {**{k: job[k] for k in ("exp", "seed", "t")}, "w1": float(matches[-1]), "seconds": elapsed}
    json.dump(result, open(result_path, "w"), indent=2)
    return result


def summarize(out_root, seeds):
    results = [json.load(open(p)) for p in out_root.glob("*/*/result.json")]
    lines = [
        "| Table | Config | Runs | W1 (ours) | W1 (paper) | Avg time/run |",
        "|---|---|---|---|---|---|",
    ]
    for table, cfg_rel, paper_mean, paper_std in EXPERIMENTS:
        rs = [r for r in results if r["exp"] == cfg_rel]
        if not rs:
            continue
        # Paper: W1 averaged over left-out marginals, then mean ± std over seeds.
        per_seed = {}
        for r in rs:
            per_seed.setdefault(r["seed"], []).append(r["w1"])
        seed_means = [statistics.mean(v) for v in per_seed.values()]
        std = statistics.stdev(seed_means) if len(seed_means) > 1 else 0.0
        n_t = len(yaml.safe_load(open(REPO / "configs/single_cell" / cfg_rel))["t_exclude"])
        avg_min = statistics.mean(r["seconds"] for r in rs) / 60
        lines.append(
            f"| {table} | {cfg_rel} | {len(rs)}/{len(seeds) * n_t} | "
            f"{statistics.mean(seed_means):.3f} ± {std:.3f} | {paper_mean} ± {paper_std} | {avg_min:.1f} min |"
        )
    text = "\n".join(lines)
    (out_root / "summary.md").write_text(text + "\n")
    print(text)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(REPO / "runs/single_cell"), help="output root")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--only", nargs="+", help="substring filter on config paths, e.g. 5dims ot-mfm_eb")
    p.add_argument("--left-out", type=int, nargs="+", help="only run these left-out timepoints")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--summary-only", action="store_true")
    args, extra = p.parse_known_args()  # unknown args are forwarded to mfm.train.main

    out_root = Path(args.out).resolve()
    if args.summary_only:
        return summarize(out_root, args.seeds)

    jobs = build_jobs(args.seeds, args.only, args.left_out)
    print(f"{len(jobs)} runs, {JOBS} in parallel -> {out_root}")
    if args.dry_run:
        for j in jobs:
            print(" ", j["name"])
        return

    ensure_data({j["cfg"]["data_name"] for j in jobs})

    threads = max(1, (os.cpu_count() or 1) // JOBS)
    failed = 0
    start = time.time()
    with ThreadPoolExecutor(JOBS) as pool:
        futures = {pool.submit(run_job, j, out_root, threads, extra): j for j in jobs}
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                r = fut.result()
                print(f"[{n}/{len(jobs)}] {futures[fut]['name']}: W1={r['w1']:.4f} ({r['seconds'] / 60:.1f} min)", flush=True)
            except Exception as e:
                failed += 1
                print(f"[{n}/{len(jobs)}] FAILED: {e}", flush=True)
    print(f"Done in {(time.time() - start) / 3600:.2f} h, {failed} failed.\n")
    summarize(out_root, args.seeds)


if __name__ == "__main__":
    main()
