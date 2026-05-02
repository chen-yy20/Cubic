#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys


SCRIPT_MAP = {
    "wan": {
        "ori": "wan_exp/ori_generate.py",
        "ori_batch": "wan_exp/ori_batch_generate.py",
        "cubic": "wan_exp/cubic_generate.py",
        "teacache": "wan_exp/teacache_generate.py",
        "teacache_batch": "wan_exp/teacache_batch_generate.py",
        "pab": "wan_exp/pab_generate.py",
        "pab_batch": "wan_exp/pab_batch_generate.py",
        "tocache": "wan_exp/tocache_generate.py",
        "tocache_batch": "wan_exp/tocache_batch_generate.py",
    },
    "firered": {
        "ori": "firered_exp/ori_generate.py",
        "cubic": "firered_exp/cubic_generate.py",
        "taylorseer": "firered_exp/taylorseer_generate.py",
        "teacache": "firered_exp/teacache_generate.py",
        "pab": "firered_exp/PAB_generate.py",
        "tocache": "firered_exp/tocache_generate.py",
        "jano": "firered_exp/jano_generate.py",
        "kv_drift": "firered_exp/firered_cubic/cond_kv_drift_experiment.py",
        "pie_bench": "firered_exp/pie_bench_batch_generate.py",
    },
    "flux": {
        "ori": "flux1_exp/ori_generate.py",
        "cubic": "flux1_exp/cubic_generate.py",
        "teacache": "flux1_exp/teacache_generate.py",
    },
    "flux2k": {
        "ori": "flux2k_exp/ori_generate.py",
        "cubic": "flux2k_exp/cubic_generate.py",
    }
}


def build_parser():
    parser = argparse.ArgumentParser(description="Unified launcher for Jano scripts")
    parser.add_argument("--model", choices=SCRIPT_MAP.keys(), required=True, help="Model family")
    parser.add_argument("--method", required=True, help="Method name in selected model family")
    parser.add_argument("--launcher", choices=["python", "srun"], default="srun", help="Launch backend")
    parser.add_argument("--script", default=None, help="Optional custom script path")
    parser.add_argument("--partition", default="debug", help="SLURM partition for srun")
    parser.add_argument("--gpus", type=int, default=1, help="GPUs per node for srun")
    parser.add_argument("--job-name", default="cubic-run", help="SLURM job name for srun")
    parser.add_argument("script_args", nargs=argparse.REMAINDER, help="Extra args passed to target script")
    return parser


def resolve_script(model, method, script):
    if script:
        return script
    if method not in SCRIPT_MAP[model]:
        raise ValueError(
            f"Unknown method '{method}' for model '{model}'. "
            f"Available: {', '.join(sorted(SCRIPT_MAP[model].keys()))}"
        )
    return SCRIPT_MAP[model][method]


def main():
    args = build_parser().parse_args()
    script_args = args.script_args
    if script_args and script_args[0] == "--":
        script_args = script_args[1:]

    script = resolve_script(args.model, args.method, args.script)
    repo_root = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.copy()

    if args.method in ("ori", "ori_batch"):
        env["ENABLE_JANO"] = "0"
    elif args.method == "jano":
        env["ENABLE_JANO"] = "1"

    extra_paths = [repo_root]
    _firered_exp = os.path.join(repo_root, "firered_exp")
    if args.model == "firered" or args.method == "jano":
        if _firered_exp not in extra_paths:
            extra_paths.append(_firered_exp)
    if env.get("PYTHONPATH") and env["PYTHONPATH"].strip():
        env["PYTHONPATH"] = os.pathsep.join(extra_paths + [env["PYTHONPATH"]])
    else:
        env["PYTHONPATH"] = os.pathsep.join(extra_paths)

    if args.launcher == "python":
        cmd = [sys.executable, script, *script_args]
    else:
        env.update(
            {
                "NNODES": "1",
                "GPUS_PER_NODE": str(args.gpus),
                "PARTITION": args.partition,
                "JOB_NAME": args.job_name,
            }
        )
        cmd = [
            "./pysrun.sh",
            script,
            *script_args,
        ]

    print("Launching:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
