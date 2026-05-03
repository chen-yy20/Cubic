#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys


SCRIPT_MAP = {
    "wan": {
        "ori": "wan_exp/ori_generate.py",
        "cubic": "wan_exp/cubic_generate.py",
        "teacache": "wan_exp/teacache_generate.py",
        "pab": "wan_exp/pab_generate.py",
        "tocache": "wan_exp/tocache_generate.py",
    },
    "firered": {
        "ori": "firered_exp/ori_generate.py",
        "cubic": "firered_exp/cubic_generate.py",
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
    },
}

MODEL_ALIASES = {
    "flux2": "flux2k",
}

MODEL_PATH_CANDIDATES = {
    "wan": [
        "/home/chenyy/WORK/models/Wan2.1-T2V-1.3B",
        "/home/fit/zhaijdcyy/WORK/models/Wan2.1-T2V-1.3B",
    ],
    "firered": [
        "/home/chenyy/WORK/models/firered-image-edit",
        "/home/fit/zhaijdcyy/WORK/models/FireRed-Image-Edit-1.0",
    ],
    "flux": [
        "/home/chenyy/WORK/models/Flux-1",
        "/home/fit/zhaijdcyy/WORK/models/Flux-1",
    ],
    "flux2k": [
        "/home/chenyy/WORK/models/flux2-klein",
        "/home/fit/zhaijdcyy/WORK/models/flux2-klein",
    ],
}

WAN_PROMPT = "A couple in formal evening wear going home get caught in a heavy downpour with umbrellas, zoom in"
FLUX_PROMPT = "A photorealistic cute cat, wearing a simple blue shirt, standing against a clear sky background."
FIRERED_PROMPT = "change orange cat to a white cat, add a dog beside it, change sitting cat to standing."
FIRERED_IMAGE_PATH = "/home/chenyy/WORK/cyy/cubic/firered_exp/examples/cat.png"
FLUX2_PROMPT = "A cat holding a sign that says hello world"


LAUNCH_DEFAULTS = {
    "wan": {
        "__common__": {
            "prompt": WAN_PROMPT,
        },
    },
    "firered": {
        "__common__": {
            "prompt": FIRERED_PROMPT,
            "input_image": [FIRERED_IMAGE_PATH],
        },
    },
    "flux": {
        "__common__": {
            "prompt": FLUX_PROMPT,
        },
    },
    "flux2k": {
        "__common__": {
            "prompt": FLUX2_PROMPT,
        },
    },
}


def build_parser():
    parser = argparse.ArgumentParser(description="Unified launcher for experiment scripts")
    parser.add_argument(
        "--model",
        choices=sorted(set(SCRIPT_MAP.keys()) | set(MODEL_ALIASES.keys())),
        required=True,
        help="Model family",
    )
    parser.add_argument("--method", required=True, help="Method name in selected model family")
    parser.add_argument("--launcher", choices=["python", "srun"], default="srun", help="Launch backend")
    parser.add_argument("--model-path", default=None, help="Override the default model path for the selected family")
    parser.add_argument("--script", default=None, help="Optional custom script path")
    parser.add_argument("--partition", default="debug", help="SLURM partition for srun")
    parser.add_argument("--gpus", type=int, default=1, help="GPUs per node for srun")
    parser.add_argument("--job-name", default="cubic-run", help="SLURM job name for srun")
    parser.add_argument("script_args", nargs=argparse.REMAINDER, help="Extra args passed to target script")
    return parser


def normalize_model_name(model):
    return MODEL_ALIASES.get(model, model)


def resolve_model_path(model, model_path_override):
    if model_path_override:
        return model_path_override

    for candidate in MODEL_PATH_CANDIDATES[model]:
        if os.path.exists(candidate):
            return candidate

    return MODEL_PATH_CANDIDATES[model][0]


def resolve_script(model, method, script):
    if script:
        return script
    if method not in SCRIPT_MAP[model]:
        raise ValueError(
            f"Unknown method '{method}' for model '{model}'. "
            f"Available: {', '.join(sorted(SCRIPT_MAP[model].keys()))}"
        )
    return SCRIPT_MAP[model][method]


def _merge_launch_defaults(model, method):
    model_defaults = LAUNCH_DEFAULTS.get(model, {})
    merged = {}
    merged.update(model_defaults.get("__common__", {}))
    merged.update(model_defaults.get(method, {}))
    return merged


def _has_option(script_args, option):
    return any(arg == option or arg.startswith(f"{option}=") for arg in script_args)


def _append_option(script_args, key, value):
    option = f"--{key}"
    if _has_option(script_args, option):
        return
    if isinstance(value, bool):
        if value:
            script_args.append(option)
        return
    if value is None:
        return
    script_args.append(option)
    if isinstance(value, (list, tuple)):
        script_args.extend(str(item) for item in value)
    else:
        script_args.append(str(value))


def apply_launch_defaults(model, method, script_args):
    resolved_args = list(script_args)
    for key, value in _merge_launch_defaults(model, method).items():
        _append_option(resolved_args, key, value)
    return resolved_args


def main():
    args = build_parser().parse_args()
    model = normalize_model_name(args.model)
    script_args = args.script_args
    if script_args and script_args[0] == "--":
        script_args = script_args[1:]

    script = resolve_script(model, args.method, args.script)
    repo_root = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.copy()
    model_path = resolve_model_path(model, args.model_path)
    script_args = apply_launch_defaults(model, args.method, script_args)

    if args.method == "ori":
        env["ENABLE_JANO"] = "0"
    elif args.method == "jano":
        env["ENABLE_JANO"] = "1"

    env["CUBIC_MODEL_PATH"] = model_path
    env["CUBIC_MODEL_FAMILY"] = model

    extra_paths = [repo_root]
    if args.method == "jano":
        extra_paths.append(os.path.join(repo_root, "firered_exp"))
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
    print("Model path:", model_path, flush=True)
    if script_args:
        print("Script args:", " ".join(script_args), flush=True)
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
