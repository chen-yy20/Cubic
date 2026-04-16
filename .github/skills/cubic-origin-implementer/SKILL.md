---
name: cubic-origin-implementer
description: Implement cubic incrementally from origin model code in this repo while keeping origin runtime untouched by default.
---

# Cubic Origin Implementer

Use this skill when the user wants to add or extend cubic support for a model in this repository.

## Read first

1. `flexflow/docs/theory.md`
2. `flexflow/docs/implement.md`
3. Relevant origin entry and runtime files for the target model
4. `flexflow/reference/jano_*.py` and `flexflow/reference/wan_modules/*` only when needed as implementation patterns

## Repo rules

- Keep origin runtime modules untouched unless the user explicitly approves editing them.
- Prefer new files under the model experiment directory such as `<model>_exp/cubic_generate.py` or `<model>_exp/<model>_cubic/*`.
- If logic is model-agnostic, place it under `flexflow/`.
- Make small, reviewable patches, usually one module at a time.
- Do not install, upgrade, downgrade, or remove dependencies on your own.

## Model map

- WAN: `wan_exp/ori_generate.py` -> `wan_exp/cubic_generate.py`, runtime under `wan_exp/wan/` and `wan_exp/wan_cubic/`
- Flux1: `flux1_exp/ori_generate.py` -> `flux1_exp/cubic_generate.py`, runtime under `flux1_exp/flux/` and `flux1_exp/flux_cubic/`
- Flux2-klein: `flux2k_exp/ori_generate.py`, cubic runtime under `flux2k_exp/flux2k_cubic/`
- FireRed: `firered_exp/ori_generate.py`, runtime under `firered_exp/firered/`

## Working procedure

1. Restate the target module and acceptance bar in 2-4 lines.
2. Read the theory and implement docs sections that govern the requested change.
3. Map the origin path to the cubic path and choose placement:
   - model-specific derived code -> corresponding `*_exp/`
   - model-agnostic reusable code -> `flexflow/`
4. Implement the smallest viable patch for the requested module.
5. Run a focused smoke validation when the environment allows it.
6. If the change is WAN-related and the user wants progress tracking, use `cubic-worklog-maintainer`.

## Validation default

Activate the repo environment first:

```bash
source ./.venv/bin/activate
```

Preferred smoke command:

```bash
python launch.py --model <target_model> --method cubic
```

Notes:

- `launch.py` defaults to `srun`; for local debugging, prefer `--launcher python`.
- Stop and report clearly if validation is blocked by missing models, GPUs, or dependencies.

## Output checklist

- Assumptions
- Plan for this patch
- Applied changes
- Validation result
- Next recommended module
