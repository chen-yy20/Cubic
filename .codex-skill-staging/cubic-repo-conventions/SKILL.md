---
name: cubic-repo-conventions
description: Quick orientation for this repository's cubic experiment layout, launch flow, and documentation boundaries. Use when Codex needs fast repo orientation before changing code in this repository.
---

# Cubic Repo Conventions

## Key files

- `README.md`: environment setup, launcher usage, model mapping, output locations
- `launch.py`: unified entrypoint routing `--model` and `--method`
- `flexflow/docs/theory.md`: algorithm and terminology
- `flexflow/docs/implement.md`: implementation architecture and contracts

## Repository layout

- `wan_exp/`: WAN origin and cubic experiments
- `flux1_exp/`: Flux1 origin and cubic experiments
- `flux2k_exp/`: Flux2-klein origin and cubic work-in-progress
- `firered_exp/`: FireRed origin experiments
- `flexflow/`: model-agnostic cubic logic, docs, and references
- `utils/`: timers, metrics, logging, result helpers

## Default engineering expectations

- Prefer extending via new cubic modules instead of editing origin runtime.
- Use `rg` to locate model paths, entrypoints, and symbols quickly.
- Validate through `launch.py` when practical.
- Treat environment and dependency changes as user decisions, not autonomous fixes.

## Common commands

```bash
source ./.venv/bin/activate
python launch.py --model <model> --method <method> --launcher python
rg -n "<symbol_or_text>"
rg --files
```
