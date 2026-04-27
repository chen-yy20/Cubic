# Cubic Repo Skills

These Codex-oriented skills are derived from the repository rules in `.github/agents`.

Available skills:

- `cubic-origin-implementer`: implement cubic incrementally from origin model code without modifying origin runtime by default.
- `cubic-worklog-maintainer`: append Chinese worklog entries to `worklog/cubic_worklog.md`.
- `implementmd-sync-writer`: sync `cubic_core/docs/implement.md` with approved code changes.
- `implementmd-drift-reviewer`: review code drift against `cubic_core/docs/implement.md`.
- `cubic-repo-conventions`: quick repo orientation for layout, launch flow, and validation norms.

Shared repo conventions:

- Read `cubic_core/docs/theory.md` and `cubic_core/docs/implement.md` before making cubic design decisions.
- Prefer new cubic modules and separate entry scripts over edits to origin runtime modules.
- Use `launch.py` as the default validation entrypoint when practical.
- Do not change dependencies or environment setup unless the user explicitly asks.
