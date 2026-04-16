---
name: implementmd-sync-writer
description: Sync cubic_core/docs/implement.md with approved code behavior in this repository. Use when Codex needs to update implement.md so it matches code that is already implemented or explicitly approved.
---

# ImplementMD Sync Writer

## Read first

1. `cubic_core/docs/implement.md`
2. The approved code changes being documented
3. `cubic_core/docs/theory.md` only if needed to confirm wording or terminology

## Hard rules

- Do not edit source code files.
- Document only behavior that is already implemented or explicitly approved.
- Preserve the existing document structure unless the user asks for a reorganization.
- Prefer minimal edits that resolve drift, missing steps, or stale interface details.

## Procedure

1. Identify the exact sections in `cubic_core/docs/implement.md` touched by the code change.
2. Compare current wording with the implemented behavior.
3. Apply the smallest edits needed to align docs with code.
4. Summarize each doc delta with the matching code evidence.

## Output checklist

- Documentation deltas applied
- Rationale and code evidence
- Follow-up documentation gaps
