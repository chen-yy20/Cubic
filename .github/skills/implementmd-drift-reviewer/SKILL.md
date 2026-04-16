---
name: implementmd-drift-reviewer
description: Review code drift against flexflow/docs/implement.md contracts and integration rules.
---

# ImplementMD Drift Reviewer

Use this skill for read-only review of whether the current implementation still matches `flexflow/docs/implement.md`.

## Read first

1. Relevant sections of `flexflow/docs/implement.md`
2. The target code files under review

## Hard rules

- Do not edit files.
- Focus on concrete contract mismatches, architectural divergence, and missing integration points.
- Prioritize WAN-related modules unless the user explicitly asks for a different scope.
- Cite exact file paths and line references for every finding.

## Review procedure

1. Extract the specific contracts or architectural rules from `implement.md`.
2. Inspect the target files and compare behavior against those rules.
3. Report findings ordered by severity.
4. If no findings are present, say so explicitly and note residual risks or testing gaps.

## Output checklist

- Findings
- Open questions or assumptions
- Residual risks
