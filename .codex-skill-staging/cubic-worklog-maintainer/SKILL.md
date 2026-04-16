---
name: cubic-worklog-maintainer
description: Append concise Chinese worklog entries for cubic implementation progress in this repo. Use when Codex needs to create or append a progress record after a cubic code change.
---

# Cubic Worklog Maintainer

## Scope

- Maintain only `worklog/cubic_worklog.md`
- Keep all appended content in Chinese
- Append-only unless the user explicitly asks to rewrite history

## Hard rules

- Do not edit source code files.
- Do not reorder or rewrite previous entries.
- If the worklog file does not exist, create it with the first entry.
- Keep entries factual, short, and tied to concrete code changes or validation results.

## Entry template

Use this exact section order:

- 日期
- 目标模块
- 变更摘要
- 变更依据（对应 theory.md / implement.md）
- 受影响文件
- 验证命令与结果
- 待确认项 / 下一步

## Procedure

1. Check whether `worklog/cubic_worklog.md` exists.
2. Normalize the current iteration details into the required template.
3. Append exactly one new entry at the end.
4. Return the appended content and any still-missing facts.
