---
name: commit
description: Use when creating a git commit. Produces small atomic commits with Conventional Commits messages, one logical change each. Invoke after a test goes green or a refactor completes.
---

# Commit conventions

The user runs git. Your job is to tell them what to stage, check the diff with them, and propose the message.

One commit per logical change. Do not batch unrelated changes.

Format: type(scope): summary
- types: feat, fix, refactor, test, docs, chore, build, ci, perf
- scopes: sim, bringup, perception, data, eval, nav, mission, docs, build
- summary: imperative, lower-case, no trailing period, under ~72 chars
- body (optional): the WHY, wrapped at ~72 cols. Not what changed (the diff shows that).

Before committing:
1. Run `pixi run test` and `pixi run lint`. Do not commit red or unformatted code.
2. Have the user run `git status` and `git diff --staged`. Confirm together it is one logical change.
3. Self-review the diff (Google code review habits): tests cover the change, names are clear, no dead code,
   no debug prints, no secrets, no large files.
4. If the diff mixes concerns, split it into separate commits.

Reference the plan step in the body, e.g. "Implements step 9.2".
Never commit secrets, .env files, datasets, checkpoints, or videos.
