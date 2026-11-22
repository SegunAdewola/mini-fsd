# ADR 0005: Follow Google style guides, enforced by linters

- Status: Accepted
- Date: 2026-11-21
- Plan step: 0.2

## Context

The repo is public and will be read by the professor and by recruiters for
robotics software roles. It mixes Python, Markdown, YAML, shell, and possibly
C++ later. Style arguments and hand review waste the little time there is.
Google publishes style guides for most of these languages, and they are widely
known.

## Decision

Follow the Google guide for each language and let tools enforce it. Every tool
runs in pre-commit and in CI.

| Area            | Guide                                                                                                  | Tool                                  |
| --------------- | ------------------------------------------------------------------------------------------------------ | ------------------------------------- |
| Python          | [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)                          | ruff (lint + format), mypy            |
| Markdown        | [Google Markdown style guide](https://google.github.io/styleguide/docguide/style.html)                 | markdownlint-cli2                     |
| Prose in docs   | [Google developer documentation style guide](https://developers.google.com/style)                      | Vale, `Google` package, warnings only |
| Shell           | [Google Shell Style Guide](https://google.github.io/styleguide/shellguide.html)                        | shellcheck, shfmt                     |
| YAML            | none                                                                                                   | yamllint                              |
| GitHub Actions  | none                                                                                                   | actionlint                            |
| C++ (if needed) | [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)                            | clang-format, cpplint                 |
| Commits and PRs | [Google CL descriptions](https://google.github.io/eng-practices/review/developer/cl-descriptions.html) | commit skill, PR template             |

The linters are the authority. If a rule is wrong for this project, change the
config in a commit that says why, rather than ignoring it inline.

## Consequences

- New contributors (and agents) get style feedback on commit, not in review.
- Some Google rules are not checkable by tools, for example "import modules, not
  names". Those are covered in code review.
- Docs are wrapped at 80 columns, which makes diffs smaller but means a one-time
  reflow of the existing docs.
- Vale is warnings-only, because prose rules are judgment calls and should not
  block a commit.

## Alternatives considered

- **Black plus flake8 plus isort.** Same outcome, three tools instead of one
  (ruff), and slower.
- **pylint with Google's pylintrc.** Closest to Google's internal setup, but
  slow and largely overlaps with ruff plus mypy.
- **No doc linting.** Docs are a large part of this project (report, ADRs, study
  guide), so they get the same treatment as code.
