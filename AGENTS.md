## Agent skills

### Issue tracker

Issues live as local markdown files under `.scratch/`.

- Use one directory per feature: `.scratch/<feature-slug>/`.
- Put the spec at `.scratch/<feature-slug>/spec.md`.
- Put each ticket at `.scratch/<feature-slug>/issues/<NN>-<slug>.md`. Number from `01`.
- Record triage state in a `Status:` line near the top.
- Append history under a `## Comments` heading.

Wayfinder map:

- Put the map at `.scratch/<effort>/map.md`.
- Put each child at `.scratch/<effort>/issues/NN-<slug>.md`.
- Record type in a `Type:` line (`research`, `prototype`, `grilling`, `task`).
- Record blocking in a `Blocked by: NN, NN` line.
- Frontier is open, unblocked, unclaimed tickets. Lowest number wins.
- Set `Status: claimed` before work.
- Append the answer under `## Answer`, set `Status: resolved`, then link the gist in `map.md`.

### Triage labels

Use five labels as-is: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`.

### Domain docs

This repo has no glossary or ADRs. Proceed silently. Do not create them upfront. Use exact domain terms in titles and tests. Flag conflicts with prior decisions explicitly.
