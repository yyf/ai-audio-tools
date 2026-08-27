# Daily repo scout — automation spec

> **Status:** Active
>
> Defines a Cursor Automation for discovering new audio/music/speech AI
> repositories and proposing README additions via pull request only.
> Runs entirely on Cursor Cloud + GitHub — no local machine required.

---

## Document conventions

### Tables

Use fixed-width ASCII tables inside fenced `text` blocks — not markdown pipe
tables (`| a | b |`). Pipe tables wrap poorly in narrow editor panes.

**Rules:**

- **Max width** — 79 characters per line (fits common editor/terminal panes)
- **Label column** — 13 characters, left-aligned
- **Value column** — 60 characters; wrap at word boundaries
- **Continuations** — empty label cell; indent wrapped text in value column
- **Separators** — box drawing: `+`, `-`, `|`
- **No pipe tables** — do not use `| a | b |` in this spec (PR templates excepted)

**Template:**

```text
+-------------+-------------------------------------------------------------+
| Field       | Value                                                       |
+-------------+-------------------------------------------------------------+
| Short label | Single-line value fits here                                   |
| Long label  | First line of a value that needs to wrap onto the next row   |
|             | Second line aligned under the value column                    |
+-------------+-------------------------------------------------------------+
```

---

## Overview

```text
+-------------+-------------------------------------------------------------+
| Field       | Value                                                       |
+-------------+-------------------------------------------------------------+
| Name        | Daily audio-tools repo scout                                  |
| Purpose     | Find new high-quality audio/music/speech AI repos; propose  |
|             | README additions via PR when at least one high-quality        |
|             | candidate is found                                            |
| Repo        | yyf/ai-audio-tools (this project only)                        |
| Base branch | main                                                          |
| Bot branch  | bot/daily-candidates-YYYY-MM-DD                               |
| Schedule    | Weekdays 9:00 AM America/Los_Angeles (adjust in Automations |
|             | editor)                                                       |
| Outcome     | Silent exit when zero high-quality finds; otherwise one PR    |
|             | with scored candidates for maintainer merge                   |
+-------------+-------------------------------------------------------------+
```

---

## Cloud execution

```text
+-------------+-------------------------------------------------------------+
| Component   | Where                                                       |
+-------------+-------------------------------------------------------------+
| Scheduled   | Cursor Cloud Agent                                          |
| run         |                                                             |
| Repo        | GitHub (yyf/ai-audio-tools)                                 |
| checkout    |                                                             |
| Discovery   | GitHub search API only (v1)                                 |
| Review      | GitHub PR UI (maintainer only)                              |
| and merge   |                                                             |
+-------------+-------------------------------------------------------------+
```

**Prerequisites before first run:**

- [ ] This file committed on `main`
- [ ] Branch protection on `main` (PR required, no force push, maintainer approval)
- [ ] GitHub connected to Cursor with permission to create branches and PRs
- [ ] Cursor Automation created from this spec; dry run once before enabling cron

---

## Strict capability rules (non-negotiable)

### 1. Project scope only

```text
+---------------------------+-----------------------------------------------+
| Allowed                   | Forbidden                                     |
+---------------------------+-----------------------------------------------+
| Read/write files in this  | Access other repos, orgs, or local paths      |
| repo on a dedicated       | outside the project                           |
| branch                    |                                               |
| Read README.md and        | Clone, fork, or modify external repositories  |
| optional repo-local       |                                               |
| config (e.g. automation/  |                                               |
| rejected.md)              |                                               |
| Open a PR into this repo  | Push to main, merge PRs, delete protected     |
|                           | branches                                      |
| GitHub search read-only   | Broad account sweeps, private repo inventory, |
| for discovery (scoped     | unscoped org listing                          |
| queries)                  |                                               |
+---------------------------+-----------------------------------------------+
```

**Default:** if an action is not explicitly allowed below, it is **denied**.

### 2. Search scope — GitHub only, relevant entries, no massive crawls

**v1 discovery:** GitHub repository search **only**. No Hugging Face, arXiv,
PaperWithCode, Kaggle, or web crawls.

**In-scope topics:** audio, music, MIR, speech, ASR, TTS, voice cloning,
audio codecs, audio LLMs, audio datasets/benchmarks, audio security/watermarking.

**Search limits (hard caps per run):**

```text
+-------------+-------------------------------------------------------------+
| Limit       | Value                                                       |
+-------------+-------------------------------------------------------------+
| GitHub      | <= 5 queries, each with narrow filters                      |
| search      |                                                             |
| Results per | <= 10 repos (top by relevance/recency)                      |
| query       |                                                             |
| Max         | <= 30 total                                                 |
| candidates  |                                                             |
| Star floor  | stars:>30 (unknown authors)                                 |
| Time window | created: or pushed: within last 7 days                      |
| Crawl depth | No link following, no scraping, no pagination beyond caps   |
+-------------+-------------------------------------------------------------+
```

**Example allowed query shape:**

```
speech OR TTS stars:>30 pushed:>YYYY-MM-DD
```

**Forbidden:**

- Unbounded pagination
- “Search all of GitHub for audio”
- Hugging Face, arXiv, PaperWithCode, Kaggle, or any non-GitHub discovery
- Scraping awesome-lists or star-history sweeps
- Downloading datasets, cloning repos, or running project code
- Arbitrary `curl`, web fetch, or MCP-based external lookups

### 3. README — never overwrite without approval

```text
+-------------+-------------------------------------------------------------+
| Rule        | Behavior                                                    |
+-------------+-------------------------------------------------------------+
| Never edit  | All README changes on a bot branch only                     |
| main        |                                                             |
| Never merge | Agent opens PR; maintainer merges manually                  |
| Never       | No --force, no rewriting main history                       |
| force-push  |                                                             |
| Approval =  | README on main changes only after PR merge                  |
| merge       |                                                             |
| Dry runs    | If zero high-quality finds: no branch, no README write, no  |
|             | PR                                                          |
+-------------+-------------------------------------------------------------+
```

**Allowed write:** `README.md` on branch `bot/daily-candidates-YYYY-MM-DD` only
when opening a PR.

### 4. Privileges — least privilege by default

**Tools to enable (minimal):**

```text
+-------------+-------------------------------------------------------------+
| Tool        | Use                                                         |
+-------------+-------------------------------------------------------------+
| Git         | Checkout branch, commit, push branch (this repo only)       |
| GitHub      | Search (read), list/close PRs (bot PRs only), open PR       |
|             | (create only); scoped to this repo                          |
| File        | Read/write this repo only                                   |
| read/write  |                                                             |
+-------------+-------------------------------------------------------------+
```

**Tools to disable / not grant:**

```text
+-------------+-------------------------------------------------------------+
| Disabled    | Reason                                                      |
+-------------+-------------------------------------------------------------+
| Auto-merge  | Maintainer gates all merges                                 |
| Push to     | Prevents silent overwrites                                  |
| main        |                                                             |
| Slack /     | Out of scope for v1                                         |
| email /     |                                                             |
| webhooks    |                                                             |
| MCP servers | Not required for v1; reduce attack surface                  |
| Arbitrary   | GitHub search only — no web crawls                          |
| shell + net |                                                             |
| PR comment  | Off by default                                              |
| spam, etc.  |                                                             |
+-------------+-------------------------------------------------------------+
```

**Recommended branch protection on `main`:**

- Require maintainer approval before merge
- Block force pushes
- Require PR for all changes

---

## Discovery and quality logic

### Pipeline

1. Read `automation/SPEC.md` and follow it exactly.
2. Read `README.md` → build normalized URL set (GitHub URLs; also match HF
   links already in list for dedupe).
3. Run **bounded GitHub-only** searches (see caps above).
4. Score each candidate **0–100**; assign **one** category from existing README
   hierarchy.
5. Count **high-quality** entries: confidence **≥ 80**, in-scope, deduped,
   clear README/activity/impact.
6. **If count = 0 → stop** (no PR, no writes).
7. **If count ≥ 1 →** close any open bot PRs (branch prefix
   `bot/daily-candidates-`), edit README on a new bot branch, open PR with
   full metadata table.

### Category paths (must match README)

```
Audio  > Benchmark | Dataset | Annotation | Model | Security
Music  > Benchmark | Analysis | Production | Generation
Speech > Benchmark | Recognition | Production | Synthesis
```

### Confidence bands

```text
+-------------+-------------------------------------------------------------+
| Score       | Meaning                                                     |
+-------------+-------------------------------------------------------------+
| 90-100      | Strong fit; merge with little or no edit                    |
| 80-89       | Good fit; included in PR                                    |
| 60-79       | Borderline; not in README diff                              |
| < 60        | Skip                                                        |
+-------------+-------------------------------------------------------------+
```

### High-quality bar

An entry counts toward the PR threshold only if **all** of:

- Confidence **≥ 80**
- Clearly not already in `README.md` (normalized URL dedupe)
- README present, identifiable purpose, reasonable activity (e.g. commit in
  last ~90 days unless a major lab release)
- Non-trivial impact: stars > 30, notable org, benchmark/dataset/model
  release, or clear SoTA claim with evidence

**Does not count:** “maybe” items, duplicates, generic ML repos with a small
audio demo, empty/stale repos.

---

## PR format (when threshold met)

**Title:** `Daily candidates: YYYY-MM-DD (N entries, conf ≥80)`

**Body must include per entry:**

- Repo name + URL
- **Confidence** (0–100)
- **Category** (path from hierarchy above)
- One-line rationale + impact signal
- Dedupe note (e.g. “not duplicate of X”)

**README diff:** only entries with confidence ≥ 80, in list format:

```markdown
- [Name](url): short description
```

**PR body template** (markdown for GitHub PR description — pipe table OK there):

```markdown
## Summary
- Run date: YYYY-MM-DD
- Candidates scanned: X
- High-quality (≥80): N
- PR opened because N ≥ 1

**This PR is proposal-only. Do not auto-merge.**

## Proposed additions

| Repo | Confidence | Category | One-line rationale |
|------|------------|----------|-------------------|
| [name](url) | 92 | Speech > Synthesis | Open TTS from X; active, 2k★, not in list |

### Entry details

#### 1. [name](url) — **92** — `Speech > Synthesis`
- **Why:** ...
- **Impact:** ...
- **Dedupe:** not duplicate of ...

## Skipped
- repo — reason (off-topic / duplicate / stale)
```

---

## When threshold is not met (zero finds)

- **No PR**
- **No branch**
- **No README changes**
- **Silent exit:** log in automation run output only

---

## Agent instructions (prompt)

```
At the start of each run, read automation/SPEC.md and follow it exactly.

You maintain a curated awesome-list for yyf/ai-audio-tools. You run on a
schedule with strict limits. Base branch is main.

SCOPE
- Work only in yyf/ai-audio-tools.
- Do not access other repos, paths, or accounts beyond read-only GitHub search.

SEARCH (GitHub only — v1)
- Audio/music/speech AI only.
- Max 5 GitHub queries, 10 results each, 30 candidates total, 7-day recency window.
- Star floor: stars:>30 for unknown authors.
- No Hugging Face, arXiv, web fetch, MCP, or non-GitHub discovery.
- No bulk crawls, pagination beyond caps, or recursive link following.

README
- Read README.md first; dedupe by normalized URL.
- Never modify main. Never merge. Never force-push.
- Write README only on bot/daily-candidates-YYYY-MM-DD when opening a PR.

QUALITY GATE
- Score 0-100 and assign one existing category per candidate.
- High-quality = confidence >= 80, in-scope, deduped, clear README/activity/impact.
- Open a PR only if high-quality count >= 1.
- Otherwise exit with no file writes.

PR
- Before opening a new PR, close any open PR from branch bot/daily-candidates-*.
- Include confidence + category for every proposed entry.
- Match list format: - [Name](url): description
- State clearly: proposal only; do not auto-merge.

DENY BY DEFAULT
- If unsure whether an action is allowed, do not do it.
```

---

## Cursor Automation draft

```text
+-------------+-------------------------------------------------------------+
| Field       | Value                                                       |
+-------------+-------------------------------------------------------------+
| Name        | Daily audio-tools repo scout                                  |
| Description | Bounded daily GitHub search for new audio/music/speech      |
|             | repos; opens a PR with scored candidates when >=1           |
|             | high-quality find; never merges.                            |
| Trigger     | Cron: weekdays 9:00 AM America/Los_Angeles                   |
| Tools       | Git (this repo), GitHub search + open/close PR (this repo   |
|             | only)                                                       |
| Instructions| Agent prompt block above; follow automation/SPEC.md exactly  |
|             | each run                                                    |
| Resolved    | Repo: yyf/ai-audio-tools; base: main; bot branch:           |
| settings    | bot/daily-candidates-*                                      |
+-------------+-------------------------------------------------------------+
```

---

## Explicit deny list

The agent must **never**:

- Auto-merge or approve its own PR
- Push directly to `main`
- Edit `README.md` on `main` or without an open PR
- Run unbounded searches or use non-GitHub discovery (HF, arXiv, web crawls)
- Clone external repos or execute their code
- Create secrets, change CI, or modify unrelated files
- Delete branches on `main` or rewrite git history
- Post publicly (Slack/issues) unless explicitly enabled later

---

## v1 configuration (locked)

```text
+-------------+-------------------------------------------------------------+
| Setting     | Value                                                       |
+-------------+-------------------------------------------------------------+
| Discovery   | GitHub search only                                          |
| Star floor  | stars:>30                                                   |
| Recency     | 7 days                                                      |
| window      |                                                             |
| PR          | >= 1 high-quality find (confidence >= 80)                   |
| threshold   |                                                             |
| Schedule    | Weekdays 9:00 AM America/Los_Angeles                        |
| Stale bot   | Close open bot/daily-candidates-* PRs before opening a new  |
| PRs         | one                                                         |
| Sub-        | Silent exit (zero finds)                                    |
| threshold   |                                                             |
| runs        |                                                             |
| Notifications| PR only (GitHub)                                           |
+-------------+-------------------------------------------------------------+
```

> **Note:** Threshold is set to ≥ 1 for initial review. Raise to ≥ 6 later in
> this file and the agent prompt if PR volume is too high.

---

## Changelog

```text
+------------+--------------------------------------------------------------+
| Date       | Change                                                       |
+------------+--------------------------------------------------------------+
| 2026-08-27 | Initial draft spec                                           |
| 2026-08-27 | v1 hardening: GitHub-only, threshold >=1, stars>30, cloud    |
|            | config, status Active                                        |
| 2026-08-27 | ASCII fixed-width tables; document conventions section         |
+------------+--------------------------------------------------------------+
```
