# Daily repo scout — automation spec

> **Status:** Active
>
> Defines a **GitHub Actions** workflow for discovering new audio/music/speech
> AI repositories and proposing README additions via pull request only.
> Runs on GitHub-hosted runners — no Cursor Cloud Agent, no local machine.

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
| Schedule    | Weekdays 9:00 AM America/Los_Angeles (GitHub Actions cron)    |
| Runner      | GitHub-hosted ubuntu-latest                                   |
| Script      | automation/scout.py (rule-based, no LLM)                      |
| Outcome     | Silent exit when zero high-quality finds; otherwise one PR    |
|             | with scored candidates for maintainer merge                   |
+-------------+-------------------------------------------------------------+
```

---

## GitHub Actions execution

```text
+-------------+-------------------------------------------------------------+
| Component   | Where                                                       |
+-------------+-------------------------------------------------------------+
| Workflow    | .github/workflows/daily-repo-scout.yml                      |
| Scheduled   | GitHub Actions cron (weekdays)                                |
| run         |                                                             |
| Discovery   | automation/scout.py + GitHub Search API                     |
| Queries     | automation/queries.json                                     |
| Review      | GitHub PR UI (maintainer only)                              |
| and merge   |                                                             |
+-------------+-------------------------------------------------------------+
```

**Prerequisites before first run:**

- [ ] This file and workflow committed on `main`
- [ ] Branch protection on `main` (PR required, no force push, maintainer approval)
- [ ] Actions enabled for the repo (Settings → Actions)
- [ ] Workflow permissions: **Read and write** + **Allow GitHub Actions to
      create and approve pull requests** (Settings → Actions → General)
- [ ] Manual dry run via **Actions → Daily repo scout → Run workflow**

**One-time repo setting (required for PR creation):**

GitHub → **Settings → Actions → General → Workflow permissions**

1. Select **Read and write permissions**
2. Check **Allow GitHub Actions to create and approve pull requests**
3. Save

Without step 2, the workflow can push the bot branch but fails with HTTP 403
when opening the PR.

**Manual dry run:**

1. GitHub → **Actions** → **Daily repo scout** → **Run workflow**
2. Zero finds → green run, no PR (expected most days)
3. One or more finds → bot branch + PR with confidence/category table
4. Close test PR without merging if undesired

---

## Strict capability rules (non-negotiable)

### 1. Project scope only

```text
+---------------------------+-----------------------------------------------+
| Allowed                   | Forbidden                                     |
+---------------------------+-----------------------------------------------+
| Read/write README in this | Modify other repositories                     |
| repo on a bot branch only |                                             |
| Read automation/queries.  | Clone or execute external repo code           |
| json, automation/rejected |                                             |
| .md, this spec            |                                             |
| GitHub Search API         | Private repo inventory, unscoped org sweeps   |
| (read-only)               |                                             |
| Open/close PRs in this    | Push to main, merge PRs, delete main history  |
| repo only                 |                                             |
+---------------------------+-----------------------------------------------+
```

### 2. Search scope — GitHub only, bounded

**v1 discovery:** GitHub repository search **only** via `automation/scout.py`.
No Hugging Face, arXiv, web crawls, or LLM calls.

**Search limits (hard caps per run):**

```text
+-------------+-------------------------------------------------------------+
| Limit       | Value                                                       |
+-------------+-------------------------------------------------------------+
| GitHub      | <= 5 queries (automation/queries.json)                      |
| search      |                                                             |
| Results per | <= 10 repos per query                                       |
| query       |                                                             |
| Max         | <= 30 unique candidates total                               |
| candidates  |                                                             |
| Star floor  | stars:>10 (recency filtered in script)                      |
| Time window | pushed within last 90 days (script-side filter)             |
| Crawl depth | No pagination beyond caps, no link following                |
+-------------+-------------------------------------------------------------+
```

### 3. README — never overwrite without approval

```text
+-------------+-------------------------------------------------------------+
| Rule        | Behavior                                                    |
+-------------+-------------------------------------------------------------+
| Never edit  | All README changes on bot/daily-candidates-YYYY-MM-DD only    |
| main        |                                                             |
| Never merge | Workflow opens PR; maintainer merges manually                 |
| Never       | No force-push to main                                       |
| force-push  |                                                             |
| Approval =  | README on main changes only after PR merge                  |
| merge       |                                                             |
| Zero finds  | No branch, no README write, no PR                           |
+-------------+-------------------------------------------------------------+
```

### 4. Workflow permissions — least privilege

```text
+-------------+-------------------------------------------------------------+
| Enabled     | Disabled / not used                                           |
+-------------+-------------------------------------------------------------+
| GITHUB_TOKEN| Auto-merge                                                    |
| contents:   | Push to main                                                  |
| write       | External API keys / LLM secrets                               |
| pull-       | Slack, email, issue creation                                  |
| requests:   | Cursor Cloud Agent                                            |
| write       |                                                             |
+-------------+-------------------------------------------------------------+
```

**Recommended branch protection on `main`:**

- Require maintainer approval before merge
- Block force pushes
- Require PR for all changes

---

## Discovery and quality logic

### Pipeline (`automation/scout.py`)

1. Load `README.md` → normalized GitHub URL dedupe set
2. Load optional `automation/rejected.md` skip list
3. Run bounded searches from `automation/queries.json`
4. Rule-based **confidence** score (0–100) and **category** assignment
5. Keep candidates with confidence **≥ 40**
6. **If count = 0 →** exit 0 (no PR)
7. **If count ≥ 1 →** close open `bot/daily-candidates-*` PRs, create bot
   branch, insert README lines, push, open PR

### Category paths (must match README)

```
Audio  > Benchmark | Dataset | Annotation | Model | Security
Music  > Benchmark | Analysis | Production | Generation
Speech > Benchmark | Recognition | Production | Synthesis
```

Category assignment uses keyword matching on repo name, description, and
GitHub topics (see `CATEGORY_KEYWORDS` in `automation/scout.py`).

### Confidence scoring (rule-based)

```text
+-------------+-------------------------------------------------------------+
| Score       | Meaning                                                     |
+-------------+-------------------------------------------------------------+
| 90-100      | Strong fit; merge with little or no edit                    |
| 80-89       | Good fit; included in PR                                    |
| 60-79       | Moderate fit; included in PR for review                     |
| 40-59       | Borderline; included in PR — scrutinize before merge        |
| < 40        | Skip                                                        |
+-------------+-------------------------------------------------------------+
```

**Score inputs:** stars (>10 base, tiers at 30/100/500), recency (90 days,
script-side), description quality (≥5 chars), keyword/topic fit, notable org
list. Search queries do **not** use `pushed:` filters (too restrictive);
recency is enforced in `score_repo()`.

**High-quality bar (≥ 40):** in-scope, deduped, not archived/fork, passes
score threshold.

---

## PR format (when threshold met)

**Title:** `Daily candidates: YYYY-MM-DD (N entries, conf ≥40)`

**README diff:** only entries with confidence ≥ 40:

```markdown
- [Name](url): short description
```

**PR body:** summary table with confidence + category per entry, entry
details, and skipped list (generated by `automation/scout.py`).

**Footer:** `This PR is proposal-only. Do not auto-merge.`

---

## When threshold is not met (zero finds)

- **No PR**
- **No branch**
- **No README changes**
- Workflow completes successfully; see Actions log for skipped summary

---

## Repository files

```text
+-------------+-------------------------------------------------------------+
| File        | Role                                                        |
+-------------+-------------------------------------------------------------+
| automation/ | This spec — policy and configuration reference              |
| SPEC.md     |                                                             |
| automation/ | Rule-based scout script                                     |
| scout.py    |                                                             |
| automation/ | Search query templates (recency filtered in scout.py)       |
| queries.json|                                                             |
| automation/ | Optional: repos to never suggest again                      |
| rejected.md |                                                             |
| .github/    | Scheduled + manual workflow                                 |
| workflows/  |                                                             |
| daily-repo- |                                                             |
| scout.yml   |                                                             |
+-------------+-------------------------------------------------------------+
```

---

## Explicit deny list

The workflow and script must **never**:

- Auto-merge or approve its own PR
- Push directly to `main`
- Edit `README.md` on `main` without an open PR
- Use non-GitHub discovery or LLM scoring
- Clone external repos or run their code
- Exceed search caps in `queries.json` / `scout.py`
- Create secrets in the repo

---

## v1 configuration (locked)

```text
+-------------+-------------------------------------------------------------+
| Setting     | Value                                                       |
+-------------+-------------------------------------------------------------+
| Execution   | GitHub Actions only (no Cursor Cloud Agent)                 |
| Discovery   | GitHub search API, script-only scoring                        |
| Star floor  | stars:>10 in search; recency in script                      |
| Recency     | 90 days (pushed), enforced in scout.py                       |
| window      |                                                             |
| PR          | >= 1 high-quality find (confidence >= 40)                   |
| threshold   |                                                             |
| Max per PR  | 10 entries (top by confidence; see MAX_PR_ENTRIES)          |
| Schedule    | Weekdays ~9:00 AM America/Los_Angeles (cron 17:00 UTC)       |
| Stale bot   | Close open bot/daily-candidates-* PRs before new one        |
| PRs         |                                                             |
| Sub-        | Silent success exit (zero finds)                            |
| threshold   |                                                             |
| runs        |                                                             |
| Notifications| PR only (GitHub)                                           |
+-------------+-------------------------------------------------------------+
```

> **Note:** Confidence floor is ≥ 40 for review-friendly PR volume. Raise
> `HIGH_QUALITY_THRESHOLD` in `automation/scout.py` (and this file) if PR
> volume is too high.

> **Note:** Cron is UTC-only on GitHub. `0 17 * * 1-5` ≈ 9:00 AM PST /
> 10:00 AM PDT. Adjust workflow cron if needed.

---

## Tuning

- **Queries:** edit `automation/queries.json` (max 5 used)
- **Keywords / scoring:** edit `CATEGORY_KEYWORDS` and scoring in
  `automation/scout.py`
- **Reject list:** add URLs to `automation/rejected.md`
- **Threshold / caps:** constants at top of `automation/scout.py`

---

## Changelog

```text
+------------+--------------------------------------------------------------+
| Date       | Change                                                       |
+------------+--------------------------------------------------------------+
| 2026-08-27 | Initial draft (Cursor Automation)                            |
| 2026-08-27 | v1 hardening: GitHub-only, threshold >=1, stars>30           |
| 2026-08-27 | ASCII fixed-width tables                                     |
| 2026-08-27 | Switched to GitHub Actions + automation/scout.py; removed    |
|            | Cursor Cloud Agent path                                      |
| 2026-08-27 | Loosen filters: confidence >=60, 30-day recency, scoring   |
| 2026-08-27 | Fix empty search (drop pushed: from queries); floor >=40,  |
|            | 90-day recency, stars>10                                    |
| 2026-08-27 | Fix GHA PR 403: split PR step, cap 10 entries, docs for     |
|            | workflow permissions setting                                  |
+------------+--------------------------------------------------------------+
```
