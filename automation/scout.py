#!/usr/bin/env python3
"""Daily sound scout — rule-based GitHub search and PR proposal.

Behavior is defined in automation/SPEC.md (v1, script-only, GitHub Actions).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

# --- v1 locked config (keep in sync with automation/SPEC.md) ---

MAX_QUERIES = 5
MAX_RESULTS_PER_QUERY = 10
MAX_CANDIDATES = 30
MIN_STARS = 10
RECENCY_DAYS = 90
HIGH_QUALITY_THRESHOLD = 40
PR_THRESHOLD = 1
MAX_PR_ENTRIES = 10
BOT_BRANCH_PREFIX = "bot/daily-candidates-"
README_PATH = Path("README.md")
QUERIES_PATH = Path("automation/queries.json")
REJECTED_PATH = Path("automation/rejected.md")

NOTABLE_ORGS = {
    "nvidia",
    "meta",
    "facebookresearch",
    "google",
    "google-deepmind",
    "microsoft",
    "openai",
    "huggingface",
    "pytorch",
    "alibaba-damo-academy",
    "apple",
    "amazon",
    "stability-ai",
    "spotify",
    "adobe-research",
    "bytedance",
    "deepmind",
}

CATEGORY_KEYWORDS: dict[tuple[str, str], list[str]] = {
    ("Audio", "Benchmark"): ["benchmark", "leaderboard", "evaluation", "hear"],
    ("Audio", "Dataset"): ["dataset", "corpus", "benchmark dataset"],
    ("Audio", "Annotation"): ["annotation", "labeling", "label tool"],
    ("Audio", "Model"): ["audio model", "audio llm", "audio foundation", "clap"],
    ("Audio", "Security"): [
        "watermark",
        "steganography",
        "deepfake detection",
        "voice detection",
        "synthetic voice",
        "forensic",
    ],
    ("Music", "Benchmark"): ["music benchmark", "musiccaps", "music eval"],
    ("Music", "Analysis"): [
        "music analysis",
        "mir",
        "music information",
        "audio analysis",
        "librosa",
        "essentia",
        "tagging",
    ],
    ("Music", "Production"): [
        "daw",
        "mastering",
        "source separation",
        "spleeter",
        "demucs",
        "audio effect",
        "mixing",
    ],
    ("Music", "Generation"): [
        "music generation",
        "text-to-music",
        "musicgen",
        "midi generation",
        "symbolic music",
        "audio generation",
        "riff",
        "suno",
    ],
    ("Speech", "Benchmark"): ["speech benchmark", "asr benchmark", "tts benchmark"],
    ("Speech", "Recognition"): [
        "asr",
        "speech recognition",
        "speech-to-text",
        "stt",
        "transcription",
        "whisper",
        "funasr",
        "kaldi",
    ],
    ("Speech", "Production"): [
        "voice agent",
        "conversational",
        "pipecat",
        "codec",
        "encodec",
        "speech pipeline",
    ],
    ("Speech", "Synthesis"): [
        "tts",
        "text-to-speech",
        "voice cloning",
        "vocoder",
        "speech synthesis",
        "talking head",
        "singing voice",
    ],
}


@dataclass
class Candidate:
    full_name: str
    html_url: str
    description: str
    stars: int
    pushed_at: datetime
    category: tuple[str, str]
    confidence: int
    rationale: str
    dedupe_note: str = "not in README"


@dataclass
class RunStats:
    scanned: int = 0
    skipped: list[str] = field(default_factory=list)


def log(msg: str) -> None:
    print(msg, flush=True)


def github_request(path: str, method: str = "GET", body: dict | None = None) -> dict | list:
    token = os.environ["GITHUB_TOKEN"]
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    url = f"https://api.github.com{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "ai-audio-tools-scout",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = resp.read().decode()
            return json.loads(payload) if payload else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"GitHub API {method} {path} failed ({exc.code}): {detail}") from exc


def normalize_github_repo_url(url: str) -> str | None:
    match = re.search(r"github\.com/([^/\s?#]+/[^/\s?#]+)", url, re.I)
    if not match:
        return None
    owner_repo = match.group(1).lower().removesuffix(".git")
    return f"https://github.com/{owner_repo}"


def load_existing_repo_keys(readme: str) -> set[str]:
    keys: set[str] = set()
    for raw in re.findall(r"https?://[^\s)>]+", readme):
        gh = normalize_github_repo_url(raw)
        if gh:
            keys.add(gh)
            keys.add(gh.split("github.com/", 1)[1])
    return keys


def load_rejected_keys() -> set[str]:
    if not REJECTED_PATH.exists():
        return set()
    return load_existing_repo_keys(REJECTED_PATH.read_text(encoding="utf-8"))


def parse_queries() -> list[str]:
    templates = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    return [q.replace("{date}", "") for q in templates[:MAX_QUERIES]]


def search_repositories(query: str) -> list[dict]:
    params = urllib.parse.urlencode(
        {
            "q": query,
            "sort": "updated",
            "order": "desc",
            "per_page": str(MAX_RESULTS_PER_QUERY),
        }
    )
    result = github_request(f"/search/repositories?{params}")
    return result.get("items", [])


def categorize(full_name: str, description: str, topics: list[str]) -> tuple[tuple[str, str], int]:
    blob = " ".join([full_name, description or "", " ".join(topics)]).lower()
    best_cat = ("Audio", "Model")
    best_hits = 0
    for category, keywords in CATEGORY_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in blob)
        if hits > best_hits:
            best_hits = hits
            best_cat = category
    return best_cat, best_hits


def score_repo(
    repo: dict,
    category: tuple[str, str],
    keyword_hits: int,
) -> tuple[int, str]:
    stars = repo.get("stargazers_count", 0)
    description = (repo.get("description") or "").strip()
    owner = repo.get("owner", {}).get("login", "").lower()
    pushed_raw = repo.get("pushed_at") or ""
    pushed_at = datetime.fromisoformat(pushed_raw.replace("Z", "+00:00"))
    recency_cutoff = datetime.now(timezone.utc) - timedelta(days=RECENCY_DAYS)

    score = 0
    reasons: list[str] = []

    if stars > MIN_STARS:
        score += 15
        reasons.append(f"{stars} stars")
    if stars >= 30:
        score += 8
    if stars >= 100:
        score += 5
    if stars >= 500:
        score += 5
    if pushed_at >= recency_cutoff:
        score += 15
        reasons.append(f"pushed in last {RECENCY_DAYS} days")
    if len(description) >= 5:
        score += 10
        reasons.append("has description")
    if len(description) >= 20:
        score += 5
    if keyword_hits >= 1:
        score += 8
        reasons.append(f"topic fit ({category[0]} > {category[1]})")
    if keyword_hits >= 2:
        score += 4
    if owner in NOTABLE_ORGS:
        score += 5
        reasons.append(f"notable org ({owner})")

    score = min(score, 100)
    rationale = "; ".join(reasons) if reasons else "matched search filters"
    return score, rationale


def is_duplicate(full_name: str, html_url: str, existing: set[str]) -> bool:
    key = html_url.lower().rstrip("/")
    short = full_name.lower()
    return key in existing or short in existing


def collect_candidates(existing: set[str], rejected: set[str], stats: RunStats) -> list[Candidate]:
    seen: set[str] = set()
    candidates: list[Candidate] = []

    for query in parse_queries():
        log(f"Search query: {query}")
        try:
            items = search_repositories(query)
        except RuntimeError as exc:
            stats.skipped.append(f"query failed — {exc}")
            continue

        if not items:
            stats.skipped.append(f"query returned 0 results — {query[:60]}")
            continue

        for repo in items:
            if len(seen) >= MAX_CANDIDATES:
                break
            if repo.get("fork"):
                stats.skipped.append(f"{repo.get('full_name')} — fork")
                continue
            if repo.get("archived"):
                stats.skipped.append(f"{repo.get('full_name')} — archived")
                continue

            full_name = repo["full_name"]
            html_url = repo["html_url"].rstrip("/")
            if full_name in seen:
                continue
            seen.add(full_name)
            stats.scanned += 1

            if is_duplicate(full_name, html_url, existing):
                stats.skipped.append(f"{full_name} — duplicate of README entry")
                continue
            if is_duplicate(full_name, html_url, rejected):
                stats.skipped.append(f"{full_name} — rejected list")
                continue

            category, keyword_hits = categorize(
                full_name,
                repo.get("description") or "",
                repo.get("topics") or [],
            )
            confidence, rationale = score_repo(repo, category, keyword_hits)

            if confidence < HIGH_QUALITY_THRESHOLD:
                stats.skipped.append(
                    f"{full_name} — confidence {confidence} (< {HIGH_QUALITY_THRESHOLD})"
                )
                continue

            candidates.append(
                Candidate(
                    full_name=full_name,
                    html_url=html_url,
                    description=(repo.get("description") or full_name.split("/")[-1]).strip(),
                    stars=repo.get("stargazers_count", 0),
                    pushed_at=datetime.fromisoformat(
                        (repo.get("pushed_at") or datetime.now(timezone.utc).isoformat()).replace(
                            "Z", "+00:00"
                        )
                    ),
                    category=category,
                    confidence=confidence,
                    rationale=rationale,
                )
            )

        if len(seen) >= MAX_CANDIDATES:
            break

    candidates.sort(key=lambda c: (-c.confidence, -c.stars))
    return candidates[:MAX_PR_ENTRIES]


def format_readme_line(candidate: Candidate) -> str:
    name = candidate.full_name.split("/")[-1]
    desc = candidate.description.rstrip(".")
    return f"- [{name}]({candidate.html_url}): {desc}"


def insert_readme_entries(readme: str, entries: list[tuple[tuple[str, str], str]]) -> str:
    lines = readme.splitlines(keepends=True)
    output = list(lines)

    # Insert from bottom to top so indices stay valid.
    grouped: dict[tuple[str, str], list[str]] = {}
    for category, line in entries:
        grouped.setdefault(category, []).append(line)

    for category in sorted(grouped.keys(), key=lambda c: (c[0], c[1]), reverse=True):
        domain, subsection = category
        new_lines = [ln + "\n" for ln in grouped[category]]
        insert_at = find_section_insert_index(output, domain, subsection)
        output[insert_at:insert_at] = new_lines

    return "".join(output)


def find_section_insert_index(lines: list[str], domain: str, subsection: str) -> int:
    in_domain = False
    in_section = False
    for i, line in enumerate(lines):
        stripped = line.rstrip("\n")
        if stripped == f"# {domain}":
            in_domain = True
            in_section = False
            continue
        if not in_domain:
            continue
        if stripped == f"## {subsection}":
            in_section = True
            continue
        if in_section and (stripped.startswith("## ") or stripped.startswith("# ")):
            return i
    if in_section:
        return len(lines)
    raise ValueError(f"README section not found: {domain} > {subsection}")


def close_open_bot_prs() -> None:
    owner, repo = os.environ["GITHUB_REPOSITORY"].split("/")
    pulls = github_request(f"/repos/{owner}/{repo}/pulls?state=open&per_page=100")
    for pr in pulls:
        head = pr.get("head", {}).get("ref", "")
        if head.startswith(BOT_BRANCH_PREFIX):
            num = pr["number"]
            log(f"Closing open bot PR #{num} ({head})")
            github_request(
                f"/repos/{owner}/{repo}/pulls/{num}",
                method="PATCH",
                body={"state": "closed"},
            )


def build_pr_body(run_date: str, stats: RunStats, candidates: list[Candidate]) -> str:
    lines = [
        "## Summary",
        f"- Run date: {run_date}",
        f"- Candidates scanned: {stats.scanned}",
        f"- High-quality (≥{HIGH_QUALITY_THRESHOLD}): {len(candidates)}",
        f"- PR opened because {len(candidates)} ≥ {PR_THRESHOLD}",
        "",
        "**This PR is proposal-only. Do not auto-merge.**",
        "",
        "## Proposed additions",
        "",
        "| Repo | Confidence | Category | One-line rationale |",
        "|------|------------|----------|-------------------|",
    ]
    for c in candidates:
        cat = f"{c.category[0]} > {c.category[1]}"
        lines.append(
            f"| [{c.full_name}]({c.html_url}) | {c.confidence} | {cat} | {c.rationale} |"
        )
    lines.extend(["", "### Entry details", ""])
    for i, c in enumerate(candidates, 1):
        cat = f"{c.category[0]} > {c.category[1]}"
        lines.extend(
            [
                f"#### {i}. [{c.full_name}]({c.html_url}) — **{c.confidence}** — `{cat}`",
                f"- **Why:** {c.rationale}",
                f"- **Impact:** {c.stars} stars; updated {c.pushed_at.date().isoformat()}",
                f"- **Dedupe:** {c.dedupe_note}",
                "",
            ]
        )
    if stats.skipped:
        lines.extend(["## Skipped", ""])
        lines.extend(f"- {item}" for item in stats.skipped[:50])
    return "\n".join(lines)


def git(*args: str) -> None:
    subprocess.run(["git", *args], check=True)


def create_pr(branch: str, title: str, body: str) -> None:
    owner, repo = os.environ["GITHUB_REPOSITORY"].split("/")
    try:
        result = github_request(
            f"/repos/{owner}/{repo}/pulls",
            method="POST",
            body={
                "title": title,
                "head": branch,
                "base": "main",
                "body": body,
            },
        )
        log(f"Opened PR #{result['number']}: {result['html_url']}")
    except RuntimeError as exc:
        if "403" not in str(exc):
            raise
        compare = f"https://github.com/{owner}/{repo}/compare/main...{branch}?expand=1"
        log("ERROR: GitHub Actions is not permitted to create pull requests.")
        log("Enable: Settings → Actions → General → Workflow permissions")
        log("      → Allow GitHub Actions to create and approve pull requests")
        log(f"Branch pushed — open a PR manually: {compare}")
        raise SystemExit(1) from exc


def write_pr_artifacts(branch: str, title: str, body: str) -> None:
    """Write PR metadata for a follow-up workflow step (survives on runner)."""
    temp = Path(os.environ.get("RUNNER_TEMP", "."))
    temp.mkdir(parents=True, exist_ok=True)
    (temp / "scout-pr-body.md").write_text(body, encoding="utf-8")
    (temp / "scout-pr-branch.txt").write_text(branch, encoding="utf-8")
    (temp / "scout-pr-title.txt").write_text(title, encoding="utf-8")
    log(f"PR artifacts written to {temp}")


def main() -> int:
    if not README_PATH.exists():
        log("README.md not found")
        return 1

    readme = README_PATH.read_text(encoding="utf-8")
    existing = load_existing_repo_keys(readme)
    rejected = load_rejected_keys()
    stats = RunStats()
    candidates = collect_candidates(existing, rejected, stats)

    log(f"Candidates scanned: {stats.scanned}")
    log(f"High-quality candidates: {len(candidates)}")
    if len(candidates) < PR_THRESHOLD:
        log("Threshold not met — exiting without PR")
        if stats.skipped:
            log("Skipped summary:")
            for item in stats.skipped[:30]:
                log(f"  - {item}")
        return 0

    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    branch = f"{BOT_BRANCH_PREFIX}{run_date}"

    close_open_bot_prs()

    entries = [(c.category, format_readme_line(c)) for c in candidates]
    updated_readme = insert_readme_entries(readme, entries)
    README_PATH.write_text(updated_readme, encoding="utf-8")

    git("config", "user.name", "github-actions[bot]")
    git("config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com")
    git("checkout", "-B", branch)
    git("add", str(README_PATH))
    git(
        "commit",
        "-m",
        f"Daily candidates: {run_date} ({len(candidates)} entries, conf >={HIGH_QUALITY_THRESHOLD})",
    )
    git("push", "--force", "origin", branch)

    title = f"Daily candidates: {run_date} ({len(candidates)} entries, conf ≥{HIGH_QUALITY_THRESHOLD})"
    body = build_pr_body(run_date, stats, candidates)
    write_pr_artifacts(branch, title, body)

    if os.environ.get("SCOUT_SKIP_PR") == "1":
        log("SCOUT_SKIP_PR=1 — branch pushed; PR step runs in workflow")
        return 0

    create_pr(branch, title, body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
