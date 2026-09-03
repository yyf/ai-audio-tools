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

# Domain-first categorization aligned with README ToC:
#   Audio  > Benchmark | Dataset | Annotation | Model | Security
#   Music  > Benchmark | Analysis | Production | Generation
#   Speech > Benchmark | Recognition | Production | Synthesis
#
# Matching is weighted + word-boundary aware for short tokens so generic
# repos do not dump into Audio > Model by substring accidents (e.g. "mir"
# in "mirror", "tts" in unrelated strings).

MIN_CATEGORY_SCORE = 3  # skip candidates that cannot map cleanly to the ToC

# (phrase, weight). Phrases with word chars only and len <= 4 use \b matching.
DOMAIN_SIGNALS: dict[str, list[tuple[str, int]]] = {
    "Speech": [
        ("text-to-speech", 6),
        ("speech-to-text", 6),
        ("speech recognition", 6),
        ("speech synthesis", 6),
        ("voice cloning", 6),
        ("voice conversion", 5),
        ("automatic speech", 5),
        ("speaker verification", 4),
        ("speaker identification", 4),
        ("diarization", 5),
        ("transcription", 4),
        ("audio codec", 5),
        ("neural codec", 5),
        ("speech", 3),
        ("whisper", 3),
        ("vocoder", 4),
        ("encodec", 4),
        ("tts", 5),
        ("asr", 5),
        ("stt", 5),
        ("funasr", 4),
        ("kaldi", 3),
        ("espnet", 3),
        ("coqui", 3),
        ("piper", 3),
        ("codec", 2),
    ],
    "Music": [
        ("text-to-music", 6),
        ("music generation", 6),
        ("audio generation", 5),
        ("text-to-audio", 5),
        ("symbolic music", 5),
        ("music information", 5),
        ("music analysis", 5),
        ("source separation", 5),
        ("musicgen", 5),
        ("musiccaps", 4),
        ("music", 4),
        ("midi", 4),
        ("song", 3),
        ("mir", 3),
        ("daw", 3),
        ("riffusion", 4),
        ("jukebox", 3),
        ("essentia", 3),
        ("librosa", 3),
        ("demucs", 4),
        ("spleeter", 4),
    ],
    "Audio": [
        ("audio language model", 5),
        ("audio-language", 5),
        ("audio llm", 5),
        ("audio foundation", 4),
        ("audio model", 4),
        ("sound event", 4),
        ("audio tagging", 4),
        ("audio watermark", 5),
        ("steganography", 4),
        ("deepfake", 5),
        ("synthetic voice", 5),
        ("synthesized voice", 5),
        ("voice detection", 5),
        ("clap", 4),
        ("hear benchmark", 4),
        ("audio dataset", 4),
        ("audio", 2),
        ("sound", 2),
        ("foley", 3),
        ("forensic", 3),
        ("watermark", 4),
    ],
}

# Subsection signals are scored only after a domain is chosen.
SUBSECTION_SIGNALS: dict[tuple[str, str], list[tuple[str, int]]] = {
    ("Audio", "Benchmark"): [
        ("leaderboard", 5),
        ("benchmark", 4),
        ("evaluation", 3),
        ("hear", 3),
        ("sota", 2),
    ],
    ("Audio", "Dataset"): [
        ("dataset", 5),
        ("corpus", 4),
        ("collections of audio", 3),
    ],
    ("Audio", "Annotation"): [
        ("annotation", 5),
        ("labeling", 4),
        ("labelling", 4),
        ("label tool", 4),
        ("data augmentation", 3),
        ("audiomentations", 4),
    ],
    ("Audio", "Model"): [
        ("audio language model", 6),
        ("audio-language", 5),
        ("audio llm", 5),
        ("audio foundation", 5),
        ("audio model", 4),
        ("foundation model", 3),
        ("clap", 4),
        ("embedding", 2),
    ],
    ("Audio", "Security"): [
        ("watermark", 6),
        ("steganography", 6),
        ("deepfake", 5),
        ("synthetic voice", 5),
        ("synthesized voice", 5),
        ("voice detection", 5),
        ("forensic", 4),
        ("authenticity", 3),
        ("spoofing", 4),
    ],
    ("Music", "Benchmark"): [
        ("musiccaps", 6),
        ("music benchmark", 6),
        ("music eval", 5),
        ("leaderboard", 3),
        ("benchmark", 3),
        ("evaluation", 2),
    ],
    ("Music", "Analysis"): [
        ("music analysis", 6),
        ("music information", 6),
        ("audio analysis", 4),
        ("feature extraction", 4),
        ("pitch detection", 4),
        ("audio-to-midi", 5),
        ("mir", 4),
        ("tagging", 3),
        ("librosa", 4),
        ("essentia", 4),
        ("madmom", 3),
        ("understanding", 2),
    ],
    ("Music", "Production"): [
        ("source separation", 6),
        ("mastering", 5),
        ("audio effect", 5),
        ("stem separation", 5),
        ("demucs", 5),
        ("spleeter", 5),
        ("mixing", 3),
        ("daw", 4),
        ("foley", 4),
        ("room impulse", 4),
        ("audacity", 3),
    ],
    ("Music", "Generation"): [
        ("music generation", 6),
        ("text-to-music", 6),
        ("midi generation", 5),
        ("symbolic music", 5),
        ("audio generation", 4),
        ("text-to-audio", 4),
        ("musicgen", 5),
        ("song generation", 5),
        ("riffusion", 4),
        ("generate music", 5),
        ("generative music", 5),
    ],
    ("Speech", "Benchmark"): [
        ("speech benchmark", 6),
        ("asr benchmark", 6),
        ("tts benchmark", 6),
        ("leaderboard", 4),
        ("benchmark", 3),
        ("evaluation", 2),
    ],
    ("Speech", "Recognition"): [
        ("speech recognition", 6),
        ("speech-to-text", 6),
        ("automatic speech", 5),
        ("transcription", 5),
        ("diarization", 5),
        ("whisper", 4),
        ("funasr", 4),
        ("kaldi", 4),
        ("asr", 5),
        ("stt", 5),
        ("dictation", 4),
        ("keyword spotting", 4),
    ],
    ("Speech", "Production"): [
        ("voice agent", 6),
        ("conversational ai", 5),
        ("speech pipeline", 5),
        ("audio codec", 5),
        ("neural codec", 5),
        ("encodec", 5),
        ("pipecat", 5),
        ("webrtc", 3),
        ("voice ai platform", 5),
        ("codec", 3),
    ],
    ("Speech", "Synthesis"): [
        ("text-to-speech", 6),
        ("speech synthesis", 6),
        ("voice cloning", 6),
        ("voice conversion", 5),
        ("singing voice", 5),
        ("talking head", 4),
        ("vocoder", 4),
        ("tts", 5),
        ("bark", 3),
        ("so-vits", 4),
        ("openvoice", 4),
    ],
}

# Soft negatives: reduce a domain score when these dominate (cross-domain noise).
DOMAIN_NEGATIVES: dict[str, list[tuple[str, int]]] = {
    "Speech": [
        ("music generation", 4),
        ("text-to-music", 4),
        ("midi", 2),
        ("watermark", 4),
        ("steganography", 4),
        ("deepfake", 5),
        ("voice detection", 5),
        ("forensic", 4),
        ("synthetic voice", 3),
        ("synthesized voice", 3),
    ],
    "Music": [
        ("text-to-speech", 4),
        ("speech-to-text", 4),
        ("voice cloning", 3),
        ("asr", 3),
        ("tts", 2),
        ("watermark", 3),
        ("deepfake", 3),
    ],
    "Audio": [
        ("text-to-speech", 3),
        ("speech-to-text", 3),
        ("music generation", 3),
        ("text-to-music", 3),
        ("audio generation", 3),
    ],
}

DOMAIN_DEFAULT_SUBSECTION: dict[str, str] = {
    "Audio": "Model",
    "Music": "Analysis",
    "Speech": "Recognition",
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


def _phrase_matches(blob: str, phrase: str) -> bool:
    """Match phrase in blob; use word boundaries for short alphanumeric tokens."""
    phrase = phrase.lower()
    if " " in phrase or "-" in phrase or len(phrase) > 4:
        return phrase in blob
    return re.search(rf"\b{re.escape(phrase)}\b", blob) is not None


def _weighted_score(blob: str, signals: list[tuple[str, int]]) -> tuple[int, int]:
    """Return (total_weight, number_of_matching_phrases)."""
    total = 0
    hits = 0
    for phrase, weight in signals:
        if _phrase_matches(blob, phrase):
            total += weight
            hits += 1
    return total, hits


def categorize(full_name: str, description: str, topics: list[str]) -> tuple[tuple[str, str], int]:
    """Assign README ToC category via domain-first weighted matching.

    Returns ((Domain, Subsection), category_score). category_score is used both
    as a placement quality signal and for confidence scoring.
    """
    blob = " ".join([full_name, description or "", " ".join(topics)]).lower()

    domain_scores: dict[str, int] = {}
    for domain, signals in DOMAIN_SIGNALS.items():
        score, _ = _weighted_score(blob, signals)
        neg, _ = _weighted_score(blob, DOMAIN_NEGATIVES.get(domain, []))
        domain_scores[domain] = max(0, score - neg)

    # Prefer Speech > Music > Audio on ties — speech signals are more specific.
    domain = max(
        ("Speech", "Music", "Audio"),
        key=lambda d: (domain_scores[d], {"Speech": 3, "Music": 2, "Audio": 1}[d]),
    )
    domain_score = domain_scores[domain]

    best_sub = DOMAIN_DEFAULT_SUBSECTION[domain]
    best_sub_score = 0
    best_sub_hits = 0
    for (dom, subsection), signals in SUBSECTION_SIGNALS.items():
        if dom != domain:
            continue
        sub_score, sub_hits = _weighted_score(blob, signals)
        if sub_score > best_sub_score or (
            sub_score == best_sub_score and sub_hits > best_sub_hits
        ):
            best_sub_score = sub_score
            best_sub_hits = sub_hits
            best_sub = subsection

    # Combined score: need real domain signal; subsection adds placement confidence.
    category_score = domain_score + best_sub_score
    return (domain, best_sub), category_score


def score_repo(
    repo: dict,
    category: tuple[str, str],
    category_score: int,
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
    if category_score >= MIN_CATEGORY_SCORE:
        score += 8
        reasons.append(f"topic fit ({category[0]} > {category[1]})")
    if category_score >= MIN_CATEGORY_SCORE + 4:
        score += 4
    if category_score >= MIN_CATEGORY_SCORE + 8:
        score += 3
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

            category, category_score = categorize(
                full_name,
                repo.get("description") or "",
                repo.get("topics") or [],
            )
            if category_score < MIN_CATEGORY_SCORE:
                stats.skipped.append(
                    f"{full_name} — weak ToC fit "
                    f"({category[0]} > {category[1]}, score {category_score} "
                    f"< {MIN_CATEGORY_SCORE})"
                )
                continue

            confidence, rationale = score_repo(repo, category, category_score)

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
