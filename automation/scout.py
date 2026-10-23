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

MAX_QUERIES = 8
MAX_RESULTS_PER_QUERY = 10
MAX_CANDIDATES = 30
MIN_STARS = 10
RECENCY_DAYS = 90
HIGH_QUALITY_THRESHOLD = 55
KEEP_THRESHOLD = 70  # PR body: Keep (≥70) vs Review (55–69)
PR_THRESHOLD = 1
MAX_PR_ENTRIES = 10
MAX_PER_SECTION = 3  # max entries per Domain > Subsection in one PR
# Reserve PR slots for Audio|Music|Speech > Model (latest open-source models task).
MIN_MODEL_PR_SLOTS = 3
# Reserve PR slots for Commercial OK > Model (friendly-license featured demos).
MIN_COMMERCIAL_OK_PR_SLOTS = 2
MIN_DESCRIPTION_LEN = 15  # skip empty / placeholder descriptions
BOT_BRANCH_PREFIX = "bot/daily-candidates-"
README_PATH = Path("README.md")
QUERIES_PATH = Path("automation/queries.json")
REJECTED_PATH = Path("automation/rejected.md")

# Featured demos / Commercial OK — repo SPDX from GitHub (weights may differ).
COMMERCIAL_OK_DOMAIN = "Commercial OK"
COMMERCIAL_OK_SECTION = (COMMERCIAL_OK_DOMAIN, "Model")
FRIENDLY_LICENSES = {
    "mit",
    "apache-2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "isc",
    "unlicense",
    "0bsd",
    "cc0-1.0",
}
# Preferred for Commercial OK ranking / scoring (subset of FRIENDLY_LICENSES).
PREFERRED_COMMERCIAL_LICENSES = {"mit", "apache-2.0"}
OK_WITH_CARE_LICENSES = {
    "cc-by-4.0",
    "mpl-2.0",
    "lgpl-2.0",
    "lgpl-2.1",
    "lgpl-3.0",
}
# Topics used by the daily Commercial OK discovery / routing
# (audio + music + any speech: ASR, TTS, STS, diarization, etc.).
COMMERCIAL_OK_TOPICS = {
    "ai-audio",
    "text-to-speech",
    "speech-to-text",
    "speech-recognition",
    "automatic-speech-recognition",
    "speech-synthesis",
    "speech-to-speech",
    "speech-translation",
    "voice-cloning",
    "voice-conversion",
    "speaker-diarization",
    "diarization",
    "asr",
    "stt",
    "tts",
    "music-generation",
    "text-to-music",
    "text-to-audio",
}

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
#   Commercial OK > Model   (friendly-license featured demos; routed post-score)
#   Audio  > Benchmark | Dataset | Annotation | Model | Security | Framework
#   Music  > Model | Benchmark | Analysis | Production | Generation
#   Speech > Model | Benchmark | Recognition | Production | Synthesis
#
# Matching is weighted + word-boundary aware for short tokens so generic
# repos do not dump into * > Model by substring accidents (e.g. "mir"
# in "mirror", "tts" in unrelated strings).
# * > Model is reserved for foundation / pretrained / language-model repos;
# task toolkits stay in Recognition, Synthesis, Generation, etc.

MIN_CATEGORY_SCORE = 7  # skip candidates that cannot map cleanly to the ToC
# Subsection score required before any Domain > Model assignment.
MIN_MODEL_SECTION_SCORE = 6

# Hard rejects: generic / off-list products that match loose search terms.
# Matched against name + description + topics (same blob as categorization).
NEGATIVE_KEYWORDS: list[str] = [
    "discord bot",
    "telegram bot",
    "music player",
    "music bot",
    "homelab",
    "agent framework",
    "typescript framework",
    "screen reader",
    "stable diffusion",
    "image generation",
    "vscode extension",
    "chrome extension",
    "clipboard",
    "gguf",
    "ollama",
    "party game",
    "music quiz",
    "home assistant",
    "media library",
    "jellyfin",
    "plex plugin",
    "spotify downloader",
    "youtube downloader",
]

# Bonus when GitHub topics include these (cleaner signal than free-text).
TOPIC_BONUS: set[str] = {
    "tts",
    "text-to-speech",
    "speech-recognition",
    "speech-to-text",
    "speech-translation",
    "asr",
    "stt",
    "voice-cloning",
    "voice-conversion",
    "speaker-recognition",
    "diarization",
    "music-generation",
    "text-to-music",
    "text-to-audio",
    "audio-generation",
    "ai-audio",
    "source-separation",
    "music-information-retrieval",
    "mir",
    "audio-watermark",
    "watermarking",
    "deepfake-detection",
    "audio-language-model",
    "clap",
    "neural-audio-codec",
    "speech-synthesis",
    "singing-voice-synthesis",
    "foundation-model",
    "pretrained-model",
    "foley",
    "sound-design",
    "game-audio",
}

# Ideal applications for the latest-models task (soft preference, not a hard filter).
# Prefer models with clear potential in music / film / gaming / new-media production.
PRODUCTION_APPLICATION_SIGNALS: list[tuple[str, int]] = [
    ("music production", 5),
    ("media production", 5),
    ("post-production", 5),
    ("sound design", 5),
    ("game audio", 5),
    ("video game", 4),
    ("new media", 4),
    ("interactive media", 4),
    ("film", 4),
    ("cinema", 4),
    ("cinematic", 4),
    ("soundtrack", 4),
    ("foley", 5),
    ("gaming", 3),
    ("game", 2),
    ("movie", 3),
    ("broadcast", 3),
    ("podcast production", 3),
    ("virtual production", 4),
    ("creative production", 3),
]

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
        ("audio language model", 7),
        ("audio-language model", 7),
        ("audio-language", 5),
        ("audio llm", 6),
        ("audio foundation model", 7),
        ("audio foundation", 5),
        ("pretrained audio model", 6),
        ("foundation model", 4),
        ("audio model", 3),
        ("clap", 5),
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
    ("Audio", "Framework"): [
        ("audio framework", 6),
        ("realtime audio", 5),
        ("real-time audio", 5),
        ("gpu-native", 4),
        ("creative software", 3),
        ("audio engine", 4),
    ],
    ("Music", "Model"): [
        ("music foundation model", 8),
        ("music language model", 7),
        ("music llm", 6),
        ("pretrained music model", 7),
        ("music foundation", 6),
        ("foundation model", 6),
        ("pretrained model", 5),
        ("music model", 3),
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
    ("Speech", "Model"): [
        ("speech foundation model", 8),
        ("speech language model", 7),
        ("speech llm", 6),
        ("pretrained speech model", 7),
        ("large audio language", 6),
        ("speech foundation", 6),
        ("foundation model", 6),
        ("pretrained model", 5),
        ("speech model", 3),
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
    # Never default into Model — Model requires MIN_MODEL_SECTION_SCORE.
    "Audio": "Framework",
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
    category_score: int
    confidence: int
    rationale: str
    dedupe_note: str = "not in README"
    license_spdx: str | None = None
    license_tier: str = "skip"  # friendly | ok_with_care | skip


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


def repo_license_spdx(repo: dict) -> str | None:
    """Return normalized SPDX id from a GitHub repo payload, or None if unknown."""
    lic = repo.get("license") or {}
    raw = (lic.get("spdx_id") or lic.get("key") or "").strip()
    if not raw or raw.upper() in {"NOASSERTION", "NONE", "OTHER"}:
        return None
    return raw


def license_tier(spdx: str | None) -> str:
    """Classify license for featured demos / Commercial OK.

    friendly: MIT, Apache-2.0, BSD-2/3, ISC, Unlicense, 0BSD, CC0
    ok_with_care: CC-BY-4.0, MPL-2.0, LGPL (allowed in general ToC, not Commercial OK)
    skip: GPL/AGPL, CC-BY-NC, CC-BY-SA, RAIL/research-only, unknown/gated
    """
    if not spdx:
        return "skip"
    key = spdx.lower().replace("_", "-")
    if key in FRIENDLY_LICENSES:
        return "friendly"
    if key in OK_WITH_CARE_LICENSES:
        return "ok_with_care"
    if (
        "nc" in key
        or key.startswith("gpl")
        or key.startswith("agpl")
        or "cc-by-sa" in key
        or "rail" in key
    ):
        return "skip"
    return "skip"


def should_route_commercial_ok(
    category: tuple[str, str],
    topics: list[str] | set[str],
    tier: str,
) -> bool:
    """Route friendly-license model / topic hits into Commercial OK > Model."""
    if tier != "friendly":
        return False
    if category[1] == "Model" and category[0] != COMMERCIAL_OK_DOMAIN:
        return True
    topic_set = {t.lower() for t in topics}
    return bool(topic_set & COMMERCIAL_OK_TOPICS)


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


def negative_hit(full_name: str, description: str, topics: list[str]) -> str | None:
    """Return the first negative keyword matched, else None."""
    blob = " ".join([full_name, description or "", " ".join(topics)]).lower()
    for phrase in NEGATIVE_KEYWORDS:
        if _phrase_matches(blob, phrase):
            return phrase
    return None


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

    Domain > Model is gated: only repos with a strong model-section score
    (foundation / pretrained / language-model signals) may land there.
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

    scored: list[tuple[str, int, int]] = []
    for (dom, subsection), signals in SUBSECTION_SIGNALS.items():
        if dom != domain:
            continue
        sub_score, sub_hits = _weighted_score(blob, signals)
        scored.append((subsection, sub_score, sub_hits))

    scored.sort(key=lambda row: (row[1], row[2]), reverse=True)

    if scored and scored[0][1] > 0:
        best_sub, best_sub_score, _ = scored[0]
    else:
        best_sub = DOMAIN_DEFAULT_SUBSECTION[domain]
        best_sub_score = 0

    # Gate Model: task toolkits must not dump into Audio/Music/Speech > Model.
    if best_sub == "Model" and best_sub_score < MIN_MODEL_SECTION_SCORE:
        fallback = next(
            ((sub, score) for sub, score, _ in scored if sub != "Model" and score > 0),
            None,
        )
        if fallback:
            best_sub, best_sub_score = fallback
        else:
            best_sub = DOMAIN_DEFAULT_SUBSECTION[domain]
            best_sub_score = 0

    # Prefer Model over task sections when foundation/LLM framing is present
    # and Model cleared the gate (e.g. "music generation foundation model").
    model_row = next((row for row in scored if row[0] == "Model"), None)
    foundationish = (
        _phrase_matches(blob, "foundation model")
        or _phrase_matches(blob, "language model")
        or _phrase_matches(blob, "audio llm")
        or _phrase_matches(blob, "speech llm")
        or _phrase_matches(blob, "music llm")
        or _phrase_matches(blob, "pretrained model")
        or _phrase_matches(blob, "pretrained speech model")
        or _phrase_matches(blob, "pretrained music model")
        or _phrase_matches(blob, "pretrained audio model")
    )
    if (
        model_row
        and model_row[1] >= MIN_MODEL_SECTION_SCORE
        and best_sub in {"Generation", "Recognition", "Synthesis", "Analysis", "Production"}
        and foundationish
    ):
        best_sub, best_sub_score = model_row[0], model_row[1]

    category_score = domain_score + best_sub_score
    return (domain, best_sub), category_score


def score_repo(
    repo: dict,
    category: tuple[str, str],
    category_score: int,
) -> tuple[int, str]:
    """Score a repo with ToC fit weighted above popularity proxies."""
    stars = repo.get("stargazers_count", 0)
    description = (repo.get("description") or "").strip()
    topics = {t.lower() for t in (repo.get("topics") or [])}
    owner = repo.get("owner", {}).get("login", "").lower()
    pushed_raw = repo.get("pushed_at") or ""
    pushed_at = datetime.fromisoformat(pushed_raw.replace("Z", "+00:00"))
    recency_cutoff = datetime.now(timezone.utc) - timedelta(days=RECENCY_DAYS)

    score = 0
    reasons: list[str] = []

    # Popularity (capped lower so stars alone cannot dominate).
    if stars > MIN_STARS:
        score += 8
        reasons.append(f"{stars} stars")
    if stars >= 30:
        score += 4
    if stars >= 100:
        score += 3
    if stars >= 500:
        score += 3

    if pushed_at >= recency_cutoff:
        score += 10
        reasons.append(f"pushed in last {RECENCY_DAYS} days")

    if len(description) >= MIN_DESCRIPTION_LEN:
        score += 8
        reasons.append("has description")
    if len(description) >= 40:
        score += 4

    # ToC fit is the primary relevance signal.
    if category_score >= MIN_CATEGORY_SCORE:
        score += 12
        reasons.append(f"topic fit ({category[0]} > {category[1]})")
    if category_score >= MIN_CATEGORY_SCORE + 4:
        score += 6
    if category_score >= MIN_CATEGORY_SCORE + 8:
        score += 6
    if category_score >= MIN_CATEGORY_SCORE + 12:
        score += 4

    topic_hits = sorted(topics & TOPIC_BONUS)
    if topic_hits:
        score += 6
        reasons.append(f"github topics ({', '.join(topic_hits[:3])})")

    if owner in NOTABLE_ORGS:
        score += 4
        reasons.append(f"notable org ({owner})")

    # Latest-models task: prefer foundation/pretrained models with production upside.
    if category[1] == "Model":
        blob = " ".join(
            [
                repo.get("full_name") or "",
                description,
                " ".join(topics),
            ]
        ).lower()
        prod_score = 0
        prod_labels: list[str] = []
        for phrase, weight in PRODUCTION_APPLICATION_SIGNALS:
            if _phrase_matches(blob, phrase):
                prod_score += weight
                prod_labels.append(phrase)
        if prod_score >= 3:
            score += min(8, 3 + prod_score // 2)
            reasons.append(f"production upside ({', '.join(prod_labels[:3])})")

    # Commercial OK: prefer MIT / Apache-2.0 (repo license; weights may differ).
    spdx = repo_license_spdx(repo)
    tier = license_tier(spdx)
    if tier == "friendly" and spdx:
        key = spdx.lower().replace("_", "-")
        if key in PREFERRED_COMMERCIAL_LICENSES:
            score += 6
            reasons.append(f"preferred license ({spdx})")
        else:
            score += 3
            reasons.append(f"friendly license ({spdx})")
        if category == COMMERCIAL_OK_SECTION:
            score += 2

    score = min(score, 100)
    rationale = "; ".join(reasons) if reasons else "matched search filters"
    return score, rationale


def is_duplicate(full_name: str, html_url: str, existing: set[str]) -> bool:
    key = html_url.lower().rstrip("/")
    short = full_name.lower()
    return key in existing or short in existing


def _is_commercial_ok_candidate(c: Candidate) -> bool:
    return c.category == COMMERCIAL_OK_SECTION


def _is_model_candidate(c: Candidate) -> bool:
    """Audio|Music|Speech > Model (excludes Commercial OK featured band)."""
    return c.category[1] == "Model" and c.category[0] != COMMERCIAL_OK_DOMAIN


def _production_app_score(c: Candidate) -> int:
    blob = " ".join([c.full_name, c.description]).lower()
    score, _ = _weighted_score(blob, PRODUCTION_APPLICATION_SIGNALS)
    return score


def _preferred_license_rank(c: Candidate) -> int:
    """Lower is better: MIT/Apache first among friendly licenses."""
    if not c.license_spdx:
        return 2
    key = c.license_spdx.lower().replace("_", "-")
    if key in PREFERRED_COMMERCIAL_LICENSES:
        return 0
    if c.license_tier == "friendly":
        return 1
    return 2


def select_for_pr(candidates: list[Candidate], stats: RunStats) -> list[Candidate]:
    """Rank by ToC fit, then confidence; cap per README section.

    Reserves Commercial OK > Model slots (friendly MIT/Apache demos), then
    Audio|Music|Speech > Model slots (latest/popular). Prefer popular + recent
    models; soft-prefer production upside and MIT/Apache licenses.
    """
    ranked = sorted(
        candidates,
        key=lambda c: (-c.category_score, -c.confidence, -c.stars),
    )
    commercial = sorted(
        [c for c in ranked if _is_commercial_ok_candidate(c)],
        key=lambda c: (
            _preferred_license_rank(c),
            -c.stars,
            -c.pushed_at.timestamp(),
            -c.confidence,
        ),
    )
    models = sorted(
        [c for c in ranked if _is_model_candidate(c)],
        key=lambda c: (
            -c.stars,
            -c.pushed_at.timestamp(),
            -_production_app_score(c),
            -c.category_score,
            -c.confidence,
        ),
    )

    selected: list[Candidate] = []
    per_section: dict[tuple[str, str], int] = {}
    selected_names: set[str] = set()

    def try_add(c: Candidate) -> bool:
        if c.full_name in selected_names:
            return False
        if len(selected) >= MAX_PR_ENTRIES:
            return False
        count = per_section.get(c.category, 0)
        if count >= MAX_PER_SECTION:
            stats.skipped.append(
                f"{c.full_name} — section cap "
                f"({c.category[0]} > {c.category[1]} already has {MAX_PER_SECTION})"
            )
            return False
        per_section[c.category] = count + 1
        selected.append(c)
        selected_names.add(c.full_name)
        return True

    # 1) Seed Commercial OK (friendly-license featured demos) first.
    commercial_added = 0
    for c in commercial:
        if commercial_added >= MIN_COMMERCIAL_OK_PR_SLOTS:
            break
        if try_add(c):
            commercial_added += 1

    # 2) Seed Audio|Music|Speech > Model band.
    model_added = 0
    for c in models:
        if model_added >= MIN_MODEL_PR_SLOTS:
            break
        if try_add(c):
            model_added += 1

    # 3) Fill remaining slots with the overall ranked list.
    for c in ranked:
        if len(selected) >= MAX_PR_ENTRIES:
            break
        try_add(c)

    # Order: Commercial OK → ToC models → others.
    selected.sort(
        key=lambda c: (
            0 if _is_commercial_ok_candidate(c) else 1 if _is_model_candidate(c) else 2,
            _preferred_license_rank(c) if _is_commercial_ok_candidate(c) else 0,
            -c.stars if (_is_commercial_ok_candidate(c) or _is_model_candidate(c)) else 0,
            -c.pushed_at.timestamp()
            if (_is_commercial_ok_candidate(c) or _is_model_candidate(c))
            else 0,
            -c.category_score,
            -c.confidence,
            -c.stars,
        )
    )
    return selected


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

            description = (repo.get("description") or "").strip()
            topics = repo.get("topics") or []
            if len(description) < MIN_DESCRIPTION_LEN:
                stats.skipped.append(
                    f"{full_name} — description too short "
                    f"({len(description)} < {MIN_DESCRIPTION_LEN})"
                )
                continue

            bad = negative_hit(full_name, description, topics)
            if bad:
                stats.skipped.append(f"{full_name} — negative keyword ({bad})")
                continue

            category, category_score = categorize(
                full_name,
                description,
                topics,
            )
            if category_score < MIN_CATEGORY_SCORE:
                stats.skipped.append(
                    f"{full_name} — weak ToC fit "
                    f"({category[0]} > {category[1]}, score {category_score} "
                    f"< {MIN_CATEGORY_SCORE})"
                )
                continue

            spdx = repo_license_spdx(repo)
            tier = license_tier(spdx)
            if should_route_commercial_ok(category, topics, tier):
                category = COMMERCIAL_OK_SECTION
                # Keep ToC score; bump slightly so friendly models clear the floor.
                category_score = max(category_score, MIN_CATEGORY_SCORE)

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
                    description=description,
                    stars=repo.get("stargazers_count", 0),
                    pushed_at=datetime.fromisoformat(
                        (repo.get("pushed_at") or datetime.now(timezone.utc).isoformat()).replace(
                            "Z", "+00:00"
                        )
                    ),
                    category=category,
                    category_score=category_score,
                    confidence=confidence,
                    rationale=rationale,
                    license_spdx=spdx,
                    license_tier=tier,
                )
            )

        if len(seen) >= MAX_CANDIDATES:
            break

    return select_for_pr(candidates, stats)


def format_readme_line(candidate: Candidate) -> str:
    name = candidate.full_name.split("/")[-1]
    desc = candidate.description.rstrip(".")
    if _is_commercial_ok_candidate(candidate) and candidate.license_spdx:
        return (
            f"- [{name}]({candidate.html_url}) ({candidate.license_spdx}): {desc}"
        )
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


def _md_cell(text: str, max_len: int = 160) -> str:
    """Sanitize text for a single markdown table cell."""
    cleaned = " ".join((text or "").split())
    if len(cleaned) > max_len:
        cleaned = cleaned[: max_len - 1].rstrip() + "…"
    return cleaned.replace("|", "\\|")


def _sort_models_latest_popular(models: list[Candidate]) -> list[Candidate]:
    """Order models for the dedicated PR section: popular, then recently pushed."""
    return sorted(
        models,
        key=lambda c: (
            -c.stars,
            -c.pushed_at.timestamp(),
            -_production_app_score(c),
            -c.confidence,
        ),
    )


def build_pr_body(run_date: str, stats: RunStats, candidates: list[Candidate]) -> str:
    commercial = sorted(
        [c for c in candidates if _is_commercial_ok_candidate(c)],
        key=lambda c: (
            _preferred_license_rank(c),
            -c.stars,
            -c.pushed_at.timestamp(),
        ),
    )
    models = _sort_models_latest_popular(
        [c for c in candidates if _is_model_candidate(c)]
    )
    # Keep/Review: everything else (excludes Commercial OK + ToC models).
    others = [
        c
        for c in candidates
        if not _is_commercial_ok_candidate(c) and not _is_model_candidate(c)
    ]
    keep = [c for c in others if c.confidence >= KEEP_THRESHOLD]
    review = [c for c in others if c.confidence < KEEP_THRESHOLD]

    lines = [
        "## Summary",
        f"- Run date: {run_date}",
        f"- Candidates scanned: {stats.scanned}",
        f"- High-quality (≥{HIGH_QUALITY_THRESHOLD}): {len(candidates)}",
        f"- Commercial OK (featured demos): {len(commercial)} "
        f"(reserved up to {MIN_COMMERCIAL_OK_PR_SLOTS})",
        f"- Latest & popular models (Audio|Music|Speech > Model): {len(models)} "
        f"(reserved up to {MIN_MODEL_PR_SLOTS})",
        f"- Keep (≥{KEEP_THRESHOLD}): {len(keep)} · Review ({HIGH_QUALITY_THRESHOLD}–{KEEP_THRESHOLD - 1}): {len(review)}",
        f"- PR opened because {len(candidates)} ≥ {PR_THRESHOLD}",
        f"- Section cap: ≤{MAX_PER_SECTION} per Domain > Subsection",
        "",
        "**This PR is proposal-only. Do not auto-merge.**",
        "",
    ]

    def append_table(title: str, rows: list[Candidate]) -> None:
        lines.extend([f"## {title}", ""])
        if not rows:
            lines.extend(["_None._", ""])
            return
        lines.extend(
            [
                "| Repo | Conf | ToC | Category | Description | Rationale |",
                "|------|------|-----|----------|-------------|-----------|",
            ]
        )
        for c in rows:
            cat = f"{c.category[0]} > {c.category[1]}"
            lines.append(
                f"| [{c.full_name}]({c.html_url}) | {c.confidence} | {c.category_score} | "
                f"{cat} | {_md_cell(c.description)} | {_md_cell(c.rationale, 120)} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Commercial OK — featured demos",
            "",
            "Friendly licenses only (MIT, Apache-2.0, BSD-2/3, ISC, Unlicense, 0BSD, "
            "CC0). Prefer **MIT / Apache-2.0**. Repo license ≠ weights — verify "
            "weight terms before shipping. Ordered by preferred license, then stars.",
            "",
        ]
    )
    if not commercial:
        lines.extend(["_None in this run._", ""])
    else:
        lines.extend(
            [
                "| Repo | License | Stars | Updated | Conf | Description |",
                "|------|---------|------:|---------|------|-------------|",
            ]
        )
        for c in commercial:
            lines.append(
                f"| [{c.full_name}]({c.html_url}) | {c.license_spdx or '—'} | "
                f"{c.stars} | {c.pushed_at.date().isoformat()} | {c.confidence} | "
                f"{_md_cell(c.description)} |"
            )
        lines.append("")

    # Dedicated latest + popular models band (stars + recency called out).
    lines.extend(
        [
            "## Latest & popular models (Audio / Music / Speech)",
            "",
            "Open-source **Audio**, **Music**, and **Speech** foundation / pretrained "
            "models from this run, ordered by **stars** then **last push**. Soft "
            "preference for music / film / gaming / new media production upside.",
            "",
        ]
    )
    if not models:
        lines.extend(["_None in this run._", ""])
    else:
        lines.extend(
            [
                "| Repo | Stars | Updated | Conf | Category | Description |",
                "|------|------:|---------|------|----------|-------------|",
            ]
        )
        for c in models:
            cat = f"{c.category[0]} > {c.category[1]}"
            lines.append(
                f"| [{c.full_name}]({c.html_url}) | {c.stars} | "
                f"{c.pushed_at.date().isoformat()} | {c.confidence} | {cat} | "
                f"{_md_cell(c.description)} |"
            )
        lines.append("")

    append_table(f"Keep (confidence ≥{KEEP_THRESHOLD})", keep)
    append_table(
        f"Review (confidence {HIGH_QUALITY_THRESHOLD}–{KEEP_THRESHOLD - 1})",
        review,
    )

    lines.extend(["### Entry details", ""])
    for i, c in enumerate(candidates, 1):
        cat = f"{c.category[0]} > {c.category[1]}"
        band = "Keep" if c.confidence >= KEEP_THRESHOLD else "Review"
        license_line = (
            f"- **License:** {c.license_spdx or 'unknown'} ({c.license_tier})"
        )
        lines.extend(
            [
                f"#### {i}. [{c.full_name}]({c.html_url}) — **{c.confidence}** ({band}) — `{cat}`",
                f"- **Description:** {c.description}",
                f"- **Why:** {c.rationale}",
                license_line,
                f"- **ToC score:** {c.category_score}",
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
