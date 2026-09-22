#!/usr/bin/env python3
"""Smoke tests for domain-first ToC categorization."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scout import (  # noqa: E402
    KEEP_THRESHOLD,
    MAX_PER_SECTION,
    MIN_CATEGORY_SCORE,
    Candidate,
    RunStats,
    categorize,
    negative_hit,
    select_for_pr,
)

CASES: list[tuple[str, str, list[str], tuple[str, str]]] = [
    # Speech
    (
        "openai/whisper",
        "Robust speech recognition via large-scale weak supervision",
        ["asr", "speech-to-text"],
        ("Speech", "Recognition"),
    ),
    (
        "coqui-ai/TTS",
        "a deep learning toolkit for Text-to-Speech",
        ["tts", "text-to-speech"],
        ("Speech", "Synthesis"),
    ),
    (
        "myshell-ai/OpenVoice",
        "Instant voice cloning by MyShell",
        ["voice-cloning", "tts"],
        ("Speech", "Synthesis"),
    ),
    (
        "pipecat-ai/pipecat",
        "Open Source framework for voice and multimodal conversational AI",
        ["voice-agent", "realtime"],
        ("Speech", "Production"),
    ),
    (
        "facebookresearch/encodec",
        "State-of-the-art deep learning based audio codec",
        ["codec", "audio"],
        ("Speech", "Production"),
    ),
    (
        "lgy1027/matrix-live-diarizer",
        "Local-first meeting transcription with live captions and diarization",
        ["transcription", "diarization"],
        ("Speech", "Recognition"),
    ),
    # Music
    (
        "facebookresearch/audiocraft",
        "PyTorch library for audio generation research including MusicGen",
        ["musicgen", "audio-generation"],
        ("Music", "Generation"),
    ),
    (
        "deezer/spleeter",
        "Deezer source separation library including pretrained models",
        ["source-separation"],
        ("Music", "Production"),
    ),
    (
        "librosa/librosa",
        "Python library for audio and music analysis",
        ["mir", "audio-analysis"],
        ("Music", "Analysis"),
    ),
    (
        "spotify/basic-pitch",
        "A lightweight yet powerful audio-to-MIDI converter",
        ["mir", "midi"],
        ("Music", "Analysis"),
    ),
    (
        "riffusion/riffusion",
        "Stable diffusion for real-time music generation",
        ["music-generation"],
        ("Music", "Generation"),
    ),
    # Audio
    (
        "wavmark/wavmark",
        "AI-based Audio Watermarking Tool",
        ["watermark", "security"],
        ("Audio", "Security"),
    ),
    (
        "midas-research/audino",
        "Open source audio annotation tool for humans",
        ["annotation"],
        ("Audio", "Annotation"),
    ),
    (
        "laion-ai/clap",
        "Contrastive Language-Audio Pretraining audio foundation model",
        ["audio-llm", "clap"],
        ("Audio", "Model"),
    ),
    (
        "NVIDIA/audio-flamingo",
        "Audio Flamingo 2: An Audio-Language Model with Long-Audio Understanding",
        ["audio-language-model"],
        ("Audio", "Model"),
    ),
    (
        "multimodal-art-projection/YuE",
        "Open Full-song Music Generation Foundation Model",
        ["music-generation", "foundation-model"],
        ("Music", "Model"),
    ),
    (
        "someone/speech-foundation",
        "A pretrained speech foundation model for multilingual understanding",
        ["speech", "foundation-model"],
        ("Speech", "Model"),
    ),
    (
        "kunitoki/yup",
        "The modern framework optimized for realtime audio and GPU-native creative software",
        ["audio", "framework"],
        ("Audio", "Framework"),
    ),
    (
        "hearbenchmark/hear-eval-kit",
        "Holistic Evaluation of Audio Representations benchmark leaderboard",
        ["benchmark", "evaluation"],
        ("Audio", "Benchmark"),
    ),
    (
        "Stability-AI/stable-audio-tools",
        "Generative models for conditional audio generation",
        ["audio-generation"],
        ("Music", "Generation"),
    ),
    (
        "csun22/Synthetic-Voice-Detection-Vocoder-Artifacts",
        "AI-Synthesized Voice Detection Using Neural Vocoder Artifacts",
        ["deepfake", "forensic"],
        ("Audio", "Security"),
    ),
    (
        "suno-ai/bark",
        "Bark is Suno's open-source text-to-speech model",
        ["tts"],
        ("Speech", "Synthesis"),
    ),
]

# Should NOT clear the ToC bar (general / off-topic).
WEAK: list[tuple[str, str, list[str]]] = [
    (
        "mastra-ai/mastra",
        "Mastra is the modern TypeScript framework for AI-powered applications",
        ["typescript", "agents"],
    ),
    (
        "unslothai/unsloth",
        "Local UI to run and train LLMs and diffusion models",
        ["llm", "training"],
    ),
    (
        "off-grid-ai/OGAM",
        "Chat, see, speak, and generate images — GGUF LLMs, vision, Stable Diffusion",
        ["llm", "offline"],
    ),
]

# Hard-reject via NEGATIVE_KEYWORDS even if audio-adjacent wording appears.
NEGATIVE_CASES: list[tuple[str, str, list[str], str]] = [
    (
        "someone/cool-discord-music",
        "A Discord bot for playing music and TTS clips",
        ["discord", "bot"],
        "discord bot",
    ),
    (
        "someone/homelab-audio",
        "My homelab media stack with Whisper and TTS",
        ["homelab"],
        "homelab",
    ),
    (
        "someone/gguf-whisper-ui",
        "Run Whisper and LLMs locally with GGUF models",
        ["gguf", "whisper"],
        "gguf",
    ),
    (
        "strawberrymusicplayer/strawberry",
        "Strawberry Music Player",
        ["music", "player"],
        "music player",
    ),
    (
        "mholzi/beatify",
        "Music quiz party game for Home Assistant",
        ["homeassistant"],
        "party game",
    ),
]


def _fake_candidate(name: str, category: tuple[str, str], cat_score: int, conf: int) -> Candidate:
    from datetime import datetime, timezone

    return Candidate(
        full_name=name,
        html_url=f"https://github.com/{name}",
        description="x" * 20,
        stars=50,
        pushed_at=datetime.now(timezone.utc),
        category=category,
        category_score=cat_score,
        confidence=conf,
        rationale="test",
    )


def main() -> int:
    failed = 0
    for full_name, desc, topics, expected in CASES:
        got, score = categorize(full_name, desc, topics)
        ok = got == expected and score >= MIN_CATEGORY_SCORE
        status = "OK" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"{status}: {full_name} -> {got[0]} > {got[1]} (score={score}) expected {expected[0]} > {expected[1]}")

    for full_name, desc, topics in WEAK:
        got, score = categorize(full_name, desc, topics)
        ok = score < MIN_CATEGORY_SCORE
        status = "OK" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"{status}: weak {full_name} -> {got[0]} > {got[1]} (score={score}) expect < {MIN_CATEGORY_SCORE}")

    for full_name, desc, topics, expect_phrase in NEGATIVE_CASES:
        hit = negative_hit(full_name, desc, topics)
        ok = hit == expect_phrase
        status = "OK" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"{status}: negative {full_name} -> {hit!r} expected {expect_phrase!r}")

    # Section cap: 4 Speech>Synthesis candidates → keep MAX_PER_SECTION.
    stats = RunStats()
    pool = [
        _fake_candidate(f"org/tts-{i}", ("Speech", "Synthesis"), 30 - i, 80 - i)
        for i in range(4)
    ]
    pool.append(_fake_candidate("org/asr-1", ("Speech", "Recognition"), 40, 90))
    selected = select_for_pr(pool, stats)
    synth = [c for c in selected if c.category == ("Speech", "Synthesis")]
    recog = [c for c in selected if c.category == ("Speech", "Recognition")]
    ok_cap = len(synth) == MAX_PER_SECTION and len(recog) == 1
    # Highest ToC scores first within the section.
    ok_order = [c.category_score for c in synth] == sorted(
        (c.category_score for c in synth), reverse=True
    )
    status = "OK" if ok_cap and ok_order else "FAIL"
    if not (ok_cap and ok_order):
        failed += 1
    print(
        f"{status}: section cap synth={len(synth)} recog={len(recog)} "
        f"(max={MAX_PER_SECTION}); keep_band={KEEP_THRESHOLD}"
    )

    total = len(CASES) + len(WEAK) + len(NEGATIVE_CASES) + 1
    print(f"\n{total - failed}/{total} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
