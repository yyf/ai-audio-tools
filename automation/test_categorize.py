#!/usr/bin/env python3
"""Smoke tests for domain-first ToC categorization."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scout import MIN_CATEGORY_SCORE, categorize  # noqa: E402

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

    print(f"\n{len(CASES) + len(WEAK) - failed}/{len(CASES) + len(WEAK)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
