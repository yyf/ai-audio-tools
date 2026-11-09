#!/usr/bin/env python3
"""Smoke tests for domain-first ToC categorization."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scout import (  # noqa: E402
    AUDIO_INDUSTRY_CONFERENCE_TOPICS,
    COMMERCIAL_OK_SECTION,
    CONFERENCE_TOPICS_SOURCE,
    KEEP_THRESHOLD,
    LORA_SECTION,
    MAX_PER_SECTION,
    MIN_CATEGORY_SCORE,
    MIN_COMMERCIAL_OK_PR_SLOTS,
    MIN_LORA_PR_SLOTS,
    MIN_MODEL_PR_SLOTS,
    MIN_PIPELINE_PR_SLOTS,
    PIPELINES_SECTION,
    Candidate,
    RunStats,
    build_pr_body,
    categorize,
    format_license_annotation,
    license_tier,
    negative_hit,
    select_for_pr,
    should_route_commercial_ok,
    should_route_lora,
    should_route_pipeline,
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


def _fake_candidate(
    name: str,
    category: tuple[str, str],
    cat_score: int,
    conf: int,
    *,
    license_spdx: str | None = None,
    license_tier_name: str = "skip",
    stars: int = 50,
) -> Candidate:
    from datetime import datetime, timezone

    return Candidate(
        full_name=name,
        html_url=f"https://github.com/{name}",
        description="x" * 20,
        stars=stars,
        pushed_at=datetime.now(timezone.utc),
        category=category,
        category_score=cat_score,
        confidence=conf,
        rationale="test",
        license_spdx=license_spdx,
        license_tier=license_tier_name,
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

    # Model focus: reserve slots even when non-model scores higher;
    # prefer more popular (stars) models in the reserved band.
    stats2 = RunStats()
    model_pool = [
        _fake_candidate("org/audio-fm", ("Audio", "Model"), 20, 60, stars=100),
        _fake_candidate("org/music-fm", ("Music", "Model"), 18, 58, stars=500),
        _fake_candidate("org/speech-fm", ("Speech", "Model"), 16, 56, stars=50),
    ]
    hot_pool = [
        _fake_candidate(f"org/hot-asr-{i}", ("Speech", "Recognition"), 50 - i, 90 - i)
        for i in range(8)
    ]
    selected2 = select_for_pr(model_pool + hot_pool, stats2)
    model_count = sum(
        1
        for c in selected2
        if c.category[1] == "Model" and c.category[0] != "Commercial OK"
    )
    ok_models = model_count >= min(MIN_MODEL_PR_SLOTS, len(model_pool))
    ok_first = selected2 and selected2[0].category[1] == "Model"
    # Most-starred model should lead the models band.
    ok_popular = selected2 and selected2[0].full_name == "org/music-fm"
    body = build_pr_body("2026-10-22", RunStats(scanned=1), selected2)
    ok_body = (
        "Latest & popular models (Audio / Music / Speech)" in body
        and "| Stars | Updated |" in body
        and "org/music-fm" in body
        # Models stay in the dedicated section, not duplicated in Keep/Review.
        and body.index("## Latest & popular models")
        < body.index("## Keep")
    )
    # Keep table should not re-list Model category rows.
    keep_section = body.split("## Keep", 1)[-1].split("## Review", 1)[0]
    ok_no_dup = "org/music-fm" not in keep_section and "org/audio-fm" not in keep_section
    status = (
        "OK" if ok_models and ok_first and ok_popular and ok_body and ok_no_dup else "FAIL"
    )
    if not (ok_models and ok_first and ok_popular and ok_body and ok_no_dup):
        failed += 1
    print(
        f"{status}: model focus models={model_count} "
        f"(min={MIN_MODEL_PR_SLOTS}); first={selected2[0].full_name if selected2 else None}"
    )

    # License tiers + Commercial OK routing / reservation.
    license_cases = [
        ("MIT", "friendly"),
        ("Apache-2.0", "friendly"),
        ("BSD-3-Clause", "friendly"),
        ("CC0-1.0", "friendly"),
        ("CC-BY-4.0", "ok_with_care"),
        ("MPL-2.0", "ok_with_care"),
        ("LGPL-3.0", "ok_with_care"),
        ("GPL-3.0", "restricted"),
        ("AGPL-3.0", "restricted"),
        ("CC-BY-NC-4.0", "restricted"),
        ("CC-BY-SA-4.0", "restricted"),
        (None, "restricted"),
    ]
    ok_tiers = True
    for spdx, expect in license_cases:
        got = license_tier(spdx)
        if got != expect:
            ok_tiers = False
            print(f"FAIL: license_tier({spdx!r}) -> {got!r} expected {expect!r}")
    ok_label = format_license_annotation("GPL-3.0", "restricted") == (
        "GPL-3.0 · Restricted"
    ) and format_license_annotation("MIT", "friendly") == ("MIT · Commercial OK")
    if ok_tiers and ok_label:
        print(f"OK: license_tier ({len(license_cases)} cases) + labels")
    else:
        failed += 1
        if not ok_label:
            print("FAIL: format_license_annotation")

    ok_route = (
        should_route_commercial_ok(("Audio", "Model"), [], "friendly")
        and should_route_commercial_ok(("Audio", "Model"), [], "restricted")
        and should_route_commercial_ok(
            ("Speech", "Synthesis"), ["text-to-speech"], "ok_with_care"
        )
        and should_route_commercial_ok(
            ("Music", "Generation"), ["sound-effects", "foley"], "restricted"
        )
        and should_route_commercial_ok(
            ("Audio", "Framework"), ["acoustics", "room-acoustics"], "friendly"
        )
        and should_route_commercial_ok(
            ("Audio", "Framework"), ["spatial-audio", "ambisonics", "aes"], "restricted"
        )
        and should_route_commercial_ok(
            ("Music", "Analysis"), ["ismir", "mir", "dcase"], "friendly"
        )
        and should_route_commercial_ok(
            ("Audio", "Framework"), ["nime", "dafx", "adc"], "friendly"
        )
        and should_route_commercial_ok(
            ("Speech", "Recognition"), ["speech-translation"], "friendly"
        )
        and not should_route_commercial_ok(
            ("Speech", "Recognition"), ["unrelated-topic"], "friendly"
        )
    )
    ok_source = (
        "music-technology-conferences" in CONFERENCE_TOPICS_SOURCE
        and "nime" in AUDIO_INDUSTRY_CONFERENCE_TOPICS
        and "aes" in AUDIO_INDUSTRY_CONFERENCE_TOPICS
        and "ismir" in AUDIO_INDUSTRY_CONFERENCE_TOPICS
    )
    status = "OK" if ok_route and ok_source else "FAIL"
    if not (ok_route and ok_source):
        failed += 1
    print(f"{status}: should_route_commercial_ok (conference-repo topics)")

    stats3 = RunStats()
    commercial_pool = [
        _fake_candidate(
            "org/mit-tts",
            COMMERCIAL_OK_SECTION,
            20,
            70,
            license_spdx="MIT",
            license_tier_name="friendly",
            stars=80,
        ),
        _fake_candidate(
            "org/gpl-sfx",
            COMMERCIAL_OK_SECTION,
            18,
            68,
            license_spdx="GPL-3.0",
            license_tier_name="restricted",
            stars=300,
        ),
        _fake_candidate(
            "org/apache-music",
            COMMERCIAL_OK_SECTION,
            18,
            68,
            license_spdx="Apache-2.0",
            license_tier_name="friendly",
            stars=200,
        ),
    ]
    hot_pool3 = [
        _fake_candidate(f"org/hot-tool-{i}", ("Speech", "Recognition"), 50 - i, 90 - i)
        for i in range(8)
    ]
    selected3 = select_for_pr(commercial_pool + hot_pool3, stats3)
    commercial_count = sum(1 for c in selected3 if c.category == COMMERCIAL_OK_SECTION)
    ok_commercial = commercial_count >= min(
        MIN_COMMERCIAL_OK_PR_SLOTS, len(commercial_pool)
    )
    ok_commercial_first = selected3 and selected3[0].category == COMMERCIAL_OK_SECTION
    # Prefer MIT/Apache over higher-star Restricted when seeding Commercial OK.
    ok_prefers_friendly = selected3 and selected3[0].full_name == "org/apache-music"
    body3 = build_pr_body("2026-11-09", RunStats(scanned=1), selected3)
    ok_commercial_body = (
        "Commercial OK — featured demos" in body3
        and "| License | Tier |" in body3
        and "Restricted" in body3
        and body3.index("## Commercial OK") < body3.index("## Latest & popular models")
    )
    status = (
        "OK"
        if ok_commercial
        and ok_commercial_first
        and ok_prefers_friendly
        and ok_commercial_body
        else "FAIL"
    )
    if not (
        ok_commercial
        and ok_commercial_first
        and ok_prefers_friendly
        and ok_commercial_body
    ):
        failed += 1
    print(
        f"{status}: commercial OK count={commercial_count} "
        f"(min={MIN_COMMERCIAL_OK_PR_SLOTS}); "
        f"first={selected3[0].full_name if selected3 else None}"
    )

    # Pipelines: AI audio pipelines (speech/music/film/gaming/realtime).
    ok_pipe_route = (
        should_route_pipeline(
            "org/voice-agent",
            "Voice agent platform chaining ASR, LLM, and TTS for telephony",
            ["voice-agent", "asr", "tts"],
            ("Speech", "Production"),
        )
        and should_route_pipeline(
            "org/music-studio",
            "End-to-end music creation pipeline with musicgen and stem separation",
            ["music-generation"],
            ("Music", "Production"),
        )
        and should_route_pipeline(
            "org/game-audio",
            "Realtime game audio pipeline for foley and sound design",
            ["game-audio", "foley"],
            ("Audio", "Framework"),
        )
        and not should_route_pipeline(
            "org/plain-whisper",
            "Robust speech recognition via large-scale weak supervision",
            ["asr", "speech-to-text"],
            ("Speech", "Recognition"),
        )
        and not should_route_pipeline(
            "org/audio-fm",
            "An audio language foundation model",
            ["foundation-model"],
            ("Audio", "Model"),
        )
    )
    status = "OK" if ok_pipe_route else "FAIL"
    if not ok_pipe_route:
        failed += 1
    print(f"{status}: should_route_pipeline")

    stats4 = RunStats()
    pipeline_pool = [
        _fake_candidate(
            "org/hot-agent",
            PIPELINES_SECTION,
            22,
            75,
            stars=400,
        ),
        _fake_candidate(
            "org/new-pipeline",
            PIPELINES_SECTION,
            20,
            72,
            stars=60,
        ),
    ]
    pipeline_pool[0].description = (
        "Popular voice agent chaining speech-to-text, llm, and text-to-speech"
    )
    pipeline_pool[1].description = (
        "Music production pipeline with realtime sound design workflows"
    )
    hot_pool4 = [
        _fake_candidate(f"org/hot-asr-{i}", ("Speech", "Recognition"), 50 - i, 90 - i)
        for i in range(8)
    ]
    selected4 = select_for_pr(pipeline_pool + hot_pool4, stats4)
    pipeline_count = sum(1 for c in selected4 if c.category == PIPELINES_SECTION)
    ok_pipe_slots = pipeline_count >= min(MIN_PIPELINE_PR_SLOTS, len(pipeline_pool))
    ok_pipe_first = selected4 and selected4[0].category == PIPELINES_SECTION
    ok_pipe_popular = selected4 and selected4[0].full_name == "org/hot-agent"
    body4 = build_pr_body("2026-11-09", RunStats(scanned=1), selected4)
    ok_pipe_body = (
        "Pipelines — AI audio pipelines" in body4
        and "org/hot-agent" in body4
        and body4.index("## Pipelines") < body4.index("## Latest & popular models")
        and body4.index("## Commercial OK") < body4.index("## Pipelines")
    )
    status = (
        "OK"
        if ok_pipe_slots and ok_pipe_first and ok_pipe_popular and ok_pipe_body
        else "FAIL"
    )
    if not (ok_pipe_slots and ok_pipe_first and ok_pipe_popular and ok_pipe_body):
        failed += 1
    print(
        f"{status}: pipelines count={pipeline_count} "
        f"(min={MIN_PIPELINE_PR_SLOTS}); "
        f"first={selected4[0].full_name if selected4 else None}"
    )

    # LoRA: popular/emerging AI audio adapters.
    ok_lora_route = (
        should_route_lora(
            "org/tts-lora",
            "LoRA adapters for multilingual text-to-speech fine-tuning",
            ["lora", "tts"],
            ("Speech", "Synthesis"),
        )
        and should_route_lora(
            "org/music-peft",
            "PEFT low-rank adaptation for music generation models",
            ["peft", "music-generation"],
            ("Music", "Generation"),
        )
        and not should_route_lora(
            "org/llm-lora",
            "LoRA fine-tunes for general LLM chat",
            ["lora"],
            ("Speech", "Recognition"),
        )
    )
    status = "OK" if ok_lora_route else "FAIL"
    if not ok_lora_route:
        failed += 1
    print(f"{status}: should_route_lora")

    stats5 = RunStats()
    lora_pool = [
        _fake_candidate("org/hot-lora", LORA_SECTION, 20, 70, stars=350),
        _fake_candidate("org/new-lora", LORA_SECTION, 18, 65, stars=40),
    ]
    lora_pool[0].description = "Popular LoRA for speech-to-text ASR adaptation"
    lora_pool[1].description = "Emerging PEFT adapter for singing voice synthesis"
    hot_pool5 = [
        _fake_candidate(f"org/tool-{i}", ("Speech", "Recognition"), 50 - i, 90 - i)
        for i in range(8)
    ]
    selected5 = select_for_pr(lora_pool + hot_pool5, stats5)
    lora_count = sum(1 for c in selected5 if c.category == LORA_SECTION)
    ok_lora_slots = lora_count >= min(MIN_LORA_PR_SLOTS, len(lora_pool))
    ok_lora_first = selected5 and selected5[0].category == LORA_SECTION
    ok_lora_popular = selected5 and selected5[0].full_name == "org/hot-lora"
    body5 = build_pr_body("2026-11-09", RunStats(scanned=1), selected5)
    ok_lora_body = (
        "LoRA — AI audio adapters" in body5
        and "org/hot-lora" in body5
        and body5.index("## LoRA") < body5.index("## Latest & popular models")
        and body5.index("## Pipelines") < body5.index("## LoRA")
    )
    status = (
        "OK"
        if ok_lora_slots and ok_lora_first and ok_lora_popular and ok_lora_body
        else "FAIL"
    )
    if not (ok_lora_slots and ok_lora_first and ok_lora_popular and ok_lora_body):
        failed += 1
    print(
        f"{status}: lora count={lora_count} "
        f"(min={MIN_LORA_PR_SLOTS}); "
        f"first={selected5[0].full_name if selected5 else None}"
    )

    total = len(CASES) + len(WEAK) + len(NEGATIVE_CASES) + 9
    print(f"\n{total - failed}/{total} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
