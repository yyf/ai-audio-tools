# ai-audio-tools
Community list of open-source AI tools, models, and datasets for audio, music, and speech applications 

# To contribute to the list

Edit the README and make a PR

# Audio
## Dataset
- [HuggingFace](https://huggingface.co/datasets?modality=modality:audio): datasets with tag "audio" on Hugging Face

## Model
- [HuggingFace](https://huggingface.co/models?other=audio): models with tag "audio" on Hugging Face

# Music
## Analysis
- [Essentia](https://github.com/MTG/essentia): open-source C++ library for audio analysis and audio-based music information retrieval  
- [Librosa](https://github.com/librosa/librosa): Python library for audio and music analysis 
- [DDSP](https://github.com/magenta/ddsp): DDSP is a library of differentiable versions of common DSP functions (such as synthesizers, waveshapers, and filters). This allows these interpretable elements to be used as part of an deep learning model, especially as the output layers for audio generation
- [MIDI-DDSP](https://github.com/magenta/midi-ddsp?tab=readme-ov-file): MIDI-DDSP is a hierarchical audio generation model for synthesizing MIDI expanded from DDSP
- [TorchAudio](https://github.com/pytorch/audio): Data manipulation and transformation for audio signal processing, powered by PyTorch 
- [nnAudio](https://github.com/KinWaiCheuk/nnAudio): Audio processing by using pytorch 1D convolution network 
- [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis/): Python Audio Analysis Library: Feature Extraction, Classification, Segmentation and Applications 
- [mutagen](https://mutagen.readthedocs.io/en/latest/#): a Python module to handle audio metadata
- [dejavu](https://github.com/worldveil/dejavu): Audio fingerprinting and recognition in Python 
- [audiomentations](https://github.com/iver56/audiomentations): A Python library for audio data augmentation. Inspired by albumentations. Useful for machine learning
- [soundata](https://github.com/soundata/soundata): Python library for downloading, loading, and working with sound datasets 
- [EfficientAT](https://github.com/fschmid56/EfficientAT): This repository aims at providing efficient CNNs for Audio Tagging. We provide AudioSet pre-trained models ready for downstream training and extraction of audio embeddings
- [AugLy](https://github.com/facebookresearch/AugLy): A data augmentations library for audio, image, text, and video
- [Pedalboard](https://github.com/spotify/pedalboard?tab=readme-ov-file): A Python library for working with audio
- [TinyTag](https://github.com/devsnd/tinytag): a Python library for reading audio file metadata
- [OpenSmile](https://github.com/audeering/opensmile): The Munich Open-Source Large-Scale Multimedia Feature Extractor 
- [Madmom](https://github.com/CPJKU/madmom): Python audio and music signal processing library 
- [Beets](https://beets.io/): a music library manager and MusicBrainz tagger
- [Mirdata](https://github.com/mir-dataset-loaders/mirdata): Python library for working with Music Information Retrieval datasets 
- [Partitura](https://github.com/CPJKU/partitura): A python package for handling modern staff notation of music 
- [msaf](https://msaf.readthedocs.io/en/latest/#): a python package for the analysis of music structural segmentation algorithms
- [basic-pitch](https://github.com/spotify/basic-pitch): A lightweight yet powerful audio-to-MIDI converter with pitch bend detection 
- [jams](https://github.com/marl/jams): A JSON Annotated Music Specification for Reproducible MIR Research 
- [Audio-Flamingo-2](https://github.com/NVIDIA/audio-flamingo): Audio Flamingo 2: An Audio-Language Model with Long-Audio Understanding and Expert Reasoning Abilities

## Production
- [OpenVINO](https://github.com/intel/openvino-plugins-ai-audacity/tree/main): OpenVINO AI effects for Audacity (Windows, Linux)
- [TuneFlow](https://github.com/tuneflow/tuneflow-py): TuneFlow is a next-gen DAW that aims to boost music making productivity through the power of AI
- [Spleeter](https://github.com/deezer/spleeter): Deezer source separation library including pretrained models
- [DeepAFx](https://github.com/adobe-research/DeepAFx?tab=readme-ov-file): Third-party audio effects plugins as differentiable layers within deep neural networks
- [matchering](https://github.com/sergree/matchering): open source audio matching and mastering 
- [AudioDec](https://github.com/facebookresearch/AudioDec): An Open-source Streaming High-fidelity Neural Audio Codec 
- [USS](https://github.com/bytedance/uss): This is the PyTorch implementation of the Universal Source Separation with Weakly labelled Data
- [FAST-RIR](https://github.com/anton-jeran/FAST-RIR): This is the official implementation of our neural-network-based fast diffuse room impulse response generator (FAST-RIR) for generating room impulse responses (RIRs) for a given rectangular acoustic environment
- [FoleyCrafter](https://foleycrafter.github.io/): FoleyCrafter is a video-to-audio generation framework which can produce realistic sound effects semantically relevant and synchronized with videos. 
- [OpenVINO](https://github.com/intel/openvino-plugins-ai-audacity/tree/main): OpenVINO AI effects for Audacity (Windows, Linux)
- [TuneFlow](https://github.com/tuneflow/tuneflow-py): TuneFlow is a next-gen DAW that aims to boost music making productivity through the power of AI

## Generation
- [StableAudio](https://github.com/Stability-AI/stable-audio-tools): Generative models for conditional audio generation 
- [AudioCraft](https://github.com/facebookresearch/audiocraft): a PyTorch library for deep learning research on audio generation. AudioCraft contains inference and training code for two state-of-the-art AI generative models producing high-quality audio: AudioGen and MusicGen.
- [Jukebox](https://github.com/openai/jukebox): A generative model for music
- [Magenta](https://github.com/magenta/symbolic-music-diffusion): symbolic music generation with diffusion models 
- [TorchSynth](https://github.com/torchsynth/torchsynth): A GPU-optional modular synthesizer in pytorch, 16200x faster than realtime, for audio ML researchers
- [audiobox](https://audiobox.metademolab.com/): Audiobox is Meta’s new foundation research model for audio generation. It can generate voices and sound effects using a combination of voice inputs and natural language text prompts
- [Amphion](https://github.com/open-mmlab/Amphion): Amphion is a toolkit for Audio, Music, and Speech Generation
- [AudioGPT](https://github.com/AIGC-Audio/AudioGPT): AudioGPT: Understanding and Generating Speech, Music, Sound, and Talking Head
- [WaveGAN](https://github.com/chrisdonahue/wavegan): WaveGAN: Learn to synthesize raw audio with generative adversarial networks 
- [RAVE](https://github.com/acids-ircam/RAVE): Official implementation of the RAVE model: a Realtime Audio Variational autoEncoder 
- [AudioLDM](https://audioldm.github.io/): This toolbox aims to unify audio generation model evaluation for easier comparison
- [Make-An-Audio](https://github.com/Text-to-Audio/Make-An-Audio): a conditional diffusion probabilistic model capable of generating high fidelity audio efficiently from X modality
- [Diffuser](https://github.com/huggingface/diffusers): Diffusers is the go-to library for state-of-the-art pretrained diffusion models for generating images, audio, and even 3D structures of molecules
- [stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools): Generative models for conditional audio generation 
- [MidiTok](https://github.com/Natooz/MidiTok): MIDI / symbolic music tokenizers for Deep Learning models
- [muspy](https://salu133445.github.io/muspy/): an open source Python library for symbolic music generation
- [MusicLM] (https://google-research.github.io/seanet/musiclm/examples/): a model generating high-fidelity music from text descriptions 
- [riffusion](https://github.com/riffusion/riffusion): Stable diffusion for real-time music generation 
- [muzic](https://github.com/microsoft/muzic): Music Understanding and Generation with Artificial Intelligence 
- [midi-lm](https://github.com/jeremyjordan/midi-lm): Generative modeling of MIDI files 
- [UniAudio](https://github.com/yangdongchao/UniAudio): The Open Source Code of UniAudio 
- [MuseGAN](https://github.com/salu133445/musegan): An AI for Music Generation 
- [YuE](https://github.com/multimodal-art-projection/YuE): Open Full-song Music Generation Foundation Model, something similar to Suno.ai but open
- [Bark](https://github.com/suno-ai/bark): Bark is Suno's open-source text-to-speech+ model. Text-Prompted Generative Audio Model
- [MG²](https://github.com/shaopengw/Awesome-Music-Generation): Awesome music generation model——MG²
- [MusicGPT](https://github.com/gabotechs/MusicGPT): Generate music based on natural language prompts using LLMs running locally
- [InspireMusic](https://github.com/FunAudioLLM/InspireMusic): InspireMusic: A Unified Framework for Music, Song, Audio Generation.
- [riffusion-hobby](https://github.com/riffusion/riffusion-hobby): Stable diffusion for real-time music generation

# Speech
## Recognition
- [Whisper](https://github.com/openai/whisper): a multitasking model that can perform multilingual speech recognition, speech translation, and language identification
- [Deep Speech](https://github.com/mozilla/DeepSpeech): Mozilla's open-source speech-to-text engine
- [Kaldi ASR](https://kaldi-asr.org/): open-source speech recognition toolkit written in C++ 
- [PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech?tab=readme-ov-file): Easy-to-use Speech Toolkit including Self-Supervised Learning model, SOTA/Streaming ASR with punctuation, Streaming TTS with text frontend, Speaker Verification System, End-to-End Speech Translation and Keyword Spotting
- [NeMo](https://github.com/NVIDIA/NeMo): a framework for generative AI 
- [julius](https://github.com/julius-speech/julius): Open-Source Large Vocabulary Continuous Speech Recognition Engine 
- [speechbrain](https://speechbrain.github.io/): an open-source and all-in-one conversational AI toolkit based on PyTorch
- [pocketsphinx](https://github.com/cmusphinx/pocketsphinx): A small speech recognizer 
- [FunASR](https://github.com/alibaba-damo-academy/FunASR): A Fundamental End-to-End Speech Recognition Toolkit and Open Source SOTA Pretrained Models
- [NeuralSpeech](https://github.com/microsoft/NeuralSpeech): a research project at Microsoft Research Asia, which focuses on neural network based speech processing, including automatic speech recognition (ASR), text-to-speech synthesis (TTS), spatial audio synthesis, video dubbing, etc
- [espnet](https://github.com/espnet/espnet): End-to-End Speech Processing Toolkit 
- [RealTimeSTT](https://github.com/KoljaB/RealtimeSTT): A robust, efficient, low-latency speech-to-text library with advanced voice activity detection, wake word activation and instant transcription

## Production
- [Descript audio codec](https://github.com/descriptinc/descript-audio-codec): State-of-the-art audio codec with 90x compression factor. Supports 44.1kHz, 24kHz, and 16kHz mono/stereo audio
- [Descript audio tools](https://github.com/descriptinc/audiotools): Object-oriented handling of audio data, with GPU-powered augmentations, and more
- [Meta encodec](https://github.com/facebookresearch/encodec): State-of-the-art deep learning based audio codec supporting both mono 24 kHz audio and stereo 48 kHz audio 
- [audino](https://github.com/midas-research/audino): Open source audio annotation tool for humans 

## Synthesis
- [Coqui TTS](https://github.com/coqui-ai/TTS): a deep learning toolkit for Text-to-Speech, battle-tested in research and production  
- [DiffSinger](https://github.com/MoonInTheRiver/DiffSinger): singing voice synthesis via shallow diffusion mechanism 
- [Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning): Clone a voice in 5 seconds to generate arbitrary speech in real-time 
- [wavenet](https://github.com/ibab/tensorflow-wavenet): A TensorFlow implementation of DeepMind's WaveNet paper 
- [FastSpeech2](https://github.com/ming024/FastSpeech2): An implementation of Microsoft's "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech" 
- [MelGAN](http://swpark.me/melgan/): Unofficial PyTorch implementation of MelGAN vocoder
- [hifi-gan](https://github.com/jik876/hifi-gan): Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis 
- [elevenlabs-pythons](https://github.com/elevenlabs/elevenlabs-python): The official Python API for ElevenLabs Text to Speech. 
- [tortoise-tts](https://github.com/neonbjb/tortoise-tts): A multi-voice TTS system trained with an emphasis on quality 
- [lyrebird](https://github.com/lyrebird-voice-changer/lyrebird): Simple and powerful voice changer for Linux, written with Python & GTK 
- [elevenlabs](https://github.com/elevenlabs/elevenlabs-python): The official Python API for ElevenLabs Text to Speech
- [piper](https://github.com/rhasspy/piper): A fast, local neural text to speech system
- [tts-generation-webui](https://github.com/rsxdalv/tts-generation-webui): TTS Generation Web UI (Bark, MusicGen + AudioGen, Tortoise, RVC, Vocos, Demucs, SeamlessM4T, MAGNet) 
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS): 1 min voice data can also be used to train a good TTS model! (few shot voice cloning) 
- [metavoice-src](https://github.com/metavoiceio/metavoice-src): Foundational model for human-like, expressive TTS 
- [Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning): Clone a voice in 5 seconds to generate arbitrary speech in real-time 
- [Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI): Voice data <= 10 mins can also be used to train a good VC model! 
- [midi2voice](https://github.com/mathigatti/midi2voice): Singing synthesis from MIDI file 
- [OpenVoice](https://github.com/myshell-ai/OpenVoice): Instant voice cloning by MyShell
- [ChatTTS](https://github.com/2noise/ChatTTS): A generative speech model for daily dialogue
- [csm](https://github.com/SesameAILabs/csm): A Conversational Speech Generation Model