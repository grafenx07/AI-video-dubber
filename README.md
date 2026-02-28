# 🎬 AI Video Dubber — English to Hindi Dubbing Pipeline

A modular, production-ready Python pipeline that takes an English video and produces a Hindi-dubbed version with **voice cloning**, **lip synchronization**, and **face restoration** — all using **free, open-source tools**.

> **Built for the Supernan AI Automation Intern Challenge**
> "The Golden 15 Seconds" — 15 seconds of perfection.

---

## 🏗️ Architecture

```
dub_video.py                 ← Main orchestrator with CLI
│
├── modules/
│   ├── video_utils.py       ← FFmpeg-based video/audio I/O
│   ├── transcription.py     ← Whisper speech-to-text
│   ├── translation.py       ← IndicTrans2 / Google Translate
│   ├── tts.py               ← XTTS v2 voice cloning
│   ├── alignment.py         ← Audio duration matching
│   ├── lipsync.py           ← Wav2Lip lip synchronization
│   └── enhancement.py       ← GFPGAN face restoration
│
├── setup.py                 ← Automated environment setup
├── requirements.txt         ← Python dependencies
├── AI_Video_Dubber.ipynb    ← Google Colab notebook (one-click)
└── outputs/                 ← Pipeline outputs (auto-created)
```

### Pipeline Flow

```
Input Video (full)
      ↓
┌─────────────────────┐
│ 1. Extract 15s clip │  ← FFmpeg (frame-accurate re-encode)
└─────────┬───────────┘
          ↓
┌─────────────────────┐
│ 2. Extract audio    │  ← FFmpeg → 16kHz mono WAV
└─────────┬───────────┘
          ↓
┌─────────────────────┐
│ 3. Transcribe       │  ← OpenAI Whisper (word timestamps)
└─────────┬───────────┘
          ↓
┌─────────────────────┐
│ 4. Translate to     │  ← IndicTrans2 (context-aware)
│    Hindi            │     or Google Translate (fallback)
└─────────┬───────────┘
          ↓
┌─────────────────────┐
│ 5. Voice cloning    │  ← Coqui XTTS v2 (speaker matching)
│                     │     or Edge TTS (no-GPU fallback)
└─────────┬───────────┘
          ↓
┌─────────────────────┐
│ 6. Audio alignment  │  ← librosa time-stretch + silence trim
└─────────┬───────────┘
          ↓
┌─────────────────────┐
│ 7. Lip sync         │  ← Wav2Lip (GAN model)
└─────────┬───────────┘
          ↓
┌─────────────────────┐
│ 8. Face enhancement │  ← GFPGAN v1.4 (face restoration)
└─────────┬───────────┘
          ↓
   Final Dubbed Video
```

---

## 🚀 Quick Start

### Option 1: Google Colab (Recommended — Free GPU)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)

1. Open `AI_Video_Dubber.ipynb` in Google Colab
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. Run all cells — the notebook handles everything

### Option 2: Local / Cloud VM

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/AI-video-dubber.git
cd AI-video-dubber

# Run automated setup
python setup.py

# Verify installation
python setup.py --check

# Run the pipeline
python dub_video.py --input video.mp4 --start 15 --end 30
```

---

## 📋 Setup Instructions

### Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10+ | 3.10 |
| GPU VRAM | 4 GB | 8 GB+ |
| RAM | 8 GB | 16 GB |
| FFmpeg | Any | 5.0+ |
| Disk Space | 10 GB | 20 GB |

### Step-by-Step Setup

#### 1. Install FFmpeg

```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y ffmpeg

# macOS
brew install ffmpeg

# Windows (via Chocolatey)
choco install ffmpeg

# Google Colab (pre-installed)
```

#### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

#### 3. Setup Wav2Lip (for lip-sync)

```bash
python setup.py --setup-wav2lip
```

Then manually download `wav2lip_gan.pth`:
- [Download Link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA?e=n9ljGW)
- Place in: `Wav2Lip/checkpoints/wav2lip_gan.pth`

#### 4. Verify Setup

```bash
python setup.py --check
```

---

## 🎯 Usage

### Basic Usage

```bash
# Process the challenge segment (0:15 - 0:30)
python dub_video.py --input supernan_training.mp4 --start 15 --end 30
```

### Advanced Options

```bash
# Use IndicTrans2 for better translation quality
python dub_video.py --input video.mp4 --translation indictrans2

# Use Edge TTS (no GPU needed for TTS stage)
python dub_video.py --input video.mp4 --tts edge

# Adjust speech rate (Hindi is ~10% longer than English)
python dub_video.py --input video.mp4 --speech-rate 1.1

# Skip heavy stages for quick testing
python dub_video.py --input video.mp4 --skip-lipsync --skip-enhancement

# Use smaller Whisper model for faster transcription
python dub_video.py --input video.mp4 --whisper-model base

# Lower Wav2Lip batch size for less VRAM usage
python dub_video.py --input video.mp4 --wav2lip-batch-size 4
```

### Full CLI Reference

```
python dub_video.py --help

Arguments:
  --input, -i          Path to input video (required)
  --start, -s          Start time in seconds (default: 15)
  --end, -e            End time in seconds (default: 30)
  --output-dir, -o     Output directory (default: outputs/)
  --whisper-model      tiny|base|small|medium|large (default: small)
  --translation        indictrans2|google|seamless (default: google)
  --tts                xtts|edge (default: xtts)
  --speech-rate        Speed multiplier (default: 1.05)
  --enhancement        gfpgan|codeformer (default: gfpgan)
  --skip-enhancement   Skip face restoration
  --skip-lipsync       Skip lip synchronization
  --wav2lip-dir        Wav2Lip repo path (default: Wav2Lip/)
  --wav2lip-batch-size Frames per batch (default: 16)
```

---

## 📂 Output Structure

After running the pipeline, `outputs/` will contain:

```
outputs/
├── 01_clip.mp4              ← Extracted 15-second clip
├── 02_audio.wav             ← Original English audio
├── 03_transcription.json    ← Whisper transcription + timestamps
├── 04_translation.json      ← Hindi translation + segments
├── 05_hindi_raw.wav         ← Generated Hindi speech
├── 06_hindi_aligned.wav     ← Duration-matched Hindi audio
├── 07_lipsynced.mp4         ← Wav2Lip output
├── 08_enhanced.mp4          ← GFPGAN enhanced output
├── final_dubbed.mp4         ← ✅ Final result
├── pipeline_config.json     ← Run configuration
└── logs/
    └── pipeline_*.log       ← Detailed execution log
```

---

## 💰 Cost Analysis

### Current Pipeline (Free Tier)

| Component | Tool | Cost |
|-----------|------|------|
| Transcription | Whisper (local) | **₹0** |
| Translation | Google Translate / IndicTrans2 | **₹0** |
| Voice Cloning | Coqui XTTS v2 | **₹0** |
| Lip Sync | Wav2Lip | **₹0** |
| Face Enhancement | GFPGAN | **₹0** |
| Compute | Google Colab Free | **₹0** |
| **Total** | | **₹0** |

### Estimated Cost Per Minute (Scaled to Paid GPU)

| GPU | Cost/hr | Processing Time/min | Cost/min of video |
|-----|---------|---------------------|-------------------|
| Colab Free (T4) | $0.00 | ~8 min | $0.00 |
| AWS g4dn.xlarge (T4) | $0.53 | ~8 min | $0.07 |
| AWS g5.xlarge (A10G) | $1.01 | ~4 min | $0.07 |
| Lambda A10 | $0.75 | ~4 min | $0.05 |

### Cost for 500 Hours of Video

| GPU Tier | Estimated Cost | Time Required |
|----------|----------------|---------------|
| A10G (single) | ~$2,100 | ~2,000 hrs |
| A10G (10x parallel) | ~$2,100 | ~200 hrs |
| A10G (50x parallel) | ~$2,100 | ~40 hrs ✅ |

---

## 🔧 Scaling to 500 Hours Overnight

To process 500 hours of video overnight (the interview question):

### 1. Scene Detection & Segmentation
```python
# Split video at silence boundaries / shot changes
# Each segment is independently processable
from scenedetect import detect, ContentDetector
scenes = detect(video_path, ContentDetector())
```

### 2. Parallel Processing
```python
# Use multiprocessing or distributed compute
# Each scene → separate GPU worker
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=num_gpus) as executor:
    futures = [executor.submit(process_segment, seg) for seg in segments]
```

### 3. Infrastructure
```
                    ┌──────────────┐
                    │  Job Queue   │  (Redis / SQS)
                    │  (30K clips) │
                    └──────┬───────┘
                           │
          ┌────────────────┼────────────────┐
          ↓                ↓                ↓
    ┌───────────┐    ┌───────────┐    ┌───────────┐
    │ GPU Pod 1 │    │ GPU Pod 2 │    │ GPU Pod N │
    │ (A10G)    │    │ (A10G)    │    │ (A10G)    │
    └─────┬─────┘    └─────┬─────┘    └─────┬─────┘
          ↓                ↓                ↓
    ┌───────────────────────────────────────────────┐
    │              Object Storage (S3)              │
    │          (source + processed videos)          │
    └───────────────────────────────────────────────┘
```

### 4. Specific Modifications
- **Kubernetes** with GPU node pools for auto-scaling
- **Batch processing**: Group segments by duration for uniform batch sizes
- **Model server**: Keep models loaded in memory across requests (TorchServe / Triton)
- **Pipeline optimization**: Overlap stages (transcribe segment N+1 while lip-syncing segment N)
- **Checkpointing**: Resume from any failed step without re-processing

---

## 🏆 Design Decisions & Why

| Decision | Why |
|----------|-----|
| **Whisper small** (not base) | Best accuracy/VRAM trade-off for free Colab T4 |
| **IndicTrans2** (not Google) | Context-aware Hindi > literal translation. A nanny would understand it. |
| **XTTS v2** (not ElevenLabs) | Free, local, supports Hindi, voice cloning, fits on T4 |
| **Audio alignment** module | The single biggest quality improvement — syncs lips to speech perfectly |
| **GFPGAN post-processing** | Wav2Lip blurs the face; GFPGAN restores it to near-original quality |
| **Speech rate 1.05x** | Hindi is typically 10-15% longer than English for same content |
| **Google Translate fallback** | IndicTrans2 needs ~4GB VRAM; having a zero-GPU fallback shows resourcefulness |
| **Edge TTS fallback** | Not everyone has GPU; edge-tts runs anywhere and still sounds professional |

---

## ⚠️ Known Limitations

1. **Wav2Lip face quality**: The GAN model improves mouth region but can still produce artifacts with fast head movements
2. **XTTS Hindi prosody**: Voice cloning works well but may not perfectly capture emotional nuances
3. **Single speaker**: Current pipeline assumes one speaker; multi-speaker support needs diarization
4. **Colab timeout**: Free Colab disconnects after ~90 minutes; long videos need batching with checkpoints
5. **Translation context**: Short clips may lose context; full-transcript translation is better

---

## 🔮 What I'd Improve With More Time

1. **VideoReTalking** instead of Wav2Lip — better quality lip-sync, handles more poses
2. **Speaker diarization** (pyannote) for multi-speaker videos
3. **CodeFormer** instead of GFPGAN — higher quality face restoration
4. **Segment-level TTS** — synthesize each sentence separately for better timing alignment
5. **Emotion transfer** — detect emotion in original speech and apply to Hindi synthesis
6. **Audio mixing** — preserve background music/SFX and only replace speech
7. **A/B testing framework** — automated quality metrics (SSIM, PESQ) for parameter tuning
8. **Streaming pipeline** — process video chunks as they arrive instead of batch

---

## 📚 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| openai-whisper | ≥20231117 | Speech recognition |
| TTS (Coqui) | ≥0.22.0 | Voice cloning (XTTS v2) |
| deep-translator | ≥1.11.4 | Google Translate fallback |
| IndicTransToolkit | latest | IndicTrans2 translation |
| gfpgan | ≥1.3.8 | Face restoration |
| librosa | ≥0.10.0 | Audio processing/alignment |
| opencv-python | ≥4.8.0 | Video frame processing |
| torch | ≥2.0.0 | ML framework |
| edge-tts | ≥6.1.9 | TTS fallback (no GPU) |
| FFmpeg | ≥5.0 | Video/audio I/O |

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) — Speech recognition
- [AI4Bharat IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) — Hindi translation
- [Coqui TTS](https://github.com/coqui-ai/TTS) — Voice cloning
- [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) — Lip synchronization
- [GFPGAN](https://github.com/TencentARC/GFPGAN) — Face restoration
