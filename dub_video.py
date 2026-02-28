#!/usr/bin/env python3
"""
AI Video Dubber — Hindi Dubbing Pipeline
==========================================
Main orchestrator that chains all modules into a complete dubbing pipeline.

Pipeline:
    Input Video (Kannada) → Extract Clip → Extract Audio → Whisper Transcription
    → IndicTrans2/Google Translation (Kannada→Hindi) → XTTS Voice Cloning
    → Audio Duration Alignment → Wav2Lip Lip Sync → GFPGAN Enhancement
    → Final Output Video (Hindi)

Usage:
    python dub_video.py --input video.mp4 --start 15 --end 30
    python dub_video.py --input video.mp4 --start 15 --end 30 --translation google
    python dub_video.py --input video.mp4 --start 15 --end 30 --skip-enhancement

Author: AI Video Dubber Team
License: MIT
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# ──────────────────────────────────────────────
#  Logging Setup
# ──────────────────────────────────────────────

def setup_logging(log_dir: str = "outputs/logs") -> logging.Logger:
    """Configure logging with both file and console handlers."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"pipeline_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger("dub_video")
    logger.info(f"Logging to: {log_file}")
    return logger


# ──────────────────────────────────────────────
#  Pipeline Configuration
# ──────────────────────────────────────────────

class PipelineConfig:
    """Central configuration for the dubbing pipeline."""

    def __init__(self, args):
        # Input/Output
        self.input_video = args.input
        self.start_time = args.start
        self.end_time = args.end
        self.output_dir = args.output_dir

        # Module settings
        self.whisper_model = args.whisper_model
        self.translation_method = args.translation
        self.tts_method = args.tts
        self.speech_rate = args.speech_rate

        # Feature flags
        self.skip_enhancement = args.skip_enhancement
        self.skip_lipsync = args.skip_lipsync
        self.enhancement_method = args.enhancement

        # Wav2Lip settings
        self.wav2lip_dir = args.wav2lip_dir
        self.wav2lip_batch_size = args.wav2lip_batch_size

        # Paths (auto-generated)
        self.clip_video = str(Path(self.output_dir) / "01_clip.mp4")
        self.clip_audio = str(Path(self.output_dir) / "02_audio.wav")
        self.transcription_json = str(Path(self.output_dir) / "03_transcription.json")
        self.translation_json = str(Path(self.output_dir) / "04_translation.json")
        self.hindi_audio_raw = str(Path(self.output_dir) / "05_hindi_raw.wav")
        self.hindi_audio_aligned = str(Path(self.output_dir) / "06_hindi_aligned.wav")
        self.lipsynced_video = str(Path(self.output_dir) / "07_lipsynced.mp4")
        self.enhanced_video = str(Path(self.output_dir) / "08_enhanced.mp4")
        self.final_output = str(Path(self.output_dir) / "final_dubbed.mp4")


# ──────────────────────────────────────────────
#  Pipeline Steps
# ──────────────────────────────────────────────

def step_1_extract_clip(config: PipelineConfig, logger: logging.Logger) -> str:
    """Step 1: Extract the target clip from the full video."""
    from modules.video_utils import extract_clip

    logger.info("=" * 60)
    logger.info("STEP 1: Extracting video clip")
    logger.info(f"  Source: {config.input_video}")
    logger.info(f"  Range: {config.start_time}s - {config.end_time}s")
    logger.info("=" * 60)

    clip_path = extract_clip(
        input_path=config.input_video,
        start_time=config.start_time,
        end_time=config.end_time,
        output_path=config.clip_video,
        reencode=True  # Frame-accurate for short clips
    )

    return clip_path


def step_2_extract_audio(config: PipelineConfig, logger: logging.Logger) -> str:
    """Step 2: Extract audio from the clip."""
    from modules.video_utils import extract_audio

    logger.info("=" * 60)
    logger.info("STEP 2: Extracting audio from clip")
    logger.info("=" * 60)

    # Extract at 16kHz for Whisper transcription
    audio_path = extract_audio(
        video_path=config.clip_video,
        audio_path=config.clip_audio,
        sample_rate=16000,  # Whisper expects 16kHz
        mono=True
    )

    # Also extract at 22050Hz for XTTS voice cloning reference
    # (higher sample rate = better voice quality for cloning)
    config.reference_audio = str(Path(config.output_dir) / "02_reference_22k.wav")
    extract_audio(
        video_path=config.clip_video,
        audio_path=config.reference_audio,
        sample_rate=22050,  # XTTS native rate
        mono=True
    )
    logger.info(f"  Reference audio for voice cloning: {config.reference_audio}")

    return audio_path


def step_3_transcribe(config: PipelineConfig, logger: logging.Logger) -> dict:
    """Step 3: Transcribe audio using Whisper."""
    from modules.transcription import Transcriber

    logger.info("=" * 60)
    logger.info("STEP 3: Transcribing Kannada speech (Whisper)")
    logger.info(f"  Model: {config.whisper_model}")
    logger.info("=" * 60)

    transcriber = Transcriber(model_size=config.whisper_model)
    transcription = transcriber.transcribe(
        audio_path=config.clip_audio,
        language="kn",
        word_timestamps=True
    )

    # Save transcription
    transcriber.save_transcription(transcription, config.transcription_json)

    logger.info(f"  Transcript: {transcription['text']}")
    logger.info(f"  Segments: {len(transcription['segments'])}")

    return transcription


def step_4_translate(config: PipelineConfig, logger: logging.Logger, transcription: dict) -> dict:
    """Step 4: Translate Kannada to Hindi."""
    from modules.translation import Translator

    logger.info("=" * 60)
    logger.info("STEP 4: Translating to Hindi")
    logger.info(f"  Method: {config.translation_method}")
    logger.info("=" * 60)

    translator = Translator(method=config.translation_method)

    translation = translator.translate(
        text=transcription["text"],
        source_segments=transcription["segments"]
    )

    # Save translation
    with open(config.translation_json, "w", encoding="utf-8") as f:
        json.dump(translation, f, indent=2, ensure_ascii=False)

    logger.info(f"  Hindi: {translation['full_text'][:100]}...")

    return translation


def step_5_voice_clone(config: PipelineConfig, logger: logging.Logger, translation: dict) -> str:
    """Step 5: Generate Hindi speech with voice cloning."""
    from modules.tts import VoiceCloner

    logger.info("=" * 60)
    logger.info("STEP 5: Generating Hindi speech (Voice Cloning)")
    logger.info(f"  Method: {config.tts_method}")
    logger.info(f"  Speech rate: {config.speech_rate}")
    logger.info("=" * 60)

    cloner = VoiceCloner(
        method=config.tts_method,
        speech_rate=config.speech_rate
    )

    # Use the 22kHz reference audio for better voice cloning quality.
    # Falls back to the 16kHz Whisper audio if reference doesn't exist.
    reference = getattr(config, 'reference_audio', config.clip_audio)
    if not Path(reference).exists():
        reference = config.clip_audio
        logger.warning("22kHz reference not found, using 16kHz audio for voice cloning")

    hindi_audio = cloner.synthesize(
        text=translation["full_text"],
        reference_audio=reference,
        output_path=config.hindi_audio_raw,
        language="hi"
    )

    return hindi_audio


def step_6_align_audio(config: PipelineConfig, logger: logging.Logger) -> str:
    """Step 6: Align Hindi audio duration to video clip."""
    from modules.alignment import align_audio_to_video, analyze_alignment_quality
    from modules.video_utils import get_media_duration

    logger.info("=" * 60)
    logger.info("STEP 6: Aligning audio duration to video")
    logger.info("=" * 60)

    # Get target duration from the video clip
    target_duration = get_media_duration(config.clip_video)

    aligned_audio = align_audio_to_video(
        generated_audio_path=config.hindi_audio_raw,
        target_duration=target_duration,
        output_path=config.hindi_audio_aligned,
        trim_first=True,
        tolerance=0.3
    )

    # Log alignment quality
    quality = analyze_alignment_quality(aligned_audio, target_duration)
    logger.info(f"  Alignment quality: {quality}")

    return aligned_audio


def step_7_lipsync(config: PipelineConfig, logger: logging.Logger) -> str:
    """Step 7: Lip-sync the video with Hindi audio."""
    from modules.lipsync import run_lipsync, check_wav2lip_setup

    logger.info("=" * 60)
    logger.info("STEP 7: Lip-syncing video (Wav2Lip)")
    logger.info("=" * 60)

    if config.skip_lipsync:
        logger.info("  Lip-sync SKIPPED (--skip-lipsync flag)")
        # Just merge audio and video without lip-sync
        from modules.video_utils import merge_audio_video
        merge_audio_video(
            config.clip_video,
            config.hindi_audio_aligned,
            config.lipsynced_video
        )
        return config.lipsynced_video

    # Check Wav2Lip setup
    status = check_wav2lip_setup(config.wav2lip_dir)
    if not status["ready"]:
        logger.warning("Wav2Lip not set up. Falling back to audio-only merge.")
        logger.warning("To enable lip-sync, run: python setup.py --setup-wav2lip")
        from modules.video_utils import merge_audio_video
        merge_audio_video(
            config.clip_video,
            config.hindi_audio_aligned,
            config.lipsynced_video
        )
        return config.lipsynced_video

    lipsynced = run_lipsync(
        video_path=config.clip_video,
        audio_path=config.hindi_audio_aligned,
        output_path=config.lipsynced_video,
        wav2lip_dir=config.wav2lip_dir,
        batch_size=config.wav2lip_batch_size
    )

    return lipsynced


def step_8_enhance(config: PipelineConfig, logger: logging.Logger) -> str:
    """Step 8: Enhance face quality with GFPGAN."""
    from modules.enhancement import FaceEnhancer

    logger.info("=" * 60)
    logger.info("STEP 8: Enhancing face quality (GFPGAN)")
    logger.info("=" * 60)

    if config.skip_enhancement:
        logger.info("  Enhancement SKIPPED (--skip-enhancement flag)")
        import shutil
        shutil.copy2(config.lipsynced_video, config.enhanced_video)
        return config.enhanced_video

    try:
        enhancer = FaceEnhancer(method=config.enhancement_method)
        enhanced = enhancer.enhance_video(
            input_video=config.lipsynced_video,
            output_video=config.enhanced_video,
            audio_path=config.hindi_audio_aligned
        )
        return enhanced
    except Exception as e:
        logger.warning(f"Enhancement failed: {e}")
        logger.warning("Continuing without enhancement...")
        import shutil
        shutil.copy2(config.lipsynced_video, config.enhanced_video)
        return config.enhanced_video


def step_9_finalize(config: PipelineConfig, logger: logging.Logger) -> str:
    """Step 9: Produce the final output video."""
    from modules.video_utils import merge_audio_video, get_media_duration

    logger.info("=" * 60)
    logger.info("STEP 9: Finalizing output")
    logger.info("=" * 60)

    import shutil

    # The enhanced video should already have audio
    # If not, merge it one more time
    if Path(config.enhanced_video).exists():
        shutil.copy2(config.enhanced_video, config.final_output)
    else:
        merge_audio_video(
            config.lipsynced_video,
            config.hindi_audio_aligned,
            config.final_output
        )

    # Log final statistics
    duration = get_media_duration(config.final_output)
    file_size = os.path.getsize(config.final_output) / (1024 * 1024)

    logger.info(f"  Final video: {config.final_output}")
    logger.info(f"  Duration: {duration:.2f}s")
    logger.info(f"  File size: {file_size:.1f} MB")

    return config.final_output


# ──────────────────────────────────────────────
#  Main Pipeline
# ──────────────────────────────────────────────

def run_pipeline(config: PipelineConfig, logger: logging.Logger) -> str:
    """Execute the full dubbing pipeline.

    Args:
        config: Pipeline configuration.
        logger: Logger instance.

    Returns:
        Path to the final output video.
    """
    pipeline_start = time.time()

    logger.info("🎬 AI Video Dubber — Hindi Dubbing Pipeline")
    logger.info(f"  Input: {config.input_video}")
    logger.info(f"  Segment: {config.start_time}s - {config.end_time}s")
    logger.info(f"  Output: {config.output_dir}")
    logger.info("")

    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_path = Path(config.output_dir) / "pipeline_config.json"
    with open(config_path, "w") as f:
        json.dump(vars(config), f, indent=2)

    try:
        # Step 1: Extract clip
        step_1_extract_clip(config, logger)

        # Step 2: Extract audio
        step_2_extract_audio(config, logger)

        # Step 3: Transcribe
        transcription = step_3_transcribe(config, logger)

        # Step 4: Translate
        translation = step_4_translate(config, logger, transcription)

        # Step 5: Voice clone
        step_5_voice_clone(config, logger, translation)

        # Step 6: Align audio
        step_6_align_audio(config, logger)

        # Step 7: Lip-sync
        step_7_lipsync(config, logger)

        # Step 8: Enhance
        step_8_enhance(config, logger)

        # Step 9: Finalize
        final_output = step_9_finalize(config, logger)

    except Exception as e:
        logger.error(f"❌ Pipeline failed at: {e}", exc_info=True)
        raise

    pipeline_elapsed = time.time() - pipeline_start

    logger.info("")
    logger.info("=" * 60)
    logger.info("✅ PIPELINE COMPLETE")
    logger.info(f"  Total time: {pipeline_elapsed:.1f}s ({pipeline_elapsed / 60:.1f} min)")
    logger.info(f"  Output: {final_output}")
    logger.info("=" * 60)

    # Generate cost estimate
    clip_duration = config.end_time - config.start_time
    cost_per_minute = estimate_cost_per_minute(pipeline_elapsed, clip_duration)
    logger.info(f"\n📊 Cost Estimate:")
    logger.info(f"  Processing time per minute of video: {cost_per_minute['time_per_min']:.0f}s")
    logger.info(f"  Estimated cost per minute (cloud GPU): ${cost_per_minute['cost_per_min']:.3f}")
    logger.info(f"  Estimated cost for 500 hours: ${cost_per_minute['cost_500hrs']:.2f}")

    return final_output


def estimate_cost_per_minute(
    processing_time: float,
    clip_duration: float
) -> dict:
    """Estimate processing costs for scaling.

    Based on current pipeline performance, extrapolate costs for
    production-scale processing.

    Args:
        processing_time: Time taken for this clip in seconds.
        clip_duration: Duration of the processed clip in seconds.

    Returns:
        Cost estimation dictionary.
    """
    time_per_min = (processing_time / clip_duration) * 60

    # Cloud GPU pricing estimates (2025 rates)
    gpu_cost_per_hour = {
        "colab_free": 0.00,
        "colab_pro": 0.10,      # ~$10/100 compute units
        "aws_g4dn": 0.526,      # g4dn.xlarge (T4)
        "aws_g5": 1.006,        # g5.xlarge (A10G)
        "lambda_a10": 0.75,     # Lambda Cloud A10
    }

    # Use A10G as reference for production
    cost_per_hour = gpu_cost_per_hour["aws_g5"]
    cost_per_second = cost_per_hour / 3600
    cost_per_min_video = time_per_min * cost_per_second

    # 500 hours = 30,000 minutes
    cost_500hrs = cost_per_min_video * 30000

    return {
        "time_per_min": time_per_min,
        "cost_per_min": cost_per_min_video,
        "cost_500hrs": cost_500hrs,
        "processing_ratio": time_per_min / 60,  # How many seconds to process 1 second of video
    }


# ──────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AI Video Dubber — Kannada to Hindi Video Dubbing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process 15-second clip (default: 0:15 - 0:30)
  python dub_video.py --input video.mp4

  # Custom time range
  python dub_video.py --input video.mp4 --start 30 --end 45

  # Use Google Translate (faster, no GPU needed for translation)
  python dub_video.py --input video.mp4 --translation google

  # Skip heavy stages for quick testing
  python dub_video.py --input video.mp4 --skip-lipsync --skip-enhancement

  # Use Edge TTS instead of XTTS (no GPU needed for TTS)
  python dub_video.py --input video.mp4 --tts edge
        """
    )

    # Required
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input video file"
    )

    # Time range
    parser.add_argument(
        "--start", "-s",
        type=float, default=15.0,
        help="Start time in seconds (default: 15)"
    )
    parser.add_argument(
        "--end", "-e",
        type=float, default=30.0,
        help="End time in seconds (default: 30)"
    )

    # Output
    parser.add_argument(
        "--output-dir", "-o",
        default="outputs",
        help="Output directory (default: outputs/)"
    )

    # Model selection
    parser.add_argument(
        "--whisper-model",
        default="small",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: small)"
    )
    parser.add_argument(
        "--translation",
        default="google",
        choices=["indictrans2", "google", "seamless"],
        help="Translation method (default: google)"
    )
    parser.add_argument(
        "--tts",
        default="xtts",
        choices=["xtts", "edge"],
        help="TTS method (default: xtts)"
    )
    parser.add_argument(
        "--speech-rate",
        type=float, default=1.05,
        help="Speech rate multiplier (default: 1.05, slight speedup for Hindi sync)"
    )

    # Enhancement
    parser.add_argument(
        "--enhancement",
        default="gfpgan",
        choices=["gfpgan", "codeformer"],
        help="Face enhancement method (default: gfpgan)"
    )

    # Feature flags
    parser.add_argument(
        "--skip-enhancement",
        action="store_true",
        help="Skip GFPGAN face enhancement"
    )
    parser.add_argument(
        "--skip-lipsync",
        action="store_true",
        help="Skip Wav2Lip lip synchronization"
    )

    # Wav2Lip
    parser.add_argument(
        "--wav2lip-dir",
        default="Wav2Lip",
        help="Path to Wav2Lip repository (default: Wav2Lip/)"
    )
    parser.add_argument(
        "--wav2lip-batch-size",
        type=int, default=16,
        help="Wav2Lip batch size — lower for less VRAM (default: 16)"
    )

    return parser.parse_args()


# ──────────────────────────────────────────────
#  Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    logger = setup_logging()

    # Auto-detect TTS availability (Coqui TTS doesn't support Python 3.12+)
    if args.tts == "xtts":
        try:
            from TTS.api import TTS as _TTS_Check
            logger.info("XTTS v2 available — using voice cloning")
        except ImportError:
            logger.warning("⚠️  Coqui TTS not installed (Python 3.12+ incompatible)")
            logger.warning("⚠️  Automatically switching to Edge TTS (still high quality)")
            args.tts = "edge"

    # Validate input
    if not Path(args.input).exists():
        logger.error(f"Input video not found: {args.input}")
        sys.exit(1)

    if args.end <= args.start:
        logger.error(f"End time ({args.end}) must be greater than start time ({args.start})")
        sys.exit(1)

    config = PipelineConfig(args)

    try:
        output = run_pipeline(config, logger)
        print(f"\n✅ Done! Output: {output}")
    except KeyboardInterrupt:
        logger.info("\n⚠️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)
