"""
Lip Synchronization Module (Wav2Lip)
=====================================
High-fidelity lip-sync using Wav2Lip: makes the speaker's lips match Hindi audio.

Architecture:
    - Primary: Wav2Lip (inference via subprocess for isolation)
    - The Wav2Lip model modifies only the lower face region
    - Combined with GFPGAN post-processing for face restoration

Setup Requirements:
    1. Clone Wav2Lip repo: git clone https://github.com/Rudrabha/Wav2Lip
    2. Download pretrained model: wav2lip_gan.pth (recommended for quality)
       - URL: https://iiitaphyd-my.sharepoint.com/...
       - Place in: Wav2Lip/checkpoints/wav2lip_gan.pth
    3. Download face detection model: s3fd.pth
       - Place in: Wav2Lip/face_detection/detection/sfd/s3fd.pth

Design for Scale:
    - For full videos, the pipeline splits into scenes/segments
    - Each segment is lip-synced independently (parallelizable)
    - GPU memory managed via batch processing of frames
"""

import logging
import os
import subprocess
import time
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default Wav2Lip repository location
DEFAULT_WAV2LIP_DIR = os.environ.get("WAV2LIP_DIR", "Wav2Lip")


def check_wav2lip_setup(wav2lip_dir: str = DEFAULT_WAV2LIP_DIR) -> dict:
    """Verify Wav2Lip installation and model files.

    Args:
        wav2lip_dir: Path to Wav2Lip repository.

    Returns:
        Dictionary with setup status flags.
    """
    wav2lip_path = Path(wav2lip_dir)

    status = {
        "repo_exists": wav2lip_path.exists(),
        "inference_script": (wav2lip_path / "inference.py").exists(),
        "wav2lip_gan_model": (wav2lip_path / "checkpoints" / "wav2lip_gan.pth").exists(),
        "wav2lip_model": (wav2lip_path / "checkpoints" / "wav2lip.pth").exists(),
        "face_detection": (wav2lip_path / "face_detection").exists(),
    }

    # Use GAN model preferentially (higher quality)
    status["best_model"] = None
    if status["wav2lip_gan_model"]:
        status["best_model"] = str(wav2lip_path / "checkpoints" / "wav2lip_gan.pth")
    elif status["wav2lip_model"]:
        status["best_model"] = str(wav2lip_path / "checkpoints" / "wav2lip.pth")

    status["ready"] = status["repo_exists"] and status["inference_script"] and status["best_model"] is not None

    for key, value in status.items():
        logger.info(f"Wav2Lip check — {key}: {value}")

    return status


def setup_wav2lip(wav2lip_dir: str = DEFAULT_WAV2LIP_DIR) -> str:
    """Clone and set up Wav2Lip repository if not present.

    Args:
        wav2lip_dir: Target directory for cloning.

    Returns:
        Path to the Wav2Lip directory.
    """
    wav2lip_path = Path(wav2lip_dir)

    if not wav2lip_path.exists():
        logger.info("Cloning Wav2Lip repository...")
        subprocess.run(
            ["git", "clone", "https://github.com/Rudrabha/Wav2Lip.git", str(wav2lip_path)],
            check=True, timeout=120
        )
        logger.info("Wav2Lip cloned successfully")

    # Create checkpoints directory
    (wav2lip_path / "checkpoints").mkdir(exist_ok=True)

    # Download models if not present
    gan_model = wav2lip_path / "checkpoints" / "wav2lip_gan.pth"
    if not gan_model.exists():
        logger.warning(
            "⚠️  wav2lip_gan.pth not found!\n"
            "Please download manually from:\n"
            "https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/"
            "EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA?e=n9ljGW\n"
            f"Place it in: {gan_model}"
        )

    return str(wav2lip_path)


def run_lipsync(
    video_path: str,
    audio_path: str,
    output_path: str,
    wav2lip_dir: str = DEFAULT_WAV2LIP_DIR,
    model_name: str = "wav2lip_gan.pth",
    pad_top: int = 0,
    pad_bottom: int = 10,
    pad_left: int = 0,
    pad_right: int = 0,
    resize_factor: int = 1,
    nosmooth: bool = False,
    batch_size: int = 16
) -> str:
    """Run Wav2Lip inference to lip-sync video with Hindi audio.

    Args:
        video_path: Path to input video clip.
        audio_path: Path to Hindi audio file.
        output_path: Path for the lip-synced output video.
        wav2lip_dir: Path to Wav2Lip repository.
        model_name: Model filename in checkpoints/.
                     'wav2lip_gan.pth' = higher quality (recommended).
                     'wav2lip.pth' = standard quality.
        pad_top/bottom/left/right: Face detection padding (pixels).
                                    Adjust if face crops look wrong.
        resize_factor: Downscale factor for processing speed.
        nosmooth: Disable face detection smoothing across frames.
        batch_size: Number of frames to process at once.
                    Lower = less VRAM usage (use 4-8 for free Colab).

    Returns:
        Path to the lip-synced video.

    Raises:
        RuntimeError: If Wav2Lip inference fails.
        FileNotFoundError: If Wav2Lip is not properly set up.
    """
    wav2lip_path = Path(wav2lip_dir)

    # Validate setup
    status = check_wav2lip_setup(wav2lip_dir)
    if not status["ready"]:
        raise FileNotFoundError(
            f"Wav2Lip not properly set up at {wav2lip_dir}. "
            "Run setup_wav2lip() first or download models manually."
        )

    checkpoint_path = str(wav2lip_path / "checkpoints" / model_name)
    if not Path(checkpoint_path).exists():
        # Fall back to whichever model is available
        checkpoint_path = status["best_model"]
        logger.warning(f"Requested model {model_name} not found, using: {checkpoint_path}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Build inference command
    cmd = [
        "python",
        str(wav2lip_path / "inference.py"),
        "--checkpoint_path", checkpoint_path,
        "--face", str(video_path),
        "--audio", str(audio_path),
        "--outfile", str(output_path),
        "--pads", str(pad_top), str(pad_bottom), str(pad_left), str(pad_right),
        "--resize_factor", str(resize_factor),
        "--wav2lip_batch_size", str(batch_size),
    ]

    if nosmooth:
        cmd.append("--nosmooth")

    logger.info(f"Running Wav2Lip inference...")
    logger.info(f"  Video: {video_path}")
    logger.info(f"  Audio: {audio_path}")
    logger.info(f"  Model: {Path(checkpoint_path).name}")
    logger.info(f"  Batch size: {batch_size}")
    logger.debug(f"  Command: {' '.join(cmd)}")

    start = time.time()

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=600,  # 10 min timeout
            cwd=str(wav2lip_path)  # Run from Wav2Lip directory
        )

        if result.returncode != 0:
            logger.error(f"Wav2Lip stdout: {result.stdout[-500:]}")
            logger.error(f"Wav2Lip stderr: {result.stderr[-500:]}")
            raise RuntimeError(f"Wav2Lip inference failed (exit {result.returncode})")

        elapsed = time.time() - start
        logger.info(f"Wav2Lip inference completed in {elapsed:.1f}s")

        # Wav2Lip may save to a default location — move if needed
        default_output = wav2lip_path / "results" / "result_voice.mp4"
        if default_output.exists() and str(default_output) != str(output_path):
            shutil.move(str(default_output), str(output_path))
            logger.info(f"Moved output to: {output_path}")

        if not Path(output_path).exists():
            raise RuntimeError(
                f"Wav2Lip completed but output file not found at {output_path}. "
                f"Check Wav2Lip logs for details."
            )

        logger.info(f"Lip-synced video saved: {output_path}")
        return output_path

    except subprocess.TimeoutExpired:
        raise RuntimeError("Wav2Lip inference timed out after 600 seconds")


def run_lipsync_batched(
    video_path: str,
    audio_segments: list,
    output_dir: str,
    wav2lip_dir: str = DEFAULT_WAV2LIP_DIR,
    batch_size: int = 8
) -> list:
    """Process multiple segments for full-video lip-sync (scale design).

    For processing 500+ hours of video, this function would:
    1. Split video into scenes at silence/cut boundaries
    2. Process each scene independently (parallelizable across GPUs)
    3. Reassemble the final video

    Args:
        video_path: Full video path.
        audio_segments: List of dicts with audio paths and timestamps.
        output_dir: Directory for segment outputs.
        wav2lip_dir: Wav2Lip directory.
        batch_size: Frames per batch for GPU processing.

    Returns:
        List of output video paths for each segment.
    """
    from modules.video_utils import extract_clip

    output_paths = []
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for i, seg in enumerate(audio_segments):
        logger.info(f"Processing segment {i + 1}/{len(audio_segments)}")

        # Extract video segment
        seg_video = str(Path(output_dir) / f"seg_{i:04d}_video.mp4")
        extract_clip(video_path, seg["start"], seg["end"], seg_video)

        # Run lip-sync on this segment
        seg_output = str(Path(output_dir) / f"seg_{i:04d}_synced.mp4")

        try:
            run_lipsync(
                video_path=seg_video,
                audio_path=seg["audio_path"],
                output_path=seg_output,
                wav2lip_dir=wav2lip_dir,
                batch_size=batch_size
            )
            output_paths.append(seg_output)
        except Exception as e:
            logger.error(f"Segment {i} lip-sync failed: {e}")
            output_paths.append(None)

    return output_paths
