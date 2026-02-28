"""
Video Utilities Module
======================
Handles all video/audio I/O operations via FFmpeg subprocess calls.

Functions:
    extract_clip      - Extract a time-bounded segment from a video
    extract_audio     - Demux audio track from a video file
    get_media_duration- Probe duration of any media file
    get_video_fps     - Get framerate of a video
    get_video_resolution - Get width x height
    merge_audio_video - Mux new audio onto a video track
    ensure_output_dir - Create output directories safely

Design Notes:
    - All FFmpeg calls use subprocess with error handling and logging
    - Designed for both local and Colab execution
    - Paths are handled via pathlib for cross-platform compatibility
"""

import subprocess
import json
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def ensure_output_dir(path: str) -> Path:
    """Create output directory if it doesn't exist.

    Args:
        path: Directory path to ensure exists.

    Returns:
        Path object for the directory.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured output directory: {p}")
    return p


def _run_ffmpeg(cmd: list, description: str = "FFmpeg operation") -> subprocess.CompletedProcess:
    """Run an FFmpeg command with standardized error handling.

    Args:
        cmd: Command list to execute.
        description: Human-readable description for logging.

    Returns:
        CompletedProcess instance.

    Raises:
        RuntimeError: If FFmpeg returns non-zero exit code.
    """
    logger.info(f"Running: {description}")
    logger.debug(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=300  # 5 min timeout for safety
        )
        if result.returncode != 0:
            logger.error(f"FFmpeg stderr: {result.stderr}")
            raise RuntimeError(
                f"{description} failed (exit code {result.returncode}): {result.stderr[:500]}"
            )
        return result
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{description} timed out after 300 seconds")
    except FileNotFoundError:
        raise RuntimeError(
            "FFmpeg not found. Install it via: apt-get install ffmpeg (Linux) "
            "or download from https://ffmpeg.org/download.html"
        )


def get_media_duration(path: str) -> float:
    """Get duration of a media file in seconds using ffprobe.

    Args:
        path: Path to audio or video file.

    Returns:
        Duration in seconds as float.

    Raises:
        RuntimeError: If ffprobe fails or duration cannot be parsed.
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        str(path)
    ]

    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30
        )
        data = json.loads(result.stdout)
        duration = float(data["format"]["duration"])
        logger.info(f"Media duration for {Path(path).name}: {duration:.2f}s")
        return duration
    except (KeyError, json.JSONDecodeError, ValueError) as e:
        raise RuntimeError(f"Could not parse duration from {path}: {e}")


def get_video_fps(path: str) -> float:
    """Get the framerate of a video file.

    Args:
        path: Path to video file.

    Returns:
        Frames per second as float.
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-select_streams", "v:0",
        "-print_format", "json",
        "-show_streams",
        str(path)
    ]

    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30
    )
    data = json.loads(result.stdout)

    # Parse r_frame_rate which is like "30/1" or "24000/1001"
    fps_str = data["streams"][0]["r_frame_rate"]
    num, den = map(int, fps_str.split("/"))
    fps = num / den
    logger.info(f"Video FPS for {Path(path).name}: {fps:.2f}")
    return fps


def get_video_resolution(path: str) -> Tuple[int, int]:
    """Get video resolution (width, height).

    Args:
        path: Path to video file.

    Returns:
        Tuple of (width, height).
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-select_streams", "v:0",
        "-print_format", "json",
        "-show_streams",
        str(path)
    ]

    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30
    )
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    width = int(stream["width"])
    height = int(stream["height"])
    logger.info(f"Video resolution for {Path(path).name}: {width}x{height}")
    return width, height


def extract_clip(
    input_path: str,
    start_time: float,
    end_time: float,
    output_path: str,
    reencode: bool = True
) -> str:
    """Extract a time-bounded segment from a video.

    Args:
        input_path: Path to source video.
        start_time: Start time in seconds.
        end_time: End time in seconds.
        output_path: Path for the output clip.
        reencode: If True, re-encode for frame accuracy. If False, use
                  stream copy (faster but may have keyframe misalignment).

    Returns:
        Path to the extracted clip.

    Raises:
        ValueError: If time range is invalid.
        RuntimeError: If FFmpeg operation fails.
    """
    if end_time <= start_time:
        raise ValueError(f"end_time ({end_time}) must be > start_time ({start_time})")

    duration = end_time - start_time
    ensure_output_dir(str(Path(output_path).parent))

    if reencode:
        # Re-encode for frame-accurate cutting — essential for short clips
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-i", str(input_path),
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",          # High quality
            "-c:a", "aac",
            "-b:a", "192k",
            "-avoid_negative_ts", "make_zero",
            str(output_path)
        ]
    else:
        # Stream copy — fast but may be inaccurate at boundaries
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-i", str(input_path),
            "-t", str(duration),
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            str(output_path)
        ]

    _run_ffmpeg(cmd, f"Extracting clip [{start_time:.1f}s - {end_time:.1f}s]")
    logger.info(f"Clip saved: {output_path} ({duration:.1f}s)")
    return output_path


def extract_audio(
    video_path: str,
    audio_path: str,
    sample_rate: int = 22050,
    mono: bool = True
) -> str:
    """Extract audio track from a video file.

    Args:
        video_path: Path to input video.
        audio_path: Path for output audio (typically .wav).
        sample_rate: Audio sample rate in Hz. 22050 is good for TTS pipelines.
        mono: Convert to mono channel.

    Returns:
        Path to the extracted audio file.
    """
    ensure_output_dir(str(Path(audio_path).parent))

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",                     # No video
        "-acodec", "pcm_s16le",   # 16-bit PCM WAV
        "-ar", str(sample_rate),
        "-ac", "1" if mono else "2",
        str(audio_path)
    ]

    _run_ffmpeg(cmd, f"Extracting audio from {Path(video_path).name}")
    logger.info(f"Audio saved: {audio_path}")
    return audio_path


def merge_audio_video(
    video_path: str,
    audio_path: str,
    output_path: str,
    keep_original_audio: bool = False
) -> str:
    """Merge a new audio track with a video file.

    Args:
        video_path: Path to video (may contain old audio).
        audio_path: Path to new audio track.
        output_path: Path for the merged output.
        keep_original_audio: If True, mix both audio tracks.

    Returns:
        Path to the output file.
    """
    ensure_output_dir(str(Path(output_path).parent))

    if keep_original_audio:
        # Mix both audio tracks
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=shortest",
            "-c:v", "copy",
            str(output_path)
        ]
    else:
        # Replace audio entirely
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            str(output_path)
        ]

    _run_ffmpeg(cmd, "Merging audio and video")
    logger.info(f"Merged output saved: {output_path}")
    return output_path


def extract_frames(
    video_path: str,
    output_dir: str,
    fps: Optional[float] = None
) -> str:
    """Extract all frames from a video as PNG images.

    Args:
        video_path: Path to source video.
        output_dir: Directory to save frames.
        fps: If specified, extract at this FPS. If None, use video's native FPS.

    Returns:
        Path to the output directory containing frames.
    """
    out_path = ensure_output_dir(output_dir)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
    ]

    if fps:
        cmd.extend(["-vf", f"fps={fps}"])

    cmd.append(str(out_path / "frame_%05d.png"))

    _run_ffmpeg(cmd, f"Extracting frames from {Path(video_path).name}")

    frame_count = len(list(out_path.glob("frame_*.png")))
    logger.info(f"Extracted {frame_count} frames to {output_dir}")
    return str(out_path)


def frames_to_video(
    frames_dir: str,
    output_path: str,
    fps: float,
    audio_path: Optional[str] = None
) -> str:
    """Reassemble frames into a video with optional audio.

    Args:
        frames_dir: Directory containing frame_XXXXX.png files.
        output_path: Path for the output video.
        fps: Framerate for the output video.
        audio_path: Optional audio file to mux in.

    Returns:
        Path to the output video.
    """
    ensure_output_dir(str(Path(output_path).parent))

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(Path(frames_dir) / "frame_%05d.png"),
    ]

    if audio_path:
        cmd.extend(["-i", str(audio_path)])

    cmd.extend([
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
    ])

    if audio_path:
        cmd.extend(["-c:a", "aac", "-b:a", "192k", "-shortest"])

    cmd.append(str(output_path))

    _run_ffmpeg(cmd, "Assembling frames into video")
    logger.info(f"Video assembled: {output_path}")
    return output_path
