"""
AI Video Dubber - Modular Hindi Dubbing Pipeline
================================================

Modules:
    video_utils   - Video/audio extraction and manipulation via FFmpeg
    transcription - Speech-to-text using OpenAI Whisper
    translation   - Kannada→Hindi translation using IndicTrans2
    tts           - Voice cloning using Coqui XTTS v2
    alignment     - Audio duration matching and silence trimming
    lipsync       - Lip synchronization using Wav2Lip
    enhancement   - Face restoration using GFPGAN/CodeFormer
"""

__version__ = "1.0.0"
