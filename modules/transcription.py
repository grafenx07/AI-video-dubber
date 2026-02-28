"""
Transcription Module (OpenAI Whisper)
=====================================
Speech-to-text transcription using OpenAI's Whisper model (local, free).

Features:
    - Loads model once and reuses for multiple transcriptions
    - Word-level timestamps for precise alignment
    - Segment-level timestamps for translation batching
    - JSON export for pipeline persistence
    - Chunked processing support for long audio files

Model Recommendations:
    - "tiny"  : Fastest, lowest accuracy (~1GB VRAM)
    - "base"  : Good balance for testing (~1GB VRAM)
    - "small" : Best for production on free Colab (~2GB VRAM)
    - "medium": Higher accuracy (~5GB VRAM)
    - "large" : Best accuracy (~10GB VRAM, needs paid GPU)
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import whisper
import torch
import numpy as np

logger = logging.getLogger(__name__)


class Transcriber:
    """Whisper-based transcription engine with caching and chunking support."""

    def __init__(self, model_size: str = "small", device: Optional[str] = None):
        """Initialize the transcriber.

        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large').
            device: 'cuda' or 'cpu'. Auto-detects if None.
        """
        self.model_size = model_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

        logger.info(f"Transcriber initialized: model={model_size}, device={self.device}")

    def _load_model(self):
        """Lazy-load the Whisper model (loaded once, reused for all calls)."""
        if self.model is None:
            logger.info(f"Loading Whisper '{self.model_size}' model on {self.device}...")
            start = time.time()
            self.model = whisper.load_model(self.model_size, device=self.device)
            elapsed = time.time() - start
            logger.info(f"Whisper model loaded in {elapsed:.1f}s")

    def transcribe(
        self,
        audio_path: str,
        language: str = "kn",
        word_timestamps: bool = True,
        initial_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Transcribe an audio file.

        Args:
            audio_path: Path to audio file (WAV, MP3, etc.).
            language: Language code (e.g., 'kn', 'hi', 'en').
            word_timestamps: Enable word-level timestamps.
            initial_prompt: Optional context prompt for better accuracy.

        Returns:
            Dictionary containing:
                - 'text': Full transcription text
                - 'segments': List of segments with timestamps
                - 'language': Detected language
                - 'duration': Audio duration in seconds
        """
        self._load_model()

        logger.info(f"Transcribing: {Path(audio_path).name}")
        start = time.time()

        # ── Transcription Strategy ──────────────────────────────────
        # Whisper's Kannada (kn) support is weak on smaller models.
        # CRITICAL FIX: Use translate mode (→ English) FIRST.
        # This gives us clean English text, which Google Translate
        # converts to high-quality Hindi. Previously, the Hindi
        # fallback produced Kannada phonetics in Devanagari script
        # (gibberish that Google Translate couldn't fix).
        # ────────────────────────────────────────────────────────────
        text = ""
        source_language = "en"  # Default: translate mode outputs English

        # Strategy 1: Translate to English (most reliable for our pipeline)
        logger.info("Trying Whisper translate mode (→ English)...")
        result = self.model.transcribe(
            str(audio_path),
            language=language,
            task="translate",
            word_timestamps=word_timestamps,
            initial_prompt=initial_prompt,
            verbose=False
        )
        text = result["text"].strip()
        if text:
            logger.info(f"Translate mode succeeded: {text[:80]}...")

        if not text:
            # Strategy 2: Auto-detect language + translate to English
            logger.warning("Translate with language hint empty. "
                           "Trying auto-detect + translate...")
            result = self.model.transcribe(
                str(audio_path),
                task="translate",
                word_timestamps=word_timestamps,
                verbose=False
            )
            text = result["text"].strip()
            if text:
                logger.info(f"Auto-detect translate succeeded: {text[:80]}...")

        if not text:
            # Strategy 3: Native Kannada transcription
            logger.warning("Translate modes empty. "
                           "Trying native Kannada transcription...")
            result = self.model.transcribe(
                str(audio_path),
                language=language,
                word_timestamps=word_timestamps,
                initial_prompt=initial_prompt,
                verbose=False
            )
            text = result["text"].strip()
            source_language = language  # "kn" — Kannada script output
            if text:
                logger.info(f"Kannada transcription: {text[:80]}...")

        if not text:
            # Strategy 4: Hindi transcription (last resort)
            logger.warning("All methods empty. Trying Hindi as last resort...")
            result = self.model.transcribe(
                str(audio_path),
                language="hi",
                word_timestamps=word_timestamps,
                verbose=False
            )
            text = result["text"].strip()
            source_language = "hi"
            if text:
                logger.info(f"Hindi transcription (last resort): {text[:80]}...")
        # ── End Strategy ────────────────────────────────────────────

        elapsed = time.time() - start
        logger.info(f"Transcription completed in {elapsed:.1f}s")
        logger.info(f"Source language for translation: {source_language}")

        # Structure the output for downstream consumption
        transcription = {
            "text": text,
            "language": result.get("language", language),
            "source_language": source_language,
            "segments": [],
            "word_segments": []
        }

        for seg in result.get("segments", []):
            segment_data = {
                "id": seg["id"],
                "start": round(seg["start"], 3),
                "end": round(seg["end"], 3),
                "text": seg["text"].strip(),
            }
            transcription["segments"].append(segment_data)

            # Extract word-level timestamps if available
            if word_timestamps and "words" in seg:
                for word in seg["words"]:
                    transcription["word_segments"].append({
                        "word": word["word"].strip(),
                        "start": round(word["start"], 3),
                        "end": round(word["end"], 3),
                    })

        logger.info(f"Transcribed {len(transcription['segments'])} segments, "
                     f"{len(transcription['word_segments'])} words")

        return transcription

    def transcribe_chunked(
        self,
        audio_path: str,
        chunk_duration: float = 30.0,
        overlap: float = 2.0,
        language: str = "kn"
    ) -> Dict[str, Any]:
        """Transcribe long audio by splitting into overlapping chunks.

        This is designed for scaling to full-length videos where processing
        the entire audio at once may exceed memory limits.

        Args:
            audio_path: Path to audio file.
            chunk_duration: Duration of each chunk in seconds.
            overlap: Overlap between chunks in seconds (prevents word splitting).
            language: Language code.

        Returns:
            Merged transcription dictionary.
        """
        import librosa

        self._load_model()

        # Load full audio
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        total_duration = len(audio) / sr

        logger.info(f"Chunked transcription: {total_duration:.1f}s audio, "
                     f"{chunk_duration}s chunks, {overlap}s overlap")

        all_segments = []
        all_words = []
        full_text_parts = []

        chunk_start = 0.0
        chunk_idx = 0

        while chunk_start < total_duration:
            chunk_end = min(chunk_start + chunk_duration, total_duration)

            # Extract chunk samples
            start_sample = int(chunk_start * sr)
            end_sample = int(chunk_end * sr)
            chunk_audio = audio[start_sample:end_sample]

            logger.info(f"Processing chunk {chunk_idx}: [{chunk_start:.1f}s - {chunk_end:.1f}s]")

            # Transcribe chunk
            result = self.model.transcribe(
                chunk_audio,
                language=language,
                word_timestamps=True,
                verbose=False
            )

            # Offset timestamps by chunk start time
            for seg in result.get("segments", []):
                adjusted_seg = {
                    "id": len(all_segments),
                    "start": round(seg["start"] + chunk_start, 3),
                    "end": round(seg["end"] + chunk_start, 3),
                    "text": seg["text"].strip(),
                }
                all_segments.append(adjusted_seg)

                if "words" in seg:
                    for word in seg["words"]:
                        all_words.append({
                            "word": word["word"].strip(),
                            "start": round(word["start"] + chunk_start, 3),
                            "end": round(word["end"] + chunk_start, 3),
                        })

            full_text_parts.append(result["text"].strip())

            # Move to next chunk (minus overlap)
            chunk_start += chunk_duration - overlap
            chunk_idx += 1

        transcription = {
            "text": " ".join(full_text_parts),
            "language": language,
            "segments": all_segments,
            "word_segments": all_words,
        }

        logger.info(f"Chunked transcription complete: {len(all_segments)} segments")
        return transcription

    def save_transcription(self, transcription: Dict, output_path: str) -> str:
        """Save transcription to a JSON file.

        Args:
            transcription: Transcription dictionary from transcribe().
            output_path: Path for the output JSON file.

        Returns:
            Path to the saved file.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(transcription, f, indent=2, ensure_ascii=False)

        logger.info(f"Transcription saved: {output_path}")
        return output_path

    @staticmethod
    def load_transcription(path: str) -> Dict:
        """Load a previously saved transcription JSON.

        Args:
            path: Path to the JSON file.

        Returns:
            Transcription dictionary.
        """
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
