"""
Voice Cloning Module (Coqui XTTS v2)
=====================================
Hindi speech synthesis with voice cloning using XTTS v2.

Features:
    - Clone any speaker's voice from a short reference audio
    - Generate Hindi speech matching the original speaker's tone/style
    - Adjustable speech rate for duration matching
    - Chunked synthesis for long text (prevents OOM on free GPUs)
    - Automatic sentence splitting for natural prosody

Model: tts_models/multilingual/multi-dataset/xtts_v2
    - Supports Hindi (hi) natively
    - Requires ~4GB VRAM (fits on free Colab T4)
    - 24kHz output quality

Fallback: Edge TTS (Microsoft) — free, no GPU, decent quality
"""

import logging
import os
import time
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class VoiceCloner:
    """XTTS v2 based voice cloning for Hindi speech synthesis."""

    def __init__(
        self,
        method: str = "xtts",
        device: Optional[str] = None,
        speech_rate: float = 1.0
    ):
        """Initialize the voice cloner.

        Args:
            method: TTS method — 'xtts' or 'edge' (fallback).
            device: 'cuda' or 'cpu'. Auto-detects if None.
            speech_rate: Speed multiplier. >1.0 = faster, <1.0 = slower.
                         1.05 is recommended for Hindi (typically longer than English).
        """
        self.method = method
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.speech_rate = speech_rate
        self._model = None

        logger.info(f"VoiceCloner initialized: method={method}, device={self.device}, "
                     f"speech_rate={speech_rate}")

    def _load_xtts(self):
        """Load the XTTS v2 model."""
        if self._model is not None:
            return

        from TTS.api import TTS

        logger.info("Loading XTTS v2 model...")
        start = time.time()

        self._model = TTS(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            progress_bar=True
        ).to(self.device)

        elapsed = time.time() - start
        logger.info(f"XTTS v2 loaded in {elapsed:.1f}s on {self.device}")

    def _split_hindi_text(self, text: str, max_chars: int = 200) -> List[str]:
        """Split Hindi text into chunks suitable for TTS.

        XTTS works best with shorter sentences. This splits on natural
        boundaries (।, |, ., !, ?) to maintain prosody.

        Args:
            text: Hindi text to split.
            max_chars: Maximum characters per chunk.

        Returns:
            List of text chunks.
        """
        # Split on Hindi sentence boundaries
        delimiters = r'[।|\.!\?]+'
        sentences = re.split(delimiters, text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= max_chars:
                current_chunk = f"{current_chunk} {sentence}".strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)

                # Handle very long sentences
                if len(sentence) > max_chars:
                    # Split on commas or spaces
                    words = sentence.split()
                    sub_chunk = ""
                    for word in words:
                        if len(sub_chunk) + len(word) + 1 <= max_chars:
                            sub_chunk = f"{sub_chunk} {word}".strip()
                        else:
                            if sub_chunk:
                                chunks.append(sub_chunk)
                            sub_chunk = word
                    if sub_chunk:
                        current_chunk = sub_chunk
                    else:
                        current_chunk = ""
                else:
                    current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)

        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks if chunks else [text]

    def synthesize(
        self,
        text: str,
        reference_audio: str,
        output_path: str,
        language: str = "hi"
    ) -> str:
        """Generate speech using XTTS v2 voice cloning.

        Args:
            text: Hindi text to synthesize.
            reference_audio: Path to reference speaker audio (6-15s recommended).
            output_path: Path for the output WAV file.
            language: Target language code.

        Returns:
            Path to the generated audio file.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if self.method == "edge":
            return self._synthesize_edge(text, output_path, language)

        return self._synthesize_xtts(text, reference_audio, output_path, language)

    def _synthesize_xtts(
        self,
        text: str,
        reference_audio: str,
        output_path: str,
        language: str = "hi"
    ) -> str:
        """XTTS v2 synthesis with chunking for long text.

        Args:
            text: Hindi text.
            reference_audio: Speaker reference audio path.
            output_path: Output WAV path.
            language: Language code.

        Returns:
            Path to output file.
        """
        import soundfile as sf

        self._load_xtts()

        logger.info(f"Synthesizing {len(text)} chars of Hindi text")
        start = time.time()

        # Split into chunks for stable generation
        chunks = self._split_hindi_text(text)

        all_audio = []

        for i, chunk in enumerate(chunks):
            logger.info(f"Generating chunk {i + 1}/{len(chunks)}: '{chunk[:50]}...'")

            try:
                wav = self._model.tts(
                    text=chunk,
                    speaker_wav=str(reference_audio),
                    language=language
                )

                # Convert to numpy array
                if isinstance(wav, list):
                    wav = np.array(wav)
                elif isinstance(wav, torch.Tensor):
                    wav = wav.cpu().numpy()

                all_audio.append(wav)

                # Small pause between chunks for natural prosody
                pause = np.zeros(int(0.15 * 24000))  # 150ms pause at 24kHz
                all_audio.append(pause)

            except Exception as e:
                logger.error(f"Failed to synthesize chunk {i + 1}: {e}")
                continue

        if not all_audio:
            raise RuntimeError("All synthesis chunks failed")

        # Concatenate all chunks
        final_audio = np.concatenate(all_audio)

        # Apply speech rate adjustment
        if abs(self.speech_rate - 1.0) > 0.01:
            import librosa
            logger.info(f"Applying speech rate: {self.speech_rate}x")
            final_audio = librosa.effects.time_stretch(
                final_audio, rate=self.speech_rate
            )

        # Save output
        sf.write(output_path, final_audio, samplerate=24000)

        elapsed = time.time() - start
        duration = len(final_audio) / 24000
        logger.info(f"Synthesis completed in {elapsed:.1f}s — output duration: {duration:.2f}s")

        return output_path

    def _synthesize_edge(
        self,
        text: str,
        output_path: str,
        language: str = "hi"
    ) -> str:
        """Fallback synthesis using Microsoft Edge TTS (free, no GPU).

        Note: This does NOT clone the speaker's voice but provides
        high-quality Hindi TTS as a fallback when XTTS is unavailable.

        Args:
            text: Hindi text.
            output_path: Output audio path.
            language: Language code.

        Returns:
            Path to output file.
        """
        import asyncio
        import edge_tts

        logger.info("Using Edge TTS fallback (no voice cloning)")

        # Best Hindi voices from Edge TTS
        voice = "hi-IN-SwaraNeural"  # Female
        # Alternative: "hi-IN-MadhurNeural"  # Male

        async def _generate():
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_path)

        # Run the async function
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                loop.run_until_complete(_generate())
            else:
                asyncio.run(_generate())
        except RuntimeError:
            asyncio.run(_generate())

        logger.info(f"Edge TTS output saved: {output_path}")
        return output_path

    def synthesize_segments(
        self,
        segments: List[dict],
        reference_audio: str,
        output_dir: str,
        language: str = "hi"
    ) -> List[dict]:
        """Synthesize multiple segments individually for precise timing.

        This is the 'production' approach for long videos — each segment
        gets its own audio file, enabling per-segment alignment.

        Args:
            segments: List of dicts with 'translated' text and timing info.
            reference_audio: Speaker reference audio path.
            output_dir: Directory for output audio files.
            language: Language code.

        Returns:
            Updated segments list with 'audio_path' added to each.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for i, seg in enumerate(segments):
            text = seg.get("translated", "")
            if not text.strip():
                continue

            output_path = str(Path(output_dir) / f"segment_{i:04d}.wav")

            try:
                self.synthesize(text, reference_audio, output_path, language)
                seg["audio_path"] = output_path
                logger.info(f"Segment {i}: synthesized '{text[:30]}...'")
            except Exception as e:
                logger.error(f"Segment {i} synthesis failed: {e}")
                seg["audio_path"] = None

        return segments
