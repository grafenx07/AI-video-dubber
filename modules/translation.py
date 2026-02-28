"""
Translation Module (IndicTrans2 + Fallback)
============================================
Kannada → Hindi translation using AI4Bharat's IndicTrans2 model.

Architecture:
    - Primary: IndicTrans2 (ai4bharat/indictrans2-indic-indic-1B) — best quality, free
    - Fallback: Deep Translator (Google Translate API) — no GPU needed, free tier
    - Both support batch processing for long documents

Design Notes:
    - Translates full sentences/paragraphs for contextual accuracy
    - Preserves sentence structure for lip-sync timing correlation
    - Batch-ready for scaling to full video transcriptions
    - IndicTrans2 provides significantly better Hindi quality than generic MT
"""

import logging
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class Translator:
    """Kannada to Hindi translator with IndicTrans2 and fallback support."""

    def __init__(self, method: str = "indictrans2", device: Optional[str] = None):
        """Initialize the translator.

        Args:
            method: Translation method — 'indictrans2' or 'google'.
            device: 'cuda' or 'cpu'. Auto-detects if None.
        """
        import torch
        self.method = method
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._tokenizer = None
        self._ip = None  # IndicProcessor

        logger.info(f"Translator initialized: method={method}, device={self.device}")

    def _load_indictrans2(self):
        """Load the IndicTrans2 model and tokenizer."""
        if self._model is not None:
            return

        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            from IndicTransToolkit import IndicProcessor

            model_name = "ai4bharat/indictrans2-indic-indic-1B"

            logger.info(f"Loading IndicTrans2 model: {model_name}")
            start = time.time()

            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, trust_remote_code=True
            ).to(self.device)

            self._ip = IndicProcessor(inference=True)

            elapsed = time.time() - start
            logger.info(f"IndicTrans2 loaded in {elapsed:.1f}s on {self.device}")

        except ImportError as e:
            logger.warning(f"IndicTrans2 dependencies not found: {e}")
            logger.warning("Falling back to Google Translate")
            self.method = "google"
        except Exception as e:
            logger.warning(f"IndicTrans2 load failed: {e}")
            logger.warning("Falling back to Google Translate")
            self.method = "google"

    def _translate_indictrans2(self, sentences: List[str]) -> List[str]:
        """Translate using IndicTrans2.

        Args:
            sentences: List of English sentences.

        Returns:
            List of Hindi translated sentences.
        """
        import torch

        self._load_indictrans2()

        src_lang = "kan_Knda"
        tgt_lang = "hin_Deva"

        # Preprocess with IndicProcessor
        batch = self._ip.preprocess_batch(
            sentences, src_lang=src_lang, tgt_lang=tgt_lang
        )

        # Tokenize
        inputs = self._tokenizer(
            batch,
            truncation=True,
            padding="longest",
            max_length=256,
            return_tensors="pt"
        ).to(self.device)

        # Generate translation
        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                num_beams=5,
                num_return_sequences=1,
                max_length=256,
            )

        # Decode
        with self._tokenizer.as_target_tokenizer():
            translations = self._tokenizer.batch_decode(
                generated, skip_special_tokens=True
            )

        # Postprocess
        translations = self._ip.postprocess_batch(
            translations, lang=tgt_lang
        )

        return translations

    def _translate_google(self, sentences: List[str]) -> List[str]:
        """Translate using Google Translate via deep_translator (free).

        Args:
            sentences: List of Kannada sentences.

        Returns:
            List of Hindi translated sentences.
        """
        from deep_translator import GoogleTranslator

        translator = GoogleTranslator(source="kn", target="hi")

        translations = []
        for sentence in sentences:
            if not sentence.strip():
                translations.append("")
                continue

            try:
                result = translator.translate(sentence)
                translations.append(result)
            except Exception as e:
                logger.warning(f"Google Translate failed for: '{sentence[:50]}...' — {e}")
                translations.append(sentence)  # Keep original as fallback

        return translations

    def _translate_seamless(self, sentences: List[str]) -> List[str]:
        """Translate using Meta's SeamlessM4T model (free, good quality).

        Args:
            sentences: List of Kannada sentences.

        Returns:
            List of Hindi translated sentences.
        """
        import torch
        from transformers import AutoProcessor, SeamlessM4TModel

        if self._model is None:
            model_name = "facebook/hf-seamless-m4t-medium"
            logger.info(f"Loading SeamlessM4T: {model_name}")
            start = time.time()

            self._processor = AutoProcessor.from_pretrained(model_name)
            self._model = SeamlessM4TModel.from_pretrained(model_name).to(self.device)

            elapsed = time.time() - start
            logger.info(f"SeamlessM4T loaded in {elapsed:.1f}s")

        translations = []
        for sentence in sentences:
            if not sentence.strip():
                translations.append("")
                continue

            inputs = self._processor(
                text=sentence, src_lang="kan", return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                output_tokens = self._model.generate(
                    **inputs,
                    tgt_lang="hin",
                    generate_speech=False
                )

            result = self._processor.decode(
                output_tokens[0].tolist()[0], skip_special_tokens=True
            )
            translations.append(result)

        return translations

    def translate(
        self,
        text: str,
        source_segments: Optional[List[Dict]] = None
    ) -> Dict:
        """Translate text from English to Hindi.

        For best results, pass the full paragraph — context-aware translation
        produces more natural Hindi than sentence-by-sentence.

        Args:
            text: Full Kannada text to translate.
            source_segments: Optional list of segments with timestamps.
                             If provided, each segment is translated individually
                             to preserve timing correlation.

        Returns:
            Dictionary with:
                - 'full_text': Complete Hindi translation
                - 'segments': Per-segment translations (if source_segments given)
        """
        logger.info(f"Translating {len(text)} chars to Hindi via {self.method}")
        start = time.time()

        result = {"full_text": "", "segments": []}

        if source_segments:
            # Translate segment by segment to preserve timing correlation
            sentences = [seg["text"] for seg in source_segments]

            # Batch translate all sentences at once for context
            if self.method == "indictrans2":
                translations = self._translate_indictrans2(sentences)
            elif self.method == "seamless":
                translations = self._translate_seamless(sentences)
            else:
                translations = self._translate_google(sentences)

            for seg, hindi_text in zip(source_segments, translations):
                result["segments"].append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "original": seg["text"],
                    "translated": hindi_text,
                })

            result["full_text"] = " ".join(translations)
        else:
            # Translate as a single block for better context
            sentences = [s.strip() for s in text.split(".") if s.strip()]
            sentences = [s + "." for s in sentences]

            if not sentences:
                sentences = [text]

            if self.method == "indictrans2":
                translations = self._translate_indictrans2(sentences)
            elif self.method == "seamless":
                translations = self._translate_seamless(sentences)
            else:
                translations = self._translate_google(sentences)

            result["full_text"] = " ".join(translations)

        elapsed = time.time() - start
        logger.info(f"Translation completed in {elapsed:.1f}s")
        logger.info(f"Hindi output: {result['full_text'][:100]}...")

        return result

    def translate_batch(
        self,
        sentences: List[str],
        batch_size: int = 16
    ) -> List[str]:
        """Translate a batch of sentences efficiently.

        Designed for scaling: processes sentences in batches to optimize
        GPU memory usage for long video transcriptions.

        Args:
            sentences: List of English sentences.
            batch_size: Number of sentences per batch.

        Returns:
            List of Hindi translations (same order as input).
        """
        all_translations = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            logger.info(f"Translating batch {i // batch_size + 1}: "
                        f"{len(batch)} sentences")

            if self.method == "indictrans2":
                translations = self._translate_indictrans2(batch)
            elif self.method == "seamless":
                translations = self._translate_seamless(batch)
            else:
                translations = self._translate_google(batch)

            all_translations.extend(translations)

        return all_translations
