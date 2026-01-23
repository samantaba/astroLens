"""
LangChain-based image annotator.

Supports OpenAI (GPT-4o) and local Ollama (LLaVA) for multimodal annotation.
"""

from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .prompts import IMAGE_ANNOTATION_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class AnnotationOutput:
    """LLM-generated annotation."""
    description: str
    hypothesis: str
    follow_up: str
    model_used: str


class ImageAnnotator:
    """
    Annotates images with LLM-generated descriptions.
    
    Supports:
    - OpenAI GPT-4o (multimodal, requires API key)
    - Ollama LLaVA (local, no API key needed)
    - None (returns placeholder text)
    """

    def __init__(
        self,
        provider: str = "openai",  # "openai", "ollama", "none"
        model: Optional[str] = None,
        temperature: float = 0.3,
    ):
        self.provider = provider
        self.temperature = temperature
        self.llm = None
        self.model = model

        if provider == "openai":
            self.model = model or "gpt-4o"
            self._init_openai()
        elif provider == "ollama":
            self.model = model or "llava"
            self._init_ollama()
        else:
            self.model = "none"
            logger.info("LLM provider set to 'none'; annotations will be placeholders")

    def _init_openai(self):
        """Initialize OpenAI client."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set; OpenAI annotations will fail")
            self.llm = None
            return

        try:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model=self.model,
                temperature=self.temperature,
                api_key=api_key,
            )
            logger.info(f"Initialized OpenAI annotator with model {self.model}")
        except ImportError:
            logger.error("langchain-openai not installed")
            self.llm = None

    def _init_ollama(self):
        """Initialize Ollama client."""
        ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        
        try:
            from langchain_community.chat_models import ChatOllama
            self.llm = ChatOllama(
                model=self.model,
                base_url=ollama_url,
                temperature=self.temperature,
            )
            logger.info(f"Initialized Ollama annotator with model {self.model}")
        except ImportError:
            logger.error("langchain-community not installed")
            self.llm = None

    def annotate(
        self,
        image_path: str,
        class_label: str = "unknown",
        confidence: float = 0.0,
        ood_score: float = 0.0,
    ) -> AnnotationOutput:
        """
        Generate annotation for an image.
        
        Args:
            image_path: Path to image file
            class_label: ML classification result
            confidence: Classification confidence
            ood_score: Anomaly detection score
        
        Returns:
            AnnotationOutput with description, hypothesis, follow-up
        """
        if self.llm is None:
            return AnnotationOutput(
                description="[LLM not configured]",
                hypothesis="[LLM not configured]",
                follow_up="[LLM not configured]",
                model_used="none",
            )

        # Build prompt
        prompt = IMAGE_ANNOTATION_PROMPT.format(
            class_label=class_label,
            confidence=confidence,
            ood_score=ood_score,
        )

        try:
            # Load image as base64 for multimodal
            image_base64 = self._load_image_base64(image_path)
            
            if self.provider == "openai" and image_base64:
                # Multimodal message with image
                from langchain_core.messages import HumanMessage

                message = HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                        },
                    ]
                )
                response = self.llm.invoke([message])
            else:
                # Text-only fallback
                response = self.llm.invoke(prompt)

            raw = response.content if hasattr(response, "content") else str(response)
            return self._parse_response(raw)

        except Exception as e:
            logger.error(f"Annotation failed: {e}")
            return AnnotationOutput(
                description=f"[Error: {e}]",
                hypothesis="",
                follow_up="",
                model_used=self.model,
            )

    def _load_image_base64(self, image_path: str) -> Optional[str]:
        """Load image as base64 string."""
        path = Path(image_path)
        if not path.exists():
            return None

        # For FITS, convert to PNG first
        if path.suffix.lower() == ".fits":
            try:
                from astropy.io import fits
                from PIL import Image
                import io
                import numpy as np

                with fits.open(path) as hdu:
                    data = hdu[0].data
                    if data is None:
                        return None
                    # Normalize to 0-255
                    data = data.astype(np.float32)
                    data = (data - data.min()) / (data.max() - data.min() + 1e-8) * 255
                    data = data.astype(np.uint8)
                    if data.ndim == 2:
                        img = Image.fromarray(data, mode="L").convert("RGB")
                    else:
                        img = Image.fromarray(data)
                    
                    buffer = io.BytesIO()
                    img.save(buffer, format="PNG")
                    return base64.b64encode(buffer.getvalue()).decode()
            except Exception as e:
                logger.warning(f"Failed to convert FITS to base64: {e}")
                return None
        else:
            try:
                with open(path, "rb") as f:
                    return base64.b64encode(f.read()).decode()
            except Exception:
                return None

    def _parse_response(self, raw: str) -> AnnotationOutput:
        """Parse structured response from LLM."""
        description = ""
        hypothesis = ""
        follow_up = ""

        for line in raw.split("\n"):
            line_lower = line.lower()
            if "**description:**" in line_lower or line_lower.startswith("description:"):
                description = line.split(":", 1)[-1].strip().strip("*")
            elif "**hypothesis:**" in line_lower or line_lower.startswith("hypothesis:"):
                hypothesis = line.split(":", 1)[-1].strip().strip("*")
            elif "**follow-up:**" in line_lower or line_lower.startswith("follow-up:"):
                follow_up = line.split(":", 1)[-1].strip().strip("*")

        # Fallback: if parsing failed, use whole response as description
        if not description and not hypothesis:
            description = raw[:500]

        return AnnotationOutput(
            description=description,
            hypothesis=hypothesis,
            follow_up=follow_up,
            model_used=self.model,
        )
