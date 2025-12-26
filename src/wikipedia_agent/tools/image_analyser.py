#!/usr/bin/env python3
"""
Image analysis agent built on top of OpenAI-compatible chat completions.

This tool provides image analysis capabilities using vision-capable LLMs.
Supports OpenAI and Hugging Face providers with OpenAI-compatible APIs.

Usage:
    from wikipedia_agent.tools import ImageAnalyzerAgent

    agent = ImageAnalyzerAgent()
    result = agent.analyze_image(Path("photo.jpg"), "Describe this image")
"""

import argparse
import base64
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import openai
from dotenv import load_dotenv

from wikipedia_agent.core.providers import ProviderFactory, get_provider_config_from_env


# Default configuration
DEFAULT_BASE_URL = "http://localhost:1234/v1/"
DEFAULT_MODEL = "qwen/qwen3-vl-4b"
DEFAULT_PROMPT = "Describe this image in detail."
DEFAULT_TEMPERATURE = 0.0
SUPPORTED_OPENAI_COMPATIBLE_PROVIDERS = {"openai", "huggingface", "hf"}


def encode_image_to_base64(image_path: Path) -> str:
    """Return a base64 string for the image specified by ``image_path``."""
    with image_path.open("rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def _detect_mime_type(image_path: Path) -> str:
    """Best-effort MIME type selection based on file extension."""
    mime_types: Dict[str, str] = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
    }
    return mime_types.get(image_path.suffix.lower(), "image/jpeg")


def _coerce_text(content: Iterable) -> str:
    """Normalize response content that may arrive as a list of parts."""
    parts = []
    for entry in content:
        if isinstance(entry, str):
            parts.append(entry)
            continue
        text = getattr(entry, "text", None)
        if text:
            parts.append(text)
            continue
        if isinstance(entry, dict) and entry.get("type") == "text":
            parts.append(entry.get("text", ""))
    return "".join(parts)


class ImageAnalyzerAgent:
    """Helper around OpenAI-compatible chat completions for vision analysis."""

    def __init__(
        self,
        provider_name: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        """
        Initialize the image analyzer agent.

        Args:
            provider_name: LLM provider (must be OpenAI-compatible: openai, huggingface)
            api_key: API key for the provider
            model: Model identifier
            base_url: Override the OpenAI-compatible base URL
            temperature: Sampling temperature
        """
        detected_provider = provider_name or ProviderFactory.auto_detect_provider() or "openai"
        if detected_provider not in SUPPORTED_OPENAI_COMPATIBLE_PROVIDERS:
            raise ValueError(
                f"Provider '{detected_provider}' is not supported for image analysis. "
                "Use an OpenAI-compatible provider such as 'openai' or 'huggingface'."
            )

        provider_config = get_provider_config_from_env(detected_provider)
        env_api_key = os.getenv("IMAGE_ANALYSER_API_KEY")
        env_model = os.getenv("IMAGE_ANALYSER_MODEL")
        env_base_url = os.getenv("IMAGE_ANALYSER_BASE_URL")

        self.provider_name = detected_provider
        self.api_key = api_key or env_api_key or provider_config.get("api_key") or "not-needed"
        self.model = model or env_model or provider_config.get("model") or DEFAULT_MODEL
        self.base_url = base_url or env_base_url or provider_config.get("base_url") or DEFAULT_BASE_URL
        self.temperature = temperature

        try:
            self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        except TypeError as exc:
            raise RuntimeError("Installed openai package does not support custom base_url") from exc

    def analyze_image(self, image_path: Path, prompt: str) -> str:
        """
        Send the image and prompt to the configured LLM and return the response.

        Args:
            image_path: Path to the image file
            prompt: Instruction for the vision model

        Returns:
            Analysis result from the vision model
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        data_url = f"data:{_detect_mime_type(image_path)};base64,{encode_image_to_base64(image_path)}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )

        message_content = response.choices[0].message.content
        if isinstance(message_content, list):
            text = _coerce_text(message_content)
        else:
            text = message_content or ""
        return text.strip()


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Analyze images with an OpenAI-compatible vision model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  wikipedia-agent-image photo.jpg\n"
            "  wikipedia-agent-image photo.jpg --prompt 'What objects do you see?'\n"
            "  wikipedia-agent-image photo.jpg --base-url http://localhost:1234/v1/\n"
        ),
    )

    parser.add_argument("image_path", help="Path to the image file to analyze")
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Instruction passed to the vision model (default: describe the image)",
    )
    parser.add_argument(
        "--provider",
        choices=sorted(SUPPORTED_OPENAI_COMPATIBLE_PROVIDERS),
        help="LLM provider to use (requires OpenAI-compatible API)",
    )
    parser.add_argument("--model", help="Model identifier, defaults to qwen/qwen3-vl-4b")
    parser.add_argument("--api-key", help="API key for the provider if required")
    parser.add_argument(
        "--base-url",
        help="Override the OpenAI-compatible base URL (default: http://localhost:1234/v1/)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature for generation (default: 0.0)",
    )

    return parser


def main() -> None:
    """CLI entry point for running the image analyzer agent."""
    load_dotenv()
    parser = build_arg_parser()
    args = parser.parse_args()

    image_path = Path(args.image_path)

    try:
        agent = ImageAnalyzerAgent(
            provider_name=args.provider,
            api_key=args.api_key,
            model=args.model,
            base_url=args.base_url,
            temperature=args.temperature,
        )

        print(f"Using provider '{agent.provider_name}' with model '{agent.model}'")
        print(f"Endpoint: {agent.base_url}\n")

        result = agent.analyze_image(image_path=image_path, prompt=args.prompt)

        print("=" * 60)
        print("VISION ANALYSIS RESULT:")
        print("=" * 60)
        print(result or "(no response)")
        print("=" * 60)

    except FileNotFoundError as error:
        print(f"Error: {error}")
        sys.exit(1)
    except ValueError as error:
        print(f"Error: {error}")
        sys.exit(1)
    except Exception as error:  # pragma: no cover - defensive catch for CLI
        print(f"Error analyzing image: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
