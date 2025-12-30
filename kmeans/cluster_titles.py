"""
Cluster Title Generation Module

Supports multiple backends for generating cluster titles:
- Ollama (llama3, mistral, phi3) - Recommended for quality
- Transformers (Flan-T5, etc.) - Lightweight fallback

Provides 1-word, 3-word, and 5-word summaries of verse clusters.
"""

import re
import requests
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class OllamaBackend:
    # Generate titles using Ollama (llama3, mistral, phi3, etc.). Requires Ollama to be installed and running.

    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434"):
        # Initialize Ollama backend.
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"

        # Test connection
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                print(f"✓ Connected to Ollama (model: {model})")
            else:
                print(
                    f"⚠️  Ollama connection warning: HTTP {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Cannot connect to Ollama at {base_url}")
            print(f"   Make sure Ollama is running: 'ollama serve'")
            print(f"   Error: {e}")

    def _create_prompt(self, verses: List[str], word_count: int) -> str:
        # Create prompt optimized for Ollama models.
        # Limit verses to avoid token overflow
        if len(verses) > 20:
            verses_sample = verses[:15] + ["..."] + verses[-5:]
        else:
            verses_sample = verses

        verses_text = '\n'.join(f"- {v}" for v in verses_sample)

        prompt = f"""You are analyzing Bible verses from the Book of Proverbs. Your task is to create a concise thematic title for this cluster of verses.

Verses:
{verses_text}

Task: Generate a {word_count}-word title that captures the main theme.

Rules:
- Use EXACTLY {word_count} word(s)
- Be descriptive and meaningful
- Do not use articles (a, an, the) for 1-word titles

Output ONLY the title, nothing else.

Title:"""

        return prompt

    def _clean_title(self, text: str) -> str:
        text = text.strip()
        # Remove quotes
        text = re.sub(r'^["\']|["\']$', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove trailing punctuation
        text = re.sub(r'[,;:!?\.]+$', '', text)
        # Remove leading/trailing whitespace again
        text = text.strip()
        return text

    def _count_words(self, text: str) -> int:
        """Count words in text."""
        return len(text.split())

    def _enforce_word_count(self, text: str, target_count: int) -> str:
        """Enforce exact word count."""
        words = text.split()
        if len(words) > target_count:
            return ' '.join(words[:target_count])
        return text

    def generate_title(
        self,
        verses: List[str],
        word_count: int,
        temperature: float = 0.3,
        max_attempts: int = 3
    ) -> str:
        # Generate a cluster title with strict word count.
        prompt = self._create_prompt(verses, word_count)

        for attempt in range(max_attempts):
            try:
                response = requests.post(
                    self.api_url,
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "num_predict": 20,  # Limit output length
                        }
                    },
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    title = result.get("response", "").strip()

                    # Clean and enforce word count
                    title = self._clean_title(title)
                    title = self._enforce_word_count(title, word_count)

                    # Check if we achieved target
                    if self._count_words(title) == word_count and title:
                        return title.title()

                    # Adjust temperature for next attempt
                    temperature = min(temperature + 0.1, 0.8)
                else:
                    print(f"⚠️  Ollama API error: HTTP {response.status_code}")

            except requests.exceptions.RequestException as e:
                print(f"⚠️  Ollama request failed: {e}")
                break

        # Fallback: create simple title from first meaningful word
        print(
            f"   Warning: Could not generate {word_count}-word title, using fallback")
        return self._create_fallback_title(verses, word_count)

    def _create_fallback_title(self, verses: List[str], word_count: int) -> str:
        """Create a fallback title if generation fails."""
        # Extract common meaningful words
        all_words = []
        for verse in verses[:5]:  # Sample first few verses
            words = verse.split()
            all_words.extend([w.strip('.,;:!?"') for w in words if len(w) > 3])

        # Return first N words
        if len(all_words) >= word_count:
            return ' '.join(all_words[:word_count]).title()
        return "Proverbs Theme"


class TransformersBackend:
    # Generate titles using HuggingFace Transformers (Flan-T5, etc.). Fallback option when Ollama is not available.

    def __init__(self, model_name: str = "google/flan-t5-small"):
        # Initialize transformers backend.
        print(f"Loading Transformers model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Model loaded on device: {self.device}")

    def _create_prompt(self, verses: List[str], word_count: int) -> str:
        """Create prompt for transformers model."""
        if len(verses) > 20:
            verses_sample = verses[:15] + ["..."] + verses[-5:]
        else:
            verses_sample = verses

        verses_text = '\n'.join(f"- {v[:100]}" for v in verses_sample)

        prompt = f"""Summarize the following Bible verses in EXACTLY {word_count} word(s).
Be precise and capture the main theme. Output ONLY the title, no explanation.

Verses:
{verses_text}

Title ({word_count} words):"""

        return prompt

    def _clean_title(self, text: str) -> str:
        """Clean generated title."""
        text = text.strip()
        text = re.sub(r'^["\']|["\']$', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[,;:!?]+$', '', text)
        return text.strip()

    def _count_words(self, text: str) -> int:
        """Count words in text."""
        return len(text.split())

    def _enforce_word_count(self, text: str, target_count: int) -> str:
        """Enforce exact word count."""
        words = text.split()
        return ' '.join(words[:target_count]) if len(words) > target_count else text

    def generate_title(
        self,
        verses: List[str],
        word_count: int,
        temperature: float = 0.3,
        max_attempts: int = 3
    ) -> str:
        # Generate a cluster title with strict word count.
        prompt = self._create_prompt(verses, word_count)

        for attempt in range(max_attempts):
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=word_count * 3 + 5,
                    min_length=word_count,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    top_p=0.9,
                    num_beams=3 if word_count > 1 else 1
                )

            title = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            title = self._clean_title(title)
            title = self._enforce_word_count(title, word_count)

            if self._count_words(title) == word_count:
                return title.title()

            temperature = min(temperature + 0.1, 0.9)

        # Fallback
        words = title.split()[:word_count]
        return ' '.join(words).title()


class ClusterTitleGenerator:
    # Unified interface for generating cluster titles with multiple backend support.

    def __init__(self, backend: str = 'ollama', model: str = None):
        # Initialize title generator with specified backend.
        self.backend_name = backend

        if backend == 'ollama':
            model = model or 'llama3'
            self.backend = OllamaBackend(model=model)
        elif backend == 'transformers':
            model = model or 'google/flan-t5-small'
            self.backend = TransformersBackend(model_name=model)
        else:
            raise ValueError(
                f"Unknown backend: {backend}. Use 'ollama' or 'transformers'")

    def generate_cluster_title(
        self,
        verses: List[str],
        word_count: int,
        temperature: float = 0.3
    ) -> str:
        """Generate a single title."""
        return self.backend.generate_title(verses, word_count, temperature)

    def generate_all_titles(
        self,
        verses: List[str],
        word_counts: List[int] = [1, 3, 4, 5]
    ) -> Dict[str, str]:
        # Generate titles for all word counts.
        titles = {}
        for word_count in word_counts:
            key = f"title_{word_count}word{'s' if word_count > 1 else ''}"
            titles[key] = self.generate_cluster_title(verses, word_count)

        return titles


# Module-level singleton for efficiency
_generator: Optional[ClusterTitleGenerator] = None


def initialize_title_generator(
    backend: str = 'ollama',
    model: str = None
) -> ClusterTitleGenerator:
    # Initialize or return the global title generator.
    global _generator
    if _generator is None:
        _generator = ClusterTitleGenerator(backend=backend, model=model)
    return _generator


def generate_titles_for_cluster(
    verses: List[str],
    word_counts: List[int] = [1, 3, 5],
    backend: str = 'ollama',
    model: str = None
) -> Dict[str, str]:
    # Convenience function to generate titles.
    generator = initialize_title_generator(backend=backend, model=model)
    return generator.generate_all_titles(verses, word_counts)
